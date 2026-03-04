"""
PIDGIN Concept Prompter.

Formats a ConceptEvidence into the context block that the LLM sees during
TextGrad optimization. This is the frozen Context Variable — it contains all
the factual evidence about a concept and does not change during optimization.

The format is designed to be:
    1. Human-readable: an LLM (or a Go player) can parse it without code.
    2. Information-dense: every field contributes something the LLM can use.
    3. Contrastive: neighbor context helps the LLM write specific descriptions.

Usage:
    from pidgin.concept_prompter import build_context
    context_text = build_context(evidence[k], all_evidence=evidence, descriptions=None)
"""

import numpy as np
from typing import Dict, List, Optional
from .data_collector import ConceptEvidence, BoardExample


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_context(
    ev: ConceptEvidence,
    all_evidence: Dict[int, "ConceptEvidence"],
    descriptions: Optional[Dict[int, str]] = None,
    n_board_examples: int = 10,
) -> str:
    """
    Build the full context block for concept ev.concept_id.

    Args:
        ev: Evidence for the target concept.
        all_evidence: Evidence for all concepts (used for neighbor context).
        descriptions: Current descriptions for other concepts (if available).
                      Pass None for the first concept; populated iteratively.
        n_board_examples: Number of board examples to include in the context.
                          Fewer than ev.examples uses whatever is available.

    Returns:
        A multiline string ready to be used as the Context Variable in TextGrad.
    """
    parts = []

    parts.append(_section_header(ev.concept_id))
    parts.append(_centroid_stats_section(ev))
    parts.append(_action_distribution_section(ev))
    parts.append(_experiment_results_section(ev))
    parts.append(_neighbor_section(ev, all_evidence, descriptions))
    parts.append(_board_examples_section(ev, n_board_examples))
    parts.append(_task_instructions())

    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_header(concept_id: int) -> str:
    return (
        f"═══════════════════════════════════════\n"
        f"CONCEPT #{concept_id:02d} — EVIDENCE PACKAGE\n"
        f"═══════════════════════════════════════"
    )


def _centroid_stats_section(ev: ConceptEvidence) -> str:
    lines = ["── CENTROID STATISTICS ──"]
    lines.append(f"Concept ID      : {ev.concept_id}")
    lines.append(f"Feature dim     : 128  (encoder output)")
    lines.append(f"Mean activation : {ev.centroid_mean_activation:.4f}")
    lines.append(
        f"Sparsity        : {ev.centroid_sparsity:.1%}  "
        f"(fraction of dims with |activation| < 0.1)"
    )

    # Top active dimensions (rough proxy for what the encoder responds to)
    top_dims = np.argsort(-np.abs(ev.centroid))[:8]
    top_vals = ev.centroid[top_dims]
    top_str = ", ".join(f"dim{d}={v:+.2f}" for d, v in zip(top_dims, top_vals))
    lines.append(f"Most active dims: {top_str}")

    return "\n".join(lines)


def _action_distribution_section(ev: ConceptEvidence) -> str:
    lines = ["── ACTION DISTRIBUTION ──"]

    # Overall stats
    lines.append(
        f"KL from uniform : {ev.kl_from_uniform:.3f} nats  "
        f"(0 = random play; higher = more strategic)"
    )
    lines.append(f"Entropy         : {ev.entropy:.3f} nats  (max possible = {np.log(50):.2f})")
    lines.append(f"Firing frequency: {ev.frequency:.2%} of all game steps")

    # Top actions
    lines.append("")
    lines.append("Top-10 actions by probability:")
    cols = "ABCDEFG"
    for rank, (action_id, prob) in enumerate(ev.top_actions, start=1):
        if action_id == 49:
            pos_str = "Pass      "
        else:
            r, c = divmod(action_id, 7)
            pos_str = f"{cols[c]}{7-r} (row={r+1},col={c+1})"
        lines.append(f"  {rank:2d}. action={action_id:2d} ({pos_str}) : {prob:.3f} = {prob:.1%}")

    # Position class breakdown
    probs = ev.action_probs
    corner_mask = _corner_mask()
    edge_mask = _edge_mask()
    interior_mask = ~(corner_mask | edge_mask)
    interior_mask[49] = False  # pass action

    corner_pct = float(np.sum(probs[corner_mask]))
    edge_pct = float(np.sum(probs[edge_mask]))
    interior_pct = float(np.sum(probs[interior_mask]))
    pass_pct = float(probs[49])

    lines.append("")
    lines.append("Position class breakdown:")
    lines.append(f"  Corner region    : {corner_pct:.1%}")
    lines.append(f"  Edge region      : {edge_pct:.1%}")
    lines.append(f"  Interior region  : {interior_pct:.1%}")
    lines.append(f"  Pass (action 49) : {pass_pct:.1%}")

    # Spatial heatmap
    lines.append("")
    lines.append("Spatial heatmap (% of probability mass per board cell):")
    heatmap_str = _format_heatmap(ev.action_heatmap)
    lines.append(heatmap_str)

    return "\n".join(lines)


def _experiment_results_section(ev: ConceptEvidence) -> str:
    lines = ["── EXPERIMENT RESULTS ──"]

    # Ablation
    if ev.ablation_win_rate_drop is not None:
        drop = ev.ablation_win_rate_drop
        rank = ev.ablation_rank
        trigger = ev.ablation_win_rate_drop  # reuse for now; could add trigger_rate
        sign = "+" if drop > 0 else ""
        lines.append(
            f"Ablation impact : win rate drop = {sign}{drop:.1%}  "
            f"(rank {rank}/64 by impact)"
        )
        importance = (
            "HIGH — top 25% most impactful" if rank is not None and rank <= 16
            else "MEDIUM — middle 50%" if rank is not None and rank <= 48
            else "LOW — bottom 25%"
        )
        lines.append(f"                  → Importance tier: {importance}")
    else:
        lines.append("Ablation impact : (data not available)")

    # Intervention
    if ev.intervention_change_rate is not None:
        lines.append(
            f"Intervention    : overall action change rate = {ev.intervention_change_rate:.1%}"
        )
    else:
        lines.append("Intervention    : (data not available)")

    if ev.concept_specificity is not None:
        lines.append(
            f"Concept specificity: {ev.concept_specificity:.1%}  "
            f"(probability of the most preferred action)"
        )

    return "\n".join(lines)


def _neighbor_section(
    ev: ConceptEvidence,
    all_evidence: Dict[int, "ConceptEvidence"],
    descriptions: Optional[Dict[int, str]],
) -> str:
    lines = ["── NEAREST NEIGHBOR CONCEPTS ──"]
    lines.append(
        "These concepts have the most similar centroid vectors "
        "(cosine similarity). Your description should distinguish "
        f"concept #{ev.concept_id} from these neighbors."
    )
    lines.append("")

    for nid, nsim in zip(ev.neighbor_ids, ev.neighbor_similarities):
        desc_str = "(not yet described)"
        if descriptions and nid in descriptions and descriptions[nid]:
            # Show only the first line (the Name field) to keep context tight
            first_line = descriptions[nid].strip().split("\n")[0]
            desc_str = first_line[:80]

        # Show neighbor's action distribution briefly
        nev = all_evidence.get(nid)
        if nev is not None:
            top_action_id, top_prob = nev.top_actions[0]
            cols = "ABCDEFG"
            if top_action_id == 49:
                top_str = f"Pass ({top_prob:.0%})"
            else:
                r, c = divmod(top_action_id, 7)
                top_str = f"{cols[c]}{7-r} ({top_prob:.0%})"
            lines.append(
                f"  Concept #{nid:02d}  sim={nsim:.3f}  "
                f"top_action={top_str}  desc={desc_str}"
            )
        else:
            lines.append(f"  Concept #{nid:02d}  sim={nsim:.3f}  desc={desc_str}")

    return "\n".join(lines)


def _board_examples_section(ev: ConceptEvidence, n: int) -> str:
    examples = ev.examples[:n]
    n_available = len(ev.examples)

    lines = [
        f"── BOARD EXAMPLES ({len(examples)} of {n_available} available) ──",
        "Each position is a game state where this concept fired.",
        "B=agent's stone (black), W=opponent's stone (white), .=empty. 'Agent played' = action taken.",
        "",
    ]

    for i, ex in enumerate(examples, start=1):
        dist_str = f"dist_to_centroid={ex.dist_to_centroid:.3f}"
        move_str = f"~move {ex.move_number + 1}"
        lines.append(f"Example {i:2d}  ({move_str}, {dist_str}):")
        lines.append(ex.ascii_board)
        lines.append("")

    if n_available < 5:
        lines.append(
            f"⚠ Only {n_available} examples available for this concept. "
            "It fires infrequently during gameplay. Descriptions may be less reliable."
        )

    return "\n".join(lines)


def _task_instructions() -> str:
    return (
        "── DESCRIPTION TASK ──\n"
        "Using the evidence above, write a description of this concept.\n"
        "The description must follow this exact format:\n"
        "\n"
        "Name: [2–5 words identifying the strategic theme]\n"
        "Description: [1–3 sentences characterising the board situation, "
        "game phase, and spatial pattern. Be specific enough to distinguish "
        "this concept from its nearest neighbors above.]\n"
        "Key Actions: [1 sentence describing the agent's preferred moves in this concept]\n"
        "Strategic Importance: [low / medium / high] — [1 sentence justification "
        "based on ablation rank and frequency]\n"
    )


# ---------------------------------------------------------------------------
# Build a max-info description (used as the empirical ceiling baseline)
# ---------------------------------------------------------------------------

def build_max_info_description(ev: ConceptEvidence) -> str:
    """
    Construct a description that contains all raw discriminative statistics
    directly, bypassing semantic compression. Used to estimate the BDM-10
    empirical ceiling in the evaluator.
    """
    top_dims = np.argsort(-np.abs(ev.centroid))[:5]
    top_dim_str = ", ".join(f"dim{d}={ev.centroid[d]:+.3f}" for d in top_dims)

    top_actions_str = ", ".join(
        f"action {a}={p:.1%}" for a, p in ev.top_actions[:5]
    )

    ablation_str = (
        f"ablation rank {ev.ablation_rank}/64 "
        f"(win rate drop {ev.ablation_win_rate_drop:+.1%})"
        if ev.ablation_win_rate_drop is not None
        else "ablation data unavailable"
    )

    neigh_str = ", ".join(
        f"#{nid}(sim={sim:.3f})"
        for nid, sim in zip(ev.neighbor_ids[:3], ev.neighbor_similarities[:3])
    )

    return (
        f"Name: Concept {ev.concept_id} [raw statistics]\n"
        f"Description: Fires on encoder states near centroid with "
        f"mean_activation={ev.centroid_mean_activation:.3f}, "
        f"sparsity={ev.centroid_sparsity:.1%}. "
        f"Top active dimensions: {top_dim_str}. "
        f"Frequency: {ev.frequency:.2%} of game steps. "
        f"Nearest neighbors: {neigh_str}.\n"
        f"Key Actions: Top-5 by probability: {top_actions_str}. "
        f"KL from uniform: {ev.kl_from_uniform:.3f} nats.\n"
        f"Strategic Importance: "
        + ("high" if ev.ablation_rank is not None and ev.ablation_rank <= 16
           else "medium" if ev.ablation_rank is not None and ev.ablation_rank <= 48
           else "low")
        + f" — {ablation_str}. "
        f"Intervention change rate: "
        + (f"{ev.intervention_change_rate:.1%}" if ev.intervention_change_rate else "n/a")
        + "."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corner_mask() -> np.ndarray:
    """Boolean mask over 50 actions: True for corner-adjacent cells."""
    mask = np.zeros(50, dtype=bool)
    # Corners of a 7x7 board (row, col) → action = row*7+col
    corner_cells = set()
    for r in range(2):
        for c in range(2):
            corner_cells.add(r * 7 + c)
            corner_cells.add(r * 7 + (6 - c))
            corner_cells.add((6 - r) * 7 + c)
            corner_cells.add((6 - r) * 7 + (6 - c))
    for a in corner_cells:
        if a < 49:
            mask[a] = True
    return mask


def _edge_mask() -> np.ndarray:
    """Boolean mask over 50 actions: True for edge (non-corner border) cells."""
    mask = np.zeros(50, dtype=bool)
    corner = _corner_mask()
    for r in range(7):
        for c in range(7):
            a = r * 7 + c
            if (r == 0 or r == 6 or c == 0 or c == 6) and not corner[a]:
                mask[a] = True
    return mask


def _format_heatmap(heatmap: np.ndarray) -> str:
    """
    Format a 7×7 probability heatmap as a compact text grid.
    Values shown as percentages (0-99%), aligned.
    """
    cols = "ABCDEFG"
    lines = ["    " + "  ".join(f" {c}" for c in cols)]
    for row in range(7):
        display_row = 7 - row
        cells = []
        for col in range(7):
            pct = int(round(heatmap[row, col] * 100))
            cells.append(f"{pct:2d}")
        lines.append(f" {display_row:2d}  " + "  ".join(cells))
    return "\n".join(lines)
