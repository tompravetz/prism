"""
PIDGIN Evaluator.

Computes all evaluation metrics defined in the design document:

    CDS (Concept Discriminativeness Score):
        BDM-10  Board-to-Description Matching, 10-way (hard distractors)
        DBM-64  Description-to-Board Matching, 64-way
        BPA     Behavioral Prediction Accuracy
        SNA     Semantic Neighbor Agreement

    Supporting metrics:
        DC      Description Consistency across seeds
        AbC     Ablation Correlation (descriptions vs. causal importance)

    Functional metric:
        TQDA    Transfer Quality via Description Alignment
                (run separately; hooks into PRISM's transfer pipeline)

    Floor / ceiling baselines for BDM-10:
        Generic template performance  (effective floor)
        Max-info description          (empirical ceiling)

All LLM evaluation calls use a FRESH engine instance (no shared context with
the optimizer) to prevent self-consistency bias.

Usage:
    ev = Evaluator(model="claude-sonnet-4-20250514")
    metrics = ev.evaluate_all(evidence, descriptions)
"""

import os
import sys
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.stats import spearmanr

from sentence_transformers import SentenceTransformer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .data_collector import ConceptEvidence, _render_board
from .concept_prompter import build_max_info_description

# We import the Anthropic client directly for judge calls — a fresh instance,
# separate from any TextGrad engine that may be alive.
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Results dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BDMResult:
    """Board-to-Description Matching results."""
    accuracy: float            # Fraction correct across all queries
    per_concept_accuracy: Dict[int, float] = field(default_factory=dict)
    n_queries_total: int = 0
    n_correct_total: int = 0
    condition: str = "textgrad"  # "generic", "single_pass", "textgrad", "max_info"


@dataclass
class DBMResult:
    """Description-to-Board Matching (64-way) results."""
    accuracy: float
    per_concept_accuracy: Dict[int, float] = field(default_factory=dict)
    n_queries_total: int = 0


@dataclass
class BPAResult:
    """Behavioral Prediction Accuracy results."""
    action_region_accuracy: float    # corner / edge / interior (3-class, chance=33%)
    importance_accuracy: float       # low / medium / high (3-class, chance=33%)
    mean_accuracy: float


@dataclass
class SNAResult:
    """Semantic Neighbor Agreement."""
    spearman_r: float          # Spearman rank correlation between desc-sim and centroid-sim
    spearman_p: float


@dataclass
class DCResult:
    """Description Consistency across seeds."""
    mean_pairwise_similarity: float
    per_concept_mean_similarity: Dict[int, float] = field(default_factory=dict)


@dataclass
class AbCResult:
    """Ablation Correlation."""
    spearman_r: float          # Correlation: predicted importance vs. actual ablation rank
    spearman_p: float
    n_concepts: int = 0


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    condition: str
    bdm10: Optional[BDMResult] = None
    dbm64: Optional[DBMResult] = None
    bpa: Optional[BPAResult] = None
    sna: Optional[SNAResult] = None
    dc: Optional[DCResult] = None
    abc: Optional[AbCResult] = None
    cds_score: Optional[float] = None   # Combined CDS (see design doc §5.1.6)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        def _safe(x):
            if x is None:
                return None
            if hasattr(x, "__dict__"):
                return {k: _safe(v) for k, v in x.__dict__.items()}
            if isinstance(x, dict):
                return {str(k): _safe(v) for k, v in x.items()}
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            return x

        return {
            "condition": self.condition,
            "bdm10": _safe(self.bdm10),
            "dbm64": _safe(self.dbm64),
            "bpa": _safe(self.bpa),
            "sna": _safe(self.sna),
            "dc": _safe(self.dc),
            "abc": _safe(self.abc),
            "cds_score": self.cds_score,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs all PIDGIN evaluation metrics.

    Uses a fresh Anthropic client for judge calls (separate from TextGrad's
    engine) to prevent self-consistency bias.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        embed_model: str = "all-MiniLM-L6-v2",
        n_bdm_queries: int = 20,   # Held-out board states per concept for BDM-10
        n_dbm_queries: int = 10,   # States per concept for DBM-64
        n_distractors_bdm: int = 9,  # Hard distractors for BDM-10 (so 10-way total)
    ):
        self.model = model
        self.n_bdm_queries = n_bdm_queries
        self.n_dbm_queries = n_dbm_queries
        self.n_distractors = n_distractors_bdm

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        # Fresh client — no shared state with the TextGrad optimizer
        self._client = Anthropic(api_key=api_key)

        print(f"Loading sentence embedder ({embed_model})...")
        self._embedder = SentenceTransformer(embed_model)
        print("Embedder ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) embedding matrix."""
        return self._embedder.encode(texts, convert_to_numpy=True)

    def _judge_call(self, prompt: str, system: str = "") -> str:
        """Make a single Claude judge call."""
        msgs = [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.model, messages=msgs, max_tokens=512, temperature=0)
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text.strip()

    # ------------------------------------------------------------------
    # BDM-10: Board-to-Description Matching
    # ------------------------------------------------------------------

    def run_bdm10(
        self,
        evidence: Dict[int, ConceptEvidence],
        descriptions: Dict[int, str],
        condition: str = "textgrad",
        rng_seed: int = 0,
    ) -> BDMResult:
        """
        Run BDM-10 for a set of descriptions.

        For each concept k:
            1. Select the 9 hardest distractors (nearest neighbors by centroid).
            2. For each held-out board state, ask the judge to pick the correct
               description from the 10 candidates.

        Args:
            evidence: Dict of ConceptEvidence.
            descriptions: Dict of description strings keyed by concept_id.
            condition: Label for this run (e.g. "generic", "textgrad", "max_info").
            rng_seed: Random seed for distractor selection and shuffling.

        Returns:
            BDMResult with per-concept and aggregate accuracy.
        """
        rng = np.random.default_rng(rng_seed)
        per_concept: Dict[int, float] = {}
        total_correct = 0
        total_queries = 0

        concept_ids = sorted(evidence.keys())

        for k in concept_ids:
            ev = evidence[k]
            if not ev.holdout_examples:
                per_concept[k] = float("nan")
                continue

            # Hard distractors: nearest neighbors by centroid similarity
            # (as encoded in ev.neighbor_ids, already sorted by similarity)
            distractor_ids = ev.neighbor_ids[:self.n_distractors]
            # Pad with random concepts if fewer than n_distractors neighbors
            if len(distractor_ids) < self.n_distractors:
                remaining = [c for c in concept_ids if c != k and c not in distractor_ids]
                extra = rng.choice(
                    remaining,
                    size=self.n_distractors - len(distractor_ids),
                    replace=False,
                ).tolist()
                distractor_ids = distractor_ids + [int(x) for x in extra]

            # Build the 10 candidate descriptions (shuffled)
            candidates = [(k, descriptions.get(k, ""))] + [
                (d, descriptions.get(d, "")) for d in distractor_ids
            ]
            correct_pos_before_shuffle = 0  # index of correct in candidates

            concept_correct = 0
            n_queries = min(self.n_bdm_queries, len(ev.holdout_examples))

            for ex_idx in range(n_queries):
                ex = ev.holdout_examples[ex_idx]

                # Shuffle candidates
                order = rng.permutation(len(candidates)).tolist()
                shuffled = [candidates[i] for i in order]
                correct_shuffled_idx = order.index(correct_pos_before_shuffle)

                # Format the prompt
                prompt = _bdm_prompt(ex.ascii_board, shuffled, k)
                response = self._judge_call(prompt, system=_BDM_SYSTEM)

                predicted = _parse_bdm_response(response, len(candidates))
                if predicted == correct_shuffled_idx:
                    concept_correct += 1

            acc = concept_correct / n_queries if n_queries > 0 else float("nan")
            per_concept[k] = acc
            total_correct += concept_correct
            total_queries += n_queries

        overall = total_correct / total_queries if total_queries > 0 else 0.0
        return BDMResult(
            accuracy=overall,
            per_concept_accuracy=per_concept,
            n_queries_total=total_queries,
            n_correct_total=total_correct,
            condition=condition,
        )

    # ------------------------------------------------------------------
    # DBM-64: Description-to-Board Matching
    # ------------------------------------------------------------------

    def run_dbm64(
        self,
        evidence: Dict[int, ConceptEvidence],
        descriptions: Dict[int, str],
        top_k_concepts: int = 20,
        rng_seed: int = 0,
    ) -> DBMResult:
        """
        Run DBM-64 for the top-K most ablation-important concepts.

        For each concept k (in top_k by ablation rank):
            Present the judge with all 64 descriptions (shuffled) + one board state.
            Ask it to identify the matching description.

        Args:
            top_k_concepts: Number of concepts to run this harder task on.
        """
        rng = np.random.default_rng(rng_seed)
        concept_ids = sorted(evidence.keys())

        # Select top-K by ablation rank (lower rank = more important)
        def _abl_rank(k):
            ev = evidence[k]
            return ev.ablation_rank if ev.ablation_rank is not None else 999

        selected = sorted(concept_ids, key=_abl_rank)[:top_k_concepts]

        all_descriptions_ordered = [(k, descriptions.get(k, "")) for k in concept_ids]

        per_concept: Dict[int, float] = {}
        total_correct = 0
        total_queries = 0

        for k in selected:
            ev = evidence[k]
            if not ev.holdout_examples:
                per_concept[k] = float("nan")
                continue

            correct_global_idx = concept_ids.index(k)
            concept_correct = 0
            n_queries = min(self.n_dbm_queries, len(ev.holdout_examples))

            for ex_idx in range(n_queries):
                ex = ev.holdout_examples[ex_idx]

                # Shuffle all 64 descriptions
                order = rng.permutation(len(concept_ids)).tolist()
                shuffled = [all_descriptions_ordered[i] for i in order]
                correct_shuffled_idx = order.index(correct_global_idx)

                prompt = _dbm_prompt(ex.ascii_board, shuffled)
                response = self._judge_call(prompt, system=_DBM_SYSTEM)

                predicted = _parse_dbm_response(response, len(concept_ids))
                if predicted == correct_shuffled_idx:
                    concept_correct += 1

            acc = concept_correct / n_queries if n_queries > 0 else float("nan")
            per_concept[k] = acc
            total_correct += concept_correct
            total_queries += n_queries

        overall = total_correct / total_queries if total_queries > 0 else 0.0
        return DBMResult(
            accuracy=overall,
            per_concept_accuracy=per_concept,
            n_queries_total=total_queries,
        )

    # ------------------------------------------------------------------
    # BPA: Behavioral Prediction Accuracy
    # ------------------------------------------------------------------

    def run_bpa(
        self,
        evidence: Dict[int, ConceptEvidence],
        descriptions: Dict[int, str],
    ) -> BPAResult:
        """
        Behavioral Prediction Accuracy.

        Given ONLY the description (no board states), ask the judge to predict:
            (a) Majority action region: corner / edge / interior  (3-class, chance=33%)
            (b) Strategic importance class: low / medium / high  (3-class, chance=33%)
        """
        region_correct = 0
        importance_correct = 0
        n = 0

        for k, desc in descriptions.items():
            ev = evidence.get(k)
            if ev is None:
                continue

            # Ground truth: majority action region
            probs = ev.action_probs
            corner_p = float(np.sum(probs[_corner_actions()]))
            edge_p = float(np.sum(probs[_edge_actions()]))
            interior_p = 1.0 - corner_p - edge_p - float(probs[49])
            gt_region = max(
                ("corner", corner_p), ("edge", edge_p), ("interior", interior_p),
                key=lambda x: x[1]
            )[0]

            # Ground truth: importance class
            rank = ev.ablation_rank
            if rank is None:
                gt_importance = "medium"
            elif rank <= 16:
                gt_importance = "high"
            elif rank <= 48:
                gt_importance = "medium"
            else:
                gt_importance = "low"

            prompt = _bpa_prompt(desc)
            response = self._judge_call(prompt, system=_BPA_SYSTEM)

            pred_region, pred_importance = _parse_bpa_response(response)
            if pred_region == gt_region:
                region_correct += 1
            if pred_importance == gt_importance:
                importance_correct += 1
            n += 1

        region_acc = region_correct / n if n > 0 else 0.0
        importance_acc = importance_correct / n if n > 0 else 0.0
        return BPAResult(
            action_region_accuracy=region_acc,
            importance_accuracy=importance_acc,
            mean_accuracy=(region_acc + importance_acc) / 2,
        )

    # ------------------------------------------------------------------
    # SNA: Semantic Neighbor Agreement
    # ------------------------------------------------------------------

    def run_sna(
        self,
        evidence: Dict[int, ConceptEvidence],
        descriptions: Dict[int, str],
    ) -> SNAResult:
        """
        Semantic Neighbor Agreement.

        Spearman rank correlation between:
            S^desc[i,j] = embedding cosine similarity between desc_i and desc_j
            S^cent[i,j] = centroid cosine similarity (already in evidence)
        """
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        concept_ids = sorted(descriptions.keys())
        descs = [descriptions[k] for k in concept_ids]
        centroids = np.stack([evidence[k].centroid for k in concept_ids])

        # Description similarity matrix
        desc_embs = self._embed(descs)
        desc_sim = cos_sim(desc_embs, desc_embs)

        # Centroid similarity matrix
        cent_sim = cos_sim(centroids, centroids)

        # Extract upper triangle (exclude diagonal)
        n = len(concept_ids)
        idx = np.triu_indices(n, k=1)
        desc_flat = desc_sim[idx]
        cent_flat = cent_sim[idx]

        r, p = spearmanr(desc_flat, cent_flat)
        return SNAResult(spearman_r=float(r), spearman_p=float(p))

    # ------------------------------------------------------------------
    # DC: Description Consistency
    # ------------------------------------------------------------------

    def compute_dc(
        self,
        descriptions_by_seed: Dict[int, Dict[int, str]],
    ) -> DCResult:
        """
        Description Consistency across multiple seeds.

        Args:
            descriptions_by_seed: Dict mapping seed_id → {concept_id: description}.
                                   Must contain at least 2 seeds.

        Returns:
            DCResult with mean pairwise cosine similarity of descriptions
            for the same concept across seeds.
        """
        seeds = list(descriptions_by_seed.keys())
        assert len(seeds) >= 2, "Need at least 2 seeds to compute DC."

        all_concept_ids = sorted(descriptions_by_seed[seeds[0]].keys())
        per_concept: Dict[int, float] = {}

        for k in all_concept_ids:
            seed_descs = [
                descriptions_by_seed[s][k]
                for s in seeds
                if k in descriptions_by_seed[s]
            ]
            if len(seed_descs) < 2:
                per_concept[k] = float("nan")
                continue

            embs = self._embed(seed_descs)
            # Pairwise cosine similarity
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sim = float(
                        np.dot(embs[i], embs[j])
                        / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-9)
                    )
                    sims.append(sim)
            per_concept[k] = float(np.mean(sims))

        valid = [v for v in per_concept.values() if not np.isnan(v)]
        return DCResult(
            mean_pairwise_similarity=float(np.mean(valid)) if valid else 0.0,
            per_concept_mean_similarity=per_concept,
        )

    # ------------------------------------------------------------------
    # AbC: Ablation Correlation
    # ------------------------------------------------------------------

    def run_abc(
        self,
        evidence: Dict[int, ConceptEvidence],
        descriptions: Dict[int, str],
    ) -> AbCResult:
        """
        Ablation Correlation.

        Spearman rank correlation between:
            predicted importance (extracted from 'Strategic Importance' field: low=1/med=2/high=3)
            actual ablation rank (lower rank = more impactful)
        """
        predicted = []
        actual_ranks = []

        for k, desc in descriptions.items():
            ev = evidence.get(k)
            if ev is None or ev.ablation_rank is None:
                continue

            importance = _extract_importance_from_description(desc)
            if importance is None:
                continue

            predicted.append(importance)
            actual_ranks.append(ev.ablation_rank)

        if len(predicted) < 5:
            return AbCResult(spearman_r=float("nan"), spearman_p=float("nan"), n_concepts=0)

        # Flip rank so higher rank_score = more important (rank 1 → score 64)
        n_concepts = len(actual_ranks)
        rank_scores = [n_concepts + 1 - r for r in actual_ranks]

        r, p = spearmanr(predicted, rank_scores)
        return AbCResult(spearman_r=float(r), spearman_p=float(p), n_concepts=len(predicted))

    # ------------------------------------------------------------------
    # Combined CDS score
    # ------------------------------------------------------------------

    def compute_cds(self, bdm10: BDMResult, dbm64: DBMResult,
                    bpa: BPAResult, sna: SNAResult) -> float:
        """
        Combined Concept Discriminativeness Score (see design doc §5.1.6).

        Normalises each metric from [chance, 1.0] → [0, 1.0]:
            BDM-10: chance = 0.10
            DBM-64: chance = 0.016
            BPA:    chance = 0.33 (mean of region and importance)
            SNA:    range [0, 1]

        CDS = 0.4*BDM + 0.3*DBM + 0.15*BPA + 0.15*SNA
        """
        def norm(val, chance):
            if val is None or np.isnan(val):
                return 0.0
            return max(0.0, (val - chance) / (1.0 - chance))

        bdm_n = norm(bdm10.accuracy if bdm10 else None, 0.10)
        dbm_n = norm(dbm64.accuracy if dbm64 else None, 0.016)
        bpa_n = norm(bpa.mean_accuracy if bpa else None, 0.33)
        sna_n = max(0.0, float(sna.spearman_r) if sna else 0.0)

        return 0.40 * bdm_n + 0.30 * dbm_n + 0.15 * bpa_n + 0.15 * sna_n

    # ------------------------------------------------------------------
    # Convenience: run all metrics at once
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        evidence: Dict[int, ConceptEvidence],
        descriptions: Dict[int, str],
        condition: str = "textgrad",
        run_dbm: bool = True,
        run_bpa: bool = True,
        run_sna: bool = True,
        run_abc: bool = True,
        verbose: bool = True,
    ) -> EvaluationReport:
        """
        Run the full evaluation suite.

        Args:
            evidence: All concept evidence.
            descriptions: Descriptions to evaluate.
            condition: Label for this condition ("generic", "single_pass", etc.)
            run_dbm / run_bpa / run_sna / run_abc: Toggle individual metrics.
                DBM-64 involves many LLM calls; disable during development.

        Returns:
            EvaluationReport with all metrics populated.
        """
        report = EvaluationReport(condition=condition)

        if verbose:
            print(f"\nEvaluating condition: {condition}")

        if verbose:
            print("  Running BDM-10...")
        report.bdm10 = self.run_bdm10(evidence, descriptions, condition=condition)
        if verbose:
            print(f"  BDM-10 accuracy: {report.bdm10.accuracy:.1%}")

        if run_dbm:
            if verbose:
                print("  Running DBM-64 (top-20 concepts)...")
            report.dbm64 = self.run_dbm64(evidence, descriptions)
            if verbose:
                print(f"  DBM-64 accuracy: {report.dbm64.accuracy:.1%}")

        if run_bpa:
            if verbose:
                print("  Running BPA...")
            report.bpa = self.run_bpa(evidence, descriptions)
            if verbose:
                print(f"  BPA region={report.bpa.action_region_accuracy:.1%} "
                      f"importance={report.bpa.importance_accuracy:.1%}")

        if run_sna:
            if verbose:
                print("  Running SNA...")
            report.sna = self.run_sna(evidence, descriptions)
            if verbose:
                print(f"  SNA Spearman r={report.sna.spearman_r:.3f} p={report.sna.spearman_p:.4f}")

        if run_abc:
            if verbose:
                print("  Running AbC...")
            report.abc = self.run_abc(evidence, descriptions)
            if verbose:
                print(f"  AbC Spearman r={report.abc.spearman_r:.3f}")

        report.cds_score = self.compute_cds(
            report.bdm10, report.dbm64, report.bpa, report.sna
        )
        if verbose:
            print(f"  CDS (combined): {report.cds_score:.3f}")

        return report


# ---------------------------------------------------------------------------
# Prompt builders for judge calls
# ---------------------------------------------------------------------------

_BDM_SYSTEM = (
    "You are a Go strategy expert. Given a board position and a list of "
    "numbered descriptions, select the description that BEST matches the "
    "strategic situation on the board. Reply with ONLY the number of the "
    "matching description (e.g. '3'). Nothing else."
)

_DBM_SYSTEM = (
    "You are a Go strategy expert. Given a board position and a numbered list "
    "of 64 strategic descriptions, identify which description BEST matches the "
    "position. Reply with ONLY the number of the matching description. Nothing else."
)

_BPA_SYSTEM = (
    "You are predicting the behavior of a Go AI agent based on a description of its "
    "current strategic concept. Answer the two questions below using only the "
    "description provided."
)


def _bdm_prompt(ascii_board: str, candidates: List[Tuple[int, str]], correct_k: int) -> str:
    lines = [
        "BOARD POSITION:",
        ascii_board,
        "",
        "CANDIDATE DESCRIPTIONS (one of these describes this position):",
    ]
    for i, (cid, desc) in enumerate(candidates, start=1):
        # Show descriptions but NOT concept IDs (would leak ground truth)
        short = desc.strip()[:300]
        lines.append(f"\n{i}. {short}")
    lines.append("\nWhich number (1-10) best matches the board position above?")
    return "\n".join(lines)


def _dbm_prompt(ascii_board: str, candidates: List[Tuple[int, str]]) -> str:
    lines = [
        "BOARD POSITION:",
        ascii_board,
        "",
        "CANDIDATE DESCRIPTIONS:",
    ]
    for i, (cid, desc) in enumerate(candidates, start=1):
        short = desc.strip()[:200]
        lines.append(f"\n{i}. {short}")
    lines.append(f"\nWhich number (1-{len(candidates)}) best matches the board position?")
    return "\n".join(lines)


def _bpa_prompt(description: str) -> str:
    return (
        f"Description:\n{description}\n\n"
        "Answer these two questions using ONLY the description above:\n\n"
        "Q1. Where does this agent MOST OFTEN play? Answer with one word: "
        "'corner', 'edge', or 'interior'.\n"
        "Q2. How strategically important is this concept? Answer with one word: "
        "'low', 'medium', or 'high'.\n\n"
        "Format your answer EXACTLY as:\n"
        "Q1: [answer]\n"
        "Q2: [answer]"
    )


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_bdm_response(response: str, n_candidates: int) -> int:
    """Parse judge response to a 0-indexed candidate index. Returns -1 on failure."""
    import re
    m = re.search(r"\b(\d+)\b", response.strip())
    if m:
        idx = int(m.group(1)) - 1  # Convert 1-indexed to 0-indexed
        if 0 <= idx < n_candidates:
            return idx
    return -1  # Parse failure


def _parse_dbm_response(response: str, n_candidates: int) -> int:
    """Same as _parse_bdm_response."""
    return _parse_bdm_response(response, n_candidates)


def _parse_bpa_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse BPA response into (region, importance)."""
    import re
    region = None
    importance = None

    q1_m = re.search(r"Q1[:\s]+(\w+)", response, re.IGNORECASE)
    if q1_m:
        val = q1_m.group(1).lower()
        if val in ("corner", "edge", "interior"):
            region = val

    q2_m = re.search(r"Q2[:\s]+(\w+)", response, re.IGNORECASE)
    if q2_m:
        val = q2_m.group(1).lower()
        if val in ("low", "medium", "high"):
            importance = val

    return region, importance


def _extract_importance_from_description(desc: str) -> Optional[int]:
    """
    Extract the Strategic Importance rating from a PIDGIN description.
    Returns 1=low, 2=medium, 3=high, or None if not found.
    """
    import re
    m = re.search(r"Strategic Importance\s*:\s*(low|medium|high)", desc, re.IGNORECASE)
    if m:
        return {"low": 1, "medium": 2, "high": 3}[m.group(1).lower()]
    return None


# ---------------------------------------------------------------------------
# Position class helpers
# ---------------------------------------------------------------------------

def _corner_actions() -> np.ndarray:
    """Indices of corner-adjacent actions (2x2 corners of 7x7)."""
    indices = []
    for r in range(2):
        for c in range(2):
            indices += [r * 7 + c, r * 7 + (6 - c), (6 - r) * 7 + c, (6 - r) * 7 + (6 - c)]
    return np.array(sorted(set(indices)))


def _edge_actions() -> np.ndarray:
    """Indices of edge (non-corner border) actions."""
    corner_set = set(_corner_actions().tolist())
    indices = []
    for r in range(7):
        for c in range(7):
            a = r * 7 + c
            if (r == 0 or r == 6 or c == 0 or c == 6) and a not in corner_set:
                indices.append(a)
    return np.array(indices)
