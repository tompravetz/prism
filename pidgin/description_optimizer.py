"""
PIDGIN Description Optimizer.

Core TextGrad optimization loop. For each concept, iteratively improves
a text description using:
    Variable (description) → TextLoss (LLM critique) → backward() → TGD.step()

The loop runs for up to T iterations or until the description stabilises
(semantic similarity between consecutive iterations exceeds a threshold).

Usage:
    optimizer = DescriptionOptimizer(model="claude-sonnet-4-20250514")
    result = optimizer.optimize_concept(evidence[k], context_text)
    # result.final_description  — the optimized text
    # result.iterations_run     — how many iterations before convergence
    # result.log                — per-iteration description + critique
"""

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import textgrad as tg
from sentence_transformers import SentenceTransformer

from .data_collector import ConceptEvidence


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """You are evaluating a strategic description of a Go board game concept.
You will be given:
  1. A package of evidence: board state examples, action distributions, and experiment results.
  2. A candidate description of the concept.

Your job is to identify specific problems with the description. Evaluate on four criteria:

1. ACCURACY: Does the description match the board examples shown? Would a Go player
   recognise these positions from the description?

2. COMPLETENESS: What important characteristics are missing? Consider: game phase
   (early/mid/late), position type (corner/edge/center), specific strategic concerns
   (territory, capture threats, connection, liberties), and spatial patterns.

3. DIFFERENTIATION: Is the description specific enough to distinguish this concept
   from its nearest neighbors (shown in the evidence)? Generic descriptions like
   "mid-game positional play" that could apply to many concepts are not acceptable.

4. ACTION CONSISTENCY: Does the "Key Actions" field match the actual action distribution
   data in the evidence? If the data shows 72% interior moves but the description says
   "corner placement", that is a factual error.

Be specific and actionable. Do NOT suggest a replacement description — only critique
the current one. Your output will be used as gradient feedback to improve the description."""

INITIAL_DESCRIPTION_TEMPLATE = (
    "Name: Concept {k}\n"
    "Description: A strategic situation in Go 7×7.\n"
    "Key Actions: The agent takes context-dependent actions.\n"
    "Strategic Importance: medium — not yet analysed.\n"
)

OPTIMIZER_CONSTRAINTS = [
    "The output MUST include exactly these four fields: "
    "Name, Description, Key Actions, Strategic Importance.",
    "Name must be 2–5 words. Do not use 'Concept N' as the name.",
    "Description must be 1–3 sentences. Reference specific spatial patterns, "
    "game phase, or strategic goals. Do not be generic.",
    "Key Actions must be 1 sentence describing the agent's preferred moves.",
    "Strategic Importance must start with 'low', 'medium', or 'high' "
    "followed by a dash and a one-sentence justification.",
    "Do not add extra fields or headers beyond the four required.",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    concept_id: int
    final_description: str
    initial_description: str
    iterations_run: int
    converged: bool               # True if stopped due to similarity threshold
    final_loss_text: str          # Last LLM critique
    log: List[Dict] = field(default_factory=list)
    # Each log entry: {"iteration": int, "description": str, "critique": str, "sim_to_prev": float}
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------

class DescriptionOptimizer:
    """
    TextGrad-based optimizer for concept descriptions.

    Each concept is optimized independently. The optimizer maintains no
    state between concepts — call optimize_concept() once per concept.

    Args:
        model: Claude model string (e.g. "claude-sonnet-4-20250514").
        max_iterations: Maximum TextGrad iterations per concept.
        convergence_threshold: Stop if cosine similarity between consecutive
            descriptions exceeds this value (after min_iterations).
        min_iterations: Always run at least this many iterations.
        gradient_memory: Number of past gradients TGD retains.
        embed_model: Sentence transformer model for convergence detection.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
        convergence_threshold: float = 0.98,
        min_iterations: int = 3,
        gradient_memory: int = 3,
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.gradient_memory = gradient_memory

        # Set the global TextGrad backward engine once
        tg.set_backward_engine(model, override=True)

        # Forward engine (same model; fresh instance for each forward call)
        self._forward_engine = tg.get_engine(model)

        # Sentence embedder for convergence detection
        print(f"Loading sentence embedder ({embed_model})...")
        self._embedder = SentenceTransformer(embed_model)
        print("Embedder ready.")

    def _embed(self, text: str) -> np.ndarray:
        """Compute a sentence embedding for convergence detection."""
        return self._embedder.encode([text], convert_to_numpy=True)[0]

    def optimize_concept(
        self,
        ev: ConceptEvidence,
        context_text: str,
        initial_description: Optional[str] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run the TextGrad optimization loop for one concept.

        Args:
            ev: ConceptEvidence for this concept (used for metadata only here;
                context_text already encodes the evidence).
            context_text: The frozen context block from concept_prompter.build_context().
            initial_description: Starting description. Uses generic template if None.
            verbose: Print progress to stdout.

        Returns:
            OptimizationResult with the final description and optimization log.
        """
        k = ev.concept_id
        t_start = time.time()

        if initial_description is None:
            initial_description = INITIAL_DESCRIPTION_TEMPLATE.format(k=k)

        if verbose:
            print(f"\n[Concept {k:02d}] Starting optimization (max {self.max_iterations} iters)")

        # ------------------------------------------------------------------
        # Build the TextGrad graph
        # ------------------------------------------------------------------

        # The frozen context variable: all evidence, no gradients
        context_var = tg.Variable(
            value=context_text,
            role_description=(
                f"Evidence package for concept #{k}: board state examples, "
                "action distribution, ablation results, and neighbor context. "
                "This is fixed and should not be changed."
            ),
            requires_grad=False,
        )

        # The description variable: what we optimise
        description_var = tg.Variable(
            value=initial_description,
            role_description=(
                f"A strategic description of Go concept #{k}, consisting of four fields: "
                "Name, Description, Key Actions, and Strategic Importance. "
                "This description should accurately characterise the board situations "
                "where this concept fires and the strategic actions it produces."
            ),
            requires_grad=True,
        )

        # Loss function: evaluates description quality against the evidence
        loss_fn = tg.TextLoss(
            eval_system_prompt=EVAL_SYSTEM_PROMPT,
            engine=self._forward_engine,
        )

        # Optimizer: TGD with constraints and gradient memory
        optimizer = tg.TGD(
            parameters=[description_var],
            engine=self._forward_engine,
            constraints=OPTIMIZER_CONSTRAINTS,
            gradient_memory=self.gradient_memory,
        )

        # ------------------------------------------------------------------
        # Optimization loop
        # ------------------------------------------------------------------
        log = []
        prev_embed = self._embed(initial_description)
        last_critique = ""
        converged = False

        for t in range(1, self.max_iterations + 1):
            iter_start = time.time()

            # Combine context + description for the loss.
            # The separator is a fresh non-grad variable each iteration
            # (not a persistent predecessor), so the + operator naturally
            # produces requires_grad=True because description_var requires grad.
            sep_var = tg.Variable(
                "\n\nCURRENT DESCRIPTION TO EVALUATE:\n",
                role_description="separator between evidence context and current description",
                requires_grad=False,
            )
            combined = context_var + sep_var + description_var

            # Forward: evaluate the description
            loss = loss_fn(combined)
            critique = loss.get_value()
            last_critique = critique

            # Backward: generate textual gradient
            optimizer.zero_grad()
            loss.backward()

            # Step: update the description
            optimizer.step()

            new_desc = description_var.get_value()

            # Convergence check via semantic similarity
            new_embed = self._embed(new_desc)
            sim = float(
                np.dot(prev_embed, new_embed)
                / (np.linalg.norm(prev_embed) * np.linalg.norm(new_embed) + 1e-9)
            )

            iter_time = time.time() - iter_start
            log.append({
                "iteration": t,
                "description": new_desc,
                "critique": critique[:500],  # truncate for log file size
                "sim_to_prev": sim,
                "iter_seconds": round(iter_time, 1),
            })

            if verbose:
                converge_str = f"(sim={sim:.3f})" if t > 1 else ""
                print(f"  iter {t:2d}/{self.max_iterations}  "
                      f"sim_to_prev={sim:.3f}  {iter_time:.1f}s  {converge_str}")

            prev_embed = new_embed

            if t >= self.min_iterations and sim >= self.convergence_threshold:
                if verbose:
                    print(f"  → Converged at iteration {t} (sim={sim:.3f} ≥ {self.convergence_threshold})")
                converged = True
                break

        final_desc = description_var.get_value()

        if verbose:
            print(f"  Final description:\n{final_desc}")

        return OptimizationResult(
            concept_id=k,
            final_description=final_desc,
            initial_description=initial_description,
            iterations_run=len(log),
            converged=converged,
            final_loss_text=last_critique,
            log=log,
            elapsed_seconds=round(time.time() - t_start, 1),
        )

    def optimize_batch(
        self,
        concepts: List[int],
        evidence: Dict[int, ConceptEvidence],
        context_texts: Dict[int, str],
        existing_descriptions: Optional[Dict[int, str]] = None,
        verbose: bool = True,
    ) -> Dict[int, OptimizationResult]:
        """
        Optimize descriptions for a batch of concepts sequentially.

        Passes already-computed descriptions to concept_prompter for neighbor
        context enrichment as the batch progresses. Concepts are processed in
        order — sort by frequency descending before calling so that common
        concepts (with richer evidence) are described first and can provide
        context to their neighbors.

        Args:
            concepts: List of concept IDs to optimize (in processing order).
            evidence: Dict of ConceptEvidence keyed by concept_id.
            context_texts: Dict of context strings keyed by concept_id.
                           Should be pre-built by the caller using concept_prompter.
            existing_descriptions: Pre-existing descriptions to seed with (optional).
            verbose: Print per-concept progress.

        Returns:
            Dict mapping concept_id → OptimizationResult.
        """
        results: Dict[int, OptimizationResult] = {}
        descriptions: Dict[int, str] = dict(existing_descriptions or {})

        for i, k in enumerate(concepts):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Concept {k:02d}  ({i+1}/{len(concepts)})")
                print(f"{'='*50}")

            ev = evidence[k]
            ctx = context_texts[k]

            result = self.optimize_concept(
                ev=ev,
                context_text=ctx,
                verbose=verbose,
            )
            results[k] = result
            descriptions[k] = result.final_description

        return results
