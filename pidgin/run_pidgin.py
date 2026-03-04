"""
PIDGIN Main Entry Point.

Orchestrates the full PIDGIN pipeline:
    1. Load PRISM artifacts
    2. Collect per-concept evidence (or load from cache)
    3. Build context blocks for all concepts
    4. Run TextGrad optimization (all 64 concepts, sequential)
    5. Run evaluation metrics
    6. Write results to pidgin/results/

Supports checkpointing: if interrupted, re-running resumes from where
it left off. Already-optimized concepts are not re-processed.

Usage:
    # Full run (optimize + evaluate)
    python -m pidgin.run_pidgin

    # Optimize only (skip evaluation)
    python -m pidgin.run_pidgin --skip-eval

    # Single concept (for development / testing)
    python -m pidgin.run_pidgin --concept 0 --max-iter 3 --verbose

    # Evaluate only (requires existing descriptions)
    python -m pidgin.run_pidgin --eval-only

    # Establish floor / ceiling baselines (no optimization)
    python -m pidgin.run_pidgin --baselines-only
"""

import argparse
import json
import os
import sys
import time
import pickle
from typing import Dict, List, Optional

# Ensure UTF-8 output on Windows (cp1252 can't encode box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pidgin.data_collector import DataCollector, ConceptEvidence, save_evidence, load_evidence
from pidgin.concept_prompter import build_context, build_max_info_description
from pidgin.description_optimizer import DescriptionOptimizer, OptimizationResult
from pidgin.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_ROOT, "pidgin", "results")
EVIDENCE_CACHE = os.path.join(RESULTS_DIR, "evidence_cache.pkl")
DESCRIPTIONS_PATH = os.path.join(RESULTS_DIR, "concept_descriptions.json")
LIBRARY_PATH = os.path.join(RESULTS_DIR, "concept_library.json")
OPT_LOGS_DIR = os.path.join(RESULTS_DIR, "optimization_logs")
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results.json")

MODEL = "claude-sonnet-4-20250514"
N_CONCEPTS = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OPT_LOGS_DIR, exist_ok=True)


def _load_checkpoint() -> Dict[int, str]:
    """Load existing descriptions from disk (checkpoint)."""
    if os.path.exists(DESCRIPTIONS_PATH):
        with open(DESCRIPTIONS_PATH) as f:
            raw = json.load(f)
        # Keys may be strings in JSON
        return {int(k): v for k, v in raw.items()}
    return {}


def _save_descriptions(descriptions: Dict[int, str]):
    """Persist descriptions dict to disk."""
    with open(DESCRIPTIONS_PATH, "w") as f:
        json.dump({str(k): v for k, v in descriptions.items()}, f, indent=2)


def _save_opt_log(result: OptimizationResult):
    """Save per-concept optimization log."""
    path = os.path.join(OPT_LOGS_DIR, f"concept_{result.concept_id:03d}_log.json")
    with open(path, "w") as f:
        json.dump({
            "concept_id": result.concept_id,
            "iterations_run": result.iterations_run,
            "converged": result.converged,
            "elapsed_seconds": result.elapsed_seconds,
            "final_description": result.final_description,
            "log": result.log,
        }, f, indent=2)


def _build_concept_library(
    evidence: Dict[int, ConceptEvidence],
    descriptions: Dict[int, str],
    opt_results: Dict[int, OptimizationResult],
) -> dict:
    """Build the structured concept library for concept_library.json."""
    library = {}
    for k in range(N_CONCEPTS):
        ev = evidence.get(k)
        res = opt_results.get(k)
        library[str(k)] = {
            "concept_id": k,
            "description": descriptions.get(k, ""),
            "frequency": ev.frequency if ev else None,
            "kl_from_uniform": ev.kl_from_uniform if ev else None,
            "ablation_rank": ev.ablation_rank if ev else None,
            "ablation_win_rate_drop": ev.ablation_win_rate_drop if ev else None,
            "neighbor_ids": ev.neighbor_ids if ev else [],
            "neighbor_similarities": ev.neighbor_similarities if ev else [],
            "n_examples_collected": len(ev.examples) if ev else 0,
            "iterations_run": res.iterations_run if res else None,
            "converged": res.converged if res else None,
            "elapsed_seconds": res.elapsed_seconds if res else None,
        }
    return library


# ---------------------------------------------------------------------------
# Stage 1: Evidence collection
# ---------------------------------------------------------------------------

def stage_collect(n_games: int = 500, force: bool = False) -> Dict[int, ConceptEvidence]:
    """Collect or load cached evidence."""
    if not force and os.path.exists(EVIDENCE_CACHE):
        print(f"Loading cached evidence from {EVIDENCE_CACHE}")
        return load_evidence(EVIDENCE_CACHE)

    print("Collecting evidence (this takes a few minutes)...")
    collector = DataCollector(n_concepts=N_CONCEPTS, n_examples=20, n_holdout=20, seed=42)
    collector.load_artifacts()
    evidence = collector.collect_all(n_games=n_games)
    save_evidence(evidence, EVIDENCE_CACHE)
    return evidence


# ---------------------------------------------------------------------------
# Stage 2: Optimization
# ---------------------------------------------------------------------------

def stage_optimize(
    evidence: Dict[int, ConceptEvidence],
    concept_ids: Optional[List[int]] = None,
    max_iterations: int = 10,
    verbose: bool = True,
    force: bool = False,
) -> Dict[int, str]:
    """
    Run TextGrad optimization for all (or specified) concepts.

    Skips already-completed concepts unless force=True.
    Saves descriptions incrementally so progress is never lost.
    """
    _ensure_dirs()
    descriptions = _load_checkpoint()

    if concept_ids is None:
        concept_ids = list(range(N_CONCEPTS))

    # Filter out already-done concepts
    if not force:
        todo = [k for k in concept_ids if k not in descriptions]
        if len(todo) < len(concept_ids):
            n_done = len(concept_ids) - len(todo)
            print(f"Resuming: {n_done}/{len(concept_ids)} concepts already done. "
                  f"Processing {len(todo)} remaining.")
        concept_ids = todo

    if not concept_ids:
        print("All concepts already optimized.")
        return descriptions

    # Sort by frequency descending so common concepts are described first
    # (neighbor context becomes available sooner)
    concept_ids.sort(
        key=lambda k: evidence[k].frequency if k in evidence else 0,
        reverse=True,
    )

    optimizer = DescriptionOptimizer(
        model=MODEL,
        max_iterations=max_iterations,
        convergence_threshold=0.98,
        min_iterations=3,
        gradient_memory=3,
    )

    opt_results: Dict[int, OptimizationResult] = {}

    for i, k in enumerate(concept_ids):
        print(f"\n{'='*60}")
        print(f"Concept {k:02d}  ({i+1}/{len(concept_ids)} remaining)")
        print(f"{'='*60}")

        ev = evidence[k]

        # Build context (pass current descriptions for neighbor context)
        context_text = build_context(
            ev=ev,
            all_evidence=evidence,
            descriptions=descriptions,
            n_board_examples=10,
        )

        result = optimizer.optimize_concept(
            ev=ev,
            context_text=context_text,
            verbose=verbose,
        )

        opt_results[k] = result
        descriptions[k] = result.final_description

        # Save incrementally
        _save_descriptions(descriptions)
        _save_opt_log(result)

        print(f"  Saved. Total completed: {len(descriptions)}/{N_CONCEPTS}")

    return descriptions


# ---------------------------------------------------------------------------
# Stage 3: Baselines
# ---------------------------------------------------------------------------

def stage_baselines(evidence: Dict[int, ConceptEvidence]) -> Dict[str, Dict[int, str]]:
    """
    Generate baseline description sets for BDM-10 floor/ceiling:
        - generic: "Concept k: A strategic situation"  (effective floor)
        - max_info: raw statistics embedded verbatim    (empirical ceiling)

    These don't require LLM calls (generic) or TextGrad (max_info).
    """
    from pidgin.description_optimizer import INITIAL_DESCRIPTION_TEMPLATE

    generic = {k: INITIAL_DESCRIPTION_TEMPLATE.format(k=k) for k in range(N_CONCEPTS)}
    max_info = {k: build_max_info_description(evidence[k]) for k in range(N_CONCEPTS)}

    baselines = {"generic": generic, "max_info": max_info}

    # Save for reference
    for name, descs in baselines.items():
        path = os.path.join(RESULTS_DIR, f"descriptions_{name}.json")
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in descs.items()}, f, indent=2)
        print(f"Saved {name} descriptions to {path}")

    return baselines


# ---------------------------------------------------------------------------
# Stage 4: Evaluation
# ---------------------------------------------------------------------------

def stage_evaluate(
    evidence: Dict[int, ConceptEvidence],
    descriptions_by_condition: Dict[str, Dict[int, str]],
    run_dbm: bool = True,
    run_bpa: bool = True,
    run_sna: bool = True,
    run_abc: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the full evaluation suite for all conditions.

    Returns a dict mapping condition → EvaluationReport.to_dict()
    """
    ev = Evaluator(model=MODEL)
    all_results = {}

    for condition, descriptions in descriptions_by_condition.items():
        print(f"\n{'─'*40}")
        print(f"Evaluating: {condition}")
        report = ev.evaluate_all(
            evidence=evidence,
            descriptions=descriptions,
            condition=condition,
            run_dbm=run_dbm,
            run_bpa=run_bpa,
            run_sna=run_sna,
            run_abc=run_abc,
            verbose=verbose,
        )
        all_results[condition] = report.to_dict()

    # Save
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation results saved to {EVAL_RESULTS_PATH}")

    # Print summary table
    _print_summary_table(all_results)
    return all_results


def _print_summary_table(results: dict):
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    header = f"{'Condition':<20} {'BDM-10':>8} {'DBM-64':>8} {'BPA':>8} {'SNA-r':>8} {'CDS':>8}"
    print(header)
    print("-"*70)
    for cond, r in results.items():
        bdm = r.get("bdm10", {}) or {}
        dbm = r.get("dbm64", {}) or {}
        bpa = r.get("bpa", {}) or {}
        sna = r.get("sna", {}) or {}
        cds = r.get("cds_score")

        def pct(x, key):
            v = x.get(key) if x else None
            return f"{v:.1%}" if v is not None else "  n/a"

        def flt(x, key):
            v = x.get(key) if x else None
            return f"{v:.3f}" if v is not None else "  n/a"

        print(f"{cond:<20} {pct(bdm,'accuracy'):>8} {pct(dbm,'accuracy'):>8} "
              f"{pct(bpa,'mean_accuracy'):>8} {flt(sna,'spearman_r'):>8} "
              f"{f'{cds:.3f}' if cds is not None else '  n/a':>8}")
    print("="*70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run PIDGIN pipeline")
    parser.add_argument("--concept", type=int, default=None,
                        help="Optimize a single concept ID (for testing)")
    parser.add_argument("--n-games", type=int, default=500,
                        help="Games for evidence collection")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Max TextGrad iterations per concept")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only optimize")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip optimization, only evaluate existing descriptions")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only generate baseline descriptions (no optimization)")
    parser.add_argument("--force-recollect", action="store_true",
                        help="Re-collect evidence even if cache exists")
    parser.add_argument("--force-reoptimize", action="store_true",
                        help="Re-optimize even if descriptions exist")
    parser.add_argument("--no-dbm", action="store_true", help="Skip DBM-64 (faster)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    _ensure_dirs()
    t_total = time.time()

    # ── Stage 1: Evidence ──────────────────────────────────────────────
    evidence = stage_collect(n_games=args.n_games, force=args.force_recollect)

    # ── Baselines ──────────────────────────────────────────────────────
    baselines = stage_baselines(evidence)

    if args.baselines_only:
        print("Baselines generated. Exiting.")
        return

    # ── Stage 2: Optimization ──────────────────────────────────────────
    if not args.eval_only:
        concept_ids = [args.concept] if args.concept is not None else None
        descriptions = stage_optimize(
            evidence=evidence,
            concept_ids=concept_ids,
            max_iterations=args.max_iter,
            verbose=args.verbose,
            force=args.force_reoptimize,
        )
    else:
        descriptions = _load_checkpoint()
        if not descriptions:
            print("ERROR: No existing descriptions found. Run without --eval-only first.")
            return

    # ── Stage 3: Evaluation ────────────────────────────────────────────
    if not args.skip_eval and args.concept is None:
        conditions = {
            "generic": baselines["generic"],
            "max_info": baselines["max_info"],
            "textgrad": descriptions,
        }
        # Also load single-pass (T=1) if it exists
        single_pass_path = os.path.join(RESULTS_DIR, "descriptions_single_pass.json")
        if os.path.exists(single_pass_path):
            with open(single_pass_path) as f:
                conditions["single_pass"] = {int(k): v for k, v in json.load(f).items()}

        stage_evaluate(
            evidence=evidence,
            descriptions_by_condition=conditions,
            run_dbm=not args.no_dbm,
            run_bpa=True,
            run_sna=True,
            run_abc=True,
            verbose=args.verbose,
        )

    elapsed = time.time() - t_total
    print(f"\nPIDGIN complete. Total time: {elapsed/60:.1f} minutes.")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
