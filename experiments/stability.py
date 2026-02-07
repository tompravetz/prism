"""
Experiment 2: Concept Stability Under Board Symmetries.

Go has 8-fold symmetry (4 rotations × 2 reflections). A good concept
representation should assign the SAME concept to symmetric board positions,
because symmetric positions are strategically equivalent.

This experiment tests:
    - Take a board state
    - Apply all 8 symmetries (rotate 0°/90°/180°/270° × flip/no-flip)
    - Check if the encoder + K-means assigns the same concept to all 8 versions

High consistency (>60%) means the encoder has learned rotation-invariant features,
which is a desirable property for Go.

Low consistency means the concepts are sensitive to board orientation, which
could indicate:
    - The encoder hasn't seen enough training data
    - The concepts are too fine-grained
    - The encoder architecture needs augmentation

Usage:
    python -m experiments.stability --algo ppo --n-states 200
    python -m experiments.stability --algo both
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from collections import Counter

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.utils import (get_device, ensure_dir, set_seed,
                       compute_board_symmetries)


def run_stability_experiment(algo="ppo", n_states=200, n_concepts=64,
                             seed=42, model_dir="models/bottleneck",
                             baseline_dir="models/baseline"):
    """
    Test concept consistency across board symmetries.

    For each test state:
        1. Compute all 8 symmetric versions of the board
        2. Assign a concept to each symmetric version
        3. Check if all 8 get the same concept
        4. Count how many unique concepts appear (ideally 1)

    We report:
        - Exact match rate: fraction of states where ALL 8 symmetries get the same concept
        - Average unique concepts: mean number of distinct concepts per state (ideally 1.0)
        - Pair-wise consistency: fraction of pairs of symmetries that agree

    Args:
        algo: "ppo" or "dqn"
        n_states: Number of test states to check
        n_concepts: Number of concept clusters
        seed: Random seed
        model_dir: Where bottleneck models are stored
        baseline_dir: Where baseline encoders are stored

    Returns:
        Dictionary of stability results.
    """
    set_seed(seed)
    device = get_device()

    # ---- Load encoder and concept manager ----
    env = GoEnv(board_size=7)
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)

    encoder_path = os.path.join(baseline_dir, f"{algo}_go_encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"ERROR: Encoder not found at {encoder_path}")
        return None
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()

    cm = ConceptManager(n_concepts=n_concepts)
    concept_path = os.path.join(model_dir, f"concepts_{algo}_k{n_concepts}.pkl")
    if not os.path.exists(concept_path):
        print(f"ERROR: Concepts not found at {concept_path}")
        return None
    cm.load(concept_path)

    # ---- Collect test states from random games ----
    print(f"Collecting {n_states} test states for stability analysis...")
    test_observations = []
    while len(test_observations) < n_states:
        obs, info = env.reset()
        done = False
        # Advance game to a random mid-game state (more interesting than empty boards)
        n_moves = np.random.randint(5, 30)
        for _ in range(n_moves):
            if done:
                break
            mask = info.get("action_mask", np.ones(50, dtype=np.int8))
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                break
            action = np.random.choice(legal)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if not done:
            test_observations.append(obs.copy())

    test_observations = test_observations[:n_states]
    print(f"Collected {len(test_observations)} test states.")

    # ---- Run stability analysis ----
    exact_matches = 0           # States where all 8 symmetries → same concept
    total_pairwise_agree = 0    # Total pairwise agreements
    total_pairwise = 0          # Total pairwise comparisons
    unique_counts = []          # Number of unique concepts per state
    per_state_results = []

    for i, obs in enumerate(test_observations):
        # Generate all 8 symmetric versions of this board
        symmetries = compute_board_symmetries(obs)
        assert len(symmetries) == 8, f"Expected 8 symmetries, got {len(symmetries)}"

        # Assign concept to each symmetric version
        concepts = []
        for sym_obs in symmetries:
            concept = cm.assign_concept_from_obs(encoder, sym_obs, device)
            concepts.append(concept)

        # Count unique concepts
        unique = len(set(concepts))
        unique_counts.append(unique)

        # Exact match: all 8 symmetries get the same concept
        if unique == 1:
            exact_matches += 1

        # Pairwise consistency: how many pairs of symmetries agree?
        # There are 8*7/2 = 28 pairs
        for j in range(8):
            for k in range(j + 1, 8):
                total_pairwise += 1
                if concepts[j] == concepts[k]:
                    total_pairwise_agree += 1

        per_state_results.append({
            "state_idx": i,
            "concepts": [int(c) for c in concepts],
            "n_unique": unique,
            "is_exact_match": unique == 1,
        })

    # ---- Compute statistics ----
    exact_match_rate = exact_matches / len(test_observations)
    pairwise_consistency = total_pairwise_agree / total_pairwise if total_pairwise > 0 else 0
    mean_unique = np.mean(unique_counts)
    median_unique = np.median(unique_counts)

    results = {
        "algo": algo,
        "n_states": len(test_observations),
        "n_concepts": n_concepts,
        "exact_match_rate": float(exact_match_rate),
        "pairwise_consistency": float(pairwise_consistency),
        "mean_unique_concepts": float(mean_unique),
        "median_unique_concepts": float(median_unique),
        "max_unique_concepts": int(max(unique_counts)),
        "unique_count_distribution": dict(Counter(unique_counts)),
    }

    # ---- Print results ----
    print(f"\n{'='*50}")
    print(f"STABILITY RESULTS — {algo.upper()}")
    print(f"{'='*50}")
    print(f"Test states:           {results['n_states']}")
    print(f"Exact match rate:      {results['exact_match_rate']:.2%}")
    print(f"  (All 8 symmetries → same concept)")
    print(f"Pairwise consistency:  {results['pairwise_consistency']:.2%}")
    print(f"  (Fraction of pairs that agree)")
    print(f"Mean unique concepts:  {results['mean_unique_concepts']:.2f} / 8")
    print(f"Median unique:         {results['median_unique_concepts']:.1f} / 8")
    print(f"")
    print(f"Unique concept distribution:")
    for k in sorted(results['unique_count_distribution'].keys()):
        v = results['unique_count_distribution'][k]
        pct = v / results['n_states'] * 100
        bar = "#" * int(pct / 2)
        print(f"  {k} unique: {v:4d} ({pct:5.1f}%) {bar}")
    print(f"{'='*50}")

    # ---- Save results ----
    ensure_dir("results")
    with open(f"results/stability_{algo}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results/stability_{algo}.json")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test concept stability under board symmetries"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]

    for algo in algos:
        run_stability_experiment(
            algo=algo, n_states=args.n_states,
            n_concepts=args.n_concepts, seed=args.seed,
            model_dir=args.model_dir, baseline_dir=args.baseline_dir,
        )


if __name__ == "__main__":
    main()
