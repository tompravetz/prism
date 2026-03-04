"""
Experiment 2: Concept Stability.

Two complementary measurements:

(A) Cross-seed stability (paper Table: stability_cross_seed)
    Fits K-means n_seeds times with different random seeds on the same feature
    corpus, then computes pairwise ARI and NMI between the resulting clusterings.
    High ARI (>0.87) means cluster assignments are not artifacts of random init.
    Also measures perturbation robustness: what fraction of assignments survive
    Gaussian noise of sigma=0.1 added to the feature vectors.

(B) Symmetry stability (additional diagnostic)
    Go has 8-fold symmetry. This checks whether the same board position, after
    rotation/reflection, gets the same concept.  Low scores here are expected
    because KMeans boundaries don't need to be orientation-invariant; we report
    this for completeness rather than as a paper claim.

Usage:
    # Cross-seed ARI (paper claim, requires --encoder-path for DAgger encoder):
    python -m experiments.stability --mode cross-seed --algo ppo --n-states 2000
    python -m experiments.stability --mode cross-seed --encoder-path models/cloned_dagger/ppo_go_encoder.pt

    # Symmetry consistency (additional diagnostic):
    python -m experiments.stability --mode symmetry --algo ppo --n-states 200
    python -m experiments.stability --mode symmetry --algo both

    # Both:
    python -m experiments.stability --mode both --algo ppo
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from collections import Counter

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.utils import (get_device, ensure_dir, set_seed,
                       compute_board_symmetries)


def _collect_features(encoder, n_states, device, seed=42):
    """
    Collect n_states feature vectors by rolling out random games.

    Returns:
        features: (n_states, features_dim) numpy array.
    """
    set_seed(seed)
    env = GoEnv(board_size=7)
    features = []
    while len(features) < n_states:
        obs, info = env.reset()
        done = False
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
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (1,H,W,C)
            with torch.no_grad():
                feat = encoder(obs_t).cpu().numpy()[0]
            features.append(feat)
    env.close()
    return np.array(features[:n_states])


def run_cross_seed_stability(algo="ppo", n_states=2000, n_concepts=64,
                              n_seeds=3, noise_sigma=0.1,
                              encoder_path=None, baseline_dir="models/baseline"):
    """
    Measure cross-seed clustering stability (paper Table: stability cross-seed ARI).

    Fits K-means n_seeds times on the same feature corpus (different random
    seeds) and computes pairwise ARI and NMI.  Also measures perturbation
    robustness (fraction of assignments unchanged after adding Gaussian noise).

    Args:
        algo: "ppo" or "dqn" (used to locate encoder if encoder_path is None).
        n_states: Number of feature vectors to collect.
        n_concepts: K for K-means.
        n_seeds: Number of K-means random seeds to compare (paper uses 3).
        noise_sigma: Std dev of Gaussian noise for perturbation robustness.
        encoder_path: Explicit path to encoder .pt file (overrides algo lookup).
        baseline_dir: Directory containing {algo}_go_encoder.pt.

    Returns:
        dict with ARI, NMI, and perturbation robustness statistics.
    """
    device = get_device()

    env = GoEnv(board_size=7)
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    env.close()

    path = encoder_path or os.path.join(baseline_dir, f"{algo}_go_encoder.pt")
    if not os.path.exists(path):
        print(f"ERROR: Encoder not found at {path}")
        return None
    encoder.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    encoder.to(device).eval()

    print(f"Collecting {n_states} feature vectors...")
    features = _collect_features(encoder, n_states, device)
    print(f"Collected {len(features)} features, shape={features.shape}")

    # --- Fit K-means n_seeds times and collect label arrays ---
    all_labels = []
    for s in range(n_seeds):
        km = MiniBatchKMeans(n_clusters=n_concepts, random_state=s,
                             batch_size=2048, n_init=3)
        labels = km.fit_predict(features)
        all_labels.append(labels)
        print(f"  Seed {s}: fitted ({n_concepts} clusters)")

    # --- Pairwise ARI and NMI ---
    ari_scores, nmi_scores = [], []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)
            nmi_scores.append(nmi)

    mean_ari = float(np.mean(ari_scores))
    std_ari  = float(np.std(ari_scores, ddof=1)) if len(ari_scores) > 1 else 0.0
    mean_nmi = float(np.mean(nmi_scores))

    # --- Perturbation robustness: add noise to features, reclassify ---
    # Use seed-0 KMeans centroids as reference
    km_ref = MiniBatchKMeans(n_clusters=n_concepts, random_state=0,
                             batch_size=2048, n_init=3)
    km_ref.fit(features)
    labels_clean = km_ref.predict(features)

    rng = np.random.RandomState(99)
    noise = rng.normal(0, noise_sigma, features.shape).astype(np.float32)
    labels_noisy = km_ref.predict(features + noise)
    perturbation_robustness = float(np.mean(labels_clean == labels_noisy))

    results = {
        "algo":                   algo,
        "encoder_path":           path,
        "n_states":               n_states,
        "n_concepts":             n_concepts,
        "n_seeds":                n_seeds,
        "noise_sigma":            noise_sigma,
        "mean_ari":               round(mean_ari, 4),
        "std_ari":                round(std_ari, 4),
        "mean_nmi":               round(mean_nmi, 4),
        "pairwise_ari_scores":    [round(v, 4) for v in ari_scores],
        "pairwise_nmi_scores":    [round(v, 4) for v in nmi_scores],
        "perturbation_robustness": round(perturbation_robustness, 4),
    }

    print(f"\n{'='*50}")
    print(f"CROSS-SEED STABILITY — {algo.upper()}")
    print(f"  (n_states={n_states}, n_seeds={n_seeds}, k={n_concepts})")
    print(f"{'='*50}")
    print(f"Cross-seed ARI  : {mean_ari:.3f} ± {std_ari:.3f}  (target >0.87)")
    print(f"Cross-seed NMI  : {mean_nmi:.3f}                  (target ~0.97)")
    print(f"Perturbation rob: {perturbation_robustness:.1%}     (sigma={noise_sigma})")
    print(f"Pairwise ARI    : {ari_scores}")
    print(f"{'='*50}")

    tag = os.path.splitext(os.path.basename(path))[0]
    out_path = f"results/stability_cross_seed_{tag}.json"
    ensure_dir("results")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


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
        description="Concept stability experiments (cross-seed ARI or symmetry)"
    )
    parser.add_argument("--mode", type=str, default="both",
                        choices=["cross-seed", "symmetry", "both"],
                        help="Which stability test to run (default: both)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Explicit encoder .pt path (overrides --algo lookup)")
    parser.add_argument("--n-states", type=int, default=None,
                        help="Feature/state count (default: 2000 cross-seed, 200 symmetry)")
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of K-means seeds for cross-seed ARI (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]

    for algo in algos:
        if args.mode in ("cross-seed", "both"):
            n = args.n_states if args.n_states is not None else 2000
            run_cross_seed_stability(
                algo=algo,
                n_states=n,
                n_concepts=args.n_concepts,
                n_seeds=args.n_seeds,
                encoder_path=args.encoder_path,
                baseline_dir=args.baseline_dir,
            )

        if args.mode in ("symmetry", "both"):
            n = args.n_states if args.n_states is not None else 200
            run_stability_experiment(
                algo=algo, n_states=n,
                n_concepts=args.n_concepts, seed=args.seed,
                model_dir=args.model_dir, baseline_dir=args.baseline_dir,
            )


if __name__ == "__main__":
    main()
