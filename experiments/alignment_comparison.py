"""
Alignment Method Comparison: Compare 5 alignment strategies on PPO->DQN transfer.

Methods:
    1. Hungarian matching (PRISM default) - optimal 1:1 bipartite matching
    2. Random permutation - random 1:1 mapping (ablation baseline)
    3. Greedy nearest-neighbor - each source maps to most similar target
    4. Identity mapping - concept i -> concept i (no alignment)
    5. Procrustes alignment - orthogonal rotation + Hungarian matching

Each method evaluated with 5 seeds, 100 games per seed.
Statistical comparison via pairwise Welch's t-tests.

Usage:
    python experiments/alignment_comparison.py
    python experiments/alignment_comparison.py --n-seeds 5 --n-eval 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir
from train_bottleneck import evaluate_agent


def load_ppo_dqn_agents(device):
    """Load the PPO (source) and DQN (target) agents."""
    env = GoEnv(board_size=7)

    # PPO (source)
    ppo_encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    ppo_encoder.load_state_dict(
        torch.load("models/baseline/ppo_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    ppo_encoder.to(device)
    ppo_encoder.eval()

    ppo_cm = ConceptManager(n_concepts=64)
    ppo_cm.load("models/bottleneck/concepts_ppo_k64.pkl")

    ppo_policy = ConceptBottleneckPolicy(
        n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50,
    )
    ppo_policy.load_state_dict(
        torch.load("models/bottleneck/ppo_bottleneck_final.pt",
                    map_location=device, weights_only=True)
    )
    ppo_policy.to(device)
    ppo_policy.eval()

    # DQN (target)
    dqn_encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    dqn_encoder.load_state_dict(
        torch.load("models/baseline/dqn_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    dqn_encoder.to(device)
    dqn_encoder.eval()

    dqn_cm = ConceptManager(n_concepts=64)
    dqn_cm.load("models/bottleneck/concepts_dqn_k64.pkl")

    env.close()

    return {
        "ppo": {"encoder": ppo_encoder, "cm": ppo_cm, "policy": ppo_policy},
        "dqn": {"encoder": dqn_encoder, "cm": dqn_cm},
    }


def evaluate_mapping(mapping, source_policy, target_encoder, target_cm,
                     aligner, n_episodes=100, device=None, gnugo_level=None):
    """Evaluate a specific concept mapping for PPO->DQN transfer."""
    from visualizer.opponents import GnuGoOpponent

    device = device or get_device()

    transferred = aligner.transfer_policy(
        source_policy, mapping,
        target_n_concepts=64, target_n_actions=50,
    )
    transferred.to(device)
    transferred.eval()

    if gnugo_level is not None:
        opponent = GnuGoOpponent(level=gnugo_level)
    else:
        opponent = None

    env = GoEnv(board_size=7, opponent_fn=opponent)

    def agent_fn(obs, action_mask):
        concept_id = target_cm.assign_concept_from_obs(target_encoder, obs, device)
        return transferred.get_action(concept_id, action_mask, deterministic=True)

    results = evaluate_agent(agent_fn, env, n_episodes=n_episodes)
    env.close()
    if opponent is not None:
        opponent.close()
    return results


def run_alignment_comparison(n_seeds=5, n_eval=100, gnugo_level=None):
    """
    Compare 5 alignment methods on the PPO->DQN transfer pair.

    Methods tested:
        1. Hungarian (optimal 1:1 matching on cosine similarity)
        2. Random permutation (5 random seeds averaged)
        3. Greedy nearest-neighbor
        4. Identity mapping (concept i -> concept i)
        5. Procrustes (orthogonal rotation + Hungarian)
    """
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Alignment Method Comparison (PPO -> DQN)")
    print(f"[{timestamp}]   Seeds: {n_seeds}, Eval games per seed: {n_eval}")
    print(f"[{timestamp}] ============================================================")

    # Load agents
    print(f"\nLoading agents...")
    agents = load_ppo_dqn_agents(device)
    ppo = agents["ppo"]
    dqn = agents["dqn"]

    aligner = ConceptAligner(ppo["cm"], dqn["cm"])

    # ============================================================
    # Generate all 5 mappings
    # ============================================================
    methods = {}

    # Method 1: Hungarian matching (PRISM default)
    print(f"\n--- Method 1: Hungarian Matching ---")
    hungarian_mapping = aligner.hungarian_alignment()
    hungarian_quality = aligner.alignment_quality(hungarian_mapping)
    methods["hungarian"] = {
        "name": "Hungarian (PRISM)",
        "mapping": hungarian_mapping,
        "alignment_sim": hungarian_quality["mean_similarity"],
    }
    print(f"  Alignment sim: {hungarian_quality['mean_similarity']:.4f}")

    # Method 2: Random permutation (multiple random seeds, averaged)
    print(f"\n--- Method 2: Random Permutation ---")
    random_mappings = []
    for rs in range(5):
        rng = np.random.RandomState(rs)
        perm = rng.permutation(64)
        random_mappings.append({int(i): int(perm[i]) for i in range(64)})
    # Use the first random mapping for the main evaluation;
    # all 5 are evaluated separately within the seed loop
    methods["random"] = {
        "name": "Random Permutation",
        "mapping": random_mappings[0],
        "all_mappings": random_mappings,
        "alignment_sim": float(np.mean([
            aligner.alignment_quality(m)["mean_similarity"] for m in random_mappings
        ])),
    }
    print(f"  Mean alignment sim (5 random): {methods['random']['alignment_sim']:.4f}")

    # Method 3: Greedy nearest-neighbor
    print(f"\n--- Method 3: Greedy Nearest-Neighbor ---")
    greedy_mapping = aligner.greedy_alignment()
    greedy_quality = aligner.alignment_quality(greedy_mapping)
    methods["greedy"] = {
        "name": "Greedy NN",
        "mapping": greedy_mapping,
        "alignment_sim": greedy_quality["mean_similarity"],
    }
    print(f"  Alignment sim: {greedy_quality['mean_similarity']:.4f}")

    # Method 4: Identity mapping (no alignment)
    print(f"\n--- Method 4: Identity Mapping ---")
    identity_mapping = {i: i for i in range(64)}
    identity_quality = aligner.alignment_quality(identity_mapping)
    methods["identity"] = {
        "name": "Identity (No Alignment)",
        "mapping": identity_mapping,
        "alignment_sim": identity_quality["mean_similarity"],
    }
    print(f"  Alignment sim: {identity_quality['mean_similarity']:.4f}")

    # Method 5: Procrustes alignment
    print(f"\n--- Method 5: Procrustes Alignment ---")
    procrustes_mapping, R = aligner.procrustes_alignment()
    procrustes_quality = aligner.alignment_quality(procrustes_mapping)
    methods["procrustes"] = {
        "name": "Procrustes",
        "mapping": procrustes_mapping,
        "alignment_sim": procrustes_quality["mean_similarity"],
    }
    print(f"  Alignment sim: {procrustes_quality['mean_similarity']:.4f}")

    # ============================================================
    # Evaluate each method with multiple seeds
    # ============================================================
    results = {}

    for method_key, method_info in methods.items():
        print(f"\n--- Evaluating: {method_info['name']} ---")
        seed_wrs = []
        seed_rewards = []
        seed_lengths = []

        for seed in range(n_seeds):
            set_seed(seed * 100)

            # For random permutation, use a different random mapping each seed
            if method_key == "random" and "all_mappings" in method_info:
                mapping = method_info["all_mappings"][seed % len(method_info["all_mappings"])]
            else:
                mapping = method_info["mapping"]

            eval_result = evaluate_mapping(
                mapping, ppo["policy"],
                dqn["encoder"], dqn["cm"], aligner,
                n_episodes=n_eval, device=device,
                gnugo_level=gnugo_level,
            )

            seed_wrs.append(eval_result["win_rate"])
            seed_rewards.append(eval_result["mean_reward"])
            seed_lengths.append(eval_result["mean_length"])
            print(f"    Seed {seed}: wr={eval_result['win_rate']:.2%}, "
                  f"reward={eval_result['mean_reward']:.3f}")

        results[method_key] = {
            "name": method_info["name"],
            "alignment_sim": float(method_info["alignment_sim"]),
            "win_rates": [float(x) for x in seed_wrs],
            "mean_wr": float(np.mean(seed_wrs)),
            "std_wr": float(np.std(seed_wrs)),
            "mean_reward": float(np.mean(seed_rewards)),
            "std_reward": float(np.std(seed_rewards)),
            "mean_length": float(np.mean(seed_lengths)),
        }

    # ============================================================
    # Pairwise statistical tests
    # ============================================================
    from scipy import stats

    method_keys = list(results.keys())
    pairwise_tests = []

    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            k1, k2 = method_keys[i], method_keys[j]
            wrs1 = results[k1]["win_rates"]
            wrs2 = results[k2]["win_rates"]
            if n_seeds >= 2:
                t_stat, p_val = stats.ttest_ind(wrs1, wrs2, equal_var=False)
            else:
                t_stat, p_val = 0.0, 1.0
            pairwise_tests.append({
                "method_1": k1,
                "method_2": k2,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            })

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/alignment_comparison.json"

    save_data = {
        "n_seeds": n_seeds,
        "n_eval": n_eval,
        "pair": "PPO -> DQN",
        "methods": {k: {kk: vv for kk, vv in v.items()}
                    for k, v in results.items()},
        "pairwise_tests": pairwise_tests,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ============================================================
    # Summary table
    # ============================================================
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Alignment Method Comparison Summary (PPO -> DQN)")
    print(f"[{ts}] {'Method':<25} {'Align Sim':>10} {'Win Rate':>14} {'Reward':>12}")
    print(f"[{ts}] {'-'*65}")
    for k, r in results.items():
        wr_str = f"{r['mean_wr']:.2%} +/- {r['std_wr']:.2%}"
        print(f"[{ts}] {r['name']:<25} {r['alignment_sim']:>10.4f} {wr_str:>14} "
              f"{r['mean_reward']:>8.3f} +/- {r['std_reward']:.3f}")

    print(f"\n[{ts}] Significant pairwise differences (p<0.05):")
    for test in pairwise_tests:
        if test["significant"]:
            print(f"[{ts}]   {test['method_1']} vs {test['method_2']}: p={test['p_value']:.4f}")

    # ============================================================
    # Visualization: bar chart with error bars
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito palette
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

        ensure_dir("results/figures")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        method_names = [results[k]["name"] for k in method_keys]
        means = [results[k]["mean_wr"] for k in method_keys]
        stds = [results[k]["std_wr"] for k in method_keys]

        bars = ax.bar(range(len(method_names)), means, yerr=stds, capsize=4,
                      color=colors[:len(method_names)],
                      edgecolor="black", linewidth=0.5)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3,
                    label="Random baseline (50%)")
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Zero-Shot Win Rate", fontsize=12)
        ax.set_title("Alignment Method Comparison: PPO -> DQN (Go 7x7)", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)

        # Add value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.0%}", ha="center",
                    fontsize=10, fontweight="bold")

        # Add significance brackets for Hungarian vs others
        hungarian_idx = method_keys.index("hungarian")
        for test in pairwise_tests:
            if test["significant"] and "hungarian" in (test["method_1"], test["method_2"]):
                other = test["method_2"] if test["method_1"] == "hungarian" else test["method_1"]
                other_idx = method_keys.index(other)
                y = max(means[hungarian_idx], means[other_idx]) + 0.08
                ax.annotate("*", xy=((hungarian_idx + other_idx) / 2, y),
                            ha="center", fontsize=14, fontweight="bold")

        plt.tight_layout()
        fig_path = "results/figures/alignment_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment Method Comparison")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of evaluation seeds (default: 5)")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Games per evaluation (default: 100)")
    parser.add_argument("--gnugo-level", type=int, default=None,
                        help="Evaluate vs GnuGo at this level instead of random "
                             "(recommended: 1 for comparability with other results)")
    args = parser.parse_args()

    run_alignment_comparison(n_seeds=args.n_seeds, n_eval=args.n_eval,
                             gnugo_level=args.gnugo_level)
