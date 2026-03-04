"""
Transitive Transfer: Test whether concept alignment composes transitively.

Core hypothesis: If concepts form a universal interface, then chaining alignments
through an intermediate agent (A->B->C) should produce transfer quality comparable
to direct alignment (A->C).

For each 3-agent chain (PPO, DQN, DAgger):
    1. Compute direct alignment A->C via Hungarian matching
    2. Compute chained alignment: align A->B, align B->C, compose mappings
    3. Transfer A's policy using direct mapping -> evaluate
    4. Transfer A's policy using chained mapping -> evaluate

All 6 possible chains are tested with 5 seeds each.

Usage:
    python experiments/transitive_transfer.py
    python experiments/transitive_transfer.py --n-seeds 5 --n-eval 100
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
from src.concept_aligner import ConceptAligner, compose_alignments
from src.utils import set_seed, get_device, ensure_dir
from train_bottleneck import evaluate_agent


# ============================================================
# Agent configurations (same as transfer_same_task.py)
# ============================================================

AGENTS = {
    "PPO": {
        "encoder_path": "models/baseline/ppo_go_encoder.pt",
        "concepts_path": "models/bottleneck/concepts_ppo_k64.pkl",
        "policy_path": "models/bottleneck/ppo_bottleneck_final.pt",
        "policy_class": "ppo",
        "n_concepts": 64,
        "n_actions": 50,
    },
    "DQN": {
        "encoder_path": "models/baseline/dqn_go_encoder.pt",
        "concepts_path": "models/bottleneck/concepts_dqn_k64.pkl",
        "policy_path": "models/bottleneck/dqn_bottleneck_final.pt",
        "policy_class": "dqn",
        "n_concepts": 64,
        "n_actions": 50,
    },
    "DAgger": {
        "encoder_path": "models/cloned_dagger/ppo_go_encoder.pt",
        "concepts_path": "models/bottleneck_dagger/concepts_ppo_k64.pkl",
        "policy_path": "models/bottleneck_dagger/ppo_bottleneck_final.pt",
        "policy_class": "ppo",
        "n_concepts": 64,
        "n_actions": 50,
    },
}


def load_agent(agent_name, device):
    """Load an agent's encoder, concept manager, and bottleneck policy."""
    config = AGENTS[agent_name]

    env = GoEnv(board_size=7)
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder.load_state_dict(
        torch.load(config["encoder_path"], map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()
    env.close()

    cm = ConceptManager(n_concepts=config["n_concepts"])
    cm.load(config["concepts_path"])

    if config["policy_class"] == "dqn":
        policy = ConceptDQNPolicy(
            n_concepts=config["n_concepts"],
            embed_dim=64, hidden_dim=128,
            n_actions=config["n_actions"],
        )
    else:
        policy = ConceptBottleneckPolicy(
            n_concepts=config["n_concepts"],
            embed_dim=64, hidden_dim=128,
            n_actions=config["n_actions"],
        )

    policy.load_state_dict(
        torch.load(config["policy_path"], map_location=device, weights_only=True)
    )
    policy.to(device)
    policy.eval()

    return encoder, cm, policy


def evaluate_transferred(encoder, cm, policy, n_episodes=100, device=None):
    """Evaluate a transferred policy using target encoder + concept manager."""
    device = device or get_device()
    env = GoEnv(board_size=7)

    def agent_fn(obs, action_mask):
        concept_id = cm.assign_concept_from_obs(encoder, obs, device)
        if isinstance(policy, ConceptDQNPolicy):
            return policy.get_action(concept_id, action_mask, epsilon=0.0)
        else:
            return policy.get_action(concept_id, action_mask, deterministic=True)

    results = evaluate_agent(agent_fn, env, n_episodes=n_episodes)
    env.close()
    return results


def run_transitive_transfer(n_seeds=5, n_eval=100):
    """
    Run the full transitive transfer experiment.

    Tests all 6 possible 3-agent chains across PPO, DQN, DAgger.
    Each chain is evaluated with multiple seeds for statistical robustness.
    """
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Transitive Transfer Experiment")
    print(f"[{timestamp}]   Seeds: {n_seeds}, Eval games per seed: {n_eval}")
    print(f"[{timestamp}] ============================================================")

    # Load all agents
    print(f"\nLoading agents...")
    agents = {}
    for name in AGENTS:
        print(f"  Loading {name}...")
        encoder, cm, policy = load_agent(name, device)
        agents[name] = {"encoder": encoder, "cm": cm, "policy": policy}

    agent_names = list(AGENTS.keys())

    # Pre-compute all pairwise alignments
    print(f"\nComputing all pairwise alignments...")
    alignments = {}  # (src, tgt) -> mapping
    align_qualities = {}  # (src, tgt) -> quality dict

    for src in agent_names:
        for tgt in agent_names:
            if src == tgt:
                continue
            key = (src, tgt)
            aligner = ConceptAligner(agents[src]["cm"], agents[tgt]["cm"])
            mapping = aligner.hungarian_alignment()
            quality = aligner.alignment_quality(mapping)
            alignments[key] = mapping
            align_qualities[key] = quality
            print(f"  {src}->{tgt}: mean_sim={quality['mean_similarity']:.4f}")

    # Generate all 3-agent chains: A->B->C where A, B, C are all different
    chains = []
    for a in agent_names:
        for b in agent_names:
            for c in agent_names:
                if a != b and b != c and a != c:
                    chains.append((a, b, c))

    print(f"\n{len(chains)} chains to evaluate")

    results = []

    for chain_idx, (a, b, c) in enumerate(chains):
        print(f"\n--- Chain {chain_idx+1}/{len(chains)}: {a} -> {b} -> {c} ---")

        # Direct alignment: A -> C
        direct_mapping = alignments[(a, c)]
        direct_quality = align_qualities[(a, c)]

        # Chained alignment: A -> B -> C
        mapping_ab = alignments[(a, b)]
        mapping_bc = alignments[(b, c)]
        chained_mapping, n_lost = compose_alignments(mapping_ab, mapping_bc)

        print(f"  Direct alignment sim:  {direct_quality['mean_similarity']:.4f}")
        print(f"  Chained: {len(chained_mapping)} mapped, {n_lost} lost")

        # Compute chained alignment quality using the similarity matrix A vs C
        chained_aligner = ConceptAligner(agents[a]["cm"], agents[c]["cm"])
        chained_quality = chained_aligner.alignment_quality(chained_mapping)
        print(f"  Chained alignment sim: {chained_quality['mean_similarity']:.4f}")

        # Transfer A's policy using direct mapping
        direct_transferred = chained_aligner.transfer_policy(
            agents[a]["policy"], direct_mapping,
            target_n_concepts=64, target_n_actions=50,
        )
        direct_transferred.to(device)
        direct_transferred.eval()

        # Transfer A's policy using chained mapping
        chained_transferred = chained_aligner.transfer_policy(
            agents[a]["policy"], chained_mapping,
            target_n_concepts=64, target_n_actions=50,
        )
        chained_transferred.to(device)
        chained_transferred.eval()

        # Evaluate both with multiple seeds
        direct_wrs = []
        chained_wrs = []

        for seed in range(n_seeds):
            set_seed(seed * 1000 + chain_idx)

            # Direct transfer evaluation
            direct_result = evaluate_transferred(
                agents[c]["encoder"], agents[c]["cm"],
                direct_transferred, n_episodes=n_eval, device=device,
            )
            direct_wrs.append(direct_result["win_rate"])

            # Chained transfer evaluation
            chained_result = evaluate_transferred(
                agents[c]["encoder"], agents[c]["cm"],
                chained_transferred, n_episodes=n_eval, device=device,
            )
            chained_wrs.append(chained_result["win_rate"])

            print(f"    Seed {seed}: direct={direct_result['win_rate']:.2%}, "
                  f"chained={chained_result['win_rate']:.2%}")

        # Statistical comparison
        from scipy import stats
        if n_seeds >= 2:
            t_stat, p_value = stats.ttest_rel(direct_wrs, chained_wrs)
        else:
            t_stat, p_value = 0.0, 1.0

        chain_result = {
            "chain": f"{a}->{b}->{c}",
            "source": a,
            "intermediate": b,
            "target": c,
            "direct": {
                "alignment_sim": float(direct_quality["mean_similarity"]),
                "win_rates": [float(x) for x in direct_wrs],
                "mean_wr": float(np.mean(direct_wrs)),
                "std_wr": float(np.std(direct_wrs)),
            },
            "chained": {
                "alignment_sim": float(chained_quality["mean_similarity"]),
                "n_mapped": int(len(chained_mapping)),
                "n_lost": int(n_lost),
                "win_rates": [float(x) for x in chained_wrs],
                "mean_wr": float(np.mean(chained_wrs)),
                "std_wr": float(np.std(chained_wrs)),
            },
            "comparison": {
                "wr_gap": float(np.mean(direct_wrs) - np.mean(chained_wrs)),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
            },
        }
        results.append(chain_result)

        print(f"  Summary: direct={np.mean(direct_wrs):.2%} +/- {np.std(direct_wrs):.2%}, "
              f"chained={np.mean(chained_wrs):.2%} +/- {np.std(chained_wrs):.2%}, "
              f"p={p_value:.4f}")

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/transitive_transfer.json"
    with open(output_path, "w") as f:
        json.dump({
            "n_seeds": n_seeds,
            "n_eval": n_eval,
            "chains": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ============================================================
    # Summary table
    # ============================================================
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Transitive Transfer Summary")
    print(f"[{ts}] {'Chain':<20} {'Direct WR':>10} {'Chained WR':>11} {'Gap':>7} {'p-value':>8}")
    print(f"[{ts}] {'-'*60}")
    for r in results:
        print(f"[{ts}] {r['chain']:<20} "
              f"{r['direct']['mean_wr']:>9.2%} "
              f"{r['chained']['mean_wr']:>10.2%} "
              f"{r['comparison']['wr_gap']:>+6.2%} "
              f"{r['comparison']['p_value']:>8.4f}")

    # ============================================================
    # Visualization: grouped bar chart
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito palette
        OI_BLUE, OI_ORANGE = "#0072B2", "#E69F00"

        ensure_dir("results/figures")
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        chain_labels = [r["chain"] for r in results]
        direct_means = [r["direct"]["mean_wr"] for r in results]
        direct_stds = [r["direct"]["std_wr"] for r in results]
        chained_means = [r["chained"]["mean_wr"] for r in results]
        chained_stds = [r["chained"]["std_wr"] for r in results]

        x = np.arange(len(chain_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, direct_means, width, yerr=direct_stds,
                        label="Direct (A->C)", color=OI_BLUE,
                        edgecolor="black", linewidth=0.5, capsize=3)
        bars2 = ax.bar(x + width/2, chained_means, width, yerr=chained_stds,
                        label="Chained (A->B->C)", color=OI_ORANGE,
                        edgecolor="black", linewidth=0.5, capsize=3)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3,
                    label="Random baseline (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels(chain_labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Win Rate vs Random", fontsize=12)
        ax.set_title("Transitive Transfer: Direct vs Chained Alignment", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)

        plt.tight_layout()
        fig_path = "results/figures/transitive_transfer.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transitive Transfer Experiment")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of evaluation seeds (default: 5)")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Games per evaluation (default: 100)")
    args = parser.parse_args()

    run_transitive_transfer(n_seeds=args.n_seeds, n_eval=args.n_eval)
