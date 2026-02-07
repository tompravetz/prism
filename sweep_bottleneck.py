"""
Information Bottleneck Sweep: Train bottleneck with K = 16, 32, 64, 128 concepts.

Produces one of the most elegant figures in the paper: the information-performance
tradeoff curve. X-axis = number of concepts (information), Y-axis = win rate
(performance). Shows how much compression the agent can tolerate before losing
strategic capability.

Usage:
    python sweep_bottleneck.py --algo ppo
    python sweep_bottleneck.py --algo ppo --self-play
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.utils import set_seed, get_device, ensure_dir

# Import the training function directly
from train_bottleneck import train_bottleneck, discover_concepts, evaluate_agent


def run_sweep(algo="ppo", k_values=None, generations=50, steps_per_gen=20000,
              self_play=False, seed=42, baseline_dir="models/baseline",
              save_dir="models/sweep"):
    """
    Train bottleneck agents with different numbers of concepts.

    Args:
        algo: "ppo" or "dqn"
        k_values: List of concept counts to test (default: [16, 32, 64, 128])
        generations: Training generations per K value
        steps_per_gen: Steps per generation
        self_play: Enable self-play training
        seed: Random seed
        baseline_dir: Where baseline encoders are stored
        save_dir: Where to save sweep results
    """
    if k_values is None:
        k_values = [16, 32, 64, 128]

    ensure_dir(save_dir)
    device = get_device()

    results = []

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"SWEEP: K={k} concepts, {algo.upper()}")
        print(f"{'='*60}")

        # Each K value gets its own subdirectory
        k_dir = os.path.join(save_dir, f"k{k}_{algo}")
        ensure_dir(k_dir)

        # Train bottleneck with this K value
        trainer, strategy_memory, metrics = train_bottleneck(
            algo=algo,
            n_generations=generations,
            steps_per_gen=steps_per_gen,
            n_concepts=k,
            seed=seed,
            save_dir=k_dir,
            baseline_dir=baseline_dir,
            self_play=self_play,
            self_play_start=10,
            self_play_ratio=0.5,
        )

        # Get best and final win rates from metrics
        best_wr = max(m["win_rate"] for m in metrics)
        final_wr = metrics[-1]["win_rate"]

        # Additional evaluation with more games for accuracy
        env = GoEnv(board_size=7)
        if algo == "ppo":
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.policy.get_action(c, mask, deterministic=True)
        else:
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.q_net.get_action(c, mask, epsilon=0.0)

        eval_result = evaluate_agent(agent_fn, env, n_episodes=200)
        env.close()

        entry = {
            "k": k,
            "algo": algo,
            "best_win_rate": best_wr,
            "final_win_rate": final_wr,
            "eval_win_rate": eval_result["win_rate"],
            "eval_mean_reward": eval_result["mean_reward"],
            "bits": np.log2(k),
            "generations": generations,
            "self_play": self_play,
        }
        results.append(entry)
        print(f"\nK={k}: best={best_wr:.2%}, final={final_wr:.2%}, "
              f"eval={eval_result['win_rate']:.2%}")

    # Save sweep results
    results_path = os.path.join(save_dir, f"sweep_{algo}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {results_path}")

    # Plot the information bottleneck curve
    plot_sweep(results, algo, save_dir)

    return results


def plot_sweep(results, algo, save_dir):
    """
    Plot the information bottleneck curve: concepts vs win rate.

    This figure shows the fundamental tradeoff between interpretability
    (fewer concepts = more interpretable) and performance (more concepts =
    better play). The "knee" of the curve reveals how many concepts are
    actually needed.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    k_values = [r["k"] for r in results]
    eval_wrs = [r["eval_win_rate"] for r in results]
    best_wrs = [r["best_win_rate"] for r in results]

    # Plot both eval and best win rates
    ax.plot(k_values, eval_wrs, 'o-', color='#2196F3', linewidth=2,
            markersize=10, label='Final Win Rate (200 games)', zorder=5)
    ax.plot(k_values, best_wrs, 's--', color='#FF9800', linewidth=1.5,
            markersize=8, label='Best Win Rate (training)', alpha=0.7)

    # Add value labels
    for k, wr in zip(k_values, eval_wrs):
        ax.annotate(f'{wr:.1%}', (k, wr), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    # Add bits axis on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    bit_ticks = k_values
    ax2.set_xticks(bit_ticks)
    ax2.set_xticklabels([f'{np.log2(k):.0f} bits' for k in bit_ticks])
    ax2.set_xlabel("Information (bits)", fontsize=11)

    ax.set_xlabel("Number of Concepts (K)", fontsize=12)
    ax.set_ylabel("Win Rate vs Random", fontsize=12)
    ax.set_title(f"Information Bottleneck Tradeoff — {algo.upper()}",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation about compression ratio
    ax.text(0.02, 0.02,
            f"Full observation: 4,704 bits (7×7×3×32)\n"
            f"K=64 concepts: 6 bits (~800× compression)",
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join("results", "figures", f"bottleneck_sweep_{algo}.png")
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bottleneck sweep figure saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bottleneck size sweep (K = 16, 32, 64, 128)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn"])
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[16, 32, 64, 128],
                        help="Concept counts to test")
    parser.add_argument("--generations", type=int, default=50,
                        help="Generations per K value (50 is enough)")
    parser.add_argument("--steps-per-gen", type=int, default=20000)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sweep(
        algo=args.algo,
        k_values=args.k_values,
        generations=args.generations,
        steps_per_gen=args.steps_per_gen,
        self_play=args.self_play,
        seed=args.seed,
    )
