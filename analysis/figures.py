"""
Generate all publication-quality figures for the ASTRIA paper.

This is the "one-click" figure generation script. Run it after all
experiments are complete to produce every figure needed for the paper.

Figures generated:
    1. Learning curves (4 variants over generations)
    2. Concept usage heatmaps (PPO vs DQN, side by side)
    3. Intervention results (bar charts of action change rates)
    4. Ablation impact charts (waterfall/bar per algorithm)
    5. Algorithm comparison table (as a figure)
    6. CartPole concept visualization
    7. VQ codebook utilization (if Phase 2 data available)

All figures are saved to results/figures/ at 150 DPI (suitable for papers).

Usage:
    python -m analysis.figures
"""

import os
import sys
import json
import numpy as np

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from analysis.visualize import (
    setup_style, COLORS,
    plot_learning_curves,
    plot_concept_usage_heatmap,
    plot_intervention_results,
    plot_ablation_results,
)
from src.utils import ensure_dir


FIGURE_DIR = "results/figures"


def generate_all_figures(results_dir="results", model_dir="models"):
    """
    Generate all paper figures.

    Checks which result files exist and generates corresponding figures.
    Missing results are skipped with a warning (so you can generate partial
    figures as experiments complete).
    """
    ensure_dir(FIGURE_DIR)
    setup_style()
    generated = []

    # ---- Figure 1: Learning Curves ----
    print("Generating Figure 1: Learning Curves...")
    metrics_files = [
        os.path.join(model_dir, "bottleneck", "metrics_ppo.json"),
        os.path.join(model_dir, "bottleneck", "metrics_dqn.json"),
        os.path.join(model_dir, "vq", "metrics_vq.json"),
    ]
    existing = [f for f in metrics_files if os.path.exists(f)]
    if existing:
        labels = []
        for f in existing:
            if "ppo" in f:
                labels.append("PPO Bottleneck")
            elif "dqn" in f:
                labels.append("DQN Bottleneck")
            elif "vq" in f:
                labels.append("VQ End-to-End")
            else:
                labels.append(os.path.basename(f))

        plot_learning_curves(
            existing, labels=labels,
            title="Concept Bottleneck Agent Learning Curves",
            save_path=os.path.join(FIGURE_DIR, "fig1_learning_curves.png"),
        )
        generated.append("fig1_learning_curves.png")
    else:
        print("  Skipped: No metrics files found.")

    # ---- Figure 2: Concept Usage Heatmaps ----
    print("Generating Figure 2: Concept Usage Heatmaps...")
    for algo in ["ppo", "dqn"]:
        sm_json = os.path.join(model_dir, "bottleneck", f"strategies_{algo}.json")
        sm_pkl = os.path.join(model_dir, "bottleneck", f"strategy_memory_{algo}.pkl")

        path = sm_json if os.path.exists(sm_json) else sm_pkl
        if os.path.exists(path):
            plot_concept_usage_heatmap(
                path,
                title=f"Concept Usage — {algo.upper()} Bottleneck",
                save_path=os.path.join(FIGURE_DIR, f"fig2_concepts_{algo}.png"),
            )
            generated.append(f"fig2_concepts_{algo}.png")
        else:
            print(f"  Skipped {algo}: No strategy data found.")

    # ---- Figure 3: Intervention Results ----
    print("Generating Figure 3: Intervention Results...")
    interv_files = []
    interv_labels = []
    for algo in ["ppo", "dqn"]:
        path = os.path.join(results_dir, f"intervention_{algo}.json")
        if os.path.exists(path):
            interv_files.append(path)
            interv_labels.append(f"{algo.upper()} Bottleneck")

    # Also check CartPole interventions
    for algo in ["ppo", "dqn"]:
        path = os.path.join(results_dir, f"intervention_cartpole_{algo}.json")
        if os.path.exists(path):
            interv_files.append(path)
            interv_labels.append(f"CartPole {algo.upper()}")

    if interv_files:
        plot_intervention_results(
            interv_files, labels=interv_labels,
            title="Causal Concept Intervention Results",
            save_path=os.path.join(FIGURE_DIR, "fig3_intervention.png"),
        )
        generated.append("fig3_intervention.png")
    else:
        print("  Skipped: No intervention results found.")

    # ---- Figure 4: Ablation Impact ----
    print("Generating Figure 4: Ablation Impact...")
    for algo in ["ppo", "dqn"]:
        path = os.path.join(results_dir, f"ablation_{algo}.json")
        if os.path.exists(path):
            plot_ablation_results(
                path,
                title=f"Strategy Ablation Impact — {algo.upper()}",
                save_path=os.path.join(FIGURE_DIR, f"fig4_ablation_{algo}.png"),
            )
            generated.append(f"fig4_ablation_{algo}.png")
        else:
            print(f"  Skipped {algo}: No ablation results found.")

    # ---- Figure 5: Stability Results ----
    print("Generating Figure 5: Stability Under Symmetry...")
    stability_results = []
    for algo in ["ppo", "dqn"]:
        path = os.path.join(results_dir, f"stability_{algo}.json")
        if os.path.exists(path):
            with open(path) as f:
                stability_results.append(json.load(f))

    if stability_results:
        fig, axes = plt.subplots(1, len(stability_results), figsize=(7 * len(stability_results), 5))
        if len(stability_results) == 1:
            axes = [axes]

        for i, sr in enumerate(stability_results):
            algo_name = sr.get("algo", "unknown").upper()
            dist = sr.get("unique_count_distribution", {})

            x_vals = sorted(int(k) for k in dist.keys())
            y_vals = [dist[str(k)] for k in x_vals]

            axes[i].bar(x_vals, y_vals, color=COLORS.get(f"{sr.get('algo', 'ppo')}_bottleneck"),
                        edgecolor="black", linewidth=0.5)
            axes[i].set_xlabel("Unique Concepts per Symmetry Group")
            axes[i].set_ylabel("Number of States")
            axes[i].set_title(f"{algo_name}: Concept Stability\n"
                             f"(Exact match: {sr.get('exact_match_rate', 0):.1%}, "
                             f"Pairwise: {sr.get('pairwise_consistency', 0):.1%})")

        fig.suptitle("Concept Stability Under Board Symmetries", fontsize=14, fontweight="bold")
        save_path = os.path.join(FIGURE_DIR, "fig5_stability.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append("fig5_stability.png")
    else:
        print("  Skipped: No stability results found.")

    # ---- Figure 6: Dynamics Model Results ----
    print("Generating Figure 6: Dynamics Model...")
    dynamics_results = []
    for algo in ["ppo", "dqn"]:
        path = os.path.join(results_dir, f"dynamics_{algo}.json")
        if os.path.exists(path):
            with open(path) as f:
                dynamics_results.append(json.load(f))

    if dynamics_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        algos = [r.get("algo", "?").upper() for r in dynamics_results]
        top1_accs = [r["dynamics_training"]["test_accuracy"] for r in dynamics_results]
        top5_accs = [r["dynamics_training"]["top5_test_accuracy"] for r in dynamics_results]

        x = np.arange(len(algos))
        width = 0.35
        ax.bar(x - width/2, top1_accs, width, label="Top-1", color="#2196F3")
        ax.bar(x + width/2, top5_accs, width, label="Top-5", color="#FF9800")

        ax.set_ylabel("Accuracy")
        ax.set_title("Concept Dynamics Prediction Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.3, label="40% threshold")

        for i, (t1, t5) in enumerate(zip(top1_accs, top5_accs)):
            ax.text(i - width/2, t1 + 0.02, f"{t1:.1%}", ha="center", fontsize=9)
            ax.text(i + width/2, t5 + 0.02, f"{t5:.1%}", ha="center", fontsize=9)

        save_path = os.path.join(FIGURE_DIR, "fig6_dynamics.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append("fig6_dynamics.png")
    else:
        print("  Skipped: No dynamics results found.")

    # ---- Summary ----
    print(f"\nGenerated {len(generated)} figures in {FIGURE_DIR}/:")
    for name in generated:
        print(f"  - {name}")

    if not generated:
        print("  No figures generated. Run experiments first!")

    return generated


if __name__ == "__main__":
    generate_all_figures()
