"""
Visualization utilities for PRISM results.

Generates individual plots and visualizations used throughout the analysis.
For the complete figure generation pipeline, see figures.py.

Available visualizations:
    - Learning curves (win rate over generations)
    - Concept usage heatmaps
    - Intervention effect charts
    - Ablation waterfall charts
    - Codebook utilization (for VQ)
"""

import os
import sys
import json
import numpy as np

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import matplotlib with non-interactive backend (works on headless servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.utils import ensure_dir


# ============================================================
# Styling
# ============================================================

# Consistent color scheme for the paper
COLORS = {
    "ppo_full": "#2196F3",      # Blue — PPO baseline
    "ppo_bottleneck": "#1565C0", # Dark blue — PPO bottleneck
    "dqn_full": "#FF9800",      # Orange — DQN baseline
    "dqn_bottleneck": "#E65100", # Dark orange — DQN bottleneck
    "vq": "#4CAF50",            # Green — VQ variant
    "highlight": "#F44336",      # Red — for emphasis
}


def setup_style():
    """Set up consistent matplotlib style for publication figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })


# ============================================================
# Learning Curves
# ============================================================

def plot_learning_curves(metrics_files, labels=None, title="Learning Curves",
                         save_path=None):
    """
    Plot win rate / reward over training generations.

    Args:
        metrics_files: List of paths to metrics JSON files (one per variant).
        labels: List of labels for each variant.
        title: Plot title.
        save_path: Where to save the figure. None = show interactively.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    color_list = list(COLORS.values())

    for i, path in enumerate(metrics_files):
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        with open(path) as f:
            metrics = json.load(f)

        gens = [m["generation"] for m in metrics]
        win_rates = [m.get("win_rate", 0) for m in metrics]
        avg_rewards = [m.get("avg_reward", 0) for m in metrics]
        label = labels[i] if labels else os.path.basename(path)
        color = color_list[i % len(color_list)]

        # Smooth the curves for readability (moving average)
        window = min(5, len(win_rates) // 5) if len(win_rates) > 10 else 1
        if window > 1:
            win_smooth = np.convolve(win_rates, np.ones(window)/window, mode='valid')
            reward_smooth = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
            gens_smooth = gens[:len(win_smooth)]
        else:
            win_smooth = win_rates
            reward_smooth = avg_rewards
            gens_smooth = gens

        ax1.plot(gens_smooth, win_smooth, label=label, color=color, linewidth=2)
        ax2.plot(gens_smooth, reward_smooth, label=label, color=color, linewidth=2)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Win Rate vs Random")
    ax1.set_title("Win Rate Over Training")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Average Episode Reward")
    ax2.set_title("Reward Over Training")
    ax2.legend()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to {save_path}")
    plt.close(fig)


# ============================================================
# Concept Usage Heatmap
# ============================================================

def plot_concept_usage_heatmap(strategy_memory_path, n_concepts=64,
                                title="Concept Usage", save_path=None):
    """
    Plot a heatmap showing how often each concept is used.

    The heatmap is arranged as an 8x8 grid (for 64 concepts).
    Darker cells = more frequently used concepts.
    """
    setup_style()

    # Load strategy data (JSON format)
    with open(strategy_memory_path) as f:
        data = json.load(f)
    concept_dist = data.get("concept_distribution", {})

    # Create grid
    grid_size = int(np.ceil(np.sqrt(n_concepts)))
    grid = np.zeros((grid_size, grid_size))
    for concept_id, count in concept_dist.items():
        idx = int(concept_id)
        if idx < n_concepts:
            row = idx // grid_size
            col = idx % grid_size
            grid[row][col] = count

    # Normalize for visualization
    if grid.max() > 0:
        grid = grid / grid.max()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="equal", vmin=0, vmax=1)

    ax.set_xlabel("Concept ID (col)")
    ax.set_ylabel("Concept ID (row)")
    ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Relative Usage Frequency")

    # Add concept ID labels for smaller grids
    if n_concepts <= 64:
        for i in range(grid_size):
            for j in range(grid_size):
                concept_id = i * grid_size + j
                if concept_id < n_concepts:
                    color = "white" if grid[i][j] > 0.5 else "black"
                    ax.text(j, i, str(concept_id), ha="center", va="center",
                            fontsize=7, color=color)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Concept heatmap saved to {save_path}")
    plt.close(fig)


# ============================================================
# Intervention Results
# ============================================================

def plot_intervention_results(results_files, labels=None,
                               title="Causal Intervention Results",
                               save_path=None):
    """
    Bar chart comparing intervention change rates across algorithms.

    Shows: overall change rate, concept specificity, active concepts.
    """
    setup_style()

    results = []
    for path in results_files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            results.append(json.load(f))

    if not results:
        print("No intervention results found.")
        return

    if labels is None:
        labels = [r.get("algo", f"variant{i}").upper() for i, r in enumerate(results)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Change rate comparison
    change_rates = [r["overall_change_rate"] for r in results]
    colors = [COLORS.get(f"{r.get('algo', 'ppo')}_bottleneck", COLORS["ppo_bottleneck"])
              for r in results]
    axes[0].bar(labels, change_rates, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    axes[0].set_ylabel("Action Change Rate")
    axes[0].set_title("Concept Override: Action Change Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Concept specificity
    specs = [r.get("mean_concept_specificity", 0) for r in results]
    axes[1].bar(labels, specs, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Mean Concept Specificity")
    axes[1].set_title("How Specific Are Concepts?")
    axes[1].set_ylim(0, 1)

    # Active concepts
    active = [r.get("n_active_concepts", 0) for r in results]
    total = [r.get("n_concepts", 64) if "n_concepts" in r else 64 for r in results]
    axes[2].bar(labels, active, color=colors, edgecolor="black", linewidth=0.5)
    for i, (a, t) in enumerate(zip(active, total)):
        axes[2].text(i, a + 1, f"{a}/{t}", ha="center", fontsize=9)
    axes[2].set_ylabel("Active Concepts")
    axes[2].set_title("Codebook Utilization")

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Intervention plot saved to {save_path}")
    plt.close(fig)


# ============================================================
# Ablation Impact Chart
# ============================================================

def plot_ablation_results(ablation_path, title="Strategy Ablation Impact",
                           save_path=None):
    """
    Waterfall/bar chart showing win rate drop for each ablated strategy.

    Strategies are sorted by impact (largest drop first).
    """
    setup_style()

    if not os.path.exists(ablation_path):
        print(f"Ablation results not found at {ablation_path}")
        return

    with open(ablation_path) as f:
        data = json.load(f)

    baseline_wr = data["baseline_win_rate"]
    results = data["ablation_results"]

    if not results:
        print("No ablation results to plot.")
        return

    # Sort by impact (biggest drop first)
    results.sort(key=lambda x: x["win_rate_drop"], reverse=True)

    # Support both concept-level (new) and concept+action (old) ablation formats
    if "action" in results[0]:
        labels = [f"C{r['concept_id']}→A{r['action']}" for r in results]
    else:
        labels = [f"C{r['concept_id']}" for r in results]
    drops = [r["win_rate_drop"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color: red for significant drops (>5%), gray for non-significant
    colors = ["#F44336" if d > 0.05 else "#9E9E9E" for d in drops]

    bars = ax.bar(range(len(labels)), drops, color=colors, edgecolor="black",
                   linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Win Rate Drop")
    ax.set_title(f"{title}\n(Baseline win rate: {baseline_wr:.1%})")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.3, label="5% threshold")
    ax.legend()

    # Add value labels on significant bars
    for i, (bar, drop) in enumerate(zip(bars, drops)):
        if drop > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{drop:.1%}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Ablation plot saved to {save_path}")
    plt.close(fig)
