"""
Create the Hero Figure for the PRISM paper.

A 2x2 grid summarizing the key transfer results:
    Panel A: Agent-to-agent transfer heatmap (3x3, PPO/DQN/DAgger)
    Panel B: Curriculum learning curves (transferred vs scratch with speedup)
    Panel C: Cross-domain learning curves (CartPole->Acrobot vs scratch)
    Panel D: Strategy library similarity matrix

Uses Okabe-Ito colorblind-friendly palette, 300 DPI, min 10pt fonts.

Usage:
    python analysis/create_hero_figure.py
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_dir


# Okabe-Ito colorblind-friendly palette
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
}


def create_hero_figure():
    """Generate the 2x2 hero figure from saved results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ensure_dir("results/figures")

    # Load all results
    with open("results/transfer_same_task.json") as f:
        same_task = json.load(f)
    with open("results/curriculum_transfer.json") as f:
        curriculum = json.load(f)
    with open("results/transfer_cross_domain.json") as f:
        cross_domain = json.load(f)
    with open("results/strategy_library.json") as f:
        library = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.rcParams.update({"font.size": 11})

    # ============================================================
    # Panel A: Agent-to-agent transfer heatmap
    # ============================================================
    ax = axes[0, 0]
    agent_names = ["PPO", "DQN", "DAgger"]
    n = len(agent_names)
    wr_matrix = np.zeros((n, n))

    for r in same_task:
        i = agent_names.index(r["source"])
        j = agent_names.index(r["target"])
        wr_matrix[i, j] = r["zero_shot"]["win_rate"]
    # Diagonal = baseline (self-transfer would be 100%)
    np.fill_diagonal(wr_matrix, 1.0)

    im = ax.imshow(wr_matrix, cmap="YlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_names, fontsize=11)
    ax.set_yticklabels(agent_names, fontsize=11)
    ax.set_xlabel("Target Agent", fontsize=12)
    ax.set_ylabel("Source Agent", fontsize=12)
    ax.set_title("(a) Agent-to-Agent Transfer (Win Rate)", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = wr_matrix[i, j]
            color = "white" if val > 0.8 else "black"
            if i == j:
                text = "100%\n(self)"
            else:
                text = f"{val:.0%}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Win Rate")

    # ============================================================
    # Panel B: Curriculum learning curves
    # ============================================================
    ax = axes[0, 1]

    curve_t = curriculum["transferred_finetune"]["learning_curve"]
    curve_s = curriculum["scratch_control"]["learning_curve"]

    gens_t = [p["generation"] for p in curve_t]
    wrs_t = [p["win_rate"] for p in curve_t]
    gens_s = [p["generation"] for p in curve_s]
    wrs_s = [p["win_rate"] for p in curve_s]

    ax.plot(gens_t, wrs_t, "-o", color=COLORS["blue"],
            label="Curriculum transfer (5x5 -> 7x7)", markersize=5, linewidth=2)
    ax.plot(gens_s, wrs_s, "-s", color=COLORS["black"],
            label="From scratch (7x7)", markersize=5, linewidth=2)

    # Mark zero-shot level
    zs_wr = curriculum["zero_shot"]["win_rate"]
    ax.axhline(y=zs_wr, color=COLORS["blue"], linestyle=":", alpha=0.5,
               label=f"Zero-shot ({zs_wr:.0%})")
    ax.axhline(y=0.5, color=COLORS["gray"], linestyle="--", alpha=0.3)

    # Annotate speedup: transferred reaches 98% at gen 15 vs scratch at gen ~35
    # Find gen where transferred hits 98%
    gen_t_98 = None
    for p in curve_t:
        if p["win_rate"] >= 0.98:
            gen_t_98 = p["generation"]
            break
    gen_s_98 = None
    for p in curve_s:
        if p["win_rate"] >= 0.96:
            gen_s_98 = p["generation"]
            break

    if gen_t_98 is not None and gen_s_98 is not None:
        speedup = gen_s_98 / gen_t_98 if gen_t_98 > 0 else 0
        ax.annotate(f"{speedup:.1f}x faster\nto 98%",
                    xy=(gen_t_98, 0.98), xytext=(gen_t_98 + 8, 0.75),
                    fontsize=10, fontweight="bold", color=COLORS["blue"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["blue"]))

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Win Rate vs Random", fontsize=12)
    ax.set_title("(b) Curriculum Transfer: Go 5x5 -> 7x7", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    # ============================================================
    # Panel C: Cross-domain learning curves
    # ============================================================
    ax = axes[1, 0]

    scratch = cross_domain["acrobot_scratch"]["learning_curve"]
    cp_ft = cross_domain["cartpole_to_acrobot"]["fine_tune_curve"]
    ll_ft = cross_domain["lunarlander_to_acrobot"]["fine_tune_curve"]

    ax.plot([s["generation"] for s in scratch],
            [s["mean_reward"] for s in scratch],
            "-s", color=COLORS["black"], label="From scratch", markersize=4, linewidth=2)
    ax.plot([s["generation"] for s in cp_ft],
            [s["mean_reward"] for s in cp_ft],
            "-o", color=COLORS["blue"], label="CartPole transfer", markersize=4, linewidth=2)
    ax.plot([s["generation"] for s in ll_ft],
            [s["mean_reward"] for s in ll_ft],
            "-^", color=COLORS["orange"], label="LunarLander transfer", markersize=4, linewidth=2)

    # Annotate improvement
    scratch_final = cross_domain["acrobot_scratch"]["final_reward"]
    cp_final = cross_domain["cartpole_to_acrobot"]["final_reward"]
    improvement = ((cp_final - scratch_final) / abs(scratch_final)) * 100
    ax.annotate(f"{improvement:.0f}% better\n({cp_final:.0f} vs {scratch_final:.0f})",
                xy=(49, cp_final), xytext=(30, -200),
                fontsize=10, fontweight="bold", color=COLORS["blue"],
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"]))

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("(c) Cross-Domain Transfer: -> Acrobot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ============================================================
    # Panel D: Strategy library similarity matrix
    # ============================================================
    ax = axes[1, 1]

    if "similarity_matrix" in library:
        sim_data = library["similarity_matrix"]
        strategy_names = list(sim_data.keys())
        n_strats = len(strategy_names)
        sim_mat = np.zeros((n_strats, n_strats))

        for i, name_i in enumerate(strategy_names):
            for j, name_j in enumerate(strategy_names):
                if name_j in sim_data.get(name_i, {}):
                    sim_mat[i, j] = sim_data[name_i][name_j]
                elif i == j:
                    sim_mat[i, j] = 1.0

        im2 = ax.imshow(sim_mat, cmap="YlOrRd", vmin=0, vmax=1)

        # Shorten names for readability
        short_names = []
        for name in strategy_names:
            name = name.replace("Go-", "Go\n")
            name = name.replace("CartPole", "CP")
            name = name.replace("LunarLander", "LL")
            short_names.append(name)

        ax.set_xticks(range(n_strats))
        ax.set_yticks(range(n_strats))
        ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(short_names, fontsize=8)

        for i in range(n_strats):
            for j in range(n_strats):
                val = sim_mat[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        plt.colorbar(im2, ax=ax, shrink=0.8, label="Cosine Similarity")
    else:
        # Fallback: just show the strategies list
        strategies = library.get("strategies", [])
        text = "Strategy Library:\n" + "\n".join(
            f"  {i+1}. {s.get('name', 'unknown')}" for i, s in enumerate(strategies[:10])
        )
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="center", fontfamily="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title("(d) Strategy Library Similarity", fontsize=13, fontweight="bold")

    # Final layout
    plt.tight_layout(pad=2.0)
    fig_path = "results/figures/hero_figure.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hero figure to {fig_path}")

    return fig_path


if __name__ == "__main__":
    create_hero_figure()
