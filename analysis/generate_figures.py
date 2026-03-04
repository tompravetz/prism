"""Generate publication figures for PRISM paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path("results")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ──────────────────────────────────────────────
# Fig 1: Fine-tuning learning curves
# ──────────────────────────────────────────────
def fig_finetune():
    with open(RESULTS / "finetune_transfer.json") as f:
        d = json.load(f)

    steps_per_gen = d["steps_per_gen"]
    thresh = d["convergence_threshold"]

    t_curve = d["transferred_curve"]
    s_curve = d["scratch_curve"]

    t_gens = [p["gen"] for p in t_curve]
    t_wrs  = [p["win_rate"] * 100 for p in t_curve]
    s_gens = [p["gen"] for p in s_curve]
    s_wrs  = [p["win_rate"] * 100 for p in s_curve]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(t_gens, t_wrs, "o-", color="#1f77b4", linewidth=1.8,
            markersize=5, label="PRISM transfer + fine-tune")
    ax.plot(s_gens, s_wrs, "s--", color="#d62728", linewidth=1.8,
            markersize=5, label="From scratch")

    # threshold line
    ax.axhline(thresh * 100, color="gray", linestyle=":", linewidth=1.2,
               label=f"{int(thresh*100)}% threshold")

    # mark convergence point
    conv_gen = d.get("transferred_convergence_gen")
    if conv_gen is not None:
        conv_wr = next(p["win_rate"] for p in t_curve if p["gen"] == conv_gen) * 100
        ax.annotate(
            f"Gen {conv_gen}\n({conv_gen * steps_per_gen // 1000}K steps)",
            xy=(conv_gen, conv_wr), xytext=(conv_gen + 4, conv_wr + 8),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            fontsize=8,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Win Rate vs. GnuGo (%)")
    ax.set_title("Fine-Tuning After Zero-Shot Transfer")
    ax.set_xlim(-1, max(t_gens) + 2)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT / "fig_finetune.pdf")
    fig.savefig(OUT / "fig_finetune.png")
    plt.close(fig)
    print("fig_finetune saved")


# ──────────────────────────────────────────────
# Fig 2: Concept ablation scatter
# ──────────────────────────────────────────────
def fig_ablation():
    with open(RESULTS / "ablation_ppo_L1.json") as f:
        d = json.load(f)

    baseline = d["baseline_win_rate"] * 100
    results = d["ablation_results"]

    freqs = [r["concept_frequency"] * 100 for r in results]
    drops = [r["win_rate_drop"] * 100 for r in results]
    cids  = [r["concept_id"] for r in results]

    # colour by drop magnitude
    colours = []
    sizes   = []
    for drop in drops:
        if drop >= 40:
            colours.append("#d62728")
            sizes.append(90)
        elif drop >= 7:
            colours.append("#ff7f0e")
            sizes.append(60)
        elif drop >= 0:
            colours.append("#1f77b4")
            sizes.append(40)
        else:
            colours.append("#aec7e8")
            sizes.append(40)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.scatter(freqs, drops, c=colours, s=sizes, alpha=0.8, zorder=3)

    # annotate notable concepts
    notable = {"C16": (0.15391279237655212 * 100, 0.482 * 100),
               "C47": (0.3300606410626624 * 100,  0.094 * 100),
               "C11": (0.1547790932717297 * 100,  0.046 * 100)}
    offsets = {"C16": (1.5, 3), "C47": (-5, 5), "C11": (1.5, -6)}
    for label, (fx, dy) in notable.items():
        ox, oy = offsets[label]
        ax.annotate(label, xy=(fx, dy), xytext=(fx + ox, dy + oy),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="#444", lw=0.7))

    # legend proxies
    patches = [
        mpatches.Patch(color="#d62728", label="Drop ≥ 40pp"),
        mpatches.Patch(color="#ff7f0e", label="Drop 7–40pp"),
        mpatches.Patch(color="#1f77b4", label="Drop 0–7pp"),
        mpatches.Patch(color="#aec7e8", label="Helps when removed"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Concept Frequency (%)")
    ax.set_ylabel("Win-Rate Drop when Ablated (pp)")
    ax.set_title("Concept Ablation: Frequency vs. Strategic Importance")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf")
    fig.savefig(OUT / "fig_ablation.png")
    plt.close(fig)
    print("fig_ablation saved")


# ──────────────────────────────────────────────
# Fig 3: K sensitivity sweep
# ──────────────────────────────────────────────
def fig_k_sweep():
    with open(RESULTS / "k_ablation.json") as f:
        d = json.load(f)

    sweep = d["sweep"]
    Ks     = [s["K"] for s in sweep]
    direct = [s["ppo_direct_win_rate"] * 100 for s in sweep]
    trans  = [s["transfer_win_rate"] * 100 for s in sweep]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(Ks, direct, "o-", color="#2ca02c", linewidth=1.8,
            markersize=6, label="Direct (PPO bottleneck vs. GnuGo L3)")
    ax.plot(Ks, trans,  "s--", color="#9467bd", linewidth=1.8,
            markersize=6, label="Transfer (PPO→DQN, Hungarian)")

    # mark K=64 operating point
    ax.axvline(64, color="gray", linestyle=":", linewidth=1.2, label="K=64 (paper)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("Number of Concepts K")
    ax.set_ylabel("Win Rate vs. GnuGo L3 (%)")
    ax.set_title("K Sensitivity: Transfer vs. Direct Performance")
    ax.set_ylim(-5, 110)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT / "fig_k_sweep.pdf")
    fig.savefig(OUT / "fig_k_sweep.png")
    plt.close(fig)
    print("fig_k_sweep saved")


# ──────────────────────────────────────────────
# Fig 4: Transfer results bar chart (Table 1 visual)
# ──────────────────────────────────────────────
def fig_transfer():
    with open(RESULTS / "transfer_same_task_10seed_L1.json") as f:
        d = json.load(f)

    # build ordered entries from data
    pairs = {
        (p["source"], p["target"]): p for p in d["pairs"]
    }

    labels = [
        "BC → DQN",
        "PPO → DQN",
        "DQN → PPO",
        "DAgger → PPO",
        "DQN → DAgger",
        "PPO → DAgger",
    ]
    pair_keys = [
        ("DAgger", "DQN"),
        ("PPO",    "DQN"),
        ("DQN",    "PPO"),
        ("DAgger", "PPO"),
        ("DQN",    "DAgger"),
        ("PPO",    "DAgger"),
    ]
    colours = ["#1f77b4", "#1f77b4", "#aec7e8", "#aec7e8", "#aec7e8", "#d62728"]
    means = []
    errs  = []
    for key in pair_keys:
        if key in pairs:
            p = pairs[key]
            means.append(p["mean_wr"] * 100)
            errs.append(p["std_wr"] * 100)
        else:
            means.append(0.0)
            errs.append(0.0)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, means, xerr=errs, color=colours, alpha=0.85,
                   height=0.6, error_kw=dict(elinewidth=1.2, capsize=3))

    # Reference lines
    ax.axvline(50, color="gray", linestyle="--", linewidth=1.0, label="50% null")
    ax.axvline(3.5, color="#888", linestyle=":", linewidth=1.0, label="Random (3.5%)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Win Rate vs. GnuGo L1 (%)")
    ax.set_title("Zero-Shot Transfer Results (10 seeds × 100 games)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer.pdf")
    fig.savefig(OUT / "fig_transfer.png")
    plt.close(fig)
    print("fig_transfer saved")


if __name__ == "__main__":
    fig_finetune()
    fig_ablation()
    fig_k_sweep()
    fig_transfer()
    print("All figures saved to", OUT)
