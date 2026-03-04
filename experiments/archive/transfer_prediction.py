"""
Transfer Quality Prediction Analysis.

Analyzes what predicts transfer success across ALL experimental results.
Uses properly normalized metrics:
  - Same-domain Go: zero-shot win rate (0-1 scale)
  - Cross-domain: improvement percentage over from-scratch baseline
  - Source quality analysis: how source agent strength predicts transfer

This script reads saved results files (no new training needed).

Usage:
    python experiments/transfer_prediction.py
"""

import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_dir


def collect_go_transfer_points():
    """
    Collect same-domain Go transfer data points where metrics are comparable
    (all zero-shot win rates on 0-1 scale).

    Sources: same_task, same_task_5seed, transitive_direct, transitive_chained,
             curriculum, alignment_comparison.

    Returns:
        List of dicts with: source, target, alignment_sim, win_rate, experiment,
                            source_type (rl/bc), is_chained (bool)
    """
    points = []

    # ---- Multi-seed same-task (preferred over single-run) ----
    path = "results/transfer_same_task_5seed.json"
    if os.path.exists(path):
        print(f"  Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        for r in data.get("pairs", []):
            if isinstance(r, dict) and "alignment" in r:
                # Classify source type: PPO/DQN = RL-trained, DAgger = BC-trained
                src = r.get("source", "?")
                points.append({
                    "source": src,
                    "target": r.get("target", "?"),
                    "alignment_sim": float(r["alignment"]["mean_similarity"]),
                    "win_rate": float(r["mean_wr"]),
                    "std_wr": float(r.get("std_wr", 0)),
                    "experiment": "same_task_5seed",
                    "source_type": "bc" if "DAgger" in src else "rl",
                    "is_chained": False,
                })
    else:
        # Fall back to single-run data
        path = "results/transfer_same_task.json"
        if os.path.exists(path):
            print(f"  Loading {path}...")
            with open(path) as f:
                data = json.load(f)
            for r in data:
                src = r["source"]
                points.append({
                    "source": src,
                    "target": r["target"],
                    "alignment_sim": float(r["alignment"]["mean_similarity"]),
                    "win_rate": float(r["zero_shot"]["win_rate"]),
                    "std_wr": 0.0,
                    "experiment": "same_task",
                    "source_type": "bc" if "DAgger" in src else "rl",
                    "is_chained": False,
                })

    # ---- Transitive transfer ----
    path = "results/transitive_transfer.json"
    if os.path.exists(path):
        print(f"  Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        for r in data.get("chains", []):
            src = r["source"]
            # Direct alignment
            points.append({
                "source": src,
                "target": r["target"],
                "alignment_sim": float(r["direct"]["alignment_sim"]),
                "win_rate": float(r["direct"]["mean_wr"]),
                "std_wr": float(r["direct"].get("std_wr", 0)),
                "experiment": "transitive_direct",
                "source_type": "bc" if "DAgger" in src else "rl",
                "is_chained": False,
            })
            # Chained alignment
            points.append({
                "source": f"{src}(via {r['intermediate']})",
                "target": r["target"],
                "alignment_sim": float(r["chained"]["alignment_sim"]),
                "win_rate": float(r["chained"]["mean_wr"]),
                "std_wr": float(r["chained"].get("std_wr", 0)),
                "experiment": "transitive_chained",
                "source_type": "bc" if "DAgger" in src else "rl",
                "is_chained": True,
            })

    # ---- Curriculum transfer ----
    path = "results/curriculum_transfer.json"
    if os.path.exists(path):
        print(f"  Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        if "alignment" in data and "zero_shot" in data:
            points.append({
                "source": "Go_5x5",
                "target": "Go_7x7",
                "alignment_sim": float(data["alignment"]["mean_similarity"]),
                "win_rate": float(data["zero_shot"]["win_rate"]),
                "std_wr": 0.0,
                "experiment": "curriculum",
                "source_type": "rl",
                "is_chained": False,
            })

    # ---- Alignment comparison (different methods, same pair) ----
    path = "results/alignment_comparison.json"
    if os.path.exists(path):
        print(f"  Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        for method_key, r in data.get("methods", {}).items():
            points.append({
                "source": f"PPO({method_key})",
                "target": "DQN",
                "alignment_sim": float(r["alignment_sim"]),
                "win_rate": float(r["mean_wr"]),
                "std_wr": float(r.get("std_wr", 0)),
                "experiment": f"alignment_{method_key}",
                "source_type": "rl",
                "is_chained": False,
            })

    return points


def collect_cross_domain_points():
    """
    Collect cross-domain transfer data normalized as improvement percentage
    over from-scratch baseline.

    Returns:
        List of dicts with: source, target, alignment_sim, improvement_pct, experiment
    """
    points = []

    # ---- Tuned cross-domain matrix (preferred) ----
    path = "results/transfer_matrix_tuned.json"
    if os.path.exists(path):
        print(f"  Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        for key, r in data.get("transfer_matrix", {}).items():
            # Skip self-transfers (alignment=1.0, not interesting)
            if r["source"] == r["target"]:
                continue
            points.append({
                "source": r["source"],
                "target": r["target"],
                "alignment_sim": float(r["alignment_sim"]),
                "improvement_pct": float(r.get("improvement_pct", 0)),
                "fine_tune_mean": float(r.get("fine_tune_mean", 0)),
                "fine_tune_std": float(r.get("fine_tune_std", 0)),
                "significant": bool(r.get("significance", {}).get("significant", False)),
                "experiment": "cross_domain_tuned",
            })

    return points


def run_prediction_analysis():
    """
    Multi-level analysis of what predicts transfer success.

    Level 1: Same-domain Go transfers (comparable WR metric)
    Level 2: Cross-domain transfers (improvement percentage)
    Level 3: Source quality analysis (RL vs BC encoders)
    """
    print("=" * 64)
    print("Transfer Quality Prediction Analysis")
    print("=" * 64)

    from scipy import stats

    # ============================================================
    # Level 1: Same-domain Go analysis (WR metric, 0-1 scale)
    # ============================================================
    print("\n--- Level 1: Same-Domain Go Transfer ---")
    go_points = collect_go_transfer_points()
    print(f"  Collected {len(go_points)} Go transfer data points")

    if len(go_points) >= 3:
        sims = np.array([p["alignment_sim"] for p in go_points])
        wrs = np.array([p["win_rate"] for p in go_points])

        # Overall regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(sims, wrs)
        r_sq = r_value ** 2
        print(f"\n  Overall regression (n={len(go_points)}):")
        print(f"    WR = {slope:.3f} * mu + {intercept:.3f}")
        print(f"    R-squared: {r_sq:.4f}, p={p_value:.6f}")

        # Direct transfers only (exclude chained)
        direct_mask = np.array([not p["is_chained"] for p in go_points])
        if direct_mask.sum() >= 3:
            d_slope, d_int, d_r, d_p, d_se = stats.linregress(
                sims[direct_mask], wrs[direct_mask]
            )
            print(f"\n  Direct transfers only (n={direct_mask.sum()}):")
            print(f"    R-squared: {d_r**2:.4f}, p={d_p:.6f}")

        # RL-source only
        rl_mask = np.array([p["source_type"] == "rl" for p in go_points])
        if rl_mask.sum() >= 3:
            rl_slope, rl_int, rl_r, rl_p, rl_se = stats.linregress(
                sims[rl_mask], wrs[rl_mask]
            )
            print(f"\n  RL-source only (n={rl_mask.sum()}):")
            print(f"    R-squared: {rl_r**2:.4f}, p={rl_p:.6f}")
        else:
            rl_r, rl_p = 0, 1

        # Source quality comparison: RL vs BC
        rl_wrs = wrs[rl_mask]
        bc_mask = ~rl_mask
        bc_wrs = wrs[bc_mask]
        if len(rl_wrs) >= 2 and len(bc_wrs) >= 2:
            t_src, p_src = stats.ttest_ind(rl_wrs, bc_wrs, equal_var=False)
            print(f"\n  Source quality comparison:")
            print(f"    RL-source mean WR: {rl_wrs.mean():.3f} (n={len(rl_wrs)})")
            print(f"    BC-source mean WR: {bc_wrs.mean():.3f} (n={len(bc_wrs)})")
            print(f"    Welch's t={t_src:.3f}, p={p_src:.6f}")
        else:
            t_src, p_src = 0, 1

        # Threshold analysis (same-domain only)
        print(f"\n  Threshold analysis:")
        best_thresh, best_acc = 0.0, 0.0
        for thresh in np.arange(0.20, 0.40, 0.01):
            predicted = sims >= thresh
            actual = wrs > 0.5
            acc = np.mean(predicted == actual)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        print(f"    Best: mu >= {best_thresh:.2f} predicts WR > 50%"
              f" (accuracy: {best_acc:.1%})")
    else:
        print("  Not enough Go data points for analysis.")
        r_sq, p_value, slope, intercept = 0, 1, 0, 0
        best_thresh, best_acc = 0.24, 0.5
        t_src, p_src = 0, 1
        rl_r, rl_p = 0, 1

    # ============================================================
    # Level 2: Cross-domain transfer (improvement %)
    # ============================================================
    print("\n--- Level 2: Cross-Domain Transfer ---")
    cd_points = collect_cross_domain_points()
    print(f"  Collected {len(cd_points)} cross-domain data points")

    if len(cd_points) >= 3:
        cd_sims = np.array([p["alignment_sim"] for p in cd_points])
        cd_imps = np.array([p["improvement_pct"] for p in cd_points])

        cd_slope, cd_int, cd_r, cd_p, cd_se = stats.linregress(cd_sims, cd_imps)
        print(f"\n  Regression: improvement% = {cd_slope:.2f} * mu + {cd_int:.2f}")
        print(f"  R-squared: {cd_r**2:.4f}, p={cd_p:.6f}")
        print(f"  Mean improvement: {cd_imps.mean():.1f}% +/- {cd_imps.std():.1f}%")

        n_positive = (cd_imps > 0).sum()
        print(f"  Positive transfer: {n_positive}/{len(cd_imps)} pairs ({n_positive/len(cd_imps):.0%})")

        # One-sample t-test: is mean improvement > 0?
        t_imp, p_imp = stats.ttest_1samp(cd_imps, 0)
        print(f"  Mean improvement vs 0: t={t_imp:.3f}, p={p_imp:.4f}")
    else:
        print("  Not enough cross-domain data points.")
        cd_r, cd_p, cd_slope, cd_int = 0, 1, 0, 0
        t_imp, p_imp = 0, 1
        cd_imps = np.array([])

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/transfer_prediction.json"

    save_data = {
        "go_analysis": {
            "n_points": len(go_points),
            "regression": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_sq),
                "p_value": float(p_value),
            },
            "source_quality": {
                "rl_mean_wr": float(np.mean([p["win_rate"] for p in go_points
                                              if p["source_type"] == "rl"])) if go_points else 0,
                "bc_mean_wr": float(np.mean([p["win_rate"] for p in go_points
                                              if p["source_type"] == "bc"])) if go_points else 0,
                "t_statistic": float(t_src),
                "p_value": float(p_src),
            },
            "threshold": {
                "best_mu": float(best_thresh),
                "accuracy": float(best_acc),
            },
        },
        "cross_domain_analysis": {
            "n_points": len(cd_points),
            "regression": {
                "slope": float(cd_slope) if len(cd_points) >= 3 else 0,
                "intercept": float(cd_int) if len(cd_points) >= 3 else 0,
                "r_squared": float(cd_r**2) if len(cd_points) >= 3 else 0,
                "p_value": float(cd_p) if len(cd_points) >= 3 else 1,
            },
            "mean_improvement_pct": float(cd_imps.mean()) if len(cd_imps) > 0 else 0,
            "improvement_test_p": float(p_imp),
        },
        "go_data_points": go_points,
        "cross_domain_data_points": [
            {k: v for k, v in p.items()} for p in cd_points
        ],
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {output_path}")

    # ============================================================
    # Visualization: two-panel figure
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito palette
        OI_BLUE = "#0072B2"
        OI_ORANGE = "#E69F00"
        OI_RED = "#D55E00"
        OI_GREEN = "#009E73"
        OI_PURPLE = "#CC79A7"

        ensure_dir("results/figures")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ---- Panel A: Same-domain Go transfer ----
        ax = axes[0]
        if len(go_points) >= 3:
            sims = np.array([p["alignment_sim"] for p in go_points])
            wrs = np.array([p["win_rate"] for p in go_points])
            is_rl = np.array([p["source_type"] == "rl" for p in go_points])
            is_chained = np.array([p["is_chained"] for p in go_points])

            # RL direct
            m1 = is_rl & ~is_chained
            if m1.any():
                ax.scatter(sims[m1], wrs[m1], s=80, color=OI_BLUE,
                           edgecolors="black", linewidths=0.5, zorder=5,
                           label="RL source (direct)")
            # RL chained
            m2 = is_rl & is_chained
            if m2.any():
                ax.scatter(sims[m2], wrs[m2], s=60, color=OI_BLUE, marker="D",
                           edgecolors="black", linewidths=0.5, zorder=5, alpha=0.6,
                           label="RL source (chained)")
            # BC direct
            m3 = ~is_rl & ~is_chained
            if m3.any():
                ax.scatter(sims[m3], wrs[m3], s=80, color=OI_ORANGE,
                           marker="^", edgecolors="black", linewidths=0.5, zorder=5,
                           label="BC source (direct)")
            # BC chained
            m4 = ~is_rl & is_chained
            if m4.any():
                ax.scatter(sims[m4], wrs[m4], s=60, color=OI_ORANGE, marker="v",
                           edgecolors="black", linewidths=0.5, zorder=5, alpha=0.6,
                           label="BC source (chained)")

            # Regression line
            x_line = np.linspace(sims.min() - 0.02, sims.max() + 0.02, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "-", color=OI_RED, linewidth=2, alpha=0.7,
                    label=f"Regression (R$^2$={r_sq:.3f})")

            # 50% baseline
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4,
                        label="Random baseline (50%)")

        ax.set_xlabel("Mean Alignment Similarity ($\\mu$)", fontsize=12)
        ax.set_ylabel("Zero-Shot Win Rate", fontsize=12)
        ax.set_title("(a) Same-Domain Go Transfer", fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.2)

        # ---- Panel B: Cross-domain improvement ----
        ax = axes[1]
        if len(cd_points) >= 3:
            cd_sims = np.array([p["alignment_sim"] for p in cd_points])
            cd_imps = np.array([p["improvement_pct"] for p in cd_points])
            cd_sig = np.array([p["significant"] for p in cd_points])

            # Significant vs not
            if cd_sig.any():
                ax.scatter(cd_sims[cd_sig], cd_imps[cd_sig], s=100, color=OI_GREEN,
                           edgecolors="black", linewidths=0.5, zorder=5, marker="*",
                           label="Significant (p<0.05)")
            if (~cd_sig).any():
                ax.scatter(cd_sims[~cd_sig], cd_imps[~cd_sig], s=80, color=OI_PURPLE,
                           edgecolors="black", linewidths=0.5, zorder=5,
                           label="Not significant")

            # Annotate
            for p in cd_points:
                label = f"{p['source'][:3]}->{p['target'][:3]}"
                ax.annotate(label, (p["alignment_sim"], p["improvement_pct"]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, alpha=0.7)

            # Zero line
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4,
                        label="No improvement")

        ax.set_xlabel("Mean Alignment Similarity ($\\mu$)", fontsize=12)
        ax.set_ylabel("Improvement over From-Scratch (%)", fontsize=12)
        ax.set_title("(b) Cross-Domain Transfer Quality", fontsize=13)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        fig_path = "results/figures/transfer_prediction.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Quality Prediction Analysis")
    args = parser.parse_args()

    run_prediction_analysis()
