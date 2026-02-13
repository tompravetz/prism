"""
Tasks 5+6: Strategy Library Demo + Transfer Efficiency Benchmark.

Demonstrates the full PRISM strategy library system:
    1. Register ALL trained agents into the library
    2. Compute cross-strategy similarity matrix
    3. Find-similar demo: query with a new agent, get nearest matches
    4. Compile transfer efficiency metrics from all experiments
    5. Generate publication-quality figures

This script should be run AFTER all other experiments (Tasks 1-4) have
completed and saved their results.

Usage:
    python experiments/strategy_library_demo.py
"""

import os
import sys
import json
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.strategy_library import StrategyLibrary
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, ensure_dir


def register_all_strategies(library):
    """
    Register all available trained agents into the strategy library.

    Scans for existing model files and registers each one found.
    Gracefully skips models that don't exist.
    """
    # ---- Go 7x7 agents ----
    go_strategies = {
        "Go-PPO": {
            "concept_path": "models/bottleneck/concepts_ppo_k64.pkl",
            "policy_path": "models/bottleneck/ppo_bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 50,
            "metadata": {"domain": "go_7x7", "algo": "PPO", "K": 64},
        },
        "Go-DQN": {
            "concept_path": "models/bottleneck/concepts_dqn_k64.pkl",
            "policy_path": "models/bottleneck/dqn_bottleneck_final.pt",
            "policy_class": "dqn",
            "n_actions": 50,
            "metadata": {"domain": "go_7x7", "algo": "DQN", "K": 64},
        },
        "Go-DAgger": {
            "concept_path": "models/bottleneck_dagger/concepts_ppo_k64.pkl",
            "policy_path": "models/bottleneck_dagger/ppo_bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 50,
            "metadata": {"domain": "go_7x7", "algo": "DAgger", "K": 64},
        },
        "Go-BC": {
            "concept_path": "models/bottleneck_bc/concepts_ppo_k64.pkl",
            "policy_path": "models/bottleneck_bc/ppo_bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 50,
            "metadata": {"domain": "go_7x7", "algo": "BC", "K": 64},
        },
    }

    # ---- Classic control agents ----
    simple_strategies = {
        "CartPole": {
            "concept_path": "models/simple/concepts_cartpole_k32.pkl",
            "policy_path": "models/simple/ppo_cartpole_bottleneck.pt",
            "policy_class": "ppo",
            "n_actions": 2,
            "metadata": {"domain": "cartpole", "algo": "PPO", "K": 32},
        },
        "LunarLander": {
            "concept_path": "models/lunar_lander/concepts_k32.pkl",
            "policy_path": "models/lunar_lander/bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 4,
            "metadata": {"domain": "lunarlander", "algo": "PPO", "K": 32},
        },
    }

    # ---- Specialist agents (from composition experiment) ----
    specialist_strategies = {
        "Go-Aggressive": {
            "concept_path": "models/specialist_aggressive/concepts_k64.pkl",
            "policy_path": "models/specialist_aggressive/bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 50,
            "metadata": {"domain": "go_7x7", "algo": "PPO-Aggressive", "K": 64},
        },
        "Go-Territorial": {
            "concept_path": "models/specialist_territorial/concepts_k64.pkl",
            "policy_path": "models/specialist_territorial/bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 50,
            "metadata": {"domain": "go_7x7", "algo": "PPO-Territorial", "K": 64},
        },
    }

    # ---- Go 5x5 (from curriculum experiment) ----
    curriculum_strategies = {
        "Go-5x5": {
            "concept_path": "models/go_5x5/concepts_k64.pkl",
            "policy_path": "models/go_5x5/ppo_bottleneck_final.pt",
            "policy_class": "ppo",
            "n_actions": 26,
            "metadata": {"domain": "go_5x5", "algo": "PPO", "K": 64},
        },
    }

    # ---- Acrobot (from cross-domain experiment) ----
    acrobot_strategies = {
        "Acrobot": {
            "concept_path": "models/acrobot/concepts_k32.pkl",
            "policy_path": "models/acrobot/bottleneck_scratch.pt",
            "policy_class": "ppo",
            "n_actions": 3,
            "metadata": {"domain": "acrobot", "algo": "PPO", "K": 32},
        },
    }

    # Register all that exist
    all_strategies = {
        **go_strategies,
        **simple_strategies,
        **specialist_strategies,
        **curriculum_strategies,
        **acrobot_strategies,
    }

    registered = 0
    for name, config in all_strategies.items():
        concept_path = config["concept_path"]
        if not os.path.exists(concept_path):
            print(f"  Skipping '{name}': {concept_path} not found")
            continue

        try:
            library.add_strategy_from_paths(
                name=name,
                concept_path=config["concept_path"],
                policy_path=config.get("policy_path"),
                policy_class=config.get("policy_class", "ppo"),
                n_actions=config.get("n_actions", 50),
                metadata=config.get("metadata", {}),
            )
            registered += 1
        except Exception as e:
            print(f"  Error loading '{name}': {e}")

    print(f"\n  Registered {registered}/{len(all_strategies)} strategies")
    return registered


def compile_transfer_results():
    """
    Compile transfer efficiency metrics from all experiment result files.

    Reads results/transfer_same_task.json, results/transfer_cross_domain.json,
    results/curriculum_transfer.json, results/strategy_composition.json and
    creates a unified comparison table.
    """
    transfer_metrics = []

    # Same-task transfer results
    if os.path.exists("results/transfer_same_task.json"):
        with open("results/transfer_same_task.json") as f:
            same_task = json.load(f)
        for r in same_task:
            warm_start = r.get("warm_start") or {}
            transfer_metrics.append({
                "experiment": "Same-Task",
                "pair": f"{r['source']}->{r['target']}",
                "alignment_sim": r["alignment"]["mean_similarity"],
                "zero_shot_perf": r["zero_shot"]["win_rate"],
                "fine_tune_perf": warm_start.get("final_win_rate"),
                "metric_type": "win_rate",
            })

    # Cross-domain results
    if os.path.exists("results/transfer_cross_domain.json"):
        with open("results/transfer_cross_domain.json") as f:
            cross = json.load(f)

        scratch = cross.get("acrobot_scratch", {}).get("final_reward", 0)

        for key in ["cartpole_to_acrobot", "lunarlander_to_acrobot"]:
            if key in cross:
                r = cross[key]
                src = key.split("_to_")[0].title()
                transfer_metrics.append({
                    "experiment": "Cross-Domain",
                    "pair": f"{src}->Acrobot",
                    "alignment_sim": r.get("alignment", {}).get("mean_similarity"),
                    "zero_shot_perf": r.get("zero_shot_reward"),
                    "fine_tune_perf": r.get("final_reward"),
                    "scratch_perf": scratch,
                    "metric_type": "reward",
                })

    # Curriculum results
    if os.path.exists("results/curriculum_transfer.json"):
        with open("results/curriculum_transfer.json") as f:
            curriculum = json.load(f)

        transfer_metrics.append({
            "experiment": "Curriculum",
            "pair": "Go5x5->Go7x7",
            "alignment_sim": curriculum.get("alignment", {}).get("mean_similarity"),
            "zero_shot_perf": curriculum.get("zero_shot", {}).get("win_rate"),
            "fine_tune_perf": curriculum.get("transferred_finetune", {}).get("final_win_rate"),
            "scratch_perf": curriculum.get("scratch_control", {}).get("final_win_rate"),
            "metric_type": "win_rate",
        })

    # Composition results
    if os.path.exists("results/strategy_composition.json"):
        with open("results/strategy_composition.json") as f:
            composition = json.load(f)

        individual = composition.get("individual", {})
        best_individual = max(
            (v.get("win_rate", 0) for v in individual.values()),
            default=0,
        )

        for method in ["embedding_average", "phase_routing"]:
            if method in composition:
                transfer_metrics.append({
                    "experiment": "Composition",
                    "pair": method.replace("_", " ").title(),
                    "alignment_sim": None,
                    "zero_shot_perf": composition[method].get("win_rate"),
                    "fine_tune_perf": None,
                    "scratch_perf": best_individual,
                    "metric_type": "win_rate",
                })

    return transfer_metrics


def run_strategy_library_demo():
    """Run the full strategy library demo and benchmark."""
    set_seed(42)
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Strategy Library Demo + Transfer Benchmark")
    print(f"[{timestamp}] ============================================================")

    results = {}

    # ============================================================
    # Step 1: Build the strategy library
    # ============================================================
    print(f"\n--- Step 1: Building Strategy Library ---")
    library = StrategyLibrary()
    n_registered = register_all_strategies(library)

    results["library"] = {
        "n_strategies": n_registered,
        "strategies": library.list_strategies(),
    }

    # ============================================================
    # Step 2: Cross-strategy similarity matrix
    # ============================================================
    if len(library) >= 2:
        print(f"\n--- Step 2: Cross-Strategy Similarity ---")
        sim_matrix, names = library.compute_cross_similarity()

        print(f"\n  Similarity matrix ({len(names)} strategies):")
        # Print header
        header = "           " + "".join(f"{n[:8]:>10}" for n in names)
        print(header)
        for i, name in enumerate(names):
            row = f"  {name[:8]:<8} " + "".join(f"{sim_matrix[i,j]:>10.3f}" for j in range(len(names)))
            print(row)

        results["similarity_matrix"] = {
            "names": names,
            "matrix": sim_matrix.tolist(),
        }

        # Find most and least similar pairs
        pairs = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                pairs.append((names[i], names[j], sim_matrix[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  Most similar: {pairs[0][0]} <-> {pairs[0][1]} ({pairs[0][2]:.4f})")
        print(f"  Least similar: {pairs[-1][0]} <-> {pairs[-1][1]} ({pairs[-1][2]:.4f})")

    # ============================================================
    # Step 3: Find-similar demo
    # ============================================================
    if len(library) >= 3:
        print(f"\n--- Step 3: Find-Similar Demo ---")

        # For each strategy, find its nearest neighbors
        find_similar_results = {}
        for name in sorted(library.strategies.keys()):
            entry = library.strategies[name]
            # Temporarily remove this strategy to avoid self-match
            temp_entry = library.strategies.pop(name)
            matches = library.find_similar(entry.concept_manager, top_k=3)
            library.strategies[name] = temp_entry

            find_similar_results[name] = [
                {"match": m[0], "similarity": m[1]} for m in matches
            ]
            if matches:
                top = matches[0]
                print(f"  Query '{name}' -> best match: '{top[0]}' (sim={top[1]:.4f})")

        results["find_similar"] = find_similar_results

    # ============================================================
    # Step 4: Compile transfer benchmark
    # ============================================================
    print(f"\n--- Step 4: Transfer Efficiency Benchmark ---")
    transfer_metrics = compile_transfer_results()

    if transfer_metrics:
        print(f"\n  {'Experiment':<14} {'Pair':<22} {'Align':>7} {'Zero-Shot':>10} {'Fine-Tune':>10} {'Scratch':>8}")
        print(f"  {'-'*75}")
        for m in transfer_metrics:
            align = f"{m['alignment_sim']:.3f}" if m['alignment_sim'] else "N/A"
            zs = f"{m['zero_shot_perf']:.3f}" if m['zero_shot_perf'] is not None else "N/A"
            ft = f"{m['fine_tune_perf']:.3f}" if m['fine_tune_perf'] is not None else "N/A"
            sc = f"{m.get('scratch_perf', 'N/A')}"
            if isinstance(sc, float):
                sc = f"{sc:.3f}"
            print(f"  {m['experiment']:<14} {m['pair']:<22} {align:>7} {zs:>10} {ft:>10} {sc:>8}")

    results["transfer_benchmark"] = transfer_metrics

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/strategy_library.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    # ============================================================
    # Visualizations
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        ensure_dir("results/figures")

        # Figure 1: Similarity heatmap
        if "similarity_matrix" in results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sm = np.array(results["similarity_matrix"]["matrix"])
            ns = results["similarity_matrix"]["names"]

            im = ax.imshow(sm, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(len(ns)))
            ax.set_yticks(range(len(ns)))
            ax.set_xticklabels(ns, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(ns, fontsize=8)
            ax.set_title("Cross-Strategy Concept Similarity")

            for i in range(len(ns)):
                for j in range(len(ns)):
                    val = sm[i][j]
                    color = "white" if val > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

            plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
            plt.tight_layout()
            plt.savefig("results/figures/strategy_similarity.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved strategy_similarity.png")

        # Figure 2: Transfer benchmark bar chart
        if transfer_metrics:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            # Group by experiment
            labels = [f"{m['experiment']}\n{m['pair']}" for m in transfer_metrics]
            zs_vals = [m["zero_shot_perf"] or 0 for m in transfer_metrics]
            ft_vals = [m["fine_tune_perf"] or 0 for m in transfer_metrics]

            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, zs_vals, width, label="Zero-Shot", color="steelblue")
            bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-Tuned", color="darkorange")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel("Performance")
            ax.set_title("Transfer Efficiency: Zero-Shot vs Fine-Tuned")
            ax.legend()
            ax.grid(True, alpha=0.2, axis="y")

            plt.tight_layout()
            plt.savefig("results/figures/transfer_benchmark.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved transfer_benchmark.png")

    except Exception as e:
        print(f"  Warning: Could not generate figures: {e}")

    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] Done!")
    return results


if __name__ == "__main__":
    run_strategy_library_demo()
