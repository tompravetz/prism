"""
Task 1: Agent-to-Agent Concept Transfer on Go 7x7.

Tests whether concepts learned by one algorithm (e.g., PPO) can be transferred
to another algorithm (e.g., DQN) or training method (e.g., DAgger) on the SAME
task (Go 7x7). This is the simplest transfer setting — same observation space,
same action space, same task, just different encoder training.

Available agents (NO new training needed):
    1. PPO:    models/baseline/ppo_go_encoder.pt + models/bottleneck/concepts_ppo_k64.pkl
    2. DQN:    models/baseline/dqn_go_encoder.pt + models/bottleneck/concepts_dqn_k64.pkl
    3. DAgger: models/cloned_dagger/ppo_go_encoder.pt + models/bottleneck_dagger/concepts_ppo_k64.pkl

For each of 6 source->target pairs:
    1. Align concepts via Hungarian matching (ConceptAligner)
    2. Report alignment quality (mean cosine sim, coverage)
    3. Zero-shot transfer: target encoder -> target concepts -> source policy (remapped)
    4. Warm-start transfer: initialize target bottleneck with transferred weights, fine-tune
    5. Control: target bottleneck trained from scratch (existing results)

Usage:
    python experiments/transfer_same_task.py
    python experiments/transfer_same_task.py --warm-start --generations 20
"""

import os
import sys
import json
import time
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
from train_bottleneck import PPOBottleneckTrainer, evaluate_agent


# ============================================================
# Agent configurations: paths to all existing models
# ============================================================

AGENTS = {
    "PPO": {
        "encoder_path": "models/baseline/ppo_go_encoder.pt",
        "concepts_path": "models/bottleneck/concepts_ppo_k64.pkl",
        "policy_path": "models/bottleneck/ppo_bottleneck_final.pt",
        "policy_class": "ppo",   # ConceptBottleneckPolicy
        "n_concepts": 64,
        "n_actions": 50,
    },
    "DQN": {
        "encoder_path": "models/baseline/dqn_go_encoder.pt",
        "concepts_path": "models/bottleneck/concepts_dqn_k64.pkl",
        "policy_path": "models/bottleneck/dqn_bottleneck_final.pt",
        "policy_class": "dqn",   # ConceptDQNPolicy
        "n_concepts": 64,
        "n_actions": 50,
    },
    "DAgger": {
        "encoder_path": "models/cloned_dagger/ppo_go_encoder.pt",
        "concepts_path": "models/bottleneck_dagger/concepts_ppo_k64.pkl",
        "policy_path": "models/bottleneck_dagger/ppo_bottleneck_final.pt",
        "policy_class": "ppo",   # ConceptBottleneckPolicy (BC-trained)
        "n_concepts": 64,
        "n_actions": 50,
    },
}


def load_agent(agent_name, device):
    """
    Load an agent's encoder, concept manager, and bottleneck policy.

    Args:
        agent_name: Key into AGENTS dict ("PPO", "DQN", or "DAgger").
        device: Torch device.

    Returns:
        (encoder, concept_manager, policy) tuple.
    """
    config = AGENTS[agent_name]

    # Load encoder
    env = GoEnv(board_size=7)
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder.load_state_dict(
        torch.load(config["encoder_path"], map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()
    env.close()

    # Load concept manager
    cm = ConceptManager(n_concepts=config["n_concepts"])
    cm.load(config["concepts_path"])

    # Load bottleneck policy
    if config["policy_class"] == "dqn":
        policy = ConceptDQNPolicy(
            n_concepts=config["n_concepts"],
            embed_dim=64,
            hidden_dim=128,
            n_actions=config["n_actions"],
        )
    else:
        policy = ConceptBottleneckPolicy(
            n_concepts=config["n_concepts"],
            embed_dim=64,
            hidden_dim=128,
            n_actions=config["n_actions"],
        )

    policy.load_state_dict(
        torch.load(config["policy_path"], map_location=device, weights_only=True)
    )
    policy.to(device)
    policy.eval()

    return encoder, cm, policy


def evaluate_transferred_policy(encoder, cm, policy, n_episodes=100, device=None):
    """
    Evaluate a transferred policy using the TARGET encoder and concept manager
    but a SOURCE-derived (remapped) policy.

    Pipeline: obs -> target_encoder -> target_concepts -> transferred_policy -> action

    Args:
        encoder: Target encoder.
        cm: Target concept manager.
        policy: Transferred policy (with remapped embeddings).
        n_episodes: Number of evaluation games.
        device: Torch device.

    Returns:
        Dict with win_rate, mean_reward, mean_length.
    """
    device = device or get_device()
    env = GoEnv(board_size=7)

    def agent_fn(obs, action_mask):
        # Convert obs -> concept via target encoder + target concepts
        concept_id = cm.assign_concept_from_obs(encoder, obs, device)
        # Get action from transferred policy
        if isinstance(policy, ConceptDQNPolicy):
            return policy.get_action(concept_id, action_mask, epsilon=0.0)
        else:
            return policy.get_action(concept_id, action_mask, deterministic=True)

    results = evaluate_agent(agent_fn, env, n_episodes=n_episodes)
    env.close()
    return results


def warm_start_finetune(source_policy, target_encoder, target_cm, mapping,
                        n_generations=20, steps_per_gen=10000, device=None):
    """
    Fine-tune a transferred policy on the target agent's pipeline.

    Initializes the target bottleneck policy with transferred weights from the
    source policy, then trains via PPO for n_generations.

    Args:
        source_policy: Source agent's trained policy.
        target_encoder: Target agent's frozen encoder.
        target_cm: Target agent's concept manager.
        mapping: Concept alignment mapping (source -> target).
        n_generations: Number of fine-tuning generations.
        steps_per_gen: Steps per generation.
        device: Torch device.

    Returns:
        (final_win_rate, learning_curve) — the win rate after fine-tuning
        and the win rate at each generation.
    """
    device = device or get_device()
    aligner = ConceptAligner(
        ConceptManager(n_concepts=len(mapping)),  # dummy, not used
        target_cm,
    )

    # Create trainer with target encoder + concepts
    trainer = PPOBottleneckTrainer(
        encoder=target_encoder,
        concept_manager=target_cm,
        n_actions=50,
        n_concepts=target_cm.n_concepts,
        lr=1e-4,  # Lower LR for fine-tuning (not training from scratch)
        device=device,
    )

    # Initialize policy with transferred weights
    transferred = ConceptAligner(
        # Need actual source cm for transfer; reconstruct from mapping size
        target_cm, target_cm  # placeholder, we'll set weights manually
    ).transfer_policy(source_policy, mapping, target_cm.n_concepts, 50)

    trainer.policy.load_state_dict(transferred.state_dict())
    trainer.policy.to(device)

    # Fine-tune
    env = GoEnv(board_size=7)
    eval_env = GoEnv(board_size=7)
    learning_curve = []

    for gen in range(n_generations):
        rollout = trainer.collect_rollout(env, n_steps=steps_per_gen)
        loss = trainer.update(rollout)

        # Evaluate every 5 generations
        if gen % 5 == 0 or gen == n_generations - 1:
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.policy.get_action(c, mask, deterministic=True)

            eval_results = evaluate_agent(agent_fn, eval_env, n_episodes=50)
            wr = eval_results["win_rate"]
            learning_curve.append({"generation": gen, "win_rate": wr})
            print(f"    Fine-tune gen {gen:3d}: win_rate={wr:.2%}")

    env.close()
    eval_env.close()
    final_wr = learning_curve[-1]["win_rate"] if learning_curve else 0.0
    return final_wr, learning_curve


def run_experiment(do_warm_start=False, warm_start_gens=20):
    """
    Run the full agent-to-agent transfer experiment.

    For each source->target pair:
        1. Align concepts (Hungarian)
        2. Report alignment quality
        3. Zero-shot transfer evaluation
        4. (Optional) Warm-start fine-tuning
    """
    set_seed(42)
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Agent-to-Agent Concept Transfer (Go 7x7)")
    print(f"[{timestamp}]   Agents: {list(AGENTS.keys())}")
    print(f"[{timestamp}]   Warm-start: {'Yes' if do_warm_start else 'No'}")
    print(f"[{timestamp}] ============================================================")

    # Load all agents
    print(f"\nLoading agents...")
    agents = {}
    for name in AGENTS:
        print(f"  Loading {name}...")
        encoder, cm, policy = load_agent(name, device)
        agents[name] = {"encoder": encoder, "cm": cm, "policy": policy}
    print(f"  All agents loaded.")

    # Generate all source->target pairs
    agent_names = list(AGENTS.keys())
    results = []

    for src_name in agent_names:
        for tgt_name in agent_names:
            if src_name == tgt_name:
                continue

            print(f"\n--- {src_name} -> {tgt_name} ---")
            src = agents[src_name]
            tgt = agents[tgt_name]

            # Step 1: Align concepts
            aligner = ConceptAligner(src["cm"], tgt["cm"])
            mapping = aligner.hungarian_alignment()
            quality = aligner.alignment_quality(mapping)

            print(f"  Alignment: mean_sim={quality['mean_similarity']:.4f}, "
                  f"min={quality['min_similarity']:.4f}, "
                  f"max={quality['max_similarity']:.4f}, "
                  f"coverage_tgt={quality['coverage_target']:.2%}")

            # Step 2: Transfer policy
            transferred_policy = aligner.transfer_policy(
                src["policy"], mapping,
                target_n_concepts=tgt["cm"].n_concepts,
                target_n_actions=50,
            )
            transferred_policy.to(device)
            transferred_policy.eval()

            # Step 3: Zero-shot evaluation
            print(f"  Zero-shot evaluation (100 games)...")
            zero_shot = evaluate_transferred_policy(
                tgt["encoder"], tgt["cm"], transferred_policy,
                n_episodes=100, device=device,
            )
            print(f"  Zero-shot: win_rate={zero_shot['win_rate']:.2%}, "
                  f"mean_reward={zero_shot['mean_reward']:.3f}")

            # Step 4: Warm-start fine-tuning (optional)
            warm_start_results = None
            if do_warm_start:
                print(f"  Warm-start fine-tuning ({warm_start_gens} generations)...")
                final_wr, curve = warm_start_finetune(
                    src["policy"], tgt["encoder"], tgt["cm"], mapping,
                    n_generations=warm_start_gens, device=device,
                )
                warm_start_results = {
                    "final_win_rate": final_wr,
                    "learning_curve": curve,
                }
                print(f"  Warm-start final: win_rate={final_wr:.2%}")

            # Compile result
            result = {
                "source": src_name,
                "target": tgt_name,
                "alignment": {
                    "mean_similarity": quality["mean_similarity"],
                    "std_similarity": quality["std_similarity"],
                    "min_similarity": quality["min_similarity"],
                    "max_similarity": quality["max_similarity"],
                    "median_similarity": quality["median_similarity"],
                    "coverage_source": quality["coverage_source"],
                    "coverage_target": quality["coverage_target"],
                    "n_mapped_pairs": quality["n_mapped_pairs"],
                },
                "zero_shot": zero_shot,
                "warm_start": warm_start_results,
            }
            results.append(result)

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/transfer_same_task.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ============================================================
    # Summary table
    # ============================================================
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] ============================================================")
    print(f"[{timestamp}] Summary: Agent-to-Agent Transfer")
    print(f"[{timestamp}] {'Source':<8} {'Target':<8} {'Align Sim':>10} {'Zero-Shot WR':>13} ", end="")
    if do_warm_start:
        print(f"{'Warm-Start WR':>14}")
    else:
        print()
    print(f"[{timestamp}] {'-'*55}")

    for r in results:
        line = (f"[{timestamp}] {r['source']:<8} {r['target']:<8} "
                f"{r['alignment']['mean_similarity']:>10.4f} "
                f"{r['zero_shot']['win_rate']:>12.2%} ")
        if r["warm_start"]:
            line += f"{r['warm_start']['final_win_rate']:>13.2%}"
        print(line)

    # ============================================================
    # Generate visualization
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito colorblind-friendly palette
        OI_BLUE, OI_ORANGE = "#0072B2", "#E69F00"
        OI_GREEN, OI_RED = "#009E73", "#D55E00"

        ensure_dir("results/figures")
        plt.rcParams.update({"font.size": 11})
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Alignment similarity heatmap
        ax = axes[0]
        sim_matrix = np.zeros((len(agent_names), len(agent_names)))
        for r in results:
            i = agent_names.index(r["source"])
            j = agent_names.index(r["target"])
            sim_matrix[i, j] = r["alignment"]["mean_similarity"]
        np.fill_diagonal(sim_matrix, 1.0)

        im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(agent_names)))
        ax.set_yticks(range(len(agent_names)))
        ax.set_xticklabels(agent_names, fontsize=11)
        ax.set_yticklabels(agent_names, fontsize=11)
        ax.set_title("Concept Alignment Similarity", fontsize=13)
        ax.set_xlabel("Target", fontsize=12)
        ax.set_ylabel("Source", fontsize=12)
        for i in range(len(agent_names)):
            for j in range(len(agent_names)):
                color = "white" if sim_matrix[i,j] > 0.7 else "black"
                ax.text(j, i, f"{sim_matrix[i,j]:.3f}",
                        ha="center", va="center", fontsize=11, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # 2. Zero-shot win rates bar chart
        ax = axes[1]
        pair_labels = [f"{r['source']}->{r['target']}" for r in results]
        zero_shot_wrs = [r["zero_shot"]["win_rate"] for r in results]
        bars = ax.bar(range(len(pair_labels)), zero_shot_wrs,
                      color=OI_BLUE, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.5, color=OI_RED, linestyle="--", alpha=0.5,
                   label="Random baseline (50%)")
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Win Rate vs Random", fontsize=12)
        ax.set_title("Zero-Shot Transfer Win Rates", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        for i, wr in enumerate(zero_shot_wrs):
            ax.text(i, wr + 0.02, f"{wr:.0%}", ha="center", fontsize=9,
                    fontweight="bold")

        # 3. Alignment vs performance scatter
        ax = axes[2]
        sims = [r["alignment"]["mean_similarity"] for r in results]
        wrs = [r["zero_shot"]["win_rate"] for r in results]
        ax.scatter(sims, wrs, s=100, color=OI_BLUE, edgecolors="black",
                   linewidths=0.5, zorder=5)
        for i, r in enumerate(results):
            ax.annotate(f"{r['source'][0]}->{r['target'][0]}",
                        (sims[i], wrs[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=10)
        ax.set_xlabel("Mean Alignment Similarity", fontsize=12)
        ax.set_ylabel("Zero-Shot Win Rate", fontsize=12)
        ax.set_title("Alignment vs Transfer Performance", fontsize=13)
        ax.axhline(y=0.5, color=OI_RED, linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig_path = "results/figures/transfer_same_task.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return results


def run_multi_seed_experiment(n_seeds=5, n_eval=100):
    """
    Run agent-to-agent transfer with multiple seeds for statistical significance.

    For each of the 6 source->target pairs, evaluates zero-shot transfer with
    n_seeds different random seeds, then reports mean +/- std with confidence
    intervals and significance tests.

    Args:
        n_seeds: Number of evaluation seeds per pair.
        n_eval: Number of games per evaluation.

    Returns:
        Dict with per-pair results, aggregate stats, and significance tests.
    """
    from scipy import stats as sp_stats

    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Multi-Seed Agent-to-Agent Transfer ({n_seeds} seeds)")
    print(f"[{timestamp}] ============================================================")

    # Load all agents
    print(f"\nLoading agents...")
    agents = {}
    for name in AGENTS:
        print(f"  Loading {name}...")
        encoder, cm, policy = load_agent(name, device)
        agents[name] = {"encoder": encoder, "cm": cm, "policy": policy}

    agent_names = list(AGENTS.keys())
    results = []

    for src_name in agent_names:
        for tgt_name in agent_names:
            if src_name == tgt_name:
                continue

            print(f"\n--- {src_name} -> {tgt_name} ---")
            src = agents[src_name]
            tgt = agents[tgt_name]

            # Align concepts
            aligner = ConceptAligner(src["cm"], tgt["cm"])
            mapping = aligner.hungarian_alignment()
            quality = aligner.alignment_quality(mapping)

            # Transfer policy
            transferred_policy = aligner.transfer_policy(
                src["policy"], mapping,
                target_n_concepts=tgt["cm"].n_concepts,
                target_n_actions=50,
            )
            transferred_policy.to(device)
            transferred_policy.eval()

            # Evaluate with multiple seeds
            seed_wrs = []
            seed_rewards = []

            for seed in range(n_seeds):
                set_seed(seed * 1000)
                eval_result = evaluate_transferred_policy(
                    tgt["encoder"], tgt["cm"], transferred_policy,
                    n_episodes=n_eval, device=device,
                )
                seed_wrs.append(eval_result["win_rate"])
                seed_rewards.append(eval_result["mean_reward"])
                print(f"    Seed {seed}: wr={eval_result['win_rate']:.2%}")

            # Statistical test: one-sample t-test vs 50% (random baseline)
            if n_seeds >= 2:
                t_stat, p_value = sp_stats.ttest_1samp(seed_wrs, 0.5)
            else:
                t_stat, p_value = 0.0, 1.0

            # 95% confidence interval
            mean_wr = float(np.mean(seed_wrs))
            std_wr = float(np.std(seed_wrs))
            if n_seeds >= 2:
                ci_95 = sp_stats.t.interval(0.95, df=n_seeds-1,
                                             loc=mean_wr, scale=std_wr/np.sqrt(n_seeds))
            else:
                ci_95 = (mean_wr, mean_wr)

            result = {
                "source": src_name,
                "target": tgt_name,
                "alignment": {
                    "mean_similarity": float(quality["mean_similarity"]),
                    "std_similarity": float(quality["std_similarity"]),
                    "min_similarity": float(quality["min_similarity"]),
                    "max_similarity": float(quality["max_similarity"]),
                },
                "win_rates": [float(x) for x in seed_wrs],
                "mean_wr": float(mean_wr),
                "std_wr": float(std_wr),
                "ci_95_lower": float(ci_95[0]),
                "ci_95_upper": float(ci_95[1]),
                "mean_reward": float(np.mean(seed_rewards)),
                "std_reward": float(np.std(seed_rewards)),
                "vs_random_ttest": {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                },
            }
            results.append(result)

            print(f"  Result: {mean_wr:.2%} +/- {std_wr:.2%}, "
                  f"CI=[{ci_95[0]:.2%}, {ci_95[1]:.2%}], p={p_value:.4f}")

    # Save
    ensure_dir("results")
    output_path = "results/transfer_same_task_5seed.json"
    save_data = {
        "n_seeds": n_seeds,
        "n_eval": n_eval,
        "pairs": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Multi-Seed Transfer Summary ({n_seeds} seeds, {n_eval} games)")
    print(f"[{ts}] {'Source':<8} {'Target':<8} {'Mean WR':>9} {'Std':>7} {'95% CI':>16} {'p-val':>7}")
    print(f"[{ts}] {'-'*60}")
    for r in results:
        print(f"[{ts}] {r['source']:<8} {r['target']:<8} "
              f"{r['mean_wr']:>8.2%} {r['std_wr']:>6.2%} "
              f"[{r['ci_95_lower']:.2%},{r['ci_95_upper']:.2%}] "
              f"{r['vs_random_ttest']['p_value']:>7.4f}")

    print(f"\nDone!")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent-to-Agent Concept Transfer")
    parser.add_argument("--warm-start", action="store_true",
                        help="Run warm-start fine-tuning after zero-shot")
    parser.add_argument("--generations", type=int, default=20,
                        help="Warm-start fine-tuning generations")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run multi-seed evaluation for statistical significance")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for multi-seed mode (default: 5)")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Games per evaluation (default: 100)")
    args = parser.parse_args()

    if args.multi_seed:
        run_multi_seed_experiment(n_seeds=args.n_seeds, n_eval=args.n_eval)
    else:
        run_experiment(do_warm_start=args.warm_start, warm_start_gens=args.generations)
