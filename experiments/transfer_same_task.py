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


def evaluate_transferred_policy(encoder, cm, policy, n_episodes=100, device=None,
                                gnugo_level=None):
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
        gnugo_level: If set, evaluate vs GnuGo at this level instead of random.

    Returns:
        Dict with win_rate, mean_reward, mean_length.
    """
    device = device or get_device()

    opponent = None
    if gnugo_level is not None:
        from visualizer.opponents import GnuGoOpponent
        opponent = GnuGoOpponent(level=gnugo_level)

    env = GoEnv(board_size=7, opponent_fn=opponent)

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
    if opponent is not None:
        opponent.close()
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


def run_multi_seed_experiment(n_seeds=5, n_eval=100, gnugo_level=None):
    """
    Run agent-to-agent transfer with multiple seeds for statistical significance.

    For each of the 6 source->target pairs, evaluates zero-shot transfer with
    n_seeds different random seeds, then reports mean +/- std with confidence
    intervals and significance tests.

    Args:
        n_seeds: Number of evaluation seeds per pair.
        n_eval: Number of games per evaluation.
        gnugo_level: If set, evaluate vs GnuGo at this level instead of random.

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
            seed_lengths = []

            for seed in range(n_seeds):
                set_seed(seed * 1000)
                eval_result = evaluate_transferred_policy(
                    tgt["encoder"], tgt["cm"], transferred_policy,
                    n_episodes=n_eval, device=device,
                    gnugo_level=gnugo_level,
                )
                seed_wrs.append(eval_result["win_rate"])
                seed_rewards.append(eval_result["mean_reward"])
                seed_lengths.append(eval_result["mean_length"])
                print(f"    Seed {seed}: wr={eval_result['win_rate']:.2%}, "
                      f"avg_moves={eval_result['mean_length']:.1f}")

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
                "mean_length": float(np.mean(seed_lengths)),
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
    suffix = f"_L{gnugo_level}" if gnugo_level is not None else "_random"
    output_path = f"results/transfer_same_task_{n_seeds}seed{suffix}.json"
    save_data = {
        "n_seeds": n_seeds,
        "n_eval": n_eval,
        "opponent": f"gnugo_L{gnugo_level}" if gnugo_level is not None else "random",
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


def run_from_scratch_comparison(
    win_rate_threshold=0.60,
    gnugo_level=1,
    steps_per_checkpoint=10_000,
    max_total_steps=500_000,
    n_eval_games=50,
    seed=42,
):
    """
    Headline comparison: PPO-transferred concepts vs DQN from scratch.

    Trains a DQN bottleneck from scratch using the DQN encoder and DQN concepts
    (no PPO transfer).  Measures how many training steps are needed to first
    reach `win_rate_threshold` vs GnuGo Level `gnugo_level`.

    Also loads the zero-shot PPO→DQN transferred policy and evaluates it at
    step 0 (no extra training) — this is the from-transfer baseline.

    The speedup ratio = (steps for scratch to reach threshold) /
                        (steps for transfer to reach threshold, or 0 if already there).

    Outputs: results/transfer_scratch_comparison.json
    """
    from train_bottleneck import DQNBottleneckTrainer, discover_concepts
    from experiments.eval_strong import eval_agent_vs_gnugo

    set_seed(seed)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"From-Scratch vs Transfer Comparison")
    print(f"  DQN scratch vs PPO→DQN zero-shot transfer")
    print(f"  Target: {win_rate_threshold:.0%} vs GnuGo L{gnugo_level}")
    print(f"  Max steps: {max_total_steps:,}")
    print(f"{'='*60}\n")

    # ---- Load PPO→DQN transferred policy (step-0 baseline) ----
    print("Loading PPO source + DQN target agents for zero-shot transfer...")
    ppo_enc, ppo_cm, ppo_policy = load_agent("PPO", device)
    dqn_enc, dqn_cm, _          = load_agent("DQN", device)

    aligner = ConceptAligner(ppo_cm, dqn_cm)
    mapping = aligner.hungarian_alignment()
    transferred_policy = aligner.transfer_policy(
        ppo_policy, mapping,
        target_n_concepts=dqn_cm.n_concepts,
        target_n_actions=50,
    )
    transferred_policy.to(device).eval()

    def transfer_fn(obs, mask):
        c = dqn_cm.assign_concept_from_obs(dqn_enc, obs, device)
        return transferred_policy.get_action(c, mask, deterministic=True)

    # Also evaluate DAgger→DQN if DAgger model exists
    dagger_zero_shot_wr = None
    if os.path.exists(AGENTS["DAgger"]["encoder_path"]):
        dag_enc, dag_cm, dag_policy = load_agent("DAgger", device)
        dagger_aligner = ConceptAligner(dag_cm, dqn_cm)
        dagger_mapping  = dagger_aligner.hungarian_alignment()
        dagger_transferred = dagger_aligner.transfer_policy(
            dag_policy, dagger_mapping,
            target_n_concepts=dqn_cm.n_concepts,
            target_n_actions=50,
        )
        dagger_transferred.to(device).eval()

        def dagger_fn(obs, mask):
            c = dqn_cm.assign_concept_from_obs(dqn_enc, obs, device)
            return dagger_transferred.get_action(c, mask, deterministic=True)

        print(f"Evaluating DAgger→DQN zero-shot vs GnuGo L{gnugo_level}...")
        dagger_r = eval_agent_vs_gnugo(
            dagger_fn, dqn_enc, dqn_cm,
            gnugo_level=gnugo_level, n_games=n_eval_games, device=device,
        )
        dagger_zero_shot_wr = dagger_r["win_rate"]
        print(f"  DAgger→DQN zero-shot: {dagger_zero_shot_wr:.1%}")

    print(f"Evaluating PPO→DQN zero-shot vs GnuGo L{gnugo_level}...")
    transfer_r = eval_agent_vs_gnugo(
        transfer_fn, dqn_enc, dqn_cm,
        gnugo_level=gnugo_level, n_games=n_eval_games, device=device,
    )
    transfer_zero_shot_wr = transfer_r["win_rate"]
    print(f"  PPO→DQN zero-shot: {transfer_zero_shot_wr:.1%}")

    # ---- Train DQN from scratch and measure convergence ----
    print(f"\nTraining DQN from scratch (checkpoint every {steps_per_checkpoint:,} steps)...")

    scratch_trainer = DQNBottleneckTrainer(
        encoder=dqn_enc,
        concept_manager=dqn_cm,
        n_actions=50,
        n_concepts=dqn_cm.n_concepts,
        lr=1e-3,
        device=device,
    )

    env = GoEnv(board_size=7)
    obs, info = env.reset()

    scratch_curve = []        # [(steps, win_rate)]
    scratch_threshold_steps = None
    transfer_threshold_steps = 0 if transfer_zero_shot_wr >= win_rate_threshold else None
    steps_done = 0
    checkpoint_idx = 0

    # Evaluate transfer policy against thresholds at each checkpoint too
    transfer_curve = [{"steps": 0, "win_rate": transfer_zero_shot_wr}]

    while steps_done < max_total_steps:
        # One environment step
        mask = info.get("action_mask", None)
        cid = scratch_trainer.get_concept(obs)
        action = scratch_trainer.select_action(cid, mask)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info.get("action_mask", np.ones(50, dtype=np.int8))
        next_cid = scratch_trainer.get_concept(next_obs)
        if mask is None:
            mask = np.ones(50, dtype=np.int8)
        scratch_trainer.store_transition(cid, action, reward, next_cid,
                                         done, mask, next_mask)
        scratch_trainer.update()
        steps_done += 1

        if done:
            obs, info = env.reset()
        else:
            obs, info = next_obs, next_info

        # Checkpoint evaluation
        if steps_done % steps_per_checkpoint == 0:
            checkpoint_idx += 1
            scratch_trainer.q_net.eval()

            def scratch_fn(ob, msk):
                c = scratch_trainer.get_concept(ob)
                return scratch_trainer.q_net.get_action(c, msk, epsilon=0.0)

            eval_env = GoEnv(board_size=7)
            r = evaluate_agent(scratch_fn, eval_env, n_episodes=n_eval_games)
            eval_env.close()
            wr = r["win_rate"]
            scratch_curve.append({"steps": steps_done, "win_rate": wr})
            print(f"  Scratch step {steps_done:>7,}: WR={wr:.1%}")

            if scratch_threshold_steps is None and wr >= win_rate_threshold:
                scratch_threshold_steps = steps_done
                print(f"  Scratch reached {win_rate_threshold:.0%} at step {steps_done:,}!")

    env.close()

    # ---- Compute speedup ----
    if scratch_threshold_steps is not None and transfer_threshold_steps == 0:
        speedup = scratch_threshold_steps  # transfer needed 0 extra steps
        speedup_str = f"{scratch_threshold_steps:,} scratch steps vs 0 transfer steps"
    elif scratch_threshold_steps is not None and transfer_threshold_steps is not None:
        speedup = scratch_threshold_steps / max(transfer_threshold_steps, 1)
        speedup_str = f"{speedup:.1f}x speedup"
    else:
        speedup = None
        speedup_str = f"scratch never reached {win_rate_threshold:.0%} in {max_total_steps:,} steps"

    print(f"\n{'='*60}")
    print(f"From-Scratch vs Transfer Summary")
    print(f"  PPO→DQN zero-shot WR: {transfer_zero_shot_wr:.1%}")
    if dagger_zero_shot_wr is not None:
        print(f"  DAgger→DQN zero-shot WR: {dagger_zero_shot_wr:.1%}")
    print(f"  DQN scratch threshold ({win_rate_threshold:.0%}): "
          f"{'step ' + str(scratch_threshold_steps) if scratch_threshold_steps else 'not reached'}")
    print(f"  Speedup: {speedup_str}")
    print(f"{'='*60}\n")

    # ---- Save ----
    ensure_dir("results")
    result = {
        "win_rate_threshold": win_rate_threshold,
        "gnugo_level": gnugo_level,
        "n_eval_games": n_eval_games,
        "seed": seed,
        "transfer_zero_shot_wr": transfer_zero_shot_wr,
        "dagger_zero_shot_wr": dagger_zero_shot_wr,
        "scratch_threshold_steps": scratch_threshold_steps,
        "transfer_threshold_steps": transfer_threshold_steps,
        "speedup": speedup,
        "scratch_learning_curve": scratch_curve,
        "transfer_zero_shot_curve": transfer_curve,
    }
    out_path = "results/transfer_scratch_comparison.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")
    return result


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
    parser.add_argument("--scratch-comparison", action="store_true",
                        help="Run from-scratch DQN vs PPO-transfer comparison")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Win-rate threshold for speedup measurement (default: 0.60)")
    parser.add_argument("--level", type=int, default=1,
                        help="GnuGo level for scratch-comparison (default: 1)")
    parser.add_argument("--gnugo-level", type=int, default=None,
                        help="Evaluate multi-seed vs GnuGo at this level (default: random)")
    args = parser.parse_args()

    if args.scratch_comparison:
        run_from_scratch_comparison(
            win_rate_threshold=args.threshold,
            gnugo_level=args.level,
            n_eval_games=args.n_eval,
            seed=42,
        )
    elif args.multi_seed:
        run_multi_seed_experiment(n_seeds=args.n_seeds, n_eval=args.n_eval,
                                  gnugo_level=args.gnugo_level)
    else:
        run_experiment(do_warm_start=args.warm_start, warm_start_gens=args.generations)
