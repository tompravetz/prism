"""
Task 4: Curriculum Learning via Concept Transfer (Go 5x5 -> 7x7).

Tests whether concepts learned on a simpler version of a task (Go 5x5) can
bootstrap learning on the full version (Go 7x7). This is the strongest test
of the concept-as-universal-interface hypothesis:

    - Encoder weights DON'T transfer (different FC sizes: 5x5 vs 7x7)
    - Concepts DO transfer (both produce 128D features, centroids are comparable)
    - Policies DO transfer (concept_id -> action mapping, remap via alignment)

Pipeline:
    1. Train Go 5x5 baseline encoder (SB3 MaskablePPO)
    2. Discover 5x5 concepts (K=64)
    3. Train 5x5 bottleneck (100 generations)
    4. Align 5x5 concepts to existing 7x7 concepts
    5. Transfer 5x5 policy -> 7x7 bottleneck (remap embeddings)
    6. Fine-tune 7x7 with transferred init vs from scratch
    7. Compare learning curves

Usage:
    python experiments/curriculum.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir
from train_bottleneck import (PPOBottleneckTrainer, discover_concepts,
                               evaluate_agent)


def train_5x5_baseline(n_steps=200000, device=None):
    """
    Train a Go 5x5 baseline agent using MaskablePPO.

    Go 5x5:
        - Board: 5x5, obs shape: (5, 5, 3)
        - Action space: 26 (25 positions + pass)
        - Simpler game, faster training

    Returns:
        Trained GoCNNEncoder for 5x5 boards.
    """
    from stable_baselines3 import PPO as SB3_PPO
    from sb3_contrib import MaskablePPO

    device = device or get_device()
    save_dir = "models/go_5x5"
    ensure_dir(save_dir)

    encoder_path = os.path.join(save_dir, "ppo_go5x5_encoder.pt")

    # Check if already trained
    if os.path.exists(encoder_path):
        print(f"  Loading existing 5x5 encoder from {encoder_path}")
        env = GoEnv(board_size=5)
        encoder = GoCNNEncoder(env.observation_space, features_dim=128)
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        env.close()
        return encoder

    print(f"  Training Go 5x5 baseline ({n_steps} steps)...")

    from src.environments.go_env import MaskedGoEnv
    env = MaskedGoEnv(board_size=5)

    policy_kwargs = dict(
        features_extractor_class=GoCNNEncoder,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    model.learn(total_timesteps=n_steps)

    # Extract encoder
    encoder = model.policy.features_extractor
    torch.save(encoder.state_dict(), encoder_path)
    print(f"  Saved 5x5 encoder to {encoder_path}")

    # Quick eval
    eval_env = GoEnv(board_size=5)
    wins = 0
    for _ in range(50):
        obs, info = eval_env.reset()
        done = False
        ep_r = 0
        while not done:
            mask = info.get("action_mask", None)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
            ep_r += r
        if ep_r > 0:
            wins += 1
    print(f"  5x5 baseline: {wins}/50 = {wins/50:.0%} win rate vs random")
    eval_env.close()
    env.close()

    return encoder


def train_5x5_bottleneck(encoder, n_concepts=64, n_generations=100,
                         steps_per_gen=10000, device=None):
    """
    Train a concept bottleneck policy on Go 5x5.

    Returns:
        (policy, concept_manager, learning_curve)
    """
    device = device or get_device()
    save_dir = "models/go_5x5"
    ensure_dir(save_dir)

    env = GoEnv(board_size=5)
    n_actions = env.action_count  # 26 for Go 5x5

    # Discover concepts
    concepts_path = os.path.join(save_dir, f"concepts_k{n_concepts}.pkl")
    if os.path.exists(concepts_path):
        cm = ConceptManager(n_concepts=n_concepts)
        cm.load(concepts_path)
    else:
        cm = discover_concepts(encoder, env, n_concepts=n_concepts,
                               n_episodes=500, save_path=concepts_path, device=device)

    # Check if bottleneck already trained
    policy_path = os.path.join(save_dir, "ppo_bottleneck_final.pt")
    if os.path.exists(policy_path):
        print(f"  Loading existing 5x5 bottleneck from {policy_path}")
        policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts, embed_dim=64, hidden_dim=128, n_actions=n_actions,
        )
        policy.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=True)
        )
        env.close()
        return policy, cm, []

    # Train bottleneck
    print(f"  Training 5x5 bottleneck ({n_generations} generations)...")
    trainer = PPOBottleneckTrainer(
        encoder=encoder, concept_manager=cm,
        n_actions=n_actions, n_concepts=n_concepts,
        lr=3e-4, gamma=0.99, device=device,
    )

    eval_env = GoEnv(board_size=5)
    learning_curve = []

    for gen in range(n_generations):
        rollout = trainer.collect_rollout(env, n_steps=steps_per_gen)
        loss = trainer.update(rollout)

        if gen % 10 == 0 or gen == n_generations - 1:
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.policy.get_action(c, mask, deterministic=True)

            eval_results = evaluate_agent(agent_fn, eval_env, n_episodes=50)
            wr = eval_results["win_rate"]
            learning_curve.append({"generation": gen, "win_rate": wr})
            print(f"    Gen {gen:3d}: win_rate={wr:.2%}")

    # Save
    torch.save(trainer.policy.state_dict(), policy_path)
    env.close()
    eval_env.close()

    return trainer.policy, cm, learning_curve


def train_7x7_bottleneck(encoder, cm, n_concepts=64, n_generations=50,
                         steps_per_gen=20000, initial_policy=None,
                         lr=3e-4, label="scratch", device=None):
    """
    Train a 7x7 bottleneck policy, optionally initialized with transferred weights.

    Args:
        encoder: Frozen 7x7 encoder.
        cm: 7x7 concept manager.
        initial_policy: Optional state_dict to initialize the policy with
                        (from transfer). If None, trains from scratch.
        lr: Learning rate.
        label: Label for logging ("scratch" or "transferred").
        device: Torch device.

    Returns:
        (policy, learning_curve)
    """
    device = device or get_device()

    env = GoEnv(board_size=7)
    n_actions = 50

    trainer = PPOBottleneckTrainer(
        encoder=encoder, concept_manager=cm,
        n_actions=n_actions, n_concepts=n_concepts,
        lr=lr, gamma=0.99, device=device,
    )

    # Initialize with transferred weights if provided
    if initial_policy is not None:
        trainer.policy.load_state_dict(initial_policy)
        trainer.policy.to(device)

    eval_env = GoEnv(board_size=7)
    learning_curve = []

    print(f"  Training 7x7 bottleneck [{label}] ({n_generations} generations)...")
    for gen in range(n_generations):
        rollout = trainer.collect_rollout(env, n_steps=steps_per_gen)
        loss = trainer.update(rollout)

        if gen % 5 == 0 or gen == n_generations - 1:
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.policy.get_action(c, mask, deterministic=True)

            eval_results = evaluate_agent(agent_fn, eval_env, n_episodes=50)
            wr = eval_results["win_rate"]
            learning_curve.append({"generation": gen, "win_rate": wr})
            if gen % 10 == 0 or gen == n_generations - 1:
                print(f"    Gen {gen:3d}: win_rate={wr:.2%}")

    env.close()
    eval_env.close()
    return trainer.policy, learning_curve


def run_curriculum_experiment():
    """Run the full curriculum learning experiment."""
    set_seed(42)
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Curriculum Learning: Go 5x5 -> 7x7")
    print(f"[{timestamp}] ============================================================")

    results = {}

    # ============================================================
    # Step 1: Train Go 5x5 baseline + bottleneck
    # ============================================================
    print(f"\n--- Step 1: Go 5x5 Baseline ---")
    encoder_5x5 = train_5x5_baseline(n_steps=200000, device=device)
    encoder_5x5.to(device)
    encoder_5x5.eval()

    print(f"\n--- Step 2: Go 5x5 Bottleneck ---")
    policy_5x5, cm_5x5, curve_5x5 = train_5x5_bottleneck(
        encoder_5x5, n_concepts=64, n_generations=100,
        steps_per_gen=10000, device=device,
    )
    results["5x5_training"] = {
        "learning_curve": curve_5x5,
        "final_win_rate": curve_5x5[-1]["win_rate"] if curve_5x5 else 0.0,
    }

    # ============================================================
    # Step 2: Load 7x7 encoder + concepts
    # ============================================================
    print(f"\n--- Step 3: Load 7x7 Models ---")
    env_7x7 = GoEnv(board_size=7)
    encoder_7x7 = GoCNNEncoder(env_7x7.observation_space, features_dim=128)
    encoder_7x7.load_state_dict(
        torch.load("models/baseline/ppo_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    encoder_7x7.to(device)
    encoder_7x7.eval()
    env_7x7.close()

    cm_7x7 = ConceptManager(n_concepts=64)
    cm_7x7.load("models/bottleneck/concepts_ppo_k64.pkl")

    # ============================================================
    # Step 3: Align 5x5 -> 7x7 concepts
    # ============================================================
    print(f"\n--- Step 4: Concept Alignment (5x5 -> 7x7) ---")
    aligner = ConceptAligner(cm_5x5, cm_7x7)
    mapping = aligner.hungarian_alignment()
    quality = aligner.alignment_quality(mapping)

    print(f"  Alignment quality:")
    print(f"    Mean similarity: {quality['mean_similarity']:.4f}")
    print(f"    Min similarity:  {quality['min_similarity']:.4f}")
    print(f"    Max similarity:  {quality['max_similarity']:.4f}")
    print(f"    Coverage:        {quality['coverage_target']:.2%}")

    results["alignment"] = {
        "mean_similarity": quality["mean_similarity"],
        "std_similarity": quality["std_similarity"],
        "min_similarity": quality["min_similarity"],
        "max_similarity": quality["max_similarity"],
        "coverage_source": quality["coverage_source"],
        "coverage_target": quality["coverage_target"],
    }

    # ============================================================
    # Step 4: Transfer 5x5 policy to 7x7
    # ============================================================
    print(f"\n--- Step 5: Transfer Policy (5x5 -> 7x7) ---")

    # Transfer: 5x5 has 26 actions, 7x7 has 50 actions
    # Embedding layer transfers via alignment mapping
    # Action head: first 25 position actions transfer (board layout differs),
    # action head is re-initialized for the larger action space
    transferred_policy = aligner.transfer_policy(
        policy_5x5, mapping,
        target_n_concepts=64,
        target_n_actions=50,  # 7x7 action space
    )

    # Zero-shot evaluation of transferred policy
    print(f"  Zero-shot evaluation...")
    eval_env = GoEnv(board_size=7)

    def agent_fn_zs(obs, mask):
        c = cm_7x7.assign_concept_from_obs(encoder_7x7, obs, device)
        return transferred_policy.to(device).get_action(c, mask, deterministic=True)

    zs_result = evaluate_agent(agent_fn_zs, eval_env, n_episodes=100)
    eval_env.close()
    print(f"  Zero-shot: win_rate={zs_result['win_rate']:.2%}")
    results["zero_shot"] = zs_result

    # ============================================================
    # Step 5: Fine-tune transferred policy on 7x7
    # ============================================================
    print(f"\n--- Step 6: Fine-tune on 7x7 (Transferred Init) ---")
    transferred_state = transferred_policy.state_dict()

    _, curve_transferred = train_7x7_bottleneck(
        encoder_7x7, cm_7x7, n_concepts=64,
        n_generations=50, steps_per_gen=20000,
        initial_policy=transferred_state, lr=1e-4,
        label="transferred", device=device,
    )
    results["transferred_finetune"] = {
        "learning_curve": curve_transferred,
        "final_win_rate": curve_transferred[-1]["win_rate"] if curve_transferred else 0.0,
    }

    # ============================================================
    # Step 6: Train 7x7 from scratch (CONTROL)
    # ============================================================
    print(f"\n--- Step 7: Train 7x7 from Scratch (Control) ---")

    _, curve_scratch = train_7x7_bottleneck(
        encoder_7x7, cm_7x7, n_concepts=64,
        n_generations=50, steps_per_gen=20000,
        initial_policy=None, lr=3e-4,
        label="scratch", device=device,
    )
    results["scratch_control"] = {
        "learning_curve": curve_scratch,
        "final_win_rate": curve_scratch[-1]["win_rate"] if curve_scratch else 0.0,
    }

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/curriculum_transfer.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    # Summary
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Summary: Curriculum Transfer (5x5 -> 7x7)")
    print(f"[{ts}]   Alignment: {results['alignment']['mean_similarity']:.4f} mean sim")
    print(f"[{ts}]   Zero-shot: {zs_result['win_rate']:.2%}")
    print(f"[{ts}]   Transferred final: {results['transferred_finetune']['final_win_rate']:.2%}")
    print(f"[{ts}]   Scratch final:     {results['scratch_control']['final_win_rate']:.2%}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito colorblind-friendly palette
        OI_BLUE, OI_BLACK = "#0072B2", "#000000"
        OI_CYAN, OI_GRAY = "#56B4E9", "#999999"

        ensure_dir("results/figures")
        plt.rcParams.update({"font.size": 11})
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if curve_transferred:
            ax.plot([p["generation"] for p in curve_transferred],
                    [p["win_rate"] for p in curve_transferred],
                    "-o", color=OI_BLUE, label="Curriculum transfer (5x5 -> 7x7)",
                    markersize=5, linewidth=2)
        if curve_scratch:
            ax.plot([p["generation"] for p in curve_scratch],
                    [p["win_rate"] for p in curve_scratch],
                    "-s", color=OI_BLACK, label="From scratch (7x7)",
                    markersize=5, linewidth=2)

        # Mark zero-shot level
        ax.axhline(y=zs_result["win_rate"], color=OI_CYAN, linestyle=":",
                    alpha=0.7, label=f"Zero-shot ({zs_result['win_rate']:.0%})")
        ax.axhline(y=0.5, color=OI_GRAY, linestyle="--", alpha=0.3, label="50% baseline")

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Win Rate vs Random", fontsize=12)
        ax.set_title("Curriculum Transfer: Go 5x5 -> 7x7", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.05)

        fig_path = "results/figures/curriculum_transfer.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return results


def run_multi_seed_curriculum(seeds=None, n_generations=50):
    """
    Run curriculum experiment with multiple seeds for statistical significance.

    Trains both transferred and from-scratch conditions with each seed, then
    computes speedup metrics with p-values.

    Args:
        seeds: List of random seeds. Defaults to [42, 123, 456].
        n_generations: Number of training generations per condition.

    Returns:
        Dict with per-seed curves, aggregate stats, and significance tests.
    """
    from scipy import stats

    if seeds is None:
        seeds = [42, 123, 456]

    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Multi-Seed Curriculum Transfer ({len(seeds)} seeds)")
    print(f"[{timestamp}] ============================================================")

    # Load shared resources (same across seeds)
    # 5x5 encoder + concepts (trained once)
    set_seed(42)
    encoder_5x5 = train_5x5_baseline(n_steps=200000, device=device)
    encoder_5x5.to(device)
    encoder_5x5.eval()

    policy_5x5, cm_5x5, _ = train_5x5_bottleneck(
        encoder_5x5, n_concepts=64, n_generations=100,
        steps_per_gen=10000, device=device,
    )

    # Load 7x7 encoder + concepts (same across seeds)
    env_7x7 = GoEnv(board_size=7)
    encoder_7x7 = GoCNNEncoder(env_7x7.observation_space, features_dim=128)
    encoder_7x7.load_state_dict(
        torch.load("models/baseline/ppo_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    encoder_7x7.to(device)
    encoder_7x7.eval()
    env_7x7.close()

    cm_7x7 = ConceptManager(n_concepts=64)
    cm_7x7.load("models/bottleneck/concepts_ppo_k64.pkl")

    # Alignment (same across seeds)
    aligner = ConceptAligner(cm_5x5, cm_7x7)
    mapping = aligner.hungarian_alignment()
    transferred_init = aligner.transfer_policy(
        policy_5x5, mapping, target_n_concepts=64, target_n_actions=50,
    )
    transferred_state = transferred_init.state_dict()

    # Run each seed
    all_transferred = []
    all_scratch = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        # Transferred condition
        _, curve_t = train_7x7_bottleneck(
            encoder_7x7, cm_7x7, n_concepts=64,
            n_generations=n_generations, steps_per_gen=20000,
            initial_policy={k: v.clone() for k, v in transferred_state.items()},
            lr=1e-4, label=f"transferred-seed{seed}", device=device,
        )
        all_transferred.append(curve_t)

        # From-scratch condition
        _, curve_s = train_7x7_bottleneck(
            encoder_7x7, cm_7x7, n_concepts=64,
            n_generations=n_generations, steps_per_gen=20000,
            initial_policy=None, lr=3e-4,
            label=f"scratch-seed{seed}", device=device,
        )
        all_scratch.append(curve_s)

    # Compute aggregate metrics
    # For each condition, collect win rates at each evaluated generation
    def get_wr_at_gen(curves, target_gen):
        """Get win rate at or near a target generation across seeds."""
        wrs = []
        for curve in curves:
            # Find the closest generation
            best = None
            for point in curve:
                if best is None or abs(point["generation"] - target_gen) < abs(best["generation"] - target_gen):
                    best = point
            if best is not None:
                wrs.append(best["win_rate"])
        return np.array(wrs)

    # Milestone analysis: generations to reach 90%, 95%, 98%
    def gens_to_threshold(curves, threshold):
        """Find first generation where win rate >= threshold for each seed."""
        gens = []
        for curve in curves:
            found = None
            for point in curve:
                if point["win_rate"] >= threshold:
                    found = point["generation"]
                    break
            gens.append(found if found is not None else n_generations)
        return np.array(gens)

    efficiency = {}
    for threshold in [0.90, 0.95, 0.98]:
        t_gens = gens_to_threshold(all_transferred, threshold)
        s_gens = gens_to_threshold(all_scratch, threshold)

        # Speedup factor: scratch / transferred
        # Use mean values for speedup, handle case where either didn't reach threshold
        t_mean = float(np.mean(t_gens))
        s_mean = float(np.mean(s_gens))
        speedup = s_mean / t_mean if t_mean > 0 else float("inf")

        # t-test for significance (paired)
        if len(seeds) >= 2:
            t_stat, p_value = stats.ttest_ind(s_gens, t_gens, alternative="greater")
        else:
            t_stat, p_value = 0.0, 1.0

        label = f"{int(threshold*100)}pct"
        efficiency[label] = {
            "threshold": threshold,
            "transferred_gens": t_gens.tolist(),
            "scratch_gens": s_gens.tolist(),
            "transferred_mean": t_mean,
            "transferred_std": float(np.std(t_gens)),
            "scratch_mean": s_mean,
            "scratch_std": float(np.std(s_gens)),
            "speedup": speedup,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        }
        print(f"  {label}: transferred={t_mean:.1f}+/-{np.std(t_gens):.1f} "
              f"scratch={s_mean:.1f}+/-{np.std(s_gens):.1f} "
              f"speedup={speedup:.2f}x p={p_value:.4f}")

    # Compare final win rates
    final_t = [c[-1]["win_rate"] for c in all_transferred if c]
    final_s = [c[-1]["win_rate"] for c in all_scratch if c]
    if len(seeds) >= 2:
        t_stat_final, p_final = stats.ttest_ind(final_t, final_s)
    else:
        t_stat_final, p_final = 0.0, 1.0

    # Early convergence comparison (at gen 15)
    early_t = get_wr_at_gen(all_transferred, 15)
    early_s = get_wr_at_gen(all_scratch, 15)
    if len(seeds) >= 2:
        t_stat_early, p_early = stats.ttest_ind(early_t, early_s, alternative="greater")
    else:
        t_stat_early, p_early = 0.0, 1.0

    multi_seed_results = {
        "seeds": seeds,
        "n_generations": n_generations,
        "efficiency": efficiency,
        "final_wr": {
            "transferred": {"mean": float(np.mean(final_t)), "std": float(np.std(final_t)), "values": final_t},
            "scratch": {"mean": float(np.mean(final_s)), "std": float(np.std(final_s)), "values": final_s},
            "p_value": float(p_final),
        },
        "early_convergence_gen15": {
            "transferred": {"mean": float(np.mean(early_t)), "std": float(np.std(early_t))},
            "scratch": {"mean": float(np.mean(early_s)), "std": float(np.std(early_s))},
            "p_value": float(p_early),
        },
        "transferred_curves": [[{"generation": p["generation"], "win_rate": p["win_rate"]} for p in c] for c in all_transferred],
        "scratch_curves": [[{"generation": p["generation"], "win_rate": p["win_rate"]} for p in c] for c in all_scratch],
    }

    # Save
    ensure_dir("results")
    output_path = "results/curriculum_multi_seed.json"
    with open(output_path, "w") as f:
        json.dump(multi_seed_results, f, indent=2)
    print(f"\nMulti-seed results saved to {output_path}")

    # Visualization with error bars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_dir("results/figures")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Collect win rates at each generation across seeds
        all_gens = sorted(set(p["generation"] for c in all_transferred for p in c))
        t_means = [float(np.mean(get_wr_at_gen(all_transferred, g))) for g in all_gens]
        t_stds = [float(np.std(get_wr_at_gen(all_transferred, g))) for g in all_gens]
        s_means = [float(np.mean(get_wr_at_gen(all_scratch, g))) for g in all_gens]
        s_stds = [float(np.std(get_wr_at_gen(all_scratch, g))) for g in all_gens]

        ax.plot(all_gens, t_means, "b-o", label="Curriculum transfer", markersize=4)
        ax.fill_between(all_gens,
                        [m - s for m, s in zip(t_means, t_stds)],
                        [m + s for m, s in zip(t_means, t_stds)],
                        alpha=0.2, color="blue")

        ax.plot(all_gens, s_means, "k-s", label="From scratch", markersize=4)
        ax.fill_between(all_gens,
                        [m - s for m, s in zip(s_means, s_stds)],
                        [m + s for m, s in zip(s_means, s_stds)],
                        alpha=0.2, color="gray")

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="50% baseline")
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Win Rate vs Random", fontsize=12)
        ax.set_title(f"Curriculum Transfer: Go 5x5 -> 7x7 ({len(seeds)} seeds)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig_path = "results/figures/curriculum_multi_seed.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    return multi_seed_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Curriculum Learning (Go 5x5 -> 7x7)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run multi-seed experiment for statistical significance")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds to use (overrides --n-seeds)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds (default: 5). Generates seeds [42, 123, 456, 789, 1024]")
    args = parser.parse_args()

    if args.multi_seed:
        if args.seeds is not None:
            seeds = args.seeds
        else:
            # Generate n_seeds seeds from a fixed list
            seed_pool = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6789]
            seeds = seed_pool[:args.n_seeds]
        run_multi_seed_curriculum(seeds=seeds)
    else:
        run_curriculum_experiment()
