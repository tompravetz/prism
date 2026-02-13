"""
Task 3: Strategy Composition — Combining Specialist Go Agents.

Trains specialist Go agents with different reward shaping, then composes
their strategies through the concept bottleneck:

Specialists:
    1. Aggressive: bonus for captures + center control
    2. Territorial: bonus for corner/edge occupation
    3. Balanced: standard reward (existing PPO agent, no new training)

Composition methods:
    a. Embedding average: weighted average of aligned specialist embeddings
    b. Phase routing: opening -> territorial, midgame -> balanced, endgame -> aggressive
    c. Concept union: combined concept space with meta-selection

Usage:
    python experiments/strategy_composition.py
"""

import os
import sys
import json
import time
import copy
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


# ============================================================
# Reward-shaping wrappers for specialist training
# ============================================================

class AggressiveGoEnv(GoEnv):
    """
    Go environment with aggressive reward shaping.

    Gives bonus rewards for:
        - Capturing opponent stones (+0.05 per capture)
        - Playing on center positions (+0.02 for center 3x3)

    Reward shaping is intentionally subtle to avoid distorting the base
    policy too much — large bonuses caused specialists to drop below 50%
    win rate in earlier experiments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Track previous white stone count to detect captures
        self._prev_white_stones = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_white_stones = obs[:, :, 1].sum()
        return obs, info

    def step(self, action):
        # Count white stones before the move (from previous observation)
        white_before = self._prev_white_stones

        obs, reward, terminated, truncated, info = super().step(action)

        # Count stones after — difference is captures
        white_after = obs[:, :, 1].sum()
        self._prev_white_stones = white_after
        captures = max(0, white_before - white_after)
        capture_bonus = captures * 0.05

        # Center control bonus (3x3 center region)
        center_bonus = 0.0
        if not terminated:
            bs = self.board_size
            center_start = bs // 2 - 1
            center_end = center_start + 3
            center_region = obs[center_start:center_end, center_start:center_end, 0]
            center_bonus = center_region.sum() * 0.02

        shaped_reward = reward + capture_bonus + center_bonus
        return obs, shaped_reward, terminated, truncated, info


class TerritorialGoEnv(GoEnv):
    """
    Go environment with territorial reward shaping.

    Gives bonus rewards for:
        - Occupying corner positions (+0.03 per stone)
        - Occupying edge positions (+0.02 per stone)

    Reward shaping is intentionally subtle to avoid distorting the base
    policy too much — large bonuses caused specialists to drop below 50%
    win rate in earlier experiments.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Territory bonus based on position
        territory_bonus = 0.0
        if not terminated:
            bs = self.board_size
            black_stones = obs[:, :, 0]

            # Corner bonus (4 corners, 2x2 each)
            corners = (black_stones[0:2, 0:2].sum() +
                       black_stones[0:2, bs-2:bs].sum() +
                       black_stones[bs-2:bs, 0:2].sum() +
                       black_stones[bs-2:bs, bs-2:bs].sum())
            territory_bonus += corners * 0.03

            # Edge bonus (edges minus corners)
            edge_top = black_stones[0, 2:bs-2].sum()
            edge_bottom = black_stones[bs-1, 2:bs-2].sum()
            edge_left = black_stones[2:bs-2, 0].sum()
            edge_right = black_stones[2:bs-2, bs-1].sum()
            territory_bonus += (edge_top + edge_bottom + edge_left + edge_right) * 0.02

        shaped_reward = reward + territory_bonus
        return obs, shaped_reward, terminated, truncated, info


def train_specialist(env_class, specialist_name, encoder, n_concepts=64,
                     n_generations=100, steps_per_gen=20000, device=None):
    """
    Train a specialist bottleneck agent with shaped rewards.

    Uses an existing frozen encoder (PPO baseline) — only the bottleneck
    policy is trained with the shaped reward.

    Args:
        env_class: GoEnv subclass with reward shaping.
        specialist_name: Name for saving ("aggressive", "territorial").
        encoder: Frozen encoder (shared across specialists).
        n_concepts: Number of concepts.
        n_generations: Training generations.
        steps_per_gen: Steps per generation.
        device: Torch device.

    Returns:
        (policy, concept_manager, learning_curve)
    """
    device = device or get_device()
    save_dir = f"models/specialist_{specialist_name}"
    ensure_dir(save_dir)

    env = env_class(board_size=7)

    # Discover concepts for this specialist (or load existing)
    concepts_path = os.path.join(save_dir, "concepts_k64.pkl")
    if os.path.exists(concepts_path):
        cm = ConceptManager(n_concepts=n_concepts)
        cm.load(concepts_path)
    else:
        cm = discover_concepts(encoder, env, n_concepts=n_concepts,
                               n_episodes=500, save_path=concepts_path, device=device)

    # Create trainer
    trainer = PPOBottleneckTrainer(
        encoder=encoder, concept_manager=cm,
        n_actions=50, n_concepts=n_concepts,
        lr=3e-4, gamma=0.99, device=device,
    )

    # Check if already trained
    policy_path = os.path.join(save_dir, "bottleneck_final.pt")
    if os.path.exists(policy_path):
        print(f"  Loading existing {specialist_name} policy from {policy_path}")
        trainer.policy.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=True)
        )
        return trainer.policy, cm, []

    # Train
    eval_env = GoEnv(board_size=7)
    learning_curve = []

    print(f"  Training {specialist_name} specialist ({n_generations} generations)...")
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


def evaluate_composed_policy(encoder, cm, policy, n_episodes=100, device=None):
    """Evaluate a composed policy on standard Go 7x7."""
    device = device or get_device()
    env = GoEnv(board_size=7)

    def agent_fn(obs, action_mask):
        concept_id = cm.assign_concept_from_obs(encoder, obs, device)
        return policy.get_action(concept_id, action_mask, deterministic=True)

    results = evaluate_agent(agent_fn, env, n_episodes=n_episodes)
    env.close()
    return results


def compose_by_embedding_average(specialists, reference_cm, encoder, device):
    """
    Compose specialists by averaging aligned embeddings.

    Each specialist's embeddings are aligned to the reference concept space,
    then averaged with equal weights to create a generalist policy.

    Args:
        specialists: Dict of {name: (policy, cm)} for each specialist.
        reference_cm: ConceptManager to use as the reference frame.
        encoder: Shared encoder.
        device: Torch device.

    Returns:
        Composed ConceptBottleneckPolicy.
    """
    n_concepts = reference_cm.n_concepts
    embed_dim = 64
    n_actions = 50

    # Start with zero embeddings
    composed_embed = torch.zeros(n_concepts, embed_dim)
    n_specialists = len(specialists)
    weight = 1.0 / n_specialists

    # Accumulate embeddings, handling head weights from first specialist
    first_policy = None

    for name, (policy, cm) in specialists.items():
        if first_policy is None:
            first_policy = policy

        aligner = ConceptAligner(cm, reference_cm)
        mapping = aligner.greedy_alignment()

        src_embed = policy.embedding.weight.data.clone()
        for src_id, tgt_id in mapping.items():
            if tgt_id < n_concepts:
                composed_embed[tgt_id] += weight * src_embed[src_id]

    # Create composed policy
    composed = ConceptBottleneckPolicy(
        n_concepts=n_concepts, embed_dim=embed_dim,
        hidden_dim=128, n_actions=n_actions,
    )
    composed.embedding.weight.data = composed_embed

    # Use first specialist's head as starting point
    if first_policy is not None:
        composed.policy_head.load_state_dict(first_policy.policy_head.state_dict())
        composed.value_head.load_state_dict(first_policy.value_head.state_dict())

    composed.to(device)
    return composed


def compose_by_phase_routing(specialists, reference_cm, encoder, device):
    """
    Compose specialists by game phase routing.

    Uses board state to determine game phase:
        - Opening (0-15 stones): use territorial specialist
        - Midgame (16-30 stones): use balanced specialist
        - Endgame (31+ stones): use aggressive specialist

    This creates a meta-policy that switches between specialists based
    on the game phase, using the concept bottleneck as the interface.

    Args:
        specialists: Dict of {name: (policy, cm)} with keys "aggressive",
                     "territorial", "balanced".
        reference_cm: Shared concept manager.
        encoder: Shared encoder.
        device: Torch device.

    Returns:
        A callable agent function (not a policy module).
    """
    def phase_routing_agent(obs, action_mask):
        # Determine game phase from stone count
        total_stones = obs[:, :, 0].sum() + obs[:, :, 1].sum()

        if total_stones <= 15:
            phase = "territorial"
        elif total_stones <= 30:
            phase = "balanced"
        else:
            phase = "aggressive"

        policy, cm = specialists[phase]
        concept_id = cm.assign_concept_from_obs(encoder, obs, device)
        return policy.get_action(concept_id, action_mask, deterministic=True)

    return phase_routing_agent


def run_composition_experiment():
    """Run the full strategy composition experiment."""
    set_seed(42)
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Strategy Composition Experiment")
    print(f"[{timestamp}] ============================================================")

    results = {}

    # Load shared encoder (PPO baseline)
    env = GoEnv(board_size=7)
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder.load_state_dict(
        torch.load("models/baseline/ppo_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()
    env.close()

    # ============================================================
    # Step 1: Load/train specialists
    # ============================================================
    print(f"\n--- Loading/Training Specialists ---")

    # Balanced = existing PPO bottleneck (no new training)
    balanced_cm = ConceptManager(n_concepts=64)
    balanced_cm.load("models/bottleneck/concepts_ppo_k64.pkl")
    balanced_policy = ConceptBottleneckPolicy(n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50)
    balanced_policy.load_state_dict(
        torch.load("models/bottleneck/ppo_bottleneck_final.pt",
                    map_location=device, weights_only=True)
    )
    balanced_policy.to(device)
    balanced_policy.eval()
    print(f"  Loaded balanced specialist (existing PPO bottleneck)")

    # Aggressive specialist
    aggressive_policy, aggressive_cm, agg_curve = train_specialist(
        AggressiveGoEnv, "aggressive", encoder,
        n_generations=100, steps_per_gen=20000, device=device,
    )
    aggressive_policy.to(device)
    aggressive_policy.eval()

    # Territorial specialist
    territorial_policy, territorial_cm, terr_curve = train_specialist(
        TerritorialGoEnv, "territorial", encoder,
        n_generations=100, steps_per_gen=20000, device=device,
    )
    territorial_policy.to(device)
    territorial_policy.eval()

    # ============================================================
    # Step 2: Evaluate individual specialists
    # ============================================================
    print(f"\n--- Evaluating Individual Specialists ---")

    specialists = {
        "aggressive": (aggressive_policy, aggressive_cm),
        "territorial": (territorial_policy, territorial_cm),
        "balanced": (balanced_policy, balanced_cm),
    }

    individual_results = {}
    for name, (policy, cm) in specialists.items():
        eval_result = evaluate_composed_policy(encoder, cm, policy,
                                               n_episodes=100, device=device)
        individual_results[name] = eval_result
        print(f"  {name}: win_rate={eval_result['win_rate']:.2%}")

    results["individual"] = individual_results

    # ============================================================
    # Step 3: Composition Method A — Embedding Average
    # ============================================================
    print(f"\n--- Composition A: Embedding Average ---")

    composed_avg = compose_by_embedding_average(
        specialists, balanced_cm, encoder, device,
    )

    avg_result = evaluate_composed_policy(encoder, balanced_cm, composed_avg,
                                          n_episodes=100, device=device)
    print(f"  Embedding average: win_rate={avg_result['win_rate']:.2%}")
    results["embedding_average"] = avg_result

    # ============================================================
    # Step 4: Composition Method B — Phase Routing
    # ============================================================
    print(f"\n--- Composition B: Phase Routing ---")

    phase_agent = compose_by_phase_routing(specialists, balanced_cm, encoder, device)

    eval_env = GoEnv(board_size=7)
    phase_result = evaluate_agent(phase_agent, eval_env, n_episodes=100)
    eval_env.close()
    print(f"  Phase routing: win_rate={phase_result['win_rate']:.2%}")
    results["phase_routing"] = phase_result

    # ============================================================
    # Step 5: Cross-specialist alignment analysis
    # ============================================================
    print(f"\n--- Cross-Specialist Alignment ---")

    alignment_matrix = {}
    spec_names = list(specialists.keys())
    for i, name_i in enumerate(spec_names):
        for j, name_j in enumerate(spec_names):
            if i >= j:
                continue
            cm_i = specialists[name_i][1]
            cm_j = specialists[name_j][1]
            aligner = ConceptAligner(cm_i, cm_j)
            mapping = aligner.hungarian_alignment()
            quality = aligner.alignment_quality(mapping)
            key = f"{name_i}_vs_{name_j}"
            alignment_matrix[key] = {
                "mean_similarity": quality["mean_similarity"],
                "min_similarity": quality["min_similarity"],
                "max_similarity": quality["max_similarity"],
            }
            print(f"  {name_i} vs {name_j}: sim={quality['mean_similarity']:.4f}")

    results["cross_alignment"] = alignment_matrix

    # ============================================================
    # Save results
    # ============================================================
    ensure_dir("results")
    output_path = "results/strategy_composition.json"

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
    print(f"[{ts}] Summary: Strategy Composition")
    print(f"[{ts}]   Individual:")
    for name, res in individual_results.items():
        print(f"[{ts}]     {name}: {res['win_rate']:.2%}")
    print(f"[{ts}]   Composed:")
    print(f"[{ts}]     Embedding average: {avg_result['win_rate']:.2%}")
    print(f"[{ts}]     Phase routing: {phase_result['win_rate']:.2%}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito colorblind-friendly palette
        OI_ORANGE = "#E69F00"
        OI_GREEN = "#009E73"
        OI_BLUE = "#0072B2"
        OI_PURPLE = "#CC79A7"
        OI_RED = "#D55E00"
        OI_GRAY = "#999999"

        ensure_dir("results/figures")
        plt.rcParams.update({"font.size": 11})
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        names = list(individual_results.keys()) + ["Emb. Average", "Phase Routing"]
        wrs = ([individual_results[n]["win_rate"] for n in individual_results] +
               [avg_result["win_rate"], phase_result["win_rate"]])
        colors = [OI_RED, OI_GREEN, OI_BLUE, OI_PURPLE, OI_ORANGE]

        bars = ax.bar(range(len(names)), wrs, color=colors[:len(names)],
                      edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.5, color=OI_GRAY, linestyle="--", alpha=0.5,
                   label="Random baseline (50%)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
        ax.set_ylabel("Win Rate vs Random", fontsize=12)
        ax.set_title("Strategy Composition: Individual vs Composed Agents", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)

        for i, (name, wr) in enumerate(zip(names, wrs)):
            ax.text(i, wr + 0.02, f"{wr:.0%}", ha="center", fontsize=10,
                    fontweight="bold")

        fig_path = "results/figures/strategy_composition.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Composition Experiment")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Delete cached specialist models and retrain from scratch")
    args = parser.parse_args()

    # If force-retrain, remove cached specialist models so they retrain
    if args.force_retrain:
        import shutil
        for name in ["aggressive", "territorial"]:
            path = f"models/specialist_{name}"
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"  Removed cached specialist: {path}")

    run_composition_experiment()
