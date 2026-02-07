"""
Evaluate trained agents: baseline vs. bottleneck comparison.

Runs evaluation games for all trained model variants and produces a
comparison table. This is the "results summary" script that collects
performance metrics across all conditions.

Evaluation matrix:
    {PPO, DQN} × {Full baseline, Concept bottleneck} × {Go 7x7, CartPole}

For each variant:
    - Win rate vs random opponent (Go)
    - Average episode reward
    - Average episode length
    - Action diversity (entropy of action distribution)
    - Concept usage distribution (bottleneck only)

Usage:
    python evaluate.py --env go --n-episodes 200
    python evaluate.py --env cartpole --n-episodes 100
    python evaluate.py --env both
"""

import argparse
import os
import json
import numpy as np
import torch
from collections import Counter

from src.environments.go_env import GoEnv
from src.environments.simple_env import CartPoleConceptEnv
from src.networks import GoCNNEncoder, SimpleMLPEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.utils import get_device, ensure_dir


def evaluate_go_baseline_ppo(n_episodes=200, model_dir="models/baseline"):
    """
    Evaluate PPO baseline on Go 7x7.

    Loads the SB3 MaskablePPO model and runs evaluation games against
    a random opponent.
    """
    try:
        from sb3_contrib import MaskablePPO
        from src.environments.go_env import MaskedGoEnv
    except ImportError:
        print("SB3 not available for PPO evaluation")
        return None

    model_path = os.path.join(model_dir, "ppo_go_baseline.zip")
    if not os.path.exists(model_path):
        print(f"PPO model not found at {model_path}")
        return None

    env = MaskedGoEnv(board_size=7)
    model = MaskablePPO.load(model_path)

    wins = 0
    total_reward = 0.0
    episode_lengths = []
    all_actions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0.0

        while not done:
            mask = info.get("action_mask", None)
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            all_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        total_reward += ep_reward
        episode_lengths.append(ep_len)
        if ep_reward > 0:
            wins += 1

    env.close()

    # Calculate action entropy (higher = more diverse action selection)
    action_counts = Counter(all_actions)
    total_actions = sum(action_counts.values())
    action_probs = [c / total_actions for c in action_counts.values()]
    action_entropy = -sum(p * np.log(p + 1e-10) for p in action_probs)

    return {
        "variant": "PPO Full Baseline (Go)",
        "win_rate": wins / n_episodes,
        "mean_reward": total_reward / n_episodes,
        "mean_length": np.mean(episode_lengths),
        "action_entropy": action_entropy,
        "unique_actions": len(action_counts),
    }


def evaluate_go_bottleneck(algo="ppo", n_episodes=200, n_concepts=64,
                           model_dir="models/bottleneck",
                           baseline_dir="models/baseline"):
    """
    Evaluate concept bottleneck agent on Go 7x7.

    Loads the frozen encoder, concept manager, and bottleneck policy,
    then runs the full pipeline: obs → encoder → concept → action.
    Also tracks concept usage statistics.
    """
    device = get_device()
    env = GoEnv(board_size=7)
    n_actions = 50

    # Load encoder
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder_path = os.path.join(baseline_dir, f"{algo}_go_encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"Encoder not found at {encoder_path}")
        return None
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()

    # Load concept manager
    concept_path = os.path.join(model_dir, f"concepts_{algo}_k{n_concepts}.pkl")
    if not os.path.exists(concept_path):
        print(f"Concepts not found at {concept_path}")
        return None
    cm = ConceptManager(n_concepts=n_concepts)
    cm.load(concept_path)

    # Load bottleneck policy
    if algo == "ppo":
        policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts, embed_dim=64, hidden_dim=128, n_actions=n_actions
        ).to(device)
    else:
        policy = ConceptDQNPolicy(
            n_concepts=n_concepts, embed_dim=64, hidden_dim=128, n_actions=n_actions
        ).to(device)

    policy_path = os.path.join(model_dir, f"{algo}_bottleneck_final.pt")
    if not os.path.exists(policy_path):
        print(f"Policy not found at {policy_path}")
        return None
    policy.load_state_dict(
        torch.load(policy_path, map_location=device, weights_only=True)
    )
    policy.eval()

    # Evaluation loop
    wins = 0
    total_reward = 0.0
    episode_lengths = []
    all_actions = []
    all_concepts = []  # Track concept usage

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0.0

        while not done:
            # Full bottleneck pipeline
            concept_id = cm.assign_concept_from_obs(encoder, obs, device)
            mask = info.get("action_mask", None)

            if algo == "ppo":
                action = policy.get_action(concept_id, mask, deterministic=True)
            else:
                action = policy.get_action(concept_id, mask, epsilon=0.0)

            all_actions.append(action)
            all_concepts.append(concept_id)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        total_reward += ep_reward
        episode_lengths.append(ep_len)
        if ep_reward > 0:
            wins += 1

    env.close()

    # Compute concept usage statistics
    concept_counts = Counter(all_concepts)
    n_active = len(concept_counts)
    concept_usage_sorted = concept_counts.most_common()

    # Concept entropy (higher = more uniform usage of concepts)
    total_concept_uses = sum(concept_counts.values())
    concept_probs = [c / total_concept_uses for c in concept_counts.values()]
    concept_entropy = -sum(p * np.log(p + 1e-10) for p in concept_probs)

    # Action entropy
    action_counts = Counter(all_actions)
    total_actions = sum(action_counts.values())
    action_probs = [c / total_actions for c in action_counts.values()]
    action_entropy = -sum(p * np.log(p + 1e-10) for p in action_probs)

    return {
        "variant": f"{algo.upper()} Bottleneck (Go)",
        "win_rate": wins / n_episodes,
        "mean_reward": total_reward / n_episodes,
        "mean_length": np.mean(episode_lengths),
        "action_entropy": action_entropy,
        "concept_entropy": concept_entropy,
        "active_concepts": n_active,
        "total_concepts": n_concepts,
        "top_5_concepts": concept_usage_sorted[:5],
    }


def print_results_table(results):
    """Print a formatted comparison table of all results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Header
    headers = ["Variant", "Win Rate", "Avg Reward", "Avg Length",
               "Action Entropy", "Concepts Used"]
    header_str = " | ".join(f"{h:>15}" for h in headers)
    print(header_str)
    print("-" * len(header_str))

    for r in results:
        if r is None:
            continue
        concepts_str = (f"{r.get('active_concepts', 'N/A')}/{r.get('total_concepts', 'N/A')}"
                       if 'active_concepts' in r else "N/A")
        row = [
            f"{r['variant']:>15}",
            f"{r['win_rate']:>15.2%}",
            f"{r['mean_reward']:>15.3f}",
            f"{r['mean_length']:>15.1f}",
            f"{r['action_entropy']:>15.3f}",
            f"{concepts_str:>15}",
        ]
        print(" | ".join(row))

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--env", type=str, default="go",
                        choices=["go", "cartpole", "both"])
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Number of evaluation episodes")
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    parser.add_argument("--bottleneck-dir", type=str, default="models/bottleneck")
    parser.add_argument("--simple-dir", type=str, default="models/simple")
    args = parser.parse_args()

    results = []

    if args.env in ("go", "both"):
        print("Evaluating Go 7x7 agents...")
        # Baselines
        r = evaluate_go_baseline_ppo(args.n_episodes, args.baseline_dir)
        if r:
            results.append(r)

        # Bottleneck variants
        for algo in ["ppo", "dqn"]:
            r = evaluate_go_bottleneck(
                algo, args.n_episodes, args.n_concepts,
                args.bottleneck_dir, args.baseline_dir,
            )
            if r:
                results.append(r)

    if results:
        print_results_table(results)

        # Save results to JSON
        ensure_dir("results")
        with open("results/evaluation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("Results saved to results/evaluation.json")


if __name__ == "__main__":
    main()
