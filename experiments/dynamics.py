"""
Phase 2B Experiment: Concept Dynamics Accuracy.

Tests whether concept transitions are predictable:
    Given (concept_t, action_t), can we predict concept_{t+1}?

If yes → concepts capture structured game dynamics, enabling planning.
If no → transitions are too stochastic, but this is still informative.

The experiment:
    1. Collect transitions from a trained bottleneck agent
    2. Train a dynamics model: P(concept_{t+1} | concept_t, action_t)
    3. Measure prediction accuracy (top-1 and top-5)
    4. If accuracy > 40%: implement 1-step lookahead planner

The planner works by:
    1. For each legal action, predict the next concept
    2. Evaluate each predicted next concept's value
    3. Pick the action leading to the best predicted next concept

Usage:
    python -m experiments.dynamics --algo ppo
    python -m experiments.dynamics --algo both
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.dynamics_model import (ConceptDynamicsModel, TransitionCollector,
                                 train_dynamics_model)
from src.utils import get_device, ensure_dir, set_seed


def collect_transitions(encoder, concept_manager, policy, env,
                        n_episodes=1000, algo="ppo",
                        device=None):
    """
    Collect concept transitions by running the bottleneck agent.

    For each step in each episode, record:
        (concept_t, action_t, concept_{t+1})

    We use the TRAINED policy (not random) to collect transitions,
    because we want transitions that reflect actual gameplay.

    Args:
        encoder: Frozen encoder.
        concept_manager: Fitted ConceptManager.
        policy: Trained bottleneck policy.
        env: Game environment.
        n_episodes: Number of episodes to collect from.
        algo: "ppo" or "dqn".
        device: Torch device.

    Returns:
        TransitionCollector with recorded transitions.
    """
    device = device or get_device()
    collector = TransitionCollector(max_size=200_000)

    print(f"Collecting transitions from {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        prev_concept = None

        while not done:
            # Get current concept
            concept = concept_manager.assign_concept_from_obs(
                encoder, obs, device
            )

            # Record transition from previous step
            if prev_concept is not None:
                collector.add(prev_concept, prev_action, concept)

            # Select action
            mask = info.get("action_mask", None)
            if algo == "ppo":
                action = policy.get_action(concept, mask, deterministic=False)
            else:
                action = policy.get_action(concept, mask, epsilon=0.1)

            prev_concept = concept
            prev_action = action

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if ep % 200 == 0:
            print(f"  Collected from {ep}/{n_episodes} episodes "
                  f"({len(collector)} transitions)")

    print(f"Total transitions collected: {len(collector)}")
    return collector


def evaluate_planner(encoder, concept_manager, policy, dynamics_model,
                     env, n_episodes=100, algo="ppo", device=None):
    """
    Evaluate 1-step lookahead planner vs greedy policy.

    The planner:
        1. For each legal action a, predict next concept c' = argmax P(c' | c, a)
        2. Evaluate each c' using the policy's value function: V(c')
        3. Pick the action leading to the highest-valued next concept

    This tests whether the dynamics model can improve decision-making
    by enabling lookahead in concept space.

    Args:
        dynamics_model: Trained ConceptDynamicsModel.
        Others: standard model components.

    Returns:
        Dictionary comparing planner vs greedy performance.
    """
    device = device or get_device()

    # Evaluate greedy (normal) policy
    greedy_wins = 0
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            concept = concept_manager.assign_concept_from_obs(encoder, obs, device)
            mask = info.get("action_mask", None)
            if algo == "ppo":
                action = policy.get_action(concept, mask, deterministic=True)
            else:
                action = policy.get_action(concept, mask, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if reward > 0:
            greedy_wins += 1

    # Evaluate planner policy
    planner_wins = 0
    n_actions = 50  # Go 7x7

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            concept = concept_manager.assign_concept_from_obs(encoder, obs, device)
            mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))
            legal_actions = np.where(mask == 1)[0]

            if len(legal_actions) == 0:
                action = 49  # pass
            else:
                # For each legal action, predict next concept and evaluate it
                best_value = float('-inf')
                best_action = legal_actions[0]

                for a in legal_actions:
                    # Predict most likely next concept
                    next_probs = dynamics_model.predict(concept, int(a))
                    predicted_next = np.argmax(next_probs)

                    # Evaluate next concept using policy's value function
                    if algo == "ppo":
                        with torch.no_grad():
                            cid_t = torch.LongTensor([predicted_next]).to(device)
                            _, value = policy(cid_t)
                            v = value[0].item()
                    else:
                        with torch.no_grad():
                            cid_t = torch.LongTensor([predicted_next]).to(device)
                            q_values = policy(cid_t)
                            v = q_values[0].max().item()

                    if v > best_value:
                        best_value = v
                        best_action = int(a)

                action = best_action

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if reward > 0:
            planner_wins += 1

    return {
        "greedy_win_rate": greedy_wins / n_episodes,
        "planner_win_rate": planner_wins / n_episodes,
        "improvement": (planner_wins - greedy_wins) / n_episodes,
    }


def run_dynamics_experiment(algo="ppo", n_collect_episodes=1000,
                            n_concepts=64, seed=42,
                            model_dir="models/bottleneck",
                            baseline_dir="models/baseline"):
    """
    Complete dynamics model experiment.

    Steps:
        1. Load trained bottleneck agent
        2. Collect concept transitions
        3. Train dynamics model
        4. Report accuracy
        5. If accuracy > 40%: evaluate planner vs greedy
    """
    set_seed(seed)
    device = get_device()

    # ---- Load models ----
    env = GoEnv(board_size=7)

    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder_path = os.path.join(baseline_dir, f"{algo}_go_encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"ERROR: Encoder not found at {encoder_path}")
        return None
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()

    cm = ConceptManager(n_concepts=n_concepts)
    concept_path = os.path.join(model_dir, f"concepts_{algo}_k{n_concepts}.pkl")
    if not os.path.exists(concept_path):
        print(f"ERROR: Concepts not found at {concept_path}")
        return None
    cm.load(concept_path)

    if algo == "ppo":
        policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts, embed_dim=64, hidden_dim=128, n_actions=50
        ).to(device)
    else:
        policy = ConceptDQNPolicy(
            n_concepts=n_concepts, embed_dim=64, hidden_dim=128, n_actions=50
        ).to(device)

    policy_path = os.path.join(model_dir, f"{algo}_bottleneck_final.pt")
    if not os.path.exists(policy_path):
        print(f"ERROR: Policy not found at {policy_path}")
        return None
    policy.load_state_dict(
        torch.load(policy_path, map_location=device, weights_only=True)
    )
    policy.eval()

    # ---- Collect transitions ----
    transitions = collect_transitions(
        encoder, cm, policy, env,
        n_episodes=n_collect_episodes, algo=algo, device=device,
    )

    if len(transitions) < 1000:
        print("WARNING: Too few transitions collected. Increase n_collect_episodes.")

    # ---- Train dynamics model ----
    print("\nTraining concept dynamics model...")
    model, train_results = train_dynamics_model(
        transitions, n_concepts=n_concepts, n_actions=50,
        n_epochs=50, lr=1e-3, device=str(device),
    )

    # ---- Decision point: is accuracy high enough for planning? ----
    test_acc = train_results["test_accuracy"]
    planner_results = None

    if test_acc > 0.40:
        print(f"\nAccuracy ({test_acc:.2%}) > 40%: Testing 1-step lookahead planner...")
        planner_results = evaluate_planner(
            encoder, cm, policy, model, env,
            n_episodes=100, algo=algo, device=device,
        )
        print(f"  Greedy win rate:  {planner_results['greedy_win_rate']:.2%}")
        print(f"  Planner win rate: {planner_results['planner_win_rate']:.2%}")
        print(f"  Improvement:      {planner_results['improvement']:+.2%}")
    else:
        print(f"\nAccuracy ({test_acc:.2%}) < 40%: Concept transitions are stochastic.")
        print("This is an informative negative result — dynamics are too noisy for planning.")

    # ---- Save results ----
    ensure_dir("results")
    full_results = {
        "algo": algo,
        "n_transitions": len(transitions),
        "dynamics_training": train_results,
        "planner": planner_results,
    }
    with open(f"results/dynamics_{algo}.json", "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Results saved to results/dynamics_{algo}.json")

    # Save model
    ensure_dir(model_dir)
    torch.save(model.state_dict(),
               os.path.join(model_dir, f"dynamics_model_{algo}.pt"))

    env.close()
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Concept dynamics experiment")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--n-episodes", type=int, default=1000,
                        help="Episodes for transition collection")
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]
    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Concept Dynamics Experiment — {algo.upper()}")
        print(f"{'='*60}")
        run_dynamics_experiment(
            algo=algo, n_collect_episodes=args.n_episodes,
            n_concepts=args.n_concepts, seed=args.seed,
            model_dir=args.model_dir, baseline_dir=args.baseline_dir,
        )


if __name__ == "__main__":
    main()
