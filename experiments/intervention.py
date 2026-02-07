"""
Experiment 1: Causal Concept Override Intervention.

This is the KEY experiment that validates whether concepts are causally
meaningful. If concepts truly mediate the agent's decision-making, then
forcing a different concept should change the agent's action.

The experiment works as follows:
    1. For each test state, get the agent's NATURAL concept (the one it
       would normally assign to this state).
    2. Get the NATURAL action (what the agent would do under this concept).
    3. Force N random ALTERNATIVE concepts and record the resulting actions.
    4. Measure how often the action changes — this is the "causal impact" of
       the concept on the action.

If concepts are just noise: action change rate ≈ 0% (concepts don't matter)
If concepts are causally meaningful: action change rate >> 50% (concepts
    strongly determine actions)

We also analyze:
    - Which concepts cause which actions (concept→action mapping)
    - How action distributions shift under different concepts
    - Whether PPO and DQN produce differently causal concepts

Usage:
    python -m experiments.intervention --algo ppo --n-states 500
    python -m experiments.intervention --algo both
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from collections import defaultdict
from scipy import stats

# Add project root to path so 'src' package is importable when running
# this script directly (e.g., python experiments/intervention.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.utils import get_device, ensure_dir, set_seed


def collect_test_states(env, n_states=500, seed=42):
    """
    Collect diverse test states by playing random games.

    We want a good spread of game states: early game (empty board),
    mid game (contested positions), and late game (mostly filled).

    Args:
        env: Go environment.
        n_states: Number of states to collect.
        seed: Random seed for reproducibility.

    Returns:
        List of (observation, action_mask) tuples.
    """
    np.random.seed(seed)
    states = []

    while len(states) < n_states:
        obs, info = env.reset()
        done = False
        while not done and len(states) < n_states:
            mask = info.get("action_mask", np.ones(50, dtype=np.int8))
            states.append((obs.copy(), mask.copy()))

            # Random action to advance the game
            legal = np.where(mask == 1)[0]
            action = np.random.choice(legal) if len(legal) > 0 else 49
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return states[:n_states]


def run_intervention(encoder, concept_manager, policy, test_states,
                     n_alternatives=5, algo="ppo", device=None):
    """
    Run the causal intervention experiment.

    For each test state:
        1. Get natural concept and natural action
        2. Try n_alternatives random different concepts
        3. Record whether the action changes

    This directly tests: does changing the concept CAUSE a change in action?
    If yes → the concept is a genuine causal mediator of the decision.
    If no → the policy ignores the concept (bottleneck is ineffective).

    Args:
        encoder: Frozen encoder for concept extraction.
        concept_manager: Fitted ConceptManager.
        policy: Trained bottleneck policy.
        test_states: List of (obs, action_mask) tuples.
        n_alternatives: Number of alternative concepts to try per state.
        algo: "ppo" or "dqn" (affects how we get actions).
        device: Torch device.

    Returns:
        Dictionary of intervention results and statistics.
    """
    device = device or get_device()
    n_concepts = concept_manager.n_concepts

    # Accumulators for results
    action_changes = []                    # Did the action change? (bool per intervention)
    natural_concepts = []                  # Natural concept for each state
    natural_actions = []                   # Natural action for each state
    intervention_results = []              # Detailed results per state
    concept_action_map = defaultdict(list) # Maps concept → list of actions taken

    for i, (obs, mask) in enumerate(test_states):
        # Step 1: Get natural concept and action
        natural_concept = concept_manager.assign_concept_from_obs(
            encoder, obs, device
        )
        if algo == "ppo":
            natural_action = policy.get_action(natural_concept, mask, deterministic=True)
        else:
            natural_action = policy.get_action(natural_concept, mask, epsilon=0.0)

        natural_concepts.append(natural_concept)
        natural_actions.append(natural_action)
        concept_action_map[natural_concept].append(natural_action)

        # Step 2: Try alternative concepts
        # Pick random concepts that are DIFFERENT from the natural one
        all_concepts = list(range(n_concepts))
        all_concepts.remove(natural_concept)
        alt_concepts = np.random.choice(all_concepts, size=n_alternatives, replace=False)

        alt_actions = []
        for alt_concept in alt_concepts:
            if algo == "ppo":
                alt_action = policy.get_action(int(alt_concept), mask, deterministic=True)
            else:
                alt_action = policy.get_action(int(alt_concept), mask, epsilon=0.0)
            alt_actions.append(alt_action)

            # Record whether this intervention changed the action
            action_changes.append(alt_action != natural_action)
            concept_action_map[int(alt_concept)].append(alt_action)

        # Step 3: Store detailed result for this state
        intervention_results.append({
            "state_idx": i,
            "natural_concept": int(natural_concept),
            "natural_action": int(natural_action),
            "alt_concepts": [int(c) for c in alt_concepts],
            "alt_actions": [int(a) for a in alt_actions],
            "n_changes": sum(1 for a in alt_actions if a != natural_action),
        })

    # ---- Compute Statistics ----

    # Overall action change rate: what fraction of interventions changed the action?
    overall_change_rate = np.mean(action_changes)

    # Per-state change rate: for each state, what fraction of alternatives changed?
    per_state_change_rates = [
        r["n_changes"] / n_alternatives for r in intervention_results
    ]

    # Concept specificity: how consistent is each concept's preferred action?
    concept_specificity = {}
    for concept, actions in concept_action_map.items():
        if len(actions) > 0:
            # Most common action and how often it occurs
            action_counts = defaultdict(int)
            for a in actions:
                action_counts[a] += 1
            most_common = max(action_counts.values())
            # Specificity = fraction of times the most common action is chosen
            concept_specificity[concept] = most_common / len(actions)

    # Statistical significance test: is the change rate significantly > 0?
    # Use binomial test: null hypothesis is that changes happen by chance
    n_changes_total = sum(action_changes)
    n_interventions_total = len(action_changes)

    # Under null hypothesis (concept doesn't matter), change rate = 0
    # We test against this with a one-sided binomial test
    binom_p_value = stats.binomtest(
        n_changes_total, n_interventions_total, p=0.5,
        alternative='greater'
    ).pvalue if n_interventions_total > 0 else 1.0

    return {
        "algo": algo,
        "n_states": len(test_states),
        "n_alternatives": n_alternatives,
        "n_interventions": n_interventions_total,
        "overall_change_rate": float(overall_change_rate),
        "mean_per_state_change_rate": float(np.mean(per_state_change_rates)),
        "std_per_state_change_rate": float(np.std(per_state_change_rates)),
        "median_per_state_change_rate": float(np.median(per_state_change_rates)),
        "p_value_vs_chance": float(binom_p_value),
        "mean_concept_specificity": float(np.mean(list(concept_specificity.values()))),
        "n_active_concepts": len(concept_specificity),
        "detailed_results": intervention_results,
    }


def run_experiment(algo="ppo", n_states=500, n_alternatives=5,
                   n_concepts=64, seed=42,
                   model_dir="models/bottleneck",
                   baseline_dir="models/baseline"):
    """
    Complete intervention experiment for one algorithm.

    Loads all required models, collects test states, runs interventions,
    and saves results.
    """
    set_seed(seed)
    device = get_device()

    # ---- Load models ----
    env = GoEnv(board_size=7)

    # Encoder
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

    # Concept manager
    cm = ConceptManager(n_concepts=n_concepts)
    concept_path = os.path.join(model_dir, f"concepts_{algo}_k{n_concepts}.pkl")
    if not os.path.exists(concept_path):
        print(f"ERROR: Concepts not found at {concept_path}")
        return None
    cm.load(concept_path)

    # Policy
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

    # ---- Collect test states ----
    print(f"Collecting {n_states} test states...")
    test_states = collect_test_states(env, n_states=n_states, seed=seed)
    print(f"Collected {len(test_states)} test states.")

    # ---- Run intervention ----
    print(f"Running intervention experiment ({algo.upper()})...")
    results = run_intervention(
        encoder, cm, policy, test_states,
        n_alternatives=n_alternatives, algo=algo, device=device,
    )

    # ---- Print summary ----
    print(f"\n{'='*50}")
    print(f"INTERVENTION RESULTS — {algo.upper()}")
    print(f"{'='*50}")
    print(f"Test states:          {results['n_states']}")
    print(f"Alternatives tested:  {results['n_alternatives']} per state")
    print(f"Total interventions:  {results['n_interventions']}")
    print(f"")
    print(f"Overall change rate:  {results['overall_change_rate']:.2%}")
    print(f"  (> 50% means concepts are causally meaningful)")
    print(f"Mean per-state rate:  {results['mean_per_state_change_rate']:.2%}")
    print(f"Median per-state:     {results['median_per_state_change_rate']:.2%}")
    print(f"p-value vs chance:    {results['p_value_vs_chance']:.6f}")
    print(f"  (< 0.05 means statistically significant)")
    print(f"Mean specificity:     {results['mean_concept_specificity']:.2%}")
    print(f"Active concepts:      {results['n_active_concepts']}/{n_concepts}")
    print(f"{'='*50}")

    # ---- Save results ----
    ensure_dir("results")
    output_path = f"results/intervention_{algo}.json"
    # Save summary (without detailed per-state results to keep file manageable)
    summary = {k: v for k, v in results.items() if k != "detailed_results"}
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run causal concept intervention experiment"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--n-states", type=int, default=500,
                        help="Number of test states")
    parser.add_argument("--n-alternatives", type=int, default=5,
                        help="Alternative concepts per state")
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]
    all_results = {}

    for algo in algos:
        results = run_experiment(
            algo=algo, n_states=args.n_states,
            n_alternatives=args.n_alternatives,
            n_concepts=args.n_concepts, seed=args.seed,
            model_dir=args.model_dir, baseline_dir=args.baseline_dir,
        )
        if results:
            all_results[algo] = results

    # ---- Cross-algorithm comparison ----
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print("CROSS-ALGORITHM COMPARISON")
        print(f"{'='*60}")
        for key in ["overall_change_rate", "mean_concept_specificity", "n_active_concepts"]:
            ppo_val = all_results["ppo"].get(key, "N/A")
            dqn_val = all_results["dqn"].get(key, "N/A")
            print(f"{key:30s} | PPO: {ppo_val} | DQN: {dqn_val}")


if __name__ == "__main__":
    main()
