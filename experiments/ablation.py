"""
Experiment 3: Concept Ablation Study.

Tests whether specific concepts are causally important for winning.
If a concept is truly useful, disabling it should cause a measurable
drop in win rate.

Two ablation modes:
    1. CONCEPT-LEVEL (primary): When concept X appears, force uniform
       random action. Tests whether the concept's entire action mapping matters.
    2. STRATEGY-LEVEL (secondary): When concept X → action Y would be chosen,
       force a random action. Tests a specific concept-action pair.

Also measures concept importance via action distribution diversity:
    - KL divergence between each concept's action distribution and the
      uniform distribution. High KL = concept strongly prefers specific actions.
    - If all concepts produce the same action distribution, concepts are useless.

This is analogous to "lesion studies" in neuroscience: damage a specific
brain region and observe what function is impaired.

Usage:
    python experiments/ablation.py --algo ppo --n-eval 500
    python experiments/ablation.py --algo both
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.strategy_memory import StrategyMemory
from src.utils import get_device, ensure_dir, set_seed


def evaluate_with_concept_ablation(encoder, concept_manager, policy, env,
                                    ablated_concept=None, n_episodes=500,
                                    algo="ppo", device=None):
    """
    Evaluate the agent with an entire concept ablated.

    When the agent encounters the ablated concept, ALL of its actions are
    replaced with uniform random legal actions. This tests whether the
    concept's learned action mapping contributes to winning.

    Args:
        encoder: Frozen encoder.
        concept_manager: Fitted ConceptManager.
        policy: Bottleneck policy.
        env: Go environment.
        ablated_concept: Concept ID to ablate (None = no ablation).
        n_episodes: Number of evaluation games.
        algo: "ppo" or "dqn".
        device: Torch device.

    Returns:
        Dictionary with win_rate, ablation_triggers, etc.
    """
    device = device or get_device()
    wins = 0
    total_reward = 0.0
    ablation_triggers = 0
    total_steps = 0
    concept_counts = defaultdict(int)  # Track how often each concept appears

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # Get concept
            concept_id = concept_manager.assign_concept_from_obs(
                encoder, obs, device
            )
            concept_counts[concept_id] += 1
            mask = info.get("action_mask", None)

            # Check if this concept is ablated
            if ablated_concept is not None and concept_id == ablated_concept:
                # ABLATION: force a random legal action
                if mask is not None:
                    legal = np.where(mask == 1)[0]
                    action = int(np.random.choice(legal)) if len(legal) > 0 else 49
                else:
                    action = np.random.randint(0, 50)
                ablation_triggers += 1
            else:
                # Normal action from policy
                if algo == "ppo":
                    action = policy.get_action(concept_id, mask, deterministic=True)
                else:
                    action = policy.get_action(concept_id, mask, epsilon=0.0)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            total_steps += 1

        total_reward += ep_reward
        if ep_reward > 0:
            wins += 1

    return {
        "win_rate": wins / n_episodes,
        "mean_reward": total_reward / n_episodes,
        "ablation_triggers": ablation_triggers,
        "trigger_rate": ablation_triggers / max(1, total_steps),
        "concept_frequency": ablation_triggers / max(1, total_steps),
        "n_episodes": n_episodes,
    }


def compute_concept_importance(policy, n_concepts=64, n_actions=50, device=None):
    """
    Measure how much each concept's action distribution differs from uniform.

    For each concept, compute the policy's action logits and measure the
    KL divergence from a uniform distribution. High KL = the concept strongly
    prefers certain actions (it's "important"). Low KL = the concept produces
    near-uniform actions (it's useless — might as well pick randomly).

    Also computes pairwise distinctness: average KL divergence between all
    pairs of concepts. If all concepts produce the same distribution,
    distinctness ≈ 0 → concepts aren't differentiated.

    Returns:
        Dictionary with per-concept KL divergences and overall metrics.
    """
    device = device or get_device()
    policy.eval()

    concept_kls = []
    concept_distributions = []
    uniform = torch.ones(n_actions) / n_actions

    with torch.no_grad():
        for c in range(n_concepts):
            cid = torch.LongTensor([c]).to(device)
            logits, _ = policy(cid)
            probs = F.softmax(logits[0], dim=-1).cpu()
            concept_distributions.append(probs)

            # KL divergence from uniform: how much does this concept "know"?
            # KL(P || Uniform) = sum(P * log(P / U)) = sum(P * log(P)) + log(n_actions)
            kl = F.kl_div(uniform.log(), probs, reduction='sum').item()
            concept_kls.append(kl)

    # Pairwise KL divergence between concepts
    # Measures how different concepts are from each other
    pairwise_kls = []
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            p = concept_distributions[i]
            q = concept_distributions[j]
            # Symmetric KL: (KL(P||Q) + KL(Q||P)) / 2
            kl_pq = F.kl_div(q.log().clamp(min=-100), p, reduction='sum').item()
            kl_qp = F.kl_div(p.log().clamp(min=-100), q, reduction='sum').item()
            pairwise_kls.append((kl_pq + kl_qp) / 2)

    return {
        "per_concept_kl": concept_kls,
        "mean_kl_from_uniform": float(np.mean(concept_kls)),
        "max_kl_from_uniform": float(np.max(concept_kls)),
        "mean_pairwise_kl": float(np.mean(pairwise_kls)) if pairwise_kls else 0.0,
        "n_informative_concepts": int(sum(1 for kl in concept_kls if kl > 0.1)),
    }


def find_most_frequent_concepts(encoder, concept_manager, policy, env,
                                 n_episodes=200, algo="ppo", device=None):
    """
    Find the most frequently occurring concepts during gameplay.

    Rather than ablating strategies from the strategy memory (which may
    not trigger often), we find concepts that actually appear frequently
    during evaluation and ablate those.

    Returns:
        List of (concept_id, frequency) sorted by frequency descending.
    """
    device = device or get_device()
    concept_counts = defaultdict(int)
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            concept_id = concept_manager.assign_concept_from_obs(
                encoder, obs, device
            )
            concept_counts[concept_id] += 1
            mask = info.get("action_mask", None)

            if algo == "ppo":
                action = policy.get_action(concept_id, mask, deterministic=True)
            else:
                action = policy.get_action(concept_id, mask, epsilon=0.0)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

    # Sort by frequency
    freq_list = [(cid, count / total_steps) for cid, count in concept_counts.items()]
    freq_list.sort(key=lambda x: x[1], reverse=True)
    return freq_list


def run_ablation_experiment(algo="ppo", n_eval=500, top_k=20,
                            n_concepts=64, seed=42,
                            model_dir="models/bottleneck",
                            baseline_dir="models/baseline"):
    """
    Run the complete concept ablation experiment.

    Steps:
        1. Compute concept importance (KL divergence analysis)
        2. Find most frequent concepts in gameplay
        3. Evaluate baseline (no ablation)
        4. For each of the top-K most frequent concepts: ablate and measure drop
        5. Report results sorted by impact
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

    # ---- Step 1: Concept importance analysis ----
    print("Computing concept importance (KL divergence analysis)...")
    importance = compute_concept_importance(policy, n_concepts, 50, device)
    print(f"  Mean KL from uniform: {importance['mean_kl_from_uniform']:.4f}")
    print(f"  Mean pairwise KL:     {importance['mean_pairwise_kl']:.4f}")
    print(f"  Informative concepts: {importance['n_informative_concepts']}/{n_concepts}")

    # ---- Step 2: Find most frequent concepts ----
    print(f"\nProfiling concept frequency over 200 games...")
    freq_list = find_most_frequent_concepts(
        encoder, cm, policy, env, n_episodes=200, algo=algo, device=device
    )
    print(f"  Top 5 concepts by frequency: "
          + ", ".join(f"C{c}({f:.1%})" for c, f in freq_list[:5]))

    # Select top-K most frequent concepts for ablation
    concepts_to_ablate = [cid for cid, _ in freq_list[:top_k]]

    # ---- Step 3: Baseline evaluation ----
    print(f"\nEvaluating baseline (no ablation, {n_eval} games)...")
    baseline = evaluate_with_concept_ablation(
        encoder, cm, policy, env,
        ablated_concept=None, n_episodes=n_eval,
        algo=algo, device=device,
    )
    baseline_win_rate = baseline["win_rate"]
    print(f"Baseline win rate: {baseline_win_rate:.2%}")

    # ---- Step 4: Ablate each concept ----
    ablation_results = []
    print(f"\nAblating {len(concepts_to_ablate)} most frequent concepts...")

    for i, concept_id in enumerate(concepts_to_ablate):
        freq = dict(freq_list).get(concept_id, 0.0)
        result = evaluate_with_concept_ablation(
            encoder, cm, policy, env,
            ablated_concept=concept_id, n_episodes=n_eval,
            algo=algo, device=device,
        )

        win_rate_drop = baseline_win_rate - result["win_rate"]

        ablation_results.append({
            "rank": i + 1,
            "concept_id": int(concept_id),
            "concept_frequency": float(freq),
            "concept_kl_from_uniform": float(importance["per_concept_kl"][concept_id]),
            "ablated_win_rate": result["win_rate"],
            "win_rate_drop": win_rate_drop,
            "ablation_triggers": result["ablation_triggers"],
            "trigger_rate": result["trigger_rate"],
        })

        status = "***" if win_rate_drop > 0.05 else ""
        print(f"  Concept {concept_id:3d} (freq={freq:.1%}): "
              f"drop={win_rate_drop:+.2%}, "
              f"triggers={result['ablation_triggers']}, "
              f"KL={importance['per_concept_kl'][concept_id]:.3f} {status}")

    # ---- Sort by impact ----
    ablation_results.sort(key=lambda x: x["win_rate_drop"], reverse=True)
    significant = [r for r in ablation_results if r["win_rate_drop"] > 0.05]

    full_results = {
        "algo": algo,
        "mode": "concept_level",
        "baseline_win_rate": baseline_win_rate,
        "n_concepts_tested": len(ablation_results),
        "n_significant_concepts": len(significant),
        "concept_importance": {
            "mean_kl_from_uniform": importance["mean_kl_from_uniform"],
            "max_kl_from_uniform": importance["max_kl_from_uniform"],
            "mean_pairwise_kl": importance["mean_pairwise_kl"],
            "n_informative_concepts": importance["n_informative_concepts"],
        },
        "ablation_results": ablation_results,
    }

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"CONCEPT ABLATION RESULTS — {algo.upper()}")
    print(f"{'='*60}")
    print(f"Baseline win rate:        {baseline_win_rate:.2%}")
    print(f"Concepts tested:          {len(ablation_results)}")
    print(f"Significant (>5% drop):   {len(significant)}")
    print(f"Informative (KL > 0.1):   {importance['n_informative_concepts']}/{n_concepts}")
    print(f"Mean concept KL:          {importance['mean_kl_from_uniform']:.4f}")
    print(f"Mean pairwise KL:         {importance['mean_pairwise_kl']:.4f}")
    print(f"")
    print(f"Top 5 most impactful concepts:")
    for r in ablation_results[:5]:
        print(f"  Concept {r['concept_id']:3d}: "
              f"drop={r['win_rate_drop']:+.2%}, "
              f"freq={r['concept_frequency']:.1%}, "
              f"KL={r['concept_kl_from_uniform']:.3f}")
    print(f"{'='*60}")

    # Save
    ensure_dir("results")
    with open(f"results/ablation_{algo}.json", "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Results saved to results/ablation_{algo}.json")

    env.close()
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run concept ablation experiment")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--n-eval", type=int, default=500,
                        help="Evaluation games per ablation (default 500)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of concepts to ablate")
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Concept Ablation — {algo.upper()}")
        print(f"{'='*60}")
        run_ablation_experiment(
            algo=algo, n_eval=args.n_eval, top_k=args.top_k,
            n_concepts=args.n_concepts, seed=args.seed,
            model_dir=args.model_dir, baseline_dir=args.baseline_dir,
        )


if __name__ == "__main__":
    main()
