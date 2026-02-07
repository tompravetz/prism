"""
CartPole Intervention Experiment.

Replicates the causal intervention experiment from Go on CartPole to
demonstrate task-agnosticism: the concept bottleneck architecture
produces causally meaningful concepts regardless of the domain.

CartPole concepts should capture states like:
    - "pole tilting left, need to push left"
    - "pole nearly balanced, either action OK"
    - "pole tilting right fast, urgent push right needed"

If the bottleneck works, overriding these concepts should dramatically
change the agent's behavior.

Usage:
    python -m experiments.simple_intervention --n-states 200
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

from src.environments.simple_env import CartPoleConceptEnv
from src.networks import SimpleMLPEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.utils import get_device, ensure_dir, set_seed


def collect_cartpole_test_states(env, n_states=200, seed=42):
    """
    Collect test states from CartPole episodes.

    Includes states from throughout episodes — early (balanced),
    middle (slight tilt), and late (extreme tilt before failure).
    """
    np.random.seed(seed)
    states = []

    while len(states) < n_states:
        obs, info = env.reset()
        done = False
        while not done and len(states) < n_states:
            mask = info.get("action_mask", np.ones(2, dtype=np.int8))
            states.append((obs.copy(), mask.copy()))
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return states[:n_states]


def run_cartpole_intervention(algo="ppo", n_states=200, n_alternatives=5,
                               n_concepts=32, seed=42,
                               model_dir="models/simple"):
    """
    Run causal intervention on CartPole bottleneck agent.

    Same logic as Go intervention:
        1. Get natural concept and action
        2. Override with random alternative concepts
        3. Measure action change rate
    """
    set_seed(seed)
    device = get_device()

    env = CartPoleConceptEnv()
    n_actions = 2

    # Load encoder
    encoder = SimpleMLPEncoder(env.observation_space, features_dim=128)
    encoder_path = os.path.join(model_dir, "ppo_cartpole_encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"ERROR: Encoder not found at {encoder_path}")
        return None
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()

    # Load concepts
    cm = ConceptManager(n_concepts=n_concepts)
    concept_path = os.path.join(model_dir, f"concepts_cartpole_k{n_concepts}.pkl")
    if not os.path.exists(concept_path):
        print(f"ERROR: Concepts not found at {concept_path}")
        return None
    cm.load(concept_path)

    # Load policy
    if algo == "ppo":
        policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts, embed_dim=32, hidden_dim=64, n_actions=n_actions
        ).to(device)
    else:
        policy = ConceptDQNPolicy(
            n_concepts=n_concepts, embed_dim=32, hidden_dim=64, n_actions=n_actions
        ).to(device)

    policy_path = os.path.join(model_dir, f"{algo}_cartpole_bottleneck.pt")
    if not os.path.exists(policy_path):
        print(f"ERROR: Policy not found at {policy_path}")
        return None
    policy.load_state_dict(
        torch.load(policy_path, map_location=device, weights_only=True)
    )
    policy.eval()

    # Collect test states
    print(f"Collecting {n_states} CartPole test states...")
    test_states = collect_cartpole_test_states(env, n_states=n_states, seed=seed)

    # Run intervention
    # We measure TWO things:
    #   1. Action change rate: did the argmax action change? (coarse)
    #   2. KL divergence: how much did the full probability distribution shift? (fine)
    # KL divergence is the better metric for CartPole because with only 2 actions,
    # even a big probability shift (0.9→0.6) might not change the argmax action.
    print(f"Running CartPole intervention ({algo.upper()})...")
    action_changes = []
    kl_divergences = []
    concept_action_map = defaultdict(list)

    for obs, mask in test_states:
        # Natural concept and action distribution
        natural_concept = cm.assign_concept_from_obs(encoder, obs, device)

        # Get FULL action distribution for natural concept (not just argmax)
        with torch.no_grad():
            cid_t = torch.LongTensor([natural_concept]).to(device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device) if mask is not None else None
            logits, _ = policy(cid_t, mask_t)
            natural_probs = F.softmax(logits[0], dim=-1)
            natural_action = natural_probs.argmax().item()

        concept_action_map[natural_concept].append(natural_action)

        # Alternative concepts
        all_concepts = list(range(n_concepts))
        if natural_concept in all_concepts:
            all_concepts.remove(natural_concept)
        n_alts = min(n_alternatives, len(all_concepts))
        alt_concepts = np.random.choice(all_concepts, size=n_alts, replace=False)

        for alt in alt_concepts:
            # Get FULL action distribution for alternative concept
            with torch.no_grad():
                alt_cid_t = torch.LongTensor([int(alt)]).to(device)
                alt_logits, _ = policy(alt_cid_t, mask_t)
                alt_probs = F.softmax(alt_logits[0], dim=-1)
                alt_action = alt_probs.argmax().item()

            action_changes.append(alt_action != natural_action)

            # KL divergence: how much did the distribution shift?
            # KL(natural || alternative) — measures info lost when using alt instead
            # Clamp to avoid log(0)
            kl = F.kl_div(
                alt_probs.log().clamp(min=-100),
                natural_probs,
                reduction='sum'
            ).item()
            kl_divergences.append(max(0.0, kl))  # Clamp negative numerical errors

    # Compute results
    change_rate = np.mean(action_changes) if action_changes else 0.0
    mean_kl = np.mean(kl_divergences) if kl_divergences else 0.0
    median_kl = np.median(kl_divergences) if kl_divergences else 0.0

    # Concept-action specificity
    specificities = []
    for concept, actions in concept_action_map.items():
        if actions:
            from collections import Counter
            counts = Counter(actions)
            specificity = counts.most_common(1)[0][1] / len(actions)
            specificities.append(specificity)

    results = {
        "algo": algo,
        "env": "CartPole",
        "n_states": n_states,
        "n_alternatives": n_alternatives,
        "n_concepts": n_concepts,
        "overall_change_rate": float(change_rate),
        "mean_kl_divergence": float(mean_kl),
        "median_kl_divergence": float(median_kl),
        "max_kl_divergence": float(np.max(kl_divergences)) if kl_divergences else 0.0,
        "frac_kl_above_0_1": float(np.mean([kl > 0.1 for kl in kl_divergences])) if kl_divergences else 0.0,
        "mean_concept_specificity": float(np.mean(specificities)) if specificities else 0.0,
        "n_active_concepts": len(concept_action_map),
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"CARTPOLE INTERVENTION RESULTS — {algo.upper()}")
    print(f"{'='*50}")
    print(f"Test states:           {results['n_states']}")
    print(f"Action change rate:    {results['overall_change_rate']:.2%}")
    print(f"Mean KL divergence:    {results['mean_kl_divergence']:.4f}")
    print(f"  (KL > 0 means concepts shift action distributions)")
    print(f"Median KL divergence:  {results['median_kl_divergence']:.4f}")
    print(f"Max KL divergence:     {results['max_kl_divergence']:.4f}")
    print(f"% interventions KL>0.1:{results['frac_kl_above_0_1']:.2%}")
    print(f"Concept specificity:   {results['mean_concept_specificity']:.2%}")
    print(f"Active concepts:       {results['n_active_concepts']}/{n_concepts}")
    print(f"{'='*50}")

    # Save results
    ensure_dir("results")
    with open(f"results/intervention_cartpole_{algo}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results/intervention_cartpole_{algo}.json")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run CartPole concept intervention experiment"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--n-alternatives", type=int, default=5)
    parser.add_argument("--n-concepts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/simple")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]
    for algo in algos:
        run_cartpole_intervention(
            algo=algo, n_states=args.n_states,
            n_alternatives=args.n_alternatives,
            n_concepts=args.n_concepts, seed=args.seed,
            model_dir=args.model_dir,
        )


if __name__ == "__main__":
    main()
