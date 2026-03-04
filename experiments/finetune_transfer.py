"""
Fine-tuning experiment: zero-shot transfer vs. from-scratch convergence on Go 7x7.

Takes the PPO->DQN zero-shot transferred policy and fine-tunes it against GnuGo L1
with the DQN concept bottleneck's RL loop. Tracks win rate every generation
and compares to the from-scratch learning curve.

This tests the practical value claim: does transferred initialization accelerate
convergence relative to training from scratch on the target agent's concept space?

Saves: results/finetune_transfer.json

Usage:
    python experiments/finetune_transfer.py
    python experiments/finetune_transfer.py --n-gen 40 --steps-per-gen 20000
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir
from train_bottleneck import evaluate_agent


def load_agents(device):
    env = GoEnv(board_size=7)

    ppo_encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    ppo_encoder.load_state_dict(
        torch.load("models/baseline/ppo_go_encoder.pt",
                   map_location=device, weights_only=True)
    )
    ppo_encoder.to(device).eval()

    ppo_cm = ConceptManager(n_concepts=64)
    ppo_cm.load("models/bottleneck/concepts_ppo_k64.pkl")

    ppo_policy = ConceptBottleneckPolicy(
        n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50
    )
    ppo_policy.load_state_dict(
        torch.load("models/bottleneck/ppo_bottleneck_final.pt",
                   map_location=device, weights_only=True)
    )
    ppo_policy.to(device).eval()

    dqn_encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    dqn_encoder.load_state_dict(
        torch.load("models/baseline/dqn_go_encoder.pt",
                   map_location=device, weights_only=True)
    )
    dqn_encoder.to(device).eval()

    dqn_cm = ConceptManager(n_concepts=64)
    dqn_cm.load("models/bottleneck/concepts_dqn_k64.pkl")

    env.close()
    return {
        "ppo": {"encoder": ppo_encoder, "cm": ppo_cm, "policy": ppo_policy},
        "dqn": {"encoder": dqn_encoder, "cm": dqn_cm},
    }


def collect_rollout(policy, encoder, cm, env, n_steps, device):
    """
    Collect (concept_id, action, reward, done) tuples for one rollout.
    Action mask is read from the info dict returned by reset/step.
    """
    transitions = []
    obs, info = env.reset()
    action_mask = info.get("action_mask", None)

    for _ in range(n_steps):
        with torch.no_grad():
            cid = cm.assign_concept_from_obs(encoder, obs, device)
            action = policy.get_action(cid, action_mask, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transitions.append((cid, action, reward, done))
        action_mask = info.get("action_mask", None)
        obs = next_obs

        if done:
            obs, info = env.reset()
            action_mask = info.get("action_mask", None)

    return transitions


def reinforce_update(policy, transitions, optimizer, gamma=0.99):
    """Simple REINFORCE update on collected transitions."""
    # Compute discounted returns
    returns = []
    G = 0.0
    for _, _, r, done in reversed(transitions):
        if done:
            G = 0.0
        G = r + gamma * G
        returns.insert(0, G)

    device = next(policy.parameters()).device
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    # Normalize returns
    if returns_t.std() > 1e-6:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    device = next(policy.parameters()).device
    concept_ids = torch.tensor([t[0] for t in transitions], dtype=torch.long, device=device)
    actions = torch.tensor([t[1] for t in transitions], dtype=torch.long, device=device)
    # returns_t already on device (set above)

    optimizer.zero_grad()
    logits, _ = policy(concept_ids)   # forward() returns (logits, values)
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs[torch.arange(len(actions), device=device), actions]
    loss = -(selected_log_probs * returns_t).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    return loss.item()


def run_finetune_experiment(n_gen=40, steps_per_gen=10000, n_eval=100,
                             gnugo_level=1, seed=42):
    device = get_device()
    set_seed(seed)

    print("=" * 60)
    print("Fine-tuning Transfer Experiment: PPO->DQN on Go 7x7")
    print(f"  Generations: {n_gen}, Steps/gen: {steps_per_gen:,}")
    print(f"  Eval games: {n_eval}, GnuGo level: {gnugo_level}")
    print("=" * 60)

    agents = load_agents(device)
    ppo = agents["ppo"]
    dqn = agents["dqn"]

    aligner = ConceptAligner(ppo["cm"], dqn["cm"])
    mapping = aligner.hungarian_alignment()

    # Build transferred policy (zero-shot starting point)
    transferred = aligner.transfer_policy(
        ppo["policy"], mapping,
        target_n_concepts=64, target_n_actions=50,
    )
    transferred.to(device)

    # Build fresh random policy (from-scratch baseline)
    scratch = ConceptBottleneckPolicy(
        n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50
    ).to(device)

    try:
        from visualizer.opponents import GnuGoOpponent
        opponent = GnuGoOpponent(level=gnugo_level)
    except Exception:
        opponent = None
        print("  Warning: GnuGo not available, evaluating vs random opponent")

    def make_env():
        return GoEnv(board_size=7, opponent_fn=opponent)

    def agent_fn_for(policy):
        def fn(obs, action_mask):
            cid = dqn["cm"].assign_concept_from_obs(dqn["encoder"], obs, device)
            return policy.get_action(cid, action_mask, deterministic=True)
        return fn

    # Evaluate zero-shot starting point
    eval_env = make_env()
    r0_transferred = evaluate_agent(agent_fn_for(transferred), eval_env, n_eval)
    r0_scratch = evaluate_agent(agent_fn_for(scratch), eval_env, n_eval)
    eval_env.close()

    print(f"\nGen 0 (before fine-tuning):")
    print(f"  Transferred (zero-shot): {r0_transferred['win_rate']:.2%}")
    print(f"  From scratch:            {r0_scratch['win_rate']:.2%}")

    transferred_curve = [{"gen": 0, "win_rate": r0_transferred["win_rate"],
                          "mean_reward": r0_transferred["mean_reward"]}]
    scratch_curve = [{"gen": 0, "win_rate": r0_scratch["win_rate"],
                      "mean_reward": r0_scratch["mean_reward"]}]

    opt_t = optim.Adam(transferred.parameters(), lr=1e-4)
    opt_s = optim.Adam(scratch.parameters(), lr=1e-4)

    train_env_t = make_env()
    train_env_s = make_env()

    for gen in range(1, n_gen + 1):
        transferred.train()
        scratch.train()

        # Collect rollouts
        transitions_t = collect_rollout(
            transferred, dqn["encoder"], dqn["cm"],
            train_env_t, steps_per_gen, device
        )
        transitions_s = collect_rollout(
            scratch, dqn["encoder"], dqn["cm"],
            train_env_s, steps_per_gen, device
        )

        # Update
        loss_t = reinforce_update(transferred, transitions_t, opt_t)
        loss_s = reinforce_update(scratch, transitions_s, opt_s)

        transferred.eval()
        scratch.eval()

        # Evaluate every 5 generations
        if gen % 5 == 0 or gen == n_gen:
            eval_env = make_env()
            rt = evaluate_agent(agent_fn_for(transferred), eval_env, n_eval)
            rs = evaluate_agent(agent_fn_for(scratch), eval_env, n_eval)
            eval_env.close()

            transferred_curve.append({
                "gen": gen, "win_rate": rt["win_rate"],
                "mean_reward": rt["mean_reward"]
            })
            scratch_curve.append({
                "gen": gen, "win_rate": rs["win_rate"],
                "mean_reward": rs["mean_reward"]
            })
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Gen {gen:3d}: "
                  f"transferred={rt['win_rate']:.2%}, "
                  f"scratch={rs['win_rate']:.2%}")

    train_env_t.close()
    train_env_s.close()
    if opponent is not None:
        opponent.close()

    # Find convergence generation (first gen to exceed 60% for each)
    threshold = 0.60
    conv_t = next((p["gen"] for p in transferred_curve if p["win_rate"] >= threshold),
                  None)
    conv_s = next((p["gen"] for p in scratch_curve if p["win_rate"] >= threshold),
                  None)

    print(f"\nConvergence to {threshold:.0%}:")
    if conv_t is not None:
        print(f"  Transferred: gen {conv_t} ({conv_t * steps_per_gen:,} steps)")
    else:
        print(f"  Transferred: did not reach {threshold:.0%} in {n_gen} generations")
    if conv_s is not None:
        print(f"  From scratch: gen {conv_s} ({conv_s * steps_per_gen:,} steps)")
    else:
        print(f"  From scratch: did not reach {threshold:.0%} in {n_gen} generations")

    ensure_dir("results")
    output = {
        "n_gen": n_gen,
        "steps_per_gen": steps_per_gen,
        "n_eval": n_eval,
        "gnugo_level": gnugo_level,
        "seed": seed,
        "convergence_threshold": threshold,
        "transferred_convergence_gen": conv_t,
        "scratch_convergence_gen": conv_s,
        "transferred_curve": transferred_curve,
        "scratch_curve": scratch_curve,
    }
    with open("results/finetune_transfer.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/finetune_transfer.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning transfer experiment")
    parser.add_argument("--n-gen", type=int, default=40,
                        help="Number of fine-tuning generations (default: 40)")
    parser.add_argument("--steps-per-gen", type=int, default=10000,
                        help="Training steps per generation (default: 10000)")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Evaluation games per checkpoint (default: 100)")
    parser.add_argument("--gnugo-level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_finetune_experiment(
        n_gen=args.n_gen,
        steps_per_gen=args.steps_per_gen,
        n_eval=args.n_eval,
        gnugo_level=args.gnugo_level,
        seed=args.seed,
    )
