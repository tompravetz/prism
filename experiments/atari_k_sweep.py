"""
K-value sensitivity sweep for Breakout concept bottleneck transfer.

For each K in {16, 32, 64, 128}:
  - Fit concept managers on PPO and DQN Breakout features
  - Train bottleneck via BC
  - Evaluate own-bottleneck performance
  - Evaluate PPO->DQN zero-shot transfer

Requires: models/atari/ppo_breakout.zip and models/atari/dqn_breakout.zip (Breakout models)
Saves: results/atari_k_sweep.json

Usage:
    python experiments/atari_k_sweep.py
    python experiments/atari_k_sweep.py --n-eval 30
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, ensure_dir
from train_atari import make_env, SAVE_DIR
from train_atari_bottleneck import (
    get_encoder, extract_features_and_actions, train_bottleneck_bc, preprocess
)

K_VALUES = [16, 32, 64, 128]
N_ACTIONS = 4


def evaluate_own(model, encoder, cm, bottleneck, n_episodes, seed, device):
    bottleneck.eval()
    env = make_env(n_envs=1, seed=seed + 6000)
    obs = env.reset()
    ep_rewards, ep_r = [], 0.0
    while len(ep_rewards) < n_episodes:
        with torch.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            features = encoder(preprocess(model, obs_tensor)).cpu().numpy()
            cid = cm.assign_concept(features)[0]
            action = bottleneck.get_action(cid, action_mask=None, deterministic=True)
        obs, reward, done, _ = env.step([action])
        ep_r += reward[0]
        if done[0]:
            ep_rewards.append(ep_r)
            ep_r = 0.0
    env.close()
    return float(np.mean(ep_rewards))


def evaluate_transfer(src_policy, tgt_model, tgt_encoder, tgt_cm,
                      mapping, n_episodes, seed, device):
    from src.concept_aligner import ConceptAligner
    aligner = ConceptAligner(tgt_cm, tgt_cm)  # src_cm unused for transfer_policy
    transferred = aligner.transfer_policy(
        src_policy, mapping,
        target_n_concepts=tgt_cm.n_concepts,
        target_n_actions=N_ACTIONS,
    )
    transferred.to(device)
    transferred.eval()

    env = make_env(n_envs=1, seed=seed + 7000)
    obs = env.reset()
    ep_rewards, ep_r = [], 0.0
    while len(ep_rewards) < n_episodes:
        with torch.no_grad():
            obs_tensor, _ = tgt_model.policy.obs_to_tensor(obs)
            features = tgt_encoder(preprocess(tgt_model, obs_tensor)).cpu().numpy()
            cid = tgt_cm.assign_concept(features)[0]
            action = transferred.get_action(cid, action_mask=None, deterministic=True)
        obs, reward, done, _ = env.step([action])
        ep_r += reward[0]
        if done[0]:
            ep_rewards.append(ep_r)
            ep_r = 0.0
    env.close()
    return float(np.mean(ep_rewards))


def run_k_sweep(n_eval=30, seed=42):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"K-sweep on Breakout (device={device_str})")
    print(f"K values: {K_VALUES}, eval episodes per K: {n_eval}\n")

    # Load baseline models once
    ppo_model = PPO.load(f"{SAVE_DIR}/ppo_breakout", device=device)
    ppo_model.policy.set_training_mode(False)
    dqn_model = DQN.load(f"{SAVE_DIR}/dqn_breakout", device=device)
    dqn_model.policy.set_training_mode(False)

    ppo_encoder = get_encoder(ppo_model, "ppo")
    dqn_encoder = get_encoder(dqn_model, "dqn")

    # Collect features once (reused across K values)
    print("Collecting features (300 episodes each)...")
    ppo_features, ppo_actions = extract_features_and_actions(
        ppo_model, "ppo", n_episodes=300, seed=seed
    )
    dqn_features, dqn_actions = extract_features_and_actions(
        dqn_model, "dqn", n_episodes=300, seed=seed
    )
    print(f"  PPO: {len(ppo_features):,} frames, DQN: {len(dqn_features):,} frames\n")

    results = []

    for K in K_VALUES:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] K={K}")

        # Fit concept managers
        ppo_cm = ConceptManager(n_concepts=K)
        ppo_cm.fit_from_array(ppo_features)
        dqn_cm = ConceptManager(n_concepts=K)
        dqn_cm.fit_from_array(dqn_features)

        # Train bottlenecks via BC
        ppo_bottleneck = train_bottleneck_bc(
            ppo_cm, ppo_features, ppo_actions,
            n_epochs=20, lr=1e-3, device=device_str
        )
        dqn_bottleneck = train_bottleneck_bc(
            dqn_cm, dqn_features, dqn_actions,
            n_epochs=20, lr=1e-3, device=device_str
        )
        ppo_bottleneck.to(device)
        dqn_bottleneck.to(device)

        # Alignment
        aligner = ConceptAligner(ppo_cm, dqn_cm)
        mapping = aligner.hungarian_alignment()
        align_sim = aligner.alignment_quality(mapping)["mean_similarity"]

        # Evaluate
        ppo_own = evaluate_own(
            ppo_model, ppo_encoder, ppo_cm, ppo_bottleneck, n_eval, seed, device
        )
        dqn_own = evaluate_own(
            dqn_model, dqn_encoder, dqn_cm, dqn_bottleneck, n_eval, seed, device
        )
        ppo_to_dqn = evaluate_transfer(
            ppo_bottleneck, dqn_model, dqn_encoder, dqn_cm,
            mapping, n_eval, seed, device
        )

        row = {
            "K": K,
            "alignment_sim": float(align_sim),
            "ppo_own_reward": ppo_own,
            "dqn_own_reward": dqn_own,
            "ppo_to_dqn_reward": ppo_to_dqn,
        }
        results.append(row)
        print(f"  align_sim={align_sim:.4f}  ppo_own={ppo_own:.1f}  "
              f"dqn_own={dqn_own:.1f}  transfer={ppo_to_dqn:.1f}")

    ensure_dir("results")
    output = {
        "K_values": K_VALUES,
        "n_eval": n_eval,
        "env": "ALE/Breakout-v5",
        "seed": seed,
        "sweep": results,
    }
    with open("results/atari_k_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/atari_k_sweep.json")

    # Summary table
    print(f"\n{'K':>5} {'align_sim':>10} {'ppo_own':>10} {'dqn_own':>10} {'transfer':>10}")
    print("-" * 50)
    for row in results:
        print(f"{row['K']:>5} {row['alignment_sim']:>10.4f} "
              f"{row['ppo_own_reward']:>10.1f} {row['dqn_own_reward']:>10.1f} "
              f"{row['ppo_to_dqn_reward']:>10.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=30,
                        help="Episodes per evaluation (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    run_k_sweep(n_eval=args.n_eval, seed=args.seed)
