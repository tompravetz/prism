"""
Fit K-means concept managers and train concept bottleneck policies
on Breakout via behavioral cloning from the trained PPO/DQN baselines.

Requires: models/atari/ppo_breakout.zip and models/atari/dqn_breakout.zip

Saves:
    models/atari/concepts_ppo_k64.pkl
    models/atari/concepts_dqn_k64.pkl
    models/atari/ppo_bottleneck.pt
    models/atari/dqn_bottleneck.pt

Usage:
    python train_atari_bottleneck.py --algo ppo
    python train_atari_bottleneck.py --algo dqn
    python train_atari_bottleneck.py --algo both
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.preprocessing import preprocess_obs as sb3_preprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.utils import set_seed, ensure_dir
from train_atari import make_env, evaluate_model, SAVE_DIR

N_CONCEPTS = 64
N_ACTIONS = 4      # Breakout minimal action space: NOOP, FIRE, RIGHT, LEFT
FEATURES_DIM = 512  # NatureCNN output


def get_encoder(model, algo):
    if algo == "ppo":
        return model.policy.features_extractor
    else:
        return model.q_net.features_extractor


def preprocess(model, obs_tensor):
    """Normalize uint8 observations to float — SB3 2.x no longer does this inside NatureCNN."""
    return sb3_preprocess(obs_tensor, model.observation_space,
                          normalize_images=model.policy.normalize_images)


def extract_features_and_actions(model, algo, n_episodes=500, seed=42):
    """
    Collect (feature_vector, action) pairs from the baseline policy.
    Returns: features (N, 512), actions (N,)
    """
    encoder = get_encoder(model, algo)
    encoder.eval()

    env = make_env(n_envs=1, seed=seed + 5000)
    obs = env.reset()

    all_features, all_actions = [], []
    ep_count = 0

    while ep_count < n_episodes:
        with torch.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            features = encoder(preprocess(model, obs_tensor))
            all_features.append(features.cpu().numpy())

        action, _ = model.predict(obs, deterministic=True)
        all_actions.append(action[0])

        obs, _, done, _ = env.step(action)
        if done[0]:
            ep_count += 1

    env.close()
    return np.vstack(all_features), np.array(all_actions)


def train_bottleneck_bc(concept_manager, features, actions,
                        n_epochs=20, lr=1e-3, device="cpu"):
    """
    Train ConceptBottleneckPolicy via behavioral cloning:
    concept_id → action (cross-entropy loss against baseline actions).
    """
    device = torch.device(device)
    policy = ConceptBottleneckPolicy(
        n_concepts=N_CONCEPTS,
        embed_dim=64,
        hidden_dim=128,
        n_actions=N_ACTIONS,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Assign concept IDs for all features
    concept_ids = concept_manager.assign_concept(features)  # (N,)
    concept_ids_t = torch.tensor(concept_ids, dtype=torch.long, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)

    N = len(concept_ids_t)
    batch_size = 512
    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            cids = concept_ids_t[idx]
            acts = actions_t[idx]

            optimizer.zero_grad()
            logits, _ = policy(cids)   # forward() returns (logits, values)
            loss = criterion(logits, acts)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    policy.load_state_dict(best_state)
    return policy


def evaluate_bottleneck(model, algo, concept_manager, bottleneck,
                        n_episodes=50, seed=42):
    """Evaluate the concept bottleneck policy on Breakout."""
    encoder = get_encoder(model, algo)
    encoder.eval()
    bottleneck.eval()
    device = next(bottleneck.parameters()).device

    env = make_env(n_envs=1, seed=seed + 7777)
    obs = env.reset()
    ep_rewards, ep_reward = [], 0.0

    while len(ep_rewards) < n_episodes:
        with torch.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            features = encoder(preprocess(model, obs_tensor)).cpu().numpy()
            concept_id = concept_manager.assign_concept(features)[0]
            action = bottleneck.get_action(
                concept_id, action_mask=None, deterministic=True
            )
        obs, reward, done, _ = env.step([action])
        ep_reward += reward[0]
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0

    env.close()
    return {"mean_reward": float(np.mean(ep_rewards))}


def run_algo(algo, seed=42):
    ensure_dir(SAVE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training {algo.upper()} concept bottleneck on Breakout")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load baseline
    cls = PPO if algo == "ppo" else DQN
    model = cls.load(f"{SAVE_DIR}/{algo}_breakout", device=device)
    model.policy.set_training_mode(False)

    # Collect features + actions from baseline gameplay
    print(f"\nCollecting features from {algo.upper()} gameplay (500 episodes)...")
    features, actions = extract_features_and_actions(
        model, algo, n_episodes=500, seed=seed
    )
    print(f"  Collected {len(features):,} frames")

    # Fit K-means
    print(f"\nFitting K-means (K={N_CONCEPTS})...")
    cm = ConceptManager(n_concepts=N_CONCEPTS)
    cm.fit_from_array(features)
    cm.save(f"{SAVE_DIR}/concepts_{algo}_k{N_CONCEPTS}.pkl")
    print(f"  Saved to {SAVE_DIR}/concepts_{algo}_k{N_CONCEPTS}.pkl")

    # Train bottleneck via BC
    print(f"\nTraining concept bottleneck via behavioral cloning...")
    bottleneck = train_bottleneck_bc(
        cm, features, actions, n_epochs=30, lr=1e-3, device=device
    )
    torch.save(
        bottleneck.state_dict(),
        f"{SAVE_DIR}/{algo}_bottleneck.pt",
    )
    print(f"  Saved to {SAVE_DIR}/{algo}_bottleneck.pt")

    # Evaluate
    print(f"\nEvaluating bottleneck (50 episodes)...")
    bottleneck.to(device)
    r_bottleneck = evaluate_bottleneck(
        model, algo, cm, bottleneck, n_episodes=50, seed=seed
    )
    r_baseline = evaluate_model(model, n_episodes=50, seed=seed)
    print(f"  Baseline:   mean_reward={r_baseline['mean_reward']:.1f}")
    print(f"  Bottleneck: mean_reward={r_bottleneck['mean_reward']:.1f}")
    return cm, bottleneck


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.algo in ("ppo", "both"):
        run_algo("ppo", seed=args.seed)

    if args.algo in ("dqn", "both"):
        run_algo("dqn", seed=args.seed)

    print("\nDone.")
