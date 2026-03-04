"""
Train PPO and DQN baseline agents on Breakout (ALE/Breakout-v5).

Saves:
    models/atari/ppo_breakout.zip              — full SB3 PPO model
    models/atari/ppo_breakout_encoder.pt       — NatureCNN weights (512-dim output)
    models/atari/dqn_breakout.zip              — full SB3 DQN model
    models/atari/dqn_breakout_encoder.pt       — NatureCNN weights

Usage:
    python train_atari.py --algo ppo
    python train_atari.py --algo dqn
    python train_atari.py --algo both
"""

import os
import sys
import argparse
import torch
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import set_seed, ensure_dir

ENV_ID = "ALE/Breakout-v5"
SAVE_DIR = "models/atari"


def make_env(n_envs=8, seed=42):
    # frameskip=1 disables ALE's built-in skip; make_atari_env's MaxAndSkipEnv(4)
    # then provides the standard 4-frame skip without double-stacking.
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed,
                         env_kwargs={"full_action_space": False, "frameskip": 1})
    return VecFrameStack(env, n_stack=4)


def train_ppo(total_steps=10_000_000, seed=42, verbose=1):
    set_seed(seed)
    ensure_dir(SAVE_DIR)

    env = make_env(n_envs=8, seed=seed)

    model = PPO(
        "CnnPolicy", env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        seed=seed,
        device="auto",
        tensorboard_log="logs/atari",
    )
    print(f"Training PPO on {ENV_ID} for {total_steps:,} steps...")
    model.learn(total_timesteps=total_steps, progress_bar=True)
    env.close()

    model.save(f"{SAVE_DIR}/ppo_breakout")
    torch.save(
        model.policy.features_extractor.state_dict(),
        f"{SAVE_DIR}/ppo_breakout_encoder.pt",
    )
    print(f"Saved PPO model to {SAVE_DIR}/ppo_breakout.zip")
    return model


def train_dqn(total_steps=5_000_000, seed=42, verbose=1):
    set_seed(seed)
    ensure_dir(SAVE_DIR)

    env = make_env(n_envs=1, seed=seed)

    model = DQN(
        "CnnPolicy", env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=50_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        verbose=verbose,
        seed=seed,
        device="auto",
        tensorboard_log="logs/atari",
    )
    print(f"Training DQN on {ENV_ID} for {total_steps:,} steps...")
    model.learn(total_timesteps=total_steps, progress_bar=True)
    env.close()

    model.save(f"{SAVE_DIR}/dqn_breakout")
    torch.save(
        model.q_net.features_extractor.state_dict(),
        f"{SAVE_DIR}/dqn_breakout_encoder.pt",
    )
    print(f"Saved DQN model to {SAVE_DIR}/dqn_breakout.zip")
    return model


def evaluate_model(model, n_episodes=20, seed=42):
    """Return mean episodic clipped reward over n_episodes.

    Breakout: each brick broken = +1 (clipped). Random agent ~= 1-2 per episode.
    Competent agent = 20-50+. Expert = 300+.
    """
    env = make_env(n_envs=1, seed=seed + 9999)
    obs = env.reset()
    ep_rewards, ep_reward = [], 0.0
    while len(ep_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ep_reward += reward[0]
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
    env.close()
    import numpy as np
    return {"mean_reward": float(np.mean(ep_rewards)), "n_episodes": n_episodes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "both"], default="both")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override default steps (PPO=10M, DQN=5M)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.algo in ("ppo", "both"):
        ppo = train_ppo(total_steps=args.steps or 10_000_000, seed=args.seed)
        r = evaluate_model(ppo, n_episodes=20)
        print(f"PPO eval: mean_reward={r['mean_reward']:.1f}")

    if args.algo in ("dqn", "both"):
        dqn = train_dqn(total_steps=args.steps or 5_000_000, seed=args.seed)
        r = evaluate_model(dqn, n_episodes=20)
        print(f"DQN eval: mean_reward={r['mean_reward']:.1f}")
