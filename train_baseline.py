"""
Train baseline agents (PPO + DQN) on Go 7x7 and CartPole.

Stage 1 of the concept bottleneck pipeline:
    Board (7x7x3) → CNN Encoder → Features (128d) → Full Policy → Action

Trains TWO baselines: one with MaskablePPO, one with custom DQN.
Both use the same CNN encoder architecture for fair comparison.

Usage:
    python train_baseline.py --env go --algo ppo --steps 200000
    python train_baseline.py --env go --algo dqn --steps 200000
    python train_baseline.py --env cartpole --algo ppo --steps 100000
    python train_baseline.py --env go --algo both --steps 200000
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from src.environments.go_env import GoEnv, MaskedGoEnv, AugmentedGoEnv
from src.environments.simple_env import CartPoleConceptEnv
from src.networks import GoCNNEncoder, SimpleMLPEncoder, QNetwork
from src.utils import set_seed, get_device, ensure_dir

# SB3 imports
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 / sb3-contrib not available. "
          "Install with: pip install stable-baselines3 sb3-contrib")


# ============================================================
# SB3-based MaskablePPO Training
# ============================================================

class WinRateCallback(BaseCallback):
    """Track win rate during SB3 training."""

    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=50, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.win_rates = []
        self.best_win_rate = 0.0

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            wins = 0
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                while not done:
                    action_masks = info.get("action_mask", None)
                    action, _ = self.model.predict(obs, deterministic=True,
                                                    action_masks=action_masks)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    if terminated and reward > 0:
                        wins += 1

            win_rate = wins / self.n_eval_episodes
            self.win_rates.append(win_rate)
            self.best_win_rate = max(self.best_win_rate, win_rate)

            if self.verbose:
                print(f"  Step {self.num_timesteps}: win_rate={win_rate:.2%} "
                      f"(best={self.best_win_rate:.2%})")

            self.logger.record("eval/win_rate", win_rate)
        return True


def mask_fn(env):
    """Action mask function for SB3 ActionMasker wrapper."""
    return env.action_masks()


def train_ppo_go(total_timesteps=200_000, seed=42, save_dir="models/baseline"):
    """Train MaskablePPO on Go 7x7."""
    ensure_dir(save_dir)
    set_seed(seed)

    # Create environments
    # Training env uses augmentation (random rotation/reflection each episode)
    # to learn rotation-invariant features — critical for concept stability.
    # AlphaGo uses the same technique. Without this, concepts become
    # position-dependent artifacts instead of strategic structure.
    train_env = AugmentedGoEnv(board_size=7)
    train_env = ActionMasker(train_env, mask_fn)

    # Eval env is NOT augmented — measures true performance in canonical orientation
    eval_env = MaskedGoEnv(board_size=7)

    # Policy kwargs: use our CNN encoder
    policy_kwargs = dict(
        features_extractor_class=GoCNNEncoder,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        "MlpPolicy",  # Will be overridden by features_extractor
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        seed=seed,
    )

    callback = WinRateCallback(eval_env, eval_freq=10000, n_eval_episodes=50)

    print(f"Training MaskablePPO on Go 7x7 for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model_path = os.path.join(save_dir, "ppo_go_baseline")
    model.save(model_path)
    print(f"PPO model saved to {model_path}")

    # Save encoder separately for concept extraction
    encoder_state = model.policy.features_extractor.state_dict()
    torch.save(encoder_state, os.path.join(save_dir, "ppo_go_encoder.pt"))
    print(f"PPO encoder saved")

    train_env.close()
    eval_env.close()

    return model, callback.win_rates


def train_ppo_cartpole(total_timesteps=100_000, seed=42, save_dir="models/baseline"):
    """Train PPO on CartPole."""
    ensure_dir(save_dir)
    set_seed(seed)

    from stable_baselines3 import PPO

    train_env = CartPoleConceptEnv()
    eval_env = CartPoleConceptEnv()

    policy_kwargs = dict(
        features_extractor_class=SimpleMLPEncoder,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        seed=seed,
    )

    print(f"Training PPO on CartPole for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    model_path = os.path.join(save_dir, "ppo_cartpole_baseline")
    model.save(model_path)

    encoder_state = model.policy.features_extractor.state_dict()
    torch.save(encoder_state, os.path.join(save_dir, "ppo_cartpole_encoder.pt"))

    print(f"CartPole PPO model saved to {model_path}")
    train_env.close()
    eval_env.close()
    return model


# ============================================================
# Custom DQN Training (with proper action masking for Go)
# ============================================================

class ReplayBuffer:
    """Simple replay buffer for DQN."""

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.buffer.append((state, action, reward, next_state, done,
                           action_mask, next_action_mask))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        action_masks = np.array([b[5] for b in batch])
        next_action_masks = np.array([b[6] for b in batch])

        return (states, actions, rewards, next_states, dones,
                action_masks, next_action_masks)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Custom DQN agent with action masking for Go.

    Uses separate encoder + Q-head, matching the architecture of the PPO agent.
    """

    def __init__(self, encoder, n_actions, features_dim=128,
                 lr=1e-4, gamma=0.99, tau=0.005, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=50_000, buffer_size=50_000,
                 batch_size=64, device=None):
        self.device = device or get_device()
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Networks
        self.encoder = encoder.to(self.device)
        self.q_head = QNetwork(features_dim, n_actions).to(self.device)

        # Target networks (copies)
        # SB3's BaseFeaturesExtractor stores observation_space as _observation_space
        self.target_encoder = type(encoder)(encoder._observation_space,
                                            features_dim).to(self.device)
        self.target_encoder.load_state_dict(encoder.state_dict())
        self.target_q_head = QNetwork(features_dim, n_actions).to(self.device)
        self.target_q_head.load_state_dict(self.q_head.state_dict())

        # Optimizer (train both encoder and Q-head)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.q_head.parameters()),
            lr=lr,
        )

        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    @property
    def epsilon(self):
        """Current epsilon for epsilon-greedy exploration."""
        decay_progress = min(1.0, self.steps_done / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_progress

    def select_action(self, obs, action_mask=None):
        """Epsilon-greedy action selection with masking."""
        self.steps_done += 1

        if np.random.random() < self.epsilon:
            # Random legal action
            if action_mask is not None:
                legal = np.where(action_mask == 1)[0]
                return int(np.random.choice(legal)) if len(legal) > 0 else 0
            return np.random.randint(0, self.n_actions)

        # Greedy action
        self.encoder.eval()
        self.q_head.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            features = self.encoder(obs_t)
            mask_t = None
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            q_values = self.q_head(features, mask_t)
            return q_values[0].argmax().item()

    def update(self):
        """Perform one DQN update step."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.encoder.train()
        self.q_head.train()

        (states, actions, rewards, next_states, dones,
         action_masks, next_action_masks) = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        next_masks_t = torch.FloatTensor(next_action_masks).to(self.device)

        # Current Q-values
        features = self.encoder(states_t)
        q_values = self.q_head(features)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (with action masking on next states)
        with torch.no_grad():
            next_features = self.target_encoder(next_states_t)
            next_q_values = self.target_q_head(next_features, next_masks_t)
            # Handle all-masked case
            next_q_max = next_q_values.max(1)[0]
            next_q_max = torch.where(
                torch.isinf(next_q_max),
                torch.zeros_like(next_q_max),
                next_q_max,
            )
            target = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # Huber loss
        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.q_head.parameters()), 10.0
        )
        self.optimizer.step()

        # Soft update target networks
        self._soft_update()

        return loss.item()

    def _soft_update(self):
        """Soft update target network parameters."""
        for target_p, p in zip(self.target_encoder.parameters(),
                               self.encoder.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)
        for target_p, p in zip(self.target_q_head.parameters(),
                               self.q_head.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

    def save(self, path):
        """Save agent state."""
        ensure_dir(os.path.dirname(path))
        torch.save({
            "encoder": self.encoder.state_dict(),
            "q_head": self.q_head.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "target_q_head": self.target_q_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path):
        """Load agent state."""
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(data["encoder"])
        self.q_head.load_state_dict(data["q_head"])
        self.target_encoder.load_state_dict(data["target_encoder"])
        self.target_q_head.load_state_dict(data["target_q_head"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.steps_done = data["steps_done"]


def train_dqn_go(total_timesteps=200_000, seed=42, save_dir="models/baseline"):
    """
    Train custom DQN on Go 7x7 with reward shaping.

    Reward shaping gives DQN small intermediate rewards for capturing
    opponent stones (+0.05 per capture). This makes the sparse Go reward
    signal denser, which is critical for DQN's one-step bootstrapping.
    Without shaping, Q-values must propagate backwards through 30-50 moves
    of zeros before reaching meaningful signal — too slow for DQN.

    The eval env does NOT use reward shaping so win rate reflects true
    game performance (terminal +1/-1 only).
    """
    ensure_dir(save_dir)
    set_seed(seed)
    device = get_device()

    # Training env uses reward shaping for denser DQN signal
    # AND augmentation for rotation-invariant features
    env = AugmentedGoEnv(board_size=7, reward_shaping=True, capture_reward=0.05)
    # Eval env has NO shaping and NO augmentation — measures true performance
    eval_env = GoEnv(board_size=7, reward_shaping=False)
    n_actions = env.action_count  # 50

    # Create encoder
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)

    agent = DQNAgent(
        encoder=encoder,
        n_actions=n_actions,
        features_dim=128,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=total_timesteps // 2,
        buffer_size=50_000,
        batch_size=64,
        device=device,
    )

    # Training loop
    print(f"Training DQN on Go 7x7 for {total_timesteps} steps...")
    episode_rewards = []
    win_rates = []
    total_steps = 0
    episode = 0
    best_win_rate = 0.0

    while total_steps < total_timesteps:
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done and total_steps < total_timesteps:
            action_mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))
            action = agent.select_action(obs, action_mask)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_mask = next_info.get("action_mask",
                                      np.zeros(n_actions, dtype=np.int8) if done
                                      else np.ones(n_actions, dtype=np.int8))

            agent.replay_buffer.push(obs, action, reward, next_obs,
                                     float(done), action_mask, next_mask)

            loss = agent.update()
            episode_reward += reward
            obs = next_obs
            info = next_info
            total_steps += 1

        episode_rewards.append(episode_reward)
        episode += 1

        # Evaluate periodically
        if episode % 100 == 0:
            wins = 0
            n_eval = 50
            for _ in range(n_eval):
                e_obs, e_info = eval_env.reset()
                e_done = False
                while not e_done:
                    e_mask = e_info.get("action_mask", np.ones(n_actions, dtype=np.int8))
                    e_action = agent.select_action(e_obs, e_mask)
                    e_obs, e_reward, e_term, e_trunc, e_info = eval_env.step(e_action)
                    e_done = e_term or e_trunc
                    if e_term and e_reward > 0:
                        wins += 1

            win_rate = wins / n_eval
            win_rates.append(win_rate)
            best_win_rate = max(best_win_rate, win_rate)
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode} | Steps {total_steps} | "
                  f"ε={agent.epsilon:.3f} | Avg R={avg_reward:.3f} | "
                  f"Win rate={win_rate:.2%} (best={best_win_rate:.2%})")

    # Save
    agent.save(os.path.join(save_dir, "dqn_go_baseline.pt"))
    torch.save(agent.encoder.state_dict(),
               os.path.join(save_dir, "dqn_go_encoder.pt"))
    print(f"DQN agent saved to {save_dir}")

    env.close()
    eval_env.close()
    return agent, win_rates


def train_dqn_cartpole(total_timesteps=100_000, seed=42, save_dir="models/baseline"):
    """Train DQN on CartPole."""
    ensure_dir(save_dir)
    set_seed(seed)
    device = get_device()

    env = CartPoleConceptEnv()
    eval_env = CartPoleConceptEnv()
    n_actions = 2

    encoder = SimpleMLPEncoder(env.observation_space, features_dim=128)

    agent = DQNAgent(
        encoder=encoder,
        n_actions=n_actions,
        features_dim=128,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=total_timesteps // 3,
        buffer_size=10_000,
        batch_size=64,
        device=device,
    )

    print(f"Training DQN on CartPole for {total_timesteps} steps...")
    total_steps = 0
    episode = 0
    episode_rewards = []

    while total_steps < total_timesteps:
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done and total_steps < total_timesteps:
            action_mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))
            action = agent.select_action(obs, action_mask)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_mask = next_info.get("action_mask", np.ones(n_actions, dtype=np.int8))

            agent.replay_buffer.push(obs, action, reward, next_obs,
                                     float(done), action_mask, next_mask)
            loss = agent.update()
            episode_reward += reward
            obs = next_obs
            info = next_info
            total_steps += 1

        episode_rewards.append(episode_reward)
        episode += 1

        if episode % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode} | Steps {total_steps} | "
                  f"ε={agent.epsilon:.3f} | Avg R={avg:.1f}")

    agent.save(os.path.join(save_dir, "dqn_cartpole_baseline.pt"))
    torch.save(agent.encoder.state_dict(),
               os.path.join(save_dir, "dqn_cartpole_encoder.pt"))
    print(f"CartPole DQN saved to {save_dir}")

    env.close()
    eval_env.close()
    return agent


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train baseline RL agents")
    parser.add_argument("--env", type=str, default="go",
                        choices=["go", "cartpole", "both"],
                        help="Environment to train on")
    parser.add_argument("--algo", type=str, default="both",
                        choices=["ppo", "dqn", "both"],
                        help="Algorithm to train")
    parser.add_argument("--steps", type=int, default=200_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models/baseline",
                        help="Directory to save models")
    args = parser.parse_args()

    if not SB3_AVAILABLE and args.algo in ("ppo", "both"):
        print("ERROR: SB3 not available. Install with: "
              "pip install stable-baselines3 sb3-contrib")
        sys.exit(1)

    envs = [args.env] if args.env != "both" else ["go", "cartpole"]
    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]

    for env_name in envs:
        for algo in algos:
            print(f"\n{'='*60}")
            print(f"Training {algo.upper()} on {env_name}")
            print(f"{'='*60}")

            if env_name == "go" and algo == "ppo":
                train_ppo_go(args.steps, args.seed, args.save_dir)
            elif env_name == "go" and algo == "dqn":
                train_dqn_go(args.steps, args.seed, args.save_dir)
            elif env_name == "cartpole" and algo == "ppo":
                train_ppo_cartpole(args.steps, args.seed, args.save_dir)
            elif env_name == "cartpole" and algo == "dqn":
                train_dqn_cartpole(args.steps, args.seed, args.save_dir)

    print("\nAll baselines trained successfully!")


if __name__ == "__main__":
    main()
