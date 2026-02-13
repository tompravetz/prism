"""
Train Concept Bottleneck Agents (PPO + DQN) on Go 7x7.

Stage 3 of the concept bottleneck pipeline:
    Board → Frozen Encoder → Features → Concept ID → Bottleneck Policy → Action

The bottleneck policy ONLY sees a concept ID (single integer), not the full
128-dimensional feature vector. This forces the agent to learn strategies that
are grounded in discrete, interpretable concepts.

This script assumes you've already:
    1. Trained baseline agents (train_baseline.py) — provides the frozen encoder
    2. Concept discovery has been run (or will be done here automatically)

Four variants are trained for the comparison study:
    - PPO-bottleneck: MaskablePPO loss, concept-only input
    - DQN-bottleneck: DQN loss (Huber), concept-only input
    - PPO-full: MaskablePPO with full features (control)
    - DQN-full: DQN with full features (control)

Usage:
    python train_bottleneck.py --algo ppo --generations 100 --steps-per-gen 20000
    python train_bottleneck.py --algo dqn --generations 100 --steps-per-gen 20000
    python train_bottleneck.py --algo both --generations 50
"""

import argparse
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from src.environments.go_env import GoEnv, MaskedGoEnv
from src.networks import GoCNNEncoder, QNetwork
from src.concept_manager import ConceptManager
from src.concept_policy import (ConceptBottleneckPolicy, ConceptDQNPolicy,
                                 ConceptBottleneckAgent)
from src.utils import set_seed, get_device, ensure_dir


# ============================================================
# PPO Bottleneck Training
# ============================================================

class PPOBottleneckTrainer:
    """
    Trains a concept bottleneck policy using PPO (Proximal Policy Optimization).

    The key difference from standard PPO: instead of passing raw observations
    or feature vectors to the policy, we pass only the concept ID — a single
    integer in [0, n_concepts). The policy must learn to map these discrete
    concepts to good action distributions.

    Training loop per generation:
        1. Collect trajectories using current policy
        2. Compute advantages using GAE (Generalized Advantage Estimation)
        3. Perform multiple epochs of mini-batch PPO updates
        4. Log metrics and optionally save checkpoint
    """

    def __init__(self, encoder, concept_manager, n_actions=50,
                 n_concepts=64, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                 n_epochs=4, batch_size=64, device=None):
        """
        Args:
            encoder: Frozen CNN encoder from baseline training.
            concept_manager: Fitted ConceptManager for obs→concept mapping.
            n_actions: Number of possible actions (50 for Go 7x7).
            n_concepts: Number of concept clusters (64 default).
            lr: Learning rate for the bottleneck policy.
            gamma: Discount factor for future rewards.
            gae_lambda: GAE lambda for advantage estimation.
            clip_range: PPO clipping range (epsilon in the paper).
            ent_coef: Entropy bonus coefficient — encourages exploration.
            vf_coef: Value function loss coefficient.
            n_epochs: Number of PPO update epochs per batch of data.
            batch_size: Mini-batch size for PPO updates.
            device: Torch device (CPU or CUDA).
        """
        self.device = device or get_device()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Frozen encoder — we only use it to extract features for concept assignment
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Concept manager maps encoder features → concept IDs
        self.concept_manager = concept_manager

        # The bottleneck policy: concept_id → action distribution + value
        self.policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts,
            embed_dim=64,
            hidden_dim=128,
            n_actions=n_actions,
        ).to(self.device)

        # Optimizer only updates the bottleneck policy (encoder is frozen)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_concept(self, obs):
        """
        Convert a raw observation to a concept ID.

        Pipeline: obs → encoder → 128d features → K-means → concept_id
        """
        return self.concept_manager.assign_concept_from_obs(
            self.encoder, obs, self.device
        )

    def collect_rollout(self, env, n_steps=1024):
        """
        Collect a rollout (trajectory) of experience.

        For each step:
            1. Get current observation and convert to concept
            2. Sample action from policy given concept
            3. Step environment and record transition

        Args:
            env: Go environment.
            n_steps: Number of steps to collect.

        Returns:
            Dictionary containing all trajectory data needed for PPO update.
        """
        # Storage for trajectory data
        concepts = []       # Concept IDs for each step
        actions = []        # Actions taken
        rewards = []        # Rewards received
        values = []         # Value estimates from the policy
        log_probs = []      # Log probabilities of the actions taken
        dones = []          # Episode termination flags
        action_masks = []   # Legal action masks

        obs, info = env.reset()
        episode_reward = 0.0
        episode_rewards = []

        self.policy.eval()

        for step in range(n_steps):
            # Step 1: Convert observation to concept ID
            concept_id = self.get_concept(obs)

            # Step 2: Get action distribution from bottleneck policy
            mask = info.get("action_mask", None)
            with torch.no_grad():
                cid_t = torch.LongTensor([concept_id]).to(self.device)
                mask_t = None
                if mask is not None:
                    mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                logits, value = self.policy(cid_t, mask_t)

                # Sample action from the probability distribution
                probs = F.softmax(logits[0], dim=-1)
                # Handle case where all probs are 0 (shouldn't happen with pass action)
                if probs.sum() < 1e-8:
                    legal = np.where(mask == 1)[0] if mask is not None else np.arange(50)
                    action = int(np.random.choice(legal))
                    log_prob = torch.tensor(0.0)
                else:
                    dist = torch.distributions.Categorical(probs)
                    action_t = dist.sample()
                    action = action_t.item()
                    log_prob = dist.log_prob(action_t)

            # Store transition data
            concepts.append(concept_id)
            actions.append(action)
            values.append(value[0].item())
            log_probs.append(log_prob.item())
            if mask is not None:
                action_masks.append(mask.copy())
            else:
                action_masks.append(np.ones(50, dtype=np.int8))

            # Step 3: Take action in environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            dones.append(float(done))

            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs
                info = next_info

        # Compute the value of the final state (for bootstrapping)
        with torch.no_grad():
            final_concept = self.get_concept(obs)
            final_cid_t = torch.LongTensor([final_concept]).to(self.device)
            _, final_value = self.policy(final_cid_t)
            last_value = final_value[0].item()

        return {
            "concepts": np.array(concepts),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
            "action_masks": np.array(action_masks),
            "last_value": last_value,
            "episode_rewards": episode_rewards,
        }

    def compute_advantages(self, rollout):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE balances bias and variance in advantage estimation:
            A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Higher lambda → lower bias but higher variance (closer to Monte Carlo)
        Lower lambda → higher bias but lower variance (closer to TD(0))
        """
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        last_value = rollout["last_value"]
        n = len(rewards)

        # Initialize advantage and return arrays
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        # Compute GAE in reverse order (from last step to first)
        for t in reversed(range(n)):
            # Next step's value: either bootstrap from value function or 0 if terminal
            if t == n - 1:
                next_value = last_value
                next_nonterminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_nonterminal = 1.0 - dones[t]

            # TD error: how much better was the actual outcome than predicted?
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]

            # GAE: exponentially weighted sum of TD errors
            advantages[t] = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            last_gae = advantages[t]

        # Returns = advantages + values (used for value function training)
        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """
        Perform PPO update using collected rollout data.

        PPO's key idea: limit how much the policy can change per update by
        clipping the probability ratio. This prevents catastrophically large
        policy updates that can destabilize training.

        The total loss has three components:
            1. Policy loss (clipped surrogate objective)
            2. Value function loss (MSE between predicted and actual returns)
            3. Entropy bonus (encourages exploration)
        """
        self.policy.train()

        # Compute advantages using GAE
        advantages, returns = self.compute_advantages(rollout)

        # Normalize advantages (standard practice for stable training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert everything to tensors
        concepts_t = torch.LongTensor(rollout["concepts"]).to(self.device)
        actions_t = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs_t = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        action_masks_t = torch.FloatTensor(rollout["action_masks"]).to(self.device)

        n = len(concepts_t)
        total_loss_epoch = 0.0

        # Multiple epochs of updates on the same data (PPO can reuse data)
        for epoch in range(self.n_epochs):
            # Shuffle data into mini-batches
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                # Get current policy outputs for the batch
                logits, values = self.policy(concepts_t[batch_idx],
                                             action_masks_t[batch_idx])

                # Compute new log probabilities and entropy
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs + 1e-8)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                # ---- Policy Loss (Clipped Surrogate) ----
                # ratio = new_prob / old_prob = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                batch_advantages = advantages_t[batch_idx]

                # Unclipped objective
                surr1 = ratio * batch_advantages
                # Clipped objective: limit the ratio to [1-eps, 1+eps]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * batch_advantages
                # Take the minimum (pessimistic bound)
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value Function Loss ----
                # Simple MSE between predicted value and actual return
                value_loss = F.mse_loss(values.squeeze(-1), returns_t[batch_idx])

                # ---- Total Loss ----
                # Combine: policy loss + value loss - entropy bonus
                # (entropy is subtracted because we want to MAXIMIZE it)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss_epoch += loss.item()

        return total_loss_epoch / (self.n_epochs * max(1, n // self.batch_size))


# ============================================================
# DQN Bottleneck Training
# ============================================================

class DQNBottleneckTrainer:
    """
    Trains a concept bottleneck policy using DQN (Deep Q-Network).

    Similar to PPOBottleneckTrainer but uses:
        - Q-value learning instead of policy gradients
        - Replay buffer instead of on-policy rollouts
        - Target network with soft updates for stability
        - Epsilon-greedy exploration instead of entropy bonus
    """

    def __init__(self, encoder, concept_manager, n_actions=50,
                 n_concepts=64, lr=1e-4, gamma=0.99, tau=0.005,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50_000,
                 buffer_size=50_000, batch_size=64, device=None):
        self.device = device or get_device()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Frozen encoder for concept extraction
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.concept_manager = concept_manager

        # Online Q-network (the one we train)
        self.q_net = ConceptDQNPolicy(
            n_concepts=n_concepts,
            embed_dim=64,
            hidden_dim=128,
            n_actions=n_actions,
        ).to(self.device)

        # Target Q-network (slowly updated copy for stable targets)
        # The target network provides the "ground truth" Q-values during
        # training. By updating it slowly, we prevent the moving-target problem.
        self.target_q_net = ConceptDQNPolicy(
            n_concepts=n_concepts,
            embed_dim=64,
            hidden_dim=128,
            n_actions=n_actions,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Epsilon-greedy exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Experience replay buffer
        # Stores (concept_id, action, reward, next_concept_id, done, action_mask, next_mask)
        self.replay_buffer = deque(maxlen=buffer_size)

    @property
    def epsilon(self):
        """Current exploration rate — decays linearly from start to end."""
        progress = min(1.0, self.steps_done / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def get_concept(self, obs):
        """Convert observation to concept ID via frozen encoder + K-means."""
        return self.concept_manager.assign_concept_from_obs(
            self.encoder, obs, self.device
        )

    def select_action(self, concept_id, action_mask=None):
        """
        Select action using epsilon-greedy strategy.

        With probability epsilon: pick a random legal action (explore)
        With probability (1-epsilon): pick the highest Q-value action (exploit)
        """
        self.steps_done += 1

        if np.random.random() < self.epsilon:
            # Exploration: random legal action
            if action_mask is not None:
                legal = np.where(action_mask == 1)[0]
                return int(np.random.choice(legal)) if len(legal) > 0 else 0
            return np.random.randint(0, self.n_actions)

        # Exploitation: greedy action from Q-network
        return self.q_net.get_action(concept_id, action_mask, epsilon=0.0)

    def store_transition(self, concept, action, reward, next_concept, done,
                         action_mask, next_action_mask):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append(
            (concept, action, reward, next_concept, float(done),
             action_mask.copy(), next_action_mask.copy())
        )

    def update(self):
        """
        Perform one DQN update step.

        DQN update:
            1. Sample a mini-batch from replay buffer
            2. Compute current Q-values: Q(s, a) for the actions we took
            3. Compute target Q-values: r + gamma * max_a' Q_target(s', a')
            4. Minimize Huber loss between current and target
            5. Soft-update the target network

        The Huber loss (smooth L1) is less sensitive to outliers than MSE,
        which is important because Q-value targets can be noisy.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.q_net.train()

        # Sample random mini-batch from replay buffer
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        # Unpack batch into separate arrays
        concepts = torch.LongTensor([b[0] for b in batch]).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_concepts = torch.LongTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        next_masks = torch.FloatTensor(
            np.array([b[6] for b in batch])
        ).to(self.device)

        # Current Q-values: Q(concept, action) for the actions we actually took
        current_q = self.q_net(concepts)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(next_concept, a')
        with torch.no_grad():
            next_q = self.target_q_net(next_concepts, next_masks)
            next_q_max = next_q.max(1)[0]
            # Handle case where all actions are masked (terminal state)
            next_q_max = torch.where(
                torch.isinf(next_q_max),
                torch.zeros_like(next_q_max),
                next_q_max,
            )
            # Bellman equation: target = reward + discount * future_value
            target_q = rewards + self.gamma * next_q_max * (1.0 - dones)

        # Huber loss (smooth L1) — more robust than MSE
        loss = F.smooth_l1_loss(current_q, target_q)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network: target = tau * online + (1-tau) * target
        # This slowly moves the target towards the online network
        for target_p, online_p in zip(self.target_q_net.parameters(),
                                      self.q_net.parameters()):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data
            )

        return loss.item()


# ============================================================
# Main Training Loop (Generational)
# ============================================================

def discover_concepts(encoder, env, n_concepts=64, n_episodes=500,
                      save_path=None, device=None):
    """
    Stage 2: Discover concepts by clustering encoder features.

    Runs many episodes with the frozen encoder, collects feature vectors
    for every game state encountered, then clusters them with K-means.
    Each cluster becomes a "concept" — a discrete category that groups
    similar game states together.

    Args:
        encoder: Trained encoder network (will be frozen).
        env: Game environment.
        n_concepts: Number of concept clusters to discover.
        n_episodes: Number of episodes to collect features from.
        save_path: Optional path to save the fitted concept manager.
        device: Torch device.

    Returns:
        Fitted ConceptManager.
    """
    device = device or get_device()
    cm = ConceptManager(n_concepts=n_concepts, features_dim=128)
    cm.collect_features(encoder, env, n_episodes=n_episodes, device=device)
    cm.fit()

    if save_path:
        cm.save(save_path)

    return cm


def evaluate_agent(agent_fn, env, n_episodes=100):
    """
    Evaluate an agent's performance.

    Args:
        agent_fn: Callable(obs, action_mask) -> action
        env: Game environment.
        n_episodes: Number of evaluation games.

    Returns:
        Dictionary with win_rate, mean_reward, episode_lengths.
    """
    wins = 0
    total_reward = 0.0
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0.0

        while not done:
            mask = info.get("action_mask", None)
            action = agent_fn(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        total_reward += ep_reward
        episode_lengths.append(ep_len)
        if ep_reward > 0:
            wins += 1

    return {
        "win_rate": wins / n_episodes,
        "mean_reward": total_reward / n_episodes,
        "mean_length": np.mean(episode_lengths),
    }


def train_bottleneck(algo="ppo", n_generations=100, steps_per_gen=20_000,
                     n_concepts=64, seed=42, save_dir="models/bottleneck",
                     baseline_dir="models/baseline"):
    """
    Main generational training loop for concept bottleneck agents.

    Each generation:
        1. Collect experience (rollout for PPO, episodes for DQN)
        2. Update the bottleneck policy
        3. Evaluate against random opponent
        4. Log metrics

    Args:
        algo: "ppo" or "dqn"
        n_generations: Number of training generations.
        steps_per_gen: Training steps per generation.
        n_concepts: Number of concept clusters.
        seed: Random seed.
        save_dir: Where to save bottleneck models.
        baseline_dir: Where to load baseline encoders from.
    """
    ensure_dir(save_dir)
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- Load frozen encoder from baseline training ----
    env = GoEnv(board_size=7)
    encoder = GoCNNEncoder(env.observation_space, features_dim=128)

    encoder_path = os.path.join(baseline_dir, f"{algo}_go_encoder.pt")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        print(f"Loaded encoder from {encoder_path}")
    else:
        print(f"WARNING: No encoder found at {encoder_path}. Using random encoder.")

    encoder.to(device)
    encoder.eval()

    # ---- Stage 2: Discover concepts ----
    concept_path = os.path.join(save_dir, f"concepts_{algo}_k{n_concepts}.pkl")
    if os.path.exists(concept_path):
        # Load previously discovered concepts
        cm = ConceptManager(n_concepts=n_concepts)
        cm.load(concept_path)
    else:
        # Discover concepts from scratch
        print(f"\nDiscovering {n_concepts} concepts...")
        cm = discover_concepts(encoder, env, n_concepts=n_concepts,
                               n_episodes=500, save_path=concept_path,
                               device=device)

    # ---- Create trainer ----
    n_actions = env.action_count  # 50 for Go 7x7

    if algo == "ppo":
        trainer = PPOBottleneckTrainer(
            encoder=encoder, concept_manager=cm,
            n_actions=n_actions, n_concepts=n_concepts,
            lr=3e-4, gamma=0.99, device=device,
        )
    elif algo == "dqn":
        trainer = DQNBottleneckTrainer(
            encoder=encoder, concept_manager=cm,
            n_actions=n_actions, n_concepts=n_concepts,
            lr=1e-4, gamma=0.99, device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    eval_env = GoEnv(board_size=7)

    # ---- Generational training loop ----
    print(f"\nTraining {algo.upper()} bottleneck for {n_generations} generations...")
    best_win_rate = 0.0
    metrics_history = []

    for gen in range(n_generations):
        gen_start = time.time()

        if algo == "ppo":
            # PPO: collect a rollout, then update
            rollout = trainer.collect_rollout(env, n_steps=steps_per_gen)
            loss = trainer.update(rollout)
            episode_rewards = rollout["episode_rewards"]

        elif algo == "dqn":
            # DQN: interleave data collection and updates
            episode_rewards = []
            obs, info = env.reset()
            ep_reward = 0.0
            steps = 0

            while steps < steps_per_gen:
                concept = trainer.get_concept(obs)
                mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))
                action = trainer.select_action(concept, mask)

                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                next_concept = trainer.get_concept(next_obs) if not done else 0
                next_mask = next_info.get(
                    "action_mask",
                    np.zeros(n_actions, dtype=np.int8) if done
                    else np.ones(n_actions, dtype=np.int8)
                )

                trainer.store_transition(concept, action, reward,
                                         next_concept, done, mask, next_mask)

                loss = trainer.update()
                ep_reward += reward
                steps += 1

                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, info = env.reset()
                else:
                    obs = next_obs
                    info = next_info

        # ---- Evaluate (always against random for consistent measurement) ----
        eval_env.opponent_fn = eval_env._random_opponent

        if algo == "ppo":
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.policy.get_action(c, mask, deterministic=True)
        else:
            def agent_fn(obs, mask):
                c = trainer.get_concept(obs)
                return trainer.q_net.get_action(c, mask, epsilon=0.0)

        eval_results = evaluate_agent(agent_fn, eval_env, n_episodes=50)
        win_rate = eval_results["win_rate"]
        best_win_rate = max(best_win_rate, win_rate)

        gen_time = time.time() - gen_start
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

        metrics = {
            "generation": gen,
            "win_rate": win_rate,
            "best_win_rate": best_win_rate,
            "avg_reward": avg_reward,
            "n_episodes": len(episode_rewards),
            "time": gen_time,
        }
        metrics_history.append(metrics)

        # Print progress every 5 generations
        if gen % 5 == 0 or gen == n_generations - 1:
            eps_str = f"eps={trainer.epsilon:.3f}" if algo == "dqn" else ""
            print(f"  Gen {gen:3d}/{n_generations} | "
                  f"Win={win_rate:.2%} (best={best_win_rate:.2%}) | "
                  f"Avg R={avg_reward:.3f} | "
                  f"Eps={len(episode_rewards)} | "
                  f"{eps_str} "
                  f"Time={gen_time:.1f}s")

        # Save checkpoint periodically
        if gen % 20 == 0 or gen == n_generations - 1:
            policy_state = (trainer.policy.state_dict() if algo == "ppo"
                           else trainer.q_net.state_dict())
            checkpoint_path = os.path.join(save_dir, f"{algo}_bottleneck_gen{gen:04d}.pt")
            torch.save(policy_state, checkpoint_path)


    # ---- Save final model and strategy memory ----
    final_state = (trainer.policy.state_dict() if algo == "ppo"
                  else trainer.q_net.state_dict())
    torch.save(final_state, os.path.join(save_dir, f"{algo}_bottleneck_final.pt"))

    # Save metrics history
    import json
    with open(os.path.join(save_dir, f"metrics_{algo}.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n{algo.upper()} bottleneck training complete!")
    print(f"Best win rate: {best_win_rate:.2%}")

    env.close()
    eval_env.close()
    return trainer, metrics_history


def main():
    parser = argparse.ArgumentParser(
        description="Train concept bottleneck agents"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"],
                        help="Algorithm for bottleneck policy")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of training generations")
    parser.add_argument("--steps-per-gen", type=int, default=20_000,
                        help="Training steps per generation")
    parser.add_argument("--n-concepts", type=int, default=64,
                        help="Number of concept clusters")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models/bottleneck",
                        help="Directory to save bottleneck models")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline",
                        help="Directory with trained baseline models")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Training {algo.upper()} Concept Bottleneck")
        print(f"{'='*60}")

        train_bottleneck(
            algo=algo,
            n_generations=args.generations,
            steps_per_gen=args.steps_per_gen,
            n_concepts=args.n_concepts,
            seed=args.seed,
            save_dir=args.save_dir,
            baseline_dir=args.baseline_dir,
        )


if __name__ == "__main__":
    main()
