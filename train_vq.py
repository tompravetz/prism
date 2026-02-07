"""
Phase 2A: Train end-to-end VQ concept bottleneck agent.

Instead of the two-stage approach (train encoder → K-means → train policy),
this trains the entire pipeline end-to-end with a differentiable VQ layer:

    Board (7x7x3) → CNN Encoder → Features → VQ Layer → Concept Embedding → Policy → Action

The VQ layer uses the Straight-Through Estimator for gradient flow.
The total loss combines:
    - RL loss (PPO surrogate objective)
    - VQ commitment loss (keeps encoder outputs near codebook entries)
    - VQ codebook loss (moves codebook entries toward encoder outputs)

Comparison with K-means approach:
    - K-means: cleaner, easier to debug, but concepts are post-hoc
    - VQ: end-to-end learning, concepts optimized for task performance
    - We compare both to see which produces better/more meaningful concepts

Usage:
    python train_vq.py --generations 100 --steps-per-gen 20000
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.vq_layer import VectorQuantizer, VQConceptEncoder
from src.concept_policy import ConceptBottleneckPolicy
from src.strategy_memory import StrategyMemory
from src.utils import set_seed, get_device, ensure_dir


class VQPPOAgent(nn.Module):
    """
    End-to-end VQ concept bottleneck agent trained with PPO.

    Architecture:
        obs → CNN encoder → continuous features → VQ → quantized concept → policy → action

    The VQ layer creates a discrete bottleneck: the policy only sees quantized
    concept embeddings, not the original features. But unlike K-means, the
    entire pipeline is trained end-to-end with gradient flow through the
    Straight-Through Estimator.
    """

    def __init__(self, observation_space, n_concepts=64, features_dim=128,
                 n_actions=50, commitment_cost=0.25):
        super().__init__()

        # CNN encoder: obs → continuous features
        self.encoder = GoCNNEncoder(observation_space, features_dim=features_dim)

        # VQ layer: continuous features → discrete concept embedding
        self.vq = VectorQuantizer(
            n_concepts=n_concepts,
            embedding_dim=features_dim,
            commitment_cost=commitment_cost,
            use_ema=True,
        )

        # Policy head: concept embedding → action logits + value
        # Note: this takes the quantized embedding (features_dim), not just an ID
        self.policy_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.n_actions = n_actions
        self.n_concepts = n_concepts

    def forward(self, obs, action_mask=None):
        """
        Full forward pass.

        Returns:
            action_logits: (batch, n_actions) — policy output.
            values: (batch, 1) — state value estimate.
            vq_loss: Scalar — VQ auxiliary loss.
            concept_ids: (batch,) — discrete concept IDs (for logging).
            perplexity: Scalar — codebook utilization (for monitoring).
        """
        # Encode observation
        features = self.encoder(obs)

        # Quantize through VQ layer
        quantized, vq_loss, concept_ids, perplexity = self.vq(features)

        # Policy and value from quantized concepts (NOT from raw features)
        action_logits = self.policy_head(quantized)
        values = self.value_head(quantized)

        # Apply action mask
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

        return action_logits, values, vq_loss, concept_ids, perplexity


def train_vq_agent(n_generations=100, steps_per_gen=20_000,
                   n_concepts=64, features_dim=128,
                   lr=3e-4, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                   commitment_cost=0.25,
                   seed=42, save_dir="models/vq"):
    """
    Train VQ concept bottleneck agent with PPO.

    The training loop is similar to standard PPO, but the loss function
    includes an additional VQ term:
        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + vq_loss

    We monitor perplexity (codebook utilization) throughout training.
    If perplexity drops too low → codebook collapse → need to reset dead entries.
    """
    ensure_dir(save_dir)
    set_seed(seed)
    device = get_device()

    # Create environment and agent
    env = GoEnv(board_size=7)
    eval_env = GoEnv(board_size=7)
    n_actions = 50

    agent = VQPPOAgent(
        observation_space=env.observation_space,
        n_concepts=n_concepts,
        features_dim=features_dim,
        n_actions=n_actions,
        commitment_cost=commitment_cost,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    strategy_memory = StrategyMemory(n_concepts=n_concepts, n_actions=n_actions)

    print(f"Training VQ PPO agent for {n_generations} generations on {device}...")
    print(f"  Concepts: {n_concepts}, Features: {features_dim}")
    print(f"  Commitment cost: {commitment_cost}")

    best_win_rate = 0.0
    metrics_history = []

    for gen in range(n_generations):
        gen_start = time.time()

        # ---- Collect rollout ----
        agent.eval()
        obs_list, action_list, reward_list = [], [], []
        value_list, log_prob_list, done_list = [], [], []
        mask_list, concept_list = [], []
        vq_losses = []

        obs, info = env.reset()
        ep_reward = 0.0
        episode_rewards = []

        for step in range(steps_per_gen):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value, vq_loss, concept_ids, perplexity = agent(obs_t, mask_t)

                # Sample action from policy distribution
                probs = F.softmax(logits[0], dim=-1)
                if probs.sum() < 1e-8:
                    legal = np.where(mask == 1)[0]
                    action = int(np.random.choice(legal)) if len(legal) > 0 else 49
                    log_prob = torch.tensor(0.0)
                else:
                    dist = torch.distributions.Categorical(probs)
                    action_t = dist.sample()
                    action = action_t.item()
                    log_prob = dist.log_prob(action_t)

            concept_id = concept_ids[0].item()

            # Store
            obs_list.append(obs.copy())
            action_list.append(action)
            value_list.append(value[0].item())
            log_prob_list.append(log_prob.item())
            mask_list.append(mask.copy())
            concept_list.append(concept_id)
            vq_losses.append(vq_loss.item())

            strategy_memory.record_step(concept_id, action)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            reward_list.append(reward)
            done_list.append(float(done))
            ep_reward += reward

            if done:
                strategy_memory.end_episode(ep_reward)
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs
                info = next_info

        # ---- Compute GAE advantages ----
        with torch.no_grad():
            final_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, final_val, _, _, _ = agent(final_obs_t)
            last_value = final_val[0].item()

        rewards = np.array(reward_list)
        values = np.array(value_list)
        dones = np.array(done_list)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = last_value if t == len(rewards) - 1 else values[t + 1]
            nt = 1.0 - dones[t]
            delta = rewards[t] + gamma * nv * nt - values[t]
            advantages[t] = delta + gamma * gae_lambda * nt * last_gae
            last_gae = advantages[t]

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO + VQ update ----
        agent.train()
        obs_t = torch.FloatTensor(np.array(obs_list)).to(device)
        act_t = torch.LongTensor(action_list).to(device)
        old_lp_t = torch.FloatTensor(log_prob_list).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)
        mask_t = torch.FloatTensor(np.array(mask_list)).to(device)

        n = len(obs_t)
        total_loss_epoch = 0.0
        total_perplexity = 0.0

        for epoch in range(4):
            perm = np.random.permutation(n)
            for start in range(0, n, 64):
                batch = perm[start:start + 64]

                logits, vals, vq_loss, _, perp = agent(obs_t[batch], mask_t[batch])

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs + 1e-8)
                new_lp = dist.log_prob(act_t[batch])
                entropy = dist.entropy().mean()

                # PPO policy loss
                ratio = torch.exp(new_lp - old_lp_t[batch])
                s1 = ratio * adv_t[batch]
                s2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t[batch]
                policy_loss = -torch.min(s1, s2).mean()

                # Value loss
                value_loss = F.mse_loss(vals.squeeze(-1), ret_t[batch])

                # Total loss includes VQ loss
                # The VQ loss ensures the discrete bottleneck remains healthy
                loss = (policy_loss
                        + vf_coef * value_loss
                        - ent_coef * entropy
                        + vq_loss)  # VQ term

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                total_loss_epoch += loss.item()
                total_perplexity += perp.item()

        # ---- Periodically reset dead codebook entries ----
        if gen % 20 == 0 and gen > 0:
            with torch.no_grad():
                sample_obs = torch.FloatTensor(np.array(obs_list[:256])).to(device)
                sample_features = agent.encoder(sample_obs)
                n_reset = agent.vq.reset_dead_entries(sample_features)
                if n_reset > 0:
                    print(f"  Reset {n_reset} dead codebook entries")

        # ---- Evaluate ----
        agent.eval()
        wins = 0
        n_eval = 50
        for _ in range(n_eval):
            e_obs, e_info = eval_env.reset()
            e_done = False
            while not e_done:
                e_obs_t = torch.FloatTensor(e_obs).unsqueeze(0).to(device)
                e_mask = e_info.get("action_mask", np.ones(n_actions, dtype=np.int8))
                e_mask_t = torch.FloatTensor(e_mask).unsqueeze(0).to(device)
                with torch.no_grad():
                    e_logits, _, _, _, _ = agent(e_obs_t, e_mask_t)
                    e_action = e_logits[0].argmax().item()
                e_obs, e_reward, e_term, e_trunc, e_info = eval_env.step(e_action)
                e_done = e_term or e_trunc
                if e_term and e_reward > 0:
                    wins += 1

        win_rate = wins / n_eval
        best_win_rate = max(best_win_rate, win_rate)

        # Codebook health
        cb_stats = agent.vq.get_codebook_utilization()
        avg_vq_loss = np.mean(vq_losses)

        gen_time = time.time() - gen_start
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

        metrics = {
            "generation": gen,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "vq_loss": avg_vq_loss,
            "perplexity": total_perplexity / max(1, 4 * (n // 64)),
            "active_concepts": cb_stats["active_entries"],
            "dead_concepts": cb_stats["dead_entries"],
            "time": gen_time,
        }
        metrics_history.append(metrics)

        if gen % 5 == 0 or gen == n_generations - 1:
            print(f"  Gen {gen:3d}/{n_generations} | "
                  f"Win={win_rate:.2%} (best={best_win_rate:.2%}) | "
                  f"R={avg_reward:.3f} | "
                  f"VQ={avg_vq_loss:.4f} | "
                  f"Active={cb_stats['active_entries']}/{n_concepts} | "
                  f"Time={gen_time:.1f}s")

        # Checkpoint
        if gen % 20 == 0 or gen == n_generations - 1:
            torch.save(agent.state_dict(),
                       os.path.join(save_dir, f"vq_agent_gen{gen:04d}.pt"))

    # ---- Save final results ----
    torch.save(agent.state_dict(), os.path.join(save_dir, "vq_agent_final.pt"))
    strategy_memory.save(os.path.join(save_dir, "strategy_memory_vq.pkl"))

    import json
    with open(os.path.join(save_dir, "metrics_vq.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nVQ training complete! Best win rate: {best_win_rate:.2%}")

    env.close()
    eval_env.close()
    return agent, metrics_history


def main():
    parser = argparse.ArgumentParser(description="Train VQ concept bottleneck agent")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--steps-per-gen", type=int, default=20_000)
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/vq")
    args = parser.parse_args()

    train_vq_agent(
        n_generations=args.generations,
        steps_per_gen=args.steps_per_gen,
        n_concepts=args.n_concepts,
        commitment_cost=args.commitment_cost,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
