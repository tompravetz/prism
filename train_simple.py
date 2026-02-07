"""
Train concept bottleneck agents on CartPole (task-agnosticism test).

This script demonstrates that the concept bottleneck architecture is NOT
specific to Go — it can discover meaningful concepts and use them for
decision-making in any RL environment.

The pipeline is identical:
    1. Train a baseline agent with a standard encoder
    2. Freeze encoder, collect features, cluster with K-means
    3. Train a bottleneck policy that sees only concept IDs

The key difference: CartPole uses an MLP encoder (not CNN) because the
observation is a 1D vector (cart position, velocity, pole angle, angular vel)
rather than a 2D spatial grid.

Usage:
    python train_simple.py --algo ppo --steps 100000 --bottleneck-gens 50
    python train_simple.py --algo both
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from src.environments.simple_env import CartPoleConceptEnv
from src.networks import SimpleMLPEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.strategy_memory import StrategyMemory
from src.utils import set_seed, get_device, ensure_dir


def train_cartpole_baseline_ppo(total_timesteps=100_000, seed=42,
                                 save_dir="models/simple"):
    """
    Train a PPO baseline on CartPole using SB3.

    CartPole is a classic control problem: balance a pole on a cart by
    pushing the cart left or right. Maximum score is 500 (episode truncates).
    A well-trained agent should consistently reach 400+ steps.

    We use this as a simple testbed to verify the concept bottleneck
    architecture works beyond Go.
    """
    from stable_baselines3 import PPO

    ensure_dir(save_dir)
    set_seed(seed)

    env = CartPoleConceptEnv()

    # Use our SimpleMLPEncoder for consistency with the concept pipeline
    policy_kwargs = dict(
        features_extractor_class=SimpleMLPEncoder,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,          # Steps per rollout before each update
        batch_size=64,
        n_epochs=10,           # PPO epochs per rollout
        gamma=0.99,
        verbose=1,
        seed=seed,
    )

    print(f"Training PPO baseline on CartPole for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # Save model and encoder
    model.save(os.path.join(save_dir, "ppo_cartpole"))
    torch.save(
        model.policy.features_extractor.state_dict(),
        os.path.join(save_dir, "ppo_cartpole_encoder.pt"),
    )
    print("CartPole PPO baseline saved.")
    env.close()
    return model


def train_cartpole_bottleneck(algo="ppo", n_concepts=32, n_generations=50,
                               steps_per_gen=5000, seed=42,
                               save_dir="models/simple"):
    """
    Train a concept bottleneck agent on CartPole.

    Steps:
        1. Load frozen encoder from baseline training
        2. Discover concepts by clustering encoder features
        3. Train bottleneck policy (concept_id → action)

    CartPole observations are 4-dimensional:
        [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    With 32 concepts, we're asking: can the agent categorize these 4D states
    into 32 discrete situations and learn a good action for each?
    For CartPole, the answer should be yes — the key decision boundary is
    roughly "pole tilting left → push left, pole tilting right → push right".
    """
    ensure_dir(save_dir)
    set_seed(seed)
    device = get_device()

    env = CartPoleConceptEnv()
    n_actions = 2  # CartPole: left or right

    # ---- Load frozen encoder ----
    encoder = SimpleMLPEncoder(env.observation_space, features_dim=128)
    encoder_path = os.path.join(save_dir, "ppo_cartpole_encoder.pt")

    if os.path.exists(encoder_path):
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        print(f"Loaded CartPole encoder from {encoder_path}")
    else:
        print("WARNING: No encoder found. Training with random encoder.")

    encoder.to(device)
    encoder.eval()

    # ---- Discover concepts ----
    concept_path = os.path.join(save_dir, f"concepts_cartpole_k{n_concepts}.pkl")
    if os.path.exists(concept_path):
        cm = ConceptManager(n_concepts=n_concepts)
        cm.load(concept_path)
    else:
        print(f"Discovering {n_concepts} concepts from CartPole states...")
        cm = ConceptManager(n_concepts=n_concepts, features_dim=128)
        # Collect features from random episodes (diverse state coverage)
        cm.collect_features(encoder, env, n_episodes=200, device=device)
        cm.fit()
        cm.save(concept_path)

    # ---- Create bottleneck policy and strategy memory ----
    if algo == "ppo":
        policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts, embed_dim=32, hidden_dim=64, n_actions=n_actions
        ).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    else:
        policy = ConceptDQNPolicy(
            n_concepts=n_concepts, embed_dim=32, hidden_dim=64, n_actions=n_actions
        ).to(device)
        target_policy = ConceptDQNPolicy(
            n_concepts=n_concepts, embed_dim=32, hidden_dim=64, n_actions=n_actions
        ).to(device)
        target_policy.load_state_dict(policy.state_dict())
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        replay_buffer = deque(maxlen=10_000)
        steps_done = 0

    strategy_memory = StrategyMemory(n_concepts=n_concepts, n_actions=n_actions)

    # ---- Training loop ----
    print(f"\nTraining {algo.upper()} bottleneck on CartPole for {n_generations} generations...")
    best_avg_reward = 0.0

    for gen in range(n_generations):
        episode_rewards = []

        if algo == "ppo":
            # ---- PPO: Collect rollout, then update ----
            concepts, actions, rewards_list, values, log_probs = [], [], [], [], []
            dones, action_masks = [], []
            obs, info = env.reset()
            ep_reward = 0.0

            policy.eval()
            for step in range(steps_per_gen):
                # Get concept for current state
                concept_id = cm.assign_concept_from_obs(encoder, obs, device)
                mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))

                # Get action from bottleneck policy
                with torch.no_grad():
                    cid_t = torch.LongTensor([concept_id]).to(device)
                    mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)
                    logits, value = policy(cid_t, mask_t)
                    probs = F.softmax(logits[0], dim=-1)
                    dist = torch.distributions.Categorical(probs + 1e-8)
                    action_t = dist.sample()
                    action = action_t.item()
                    log_prob = dist.log_prob(action_t)

                # Store transition
                concepts.append(concept_id)
                actions.append(action)
                values.append(value[0].item())
                log_probs.append(log_prob.item())
                action_masks.append(mask.copy())

                strategy_memory.record_step(concept_id, action)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                rewards_list.append(reward)
                dones.append(float(done))
                ep_reward += reward

                if done:
                    strategy_memory.end_episode(ep_reward)
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, info = env.reset()
                else:
                    obs = next_obs
                    info = next_info

            # PPO update with GAE
            policy.train()
            concepts_t = torch.LongTensor(concepts).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            old_log_probs_t = torch.FloatTensor(log_probs).to(device)
            masks_t = torch.FloatTensor(np.array(action_masks)).to(device)
            rewards_arr = np.array(rewards_list)
            values_arr = np.array(values)
            dones_arr = np.array(dones)

            # Compute GAE advantages
            with torch.no_grad():
                final_cid = cm.assign_concept_from_obs(encoder, obs, device)
                _, final_val = policy(torch.LongTensor([final_cid]).to(device))
                last_val = final_val[0].item()

            advantages = np.zeros(len(rewards_arr), dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(len(rewards_arr))):
                nv = last_val if t == len(rewards_arr) - 1 else values_arr[t + 1]
                nt = 1.0 - dones_arr[t]
                delta = rewards_arr[t] + 0.99 * nv * nt - values_arr[t]
                advantages[t] = delta + 0.99 * 0.95 * nt * last_gae
                last_gae = advantages[t]

            returns = advantages + values_arr
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            adv_t = torch.FloatTensor(advantages).to(device)
            ret_t = torch.FloatTensor(returns).to(device)

            # PPO update epochs
            for epoch in range(4):
                idx = np.random.permutation(len(concepts_t))
                for start in range(0, len(idx), 64):
                    batch = idx[start:start + 64]
                    logits, vals = policy(concepts_t[batch], masks_t[batch])
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs + 1e-8)
                    new_lp = dist.log_prob(actions_t[batch])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - old_log_probs_t[batch])
                    s1 = ratio * adv_t[batch]
                    s2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[batch]
                    policy_loss = -torch.min(s1, s2).mean()
                    value_loss = F.mse_loss(vals.squeeze(-1), ret_t[batch])
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

        else:
            # ---- DQN: Interleaved collection and updates ----
            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(steps_per_gen):
                steps_done += 1
                concept = cm.assign_concept_from_obs(encoder, obs, device)
                mask = info.get("action_mask", np.ones(n_actions, dtype=np.int8))

                # Epsilon-greedy
                eps = max(0.01, 1.0 - steps_done / 20_000)
                if np.random.random() < eps:
                    action = np.random.choice(np.where(mask == 1)[0])
                else:
                    action = policy.get_action(concept, mask, epsilon=0.0)

                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                next_concept = cm.assign_concept_from_obs(encoder, next_obs, device) if not done else 0
                next_mask = next_info.get("action_mask", np.ones(n_actions, dtype=np.int8))

                replay_buffer.append((concept, action, reward, next_concept,
                                     float(done), mask.copy(), next_mask.copy()))
                strategy_memory.record_step(concept, action)
                ep_reward += reward

                # DQN update
                if len(replay_buffer) >= 64:
                    batch_idx = np.random.choice(len(replay_buffer), 64, replace=False)
                    batch = [replay_buffer[i] for i in batch_idx]
                    c_b = torch.LongTensor([b[0] for b in batch]).to(device)
                    a_b = torch.LongTensor([b[1] for b in batch]).to(device)
                    r_b = torch.FloatTensor([b[2] for b in batch]).to(device)
                    nc_b = torch.LongTensor([b[3] for b in batch]).to(device)
                    d_b = torch.FloatTensor([b[4] for b in batch]).to(device)
                    nm_b = torch.FloatTensor(np.array([b[6] for b in batch])).to(device)

                    q = policy(c_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        nq = target_policy(nc_b, nm_b).max(1)[0]
                        nq = torch.where(torch.isinf(nq), torch.zeros_like(nq), nq)
                        target = r_b + 0.99 * nq * (1.0 - d_b)

                    loss = F.smooth_l1_loss(q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Soft update target
                    for tp, op in zip(target_policy.parameters(), policy.parameters()):
                        tp.data.copy_(0.005 * op.data + 0.995 * tp.data)

                if done:
                    strategy_memory.end_episode(ep_reward)
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, info = env.reset()
                else:
                    obs = next_obs
                    info = next_info

        # ---- Log progress ----
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        best_avg_reward = max(best_avg_reward, avg_reward)

        if gen % 5 == 0:
            print(f"  Gen {gen:3d}/{n_generations} | "
                  f"Avg R={avg_reward:.1f} (best={best_avg_reward:.1f}) | "
                  f"Eps={len(episode_rewards)}")

    # ---- Save final model ----
    torch.save(policy.state_dict(),
               os.path.join(save_dir, f"{algo}_cartpole_bottleneck.pt"))
    strategy_memory.save(os.path.join(save_dir, f"strategy_memory_cartpole_{algo}.pkl"))
    strategy_memory.export_to_json(
        os.path.join(save_dir, f"strategies_cartpole_{algo}.json")
    )

    print(f"\nCartPole {algo.upper()} bottleneck training complete!")
    print(f"Best avg reward: {best_avg_reward:.1f}")

    env.close()
    return policy, strategy_memory


def main():
    parser = argparse.ArgumentParser(
        description="Train concept bottleneck on CartPole"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Baseline training steps")
    parser.add_argument("--bottleneck-gens", type=int, default=50,
                        help="Bottleneck training generations")
    parser.add_argument("--n-concepts", type=int, default=32,
                        help="Number of concepts for CartPole")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/simple")
    args = parser.parse_args()

    # Step 1: Train baseline
    print("="*60)
    print("Step 1: Training CartPole Baseline")
    print("="*60)
    train_cartpole_baseline_ppo(args.steps, args.seed, args.save_dir)

    # Step 2: Train bottleneck
    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]
    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Step 2: Training CartPole {algo.upper()} Bottleneck")
        print(f"{'='*60}")
        train_cartpole_bottleneck(
            algo=algo, n_concepts=args.n_concepts,
            n_generations=args.bottleneck_gens,
            seed=args.seed, save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
