"""
Task 2: Cross-Domain Concept Transfer.

Tests whether concepts learned in one domain can transfer to a different domain:
    - CartPole (K=32) -> Acrobot (K=32): both classic control, MLP encoder
    - LunarLander (K=32) -> Acrobot (K=32): both MLP, different dynamics

This requires NEW training:
    1. Train Acrobot baseline (SimpleMLPEncoder, ~100K steps)
    2. Discover Acrobot concepts (K=32)
    3. Train Acrobot bottleneck from scratch (control)
    4. Align source concepts to Acrobot concepts
    5. Transfer source policy -> Acrobot (remap embeddings, handle action space)
    6. Fine-tune transferred policy on Acrobot
    7. Compare learning curves: transferred vs from-scratch

Usage:
    python experiments/transfer_cross_domain.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.simple_env import CartPoleConceptEnv, LunarLanderConceptEnv
from src.environments.acrobot_env import AcrobotConceptEnv
from src.networks import SimpleMLPEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir


# ============================================================
# Lightweight PPO trainer for simple environments
# ============================================================

class SimplePPOBottleneckTrainer:
    """
    Simplified PPO bottleneck trainer for classic control environments.

    Same core algorithm as the Go PPO trainer but adapted for environments
    without action masks (CartPole, Acrobot, LunarLander all have fully
    legal action spaces).
    """

    def __init__(self, encoder, concept_manager, n_actions, n_concepts=32,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, n_epochs=4, batch_size=64,
                 device=None):
        self.device = device or get_device()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Frozen encoder
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.concept_manager = concept_manager

        # Bottleneck policy — use embed_dim=32, hidden_dim=64 to match
        # the CartPole/LunarLander bottleneck architecture for fair comparison
        self.policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts,
            embed_dim=32,
            hidden_dim=64,
            n_actions=n_actions,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def get_concept(self, obs):
        """obs -> encoder -> features -> concept_id"""
        return self.concept_manager.assign_concept_from_obs(
            self.encoder, obs, self.device
        )

    def collect_rollout(self, env, n_steps=2048):
        """Collect trajectory data for PPO update."""
        concepts, actions, rewards = [], [], []
        values, log_probs, dones = [], [], []

        obs, info = env.reset()
        episode_rewards = []
        ep_reward = 0.0
        self.policy.eval()

        for step in range(n_steps):
            concept_id = self.get_concept(obs)

            with torch.no_grad():
                cid_t = torch.LongTensor([concept_id]).to(self.device)
                logits, value = self.policy(cid_t)
                probs = F.softmax(logits[0], dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_t = dist.sample()
                action = action_t.item()
                log_prob = dist.log_prob(action_t)

            concepts.append(concept_id)
            actions.append(action)
            values.append(value[0].item())
            log_probs.append(log_prob.item())

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            dones.append(float(done))
            ep_reward += reward

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs
                info = next_info

        # Bootstrap final value
        with torch.no_grad():
            final_cid = torch.LongTensor([self.get_concept(obs)]).to(self.device)
            _, final_value = self.policy(final_cid)
            last_value = final_value[0].item()

        return {
            "concepts": np.array(concepts),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
            "last_value": last_value,
            "episode_rewards": episode_rewards,
        }

    def compute_advantages(self, rollout):
        """GAE advantage estimation."""
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        last_value = rollout["last_value"]
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            last_gae = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO clipped surrogate update."""
        self.policy.train()
        advantages, returns = self.compute_advantages(rollout)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        concepts_t = torch.LongTensor(rollout["concepts"]).to(self.device)
        actions_t = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs_t = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        n = len(concepts_t)
        total_loss = 0.0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                logits, values = self.policy(concepts_t[batch_idx])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs + 1e-8)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                batch_adv = advantages_t[batch_idx]
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns_t[batch_idx])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / (self.n_epochs * max(1, n // self.batch_size))


def evaluate_simple_agent(agent_fn, env, n_episodes=50):
    """Evaluate agent on a simple (non-Go) environment."""
    total_reward = 0.0
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action = agent_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        total_reward += ep_reward
        episode_lengths.append(ep_len)

    return {
        "mean_reward": total_reward / n_episodes,
        "mean_length": np.mean(episode_lengths),
    }


# ============================================================
# Training pipeline for a new environment
# ============================================================

def train_baseline_simple(env, env_name, n_steps=100000, save_dir="models", device=None):
    """
    Train a baseline RL agent on a simple environment (for encoder features).

    Uses PPO via SB3 to train an MLP encoder + policy, then extracts the encoder.

    Args:
        env: Gymnasium environment.
        env_name: Name for saving ("acrobot", "mountaincar").
        n_steps: Total training steps.
        save_dir: Directory to save encoder.
        device: Torch device.

    Returns:
        Trained encoder (SimpleMLPEncoder).
    """
    from stable_baselines3 import PPO as SB3_PPO
    device = device or get_device()
    ensure_dir(save_dir)

    print(f"  Training {env_name} baseline ({n_steps} steps)...")

    # SB3 PPO with custom MLP encoder
    policy_kwargs = dict(
        features_extractor_class=SimpleMLPEncoder,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = SB3_PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    model.learn(total_timesteps=n_steps)

    # Extract encoder
    encoder = model.policy.features_extractor
    encoder_path = os.path.join(save_dir, f"{env_name}_encoder.pt")
    torch.save(encoder.state_dict(), encoder_path)
    print(f"  Saved encoder to {encoder_path}")

    # Quick evaluation
    obs, info = env.reset()
    total_r = 0
    for _ in range(10):
        obs, info = env.reset()
        done = False
        ep_r = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += r
        total_r += ep_r
    print(f"  Baseline performance: {total_r/10:.1f} avg reward (10 episodes)")

    return encoder


def train_bottleneck_simple(encoder, cm, env, n_actions, n_concepts=32,
                            n_generations=50, steps_per_gen=4096,
                            lr=3e-4, device=None):
    """
    Train a bottleneck policy from scratch on a simple environment.

    Args:
        encoder: Frozen encoder.
        cm: Fitted ConceptManager.
        env: Environment.
        n_actions: Number of actions.
        n_concepts: Number of concepts.
        n_generations: Training generations.
        steps_per_gen: Steps per generation.
        lr: Learning rate.
        device: Torch device.

    Returns:
        (trainer, learning_curve) — trained policy and performance over time.
    """
    device = device or get_device()

    trainer = SimplePPOBottleneckTrainer(
        encoder=encoder, concept_manager=cm,
        n_actions=n_actions, n_concepts=n_concepts,
        lr=lr, device=device,
    )

    learning_curve = []
    eval_env = type(env)()

    for gen in range(n_generations):
        rollout = trainer.collect_rollout(env, n_steps=steps_per_gen)
        loss = trainer.update(rollout)

        if gen % 5 == 0 or gen == n_generations - 1:
            def agent_fn(obs):
                c = trainer.get_concept(obs)
                return trainer.policy.get_action(c, deterministic=True)

            eval_result = evaluate_simple_agent(agent_fn, eval_env, n_episodes=20)
            mean_r = eval_result["mean_reward"]
            learning_curve.append({
                "generation": gen,
                "mean_reward": mean_r,
                "mean_length": eval_result["mean_length"],
            })
            if gen % 10 == 0 or gen == n_generations - 1:
                print(f"    Gen {gen:3d}: reward={mean_r:.1f}")

    eval_env.close()
    return trainer, learning_curve


# ============================================================
# Main experiment
# ============================================================

def run_cross_domain_transfer():
    """Run the full cross-domain transfer experiment."""
    set_seed(42)
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Cross-Domain Concept Transfer")
    print(f"[{timestamp}] ============================================================")

    results = {}

    # ============================================================
    # Step 1: Train Acrobot baseline + discover concepts
    # ============================================================
    print(f"\n--- Step 1: Acrobot Baseline ---")
    acrobot_dir = "models/acrobot"
    ensure_dir(acrobot_dir)

    acrobot_env = AcrobotConceptEnv()
    encoder_path = os.path.join(acrobot_dir, "acrobot_encoder.pt")
    concepts_path = os.path.join(acrobot_dir, "concepts_k32.pkl")

    # Check if encoder already exists
    if os.path.exists(encoder_path):
        print(f"  Loading existing Acrobot encoder from {encoder_path}")
        acrobot_encoder = SimpleMLPEncoder(acrobot_env.observation_space, features_dim=128)
        acrobot_encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
    else:
        acrobot_encoder = train_baseline_simple(
            acrobot_env, "acrobot", n_steps=100000, save_dir=acrobot_dir, device=device,
        )

    acrobot_encoder.to(device)
    acrobot_encoder.eval()

    # Discover Acrobot concepts
    if os.path.exists(concepts_path):
        print(f"  Loading existing Acrobot concepts from {concepts_path}")
        acrobot_cm = ConceptManager(n_concepts=32)
        acrobot_cm.load(concepts_path)
    else:
        print(f"  Discovering Acrobot concepts (K=32)...")
        acrobot_cm = ConceptManager(n_concepts=32, features_dim=128)
        acrobot_cm.collect_features(acrobot_encoder, acrobot_env, n_episodes=500, device=device)
        acrobot_cm.fit()
        acrobot_cm.save(concepts_path)

    # ============================================================
    # Step 2: Train Acrobot bottleneck from scratch (CONTROL)
    # ============================================================
    print(f"\n--- Step 2: Acrobot Bottleneck (from scratch) ---")
    scratch_trainer, scratch_curve = train_bottleneck_simple(
        acrobot_encoder, acrobot_cm, acrobot_env,
        n_actions=3, n_concepts=32, n_generations=50,
        steps_per_gen=4096, lr=3e-4, device=device,
    )
    torch.save(scratch_trainer.policy.state_dict(),
               os.path.join(acrobot_dir, "bottleneck_scratch.pt"))

    results["acrobot_scratch"] = {
        "learning_curve": scratch_curve,
        "final_reward": scratch_curve[-1]["mean_reward"] if scratch_curve else 0.0,
    }

    # ============================================================
    # Step 3: CartPole -> Acrobot Transfer
    # ============================================================
    print(f"\n--- Step 3: CartPole -> Acrobot Transfer ---")

    # Load CartPole encoder + concepts + policy
    cartpole_cm = ConceptManager(n_concepts=32)
    cartpole_cm.load("models/simple/concepts_cartpole_k32.pkl")

    cartpole_encoder = SimpleMLPEncoder(
        CartPoleConceptEnv().observation_space, features_dim=128
    )
    cartpole_encoder.load_state_dict(
        torch.load("models/simple/ppo_cartpole_encoder.pt",
                    map_location=device, weights_only=True)
    )
    cartpole_encoder.eval()

    # Load CartPole bottleneck policy
    cartpole_policy = ConceptBottleneckPolicy(
        n_concepts=32, embed_dim=32, hidden_dim=64, n_actions=2,
    )
    cartpole_policy.load_state_dict(
        torch.load("models/simple/ppo_cartpole_bottleneck.pt",
                    map_location=device, weights_only=True)
    )
    cartpole_policy.eval()

    # Align CartPole -> Acrobot concepts
    aligner_cp = ConceptAligner(cartpole_cm, acrobot_cm)
    mapping_cp = aligner_cp.hungarian_alignment()
    quality_cp = aligner_cp.alignment_quality(mapping_cp)
    print(f"  Alignment: mean_sim={quality_cp['mean_similarity']:.4f}")

    # Transfer CartPole policy to Acrobot (2 actions -> 3 actions)
    transferred_cp = aligner_cp.transfer_policy(
        cartpole_policy, mapping_cp, target_n_concepts=32, target_n_actions=3,
    )
    transferred_cp.to(device)

    # Zero-shot evaluation
    def agent_fn_zero(obs):
        c = acrobot_cm.assign_concept_from_obs(acrobot_encoder, obs, device)
        return transferred_cp.get_action(c, deterministic=True)

    zero_shot_cp = evaluate_simple_agent(agent_fn_zero, AcrobotConceptEnv(), n_episodes=50)
    print(f"  Zero-shot: reward={zero_shot_cp['mean_reward']:.1f}")

    # Fine-tune transferred policy
    print(f"  Fine-tuning transferred policy...")
    ft_trainer = SimplePPOBottleneckTrainer(
        encoder=acrobot_encoder, concept_manager=acrobot_cm,
        n_actions=3, n_concepts=32, lr=1e-4, device=device,
    )
    ft_trainer.policy.load_state_dict(transferred_cp.state_dict())
    ft_trainer.policy.to(device)

    ft_env = AcrobotConceptEnv()
    ft_eval_env = AcrobotConceptEnv()
    ft_curve = []

    for gen in range(50):
        rollout = ft_trainer.collect_rollout(ft_env, n_steps=4096)
        ft_trainer.update(rollout)

        if gen % 5 == 0 or gen == 49:
            def agent_fn_ft(obs):
                c = ft_trainer.get_concept(obs)
                return ft_trainer.policy.get_action(c, deterministic=True)

            eval_result = evaluate_simple_agent(agent_fn_ft, ft_eval_env, n_episodes=20)
            ft_curve.append({
                "generation": gen,
                "mean_reward": eval_result["mean_reward"],
            })
            if gen % 10 == 0 or gen == 49:
                print(f"    Gen {gen:3d}: reward={eval_result['mean_reward']:.1f}")

    ft_env.close()
    ft_eval_env.close()

    results["cartpole_to_acrobot"] = {
        "alignment": quality_cp,
        "zero_shot_reward": zero_shot_cp["mean_reward"],
        "fine_tune_curve": ft_curve,
        "final_reward": ft_curve[-1]["mean_reward"] if ft_curve else 0.0,
    }

    # ============================================================
    # Step 4: LunarLander -> Acrobot Transfer
    # ============================================================
    print(f"\n--- Step 4: LunarLander -> Acrobot Transfer ---")

    lunar_cm = ConceptManager(n_concepts=32)
    lunar_cm.load("models/lunar_lander/concepts_k32.pkl")

    # Load LunarLander encoder — infer architecture from checkpoint
    # (LunarLander encoder may have different hidden dims than default)
    lunar_encoder_path = "models/lunar_lander/encoder.pt"
    lunar_sd = torch.load(lunar_encoder_path, map_location=device, weights_only=True)
    lunar_encoder = SimpleMLPEncoder(
        LunarLanderConceptEnv().observation_space, features_dim=128
    )
    # Check if architecture matches; if not, rebuild with correct dims
    try:
        lunar_encoder.load_state_dict(lunar_sd)
    except RuntimeError:
        # Infer first hidden dim from checkpoint
        first_hidden = lunar_sd["net.0.weight"].shape[0]
        obs_dim = lunar_sd["net.0.weight"].shape[1]
        import torch.nn as tnn
        lunar_encoder.net = tnn.Sequential(
            tnn.Linear(obs_dim, first_hidden),
            tnn.ReLU(),
            tnn.Linear(first_hidden, 128),
            tnn.ReLU(),
            tnn.Linear(128, 128),
            tnn.ReLU(),
        )
        lunar_encoder.load_state_dict(lunar_sd)
    lunar_encoder.eval()

    # Load LunarLander bottleneck policy
    lunar_policy = ConceptBottleneckPolicy(
        n_concepts=32, embed_dim=32, hidden_dim=64, n_actions=4,
    )
    lunar_policy.load_state_dict(
        torch.load("models/lunar_lander/bottleneck_final.pt",
                    map_location=device, weights_only=True)
    )
    lunar_policy.eval()

    # Align LunarLander -> Acrobot
    aligner_ll = ConceptAligner(lunar_cm, acrobot_cm)
    mapping_ll = aligner_ll.hungarian_alignment()
    quality_ll = aligner_ll.alignment_quality(mapping_ll)
    print(f"  Alignment: mean_sim={quality_ll['mean_similarity']:.4f}")

    # Transfer (4 actions -> 3 actions)
    transferred_ll = aligner_ll.transfer_policy(
        lunar_policy, mapping_ll, target_n_concepts=32, target_n_actions=3,
    )
    transferred_ll.to(device)

    # Zero-shot evaluation
    def agent_fn_zero_ll(obs):
        c = acrobot_cm.assign_concept_from_obs(acrobot_encoder, obs, device)
        return transferred_ll.get_action(c, deterministic=True)

    zero_shot_ll = evaluate_simple_agent(agent_fn_zero_ll, AcrobotConceptEnv(), n_episodes=50)
    print(f"  Zero-shot: reward={zero_shot_ll['mean_reward']:.1f}")

    # Fine-tune
    print(f"  Fine-tuning transferred policy...")
    ft_trainer_ll = SimplePPOBottleneckTrainer(
        encoder=acrobot_encoder, concept_manager=acrobot_cm,
        n_actions=3, n_concepts=32, lr=1e-4, device=device,
    )
    ft_trainer_ll.policy.load_state_dict(transferred_ll.state_dict())
    ft_trainer_ll.policy.to(device)

    ft_env_ll = AcrobotConceptEnv()
    ft_eval_ll = AcrobotConceptEnv()
    ft_curve_ll = []

    for gen in range(50):
        rollout = ft_trainer_ll.collect_rollout(ft_env_ll, n_steps=4096)
        ft_trainer_ll.update(rollout)

        if gen % 5 == 0 or gen == 49:
            def agent_fn_ft_ll(obs):
                c = ft_trainer_ll.get_concept(obs)
                return ft_trainer_ll.policy.get_action(c, deterministic=True)

            eval_result = evaluate_simple_agent(agent_fn_ft_ll, ft_eval_ll, n_episodes=20)
            ft_curve_ll.append({
                "generation": gen,
                "mean_reward": eval_result["mean_reward"],
            })
            if gen % 10 == 0 or gen == 49:
                print(f"    Gen {gen:3d}: reward={eval_result['mean_reward']:.1f}")

    ft_env_ll.close()
    ft_eval_ll.close()

    results["lunarlander_to_acrobot"] = {
        "alignment": quality_ll,
        "zero_shot_reward": zero_shot_ll["mean_reward"],
        "fine_tune_curve": ft_curve_ll,
        "final_reward": ft_curve_ll[-1]["mean_reward"] if ft_curve_ll else 0.0,
    }

    # ============================================================
    # Save results + visualize
    # ============================================================
    ensure_dir("results")
    output_path = "results/transfer_cross_domain.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    # Summary
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Summary: Cross-Domain Transfer")
    print(f"[{ts}]   Acrobot from scratch: {results['acrobot_scratch']['final_reward']:.1f}")
    print(f"[{ts}]   CartPole->Acrobot zero-shot: {results['cartpole_to_acrobot']['zero_shot_reward']:.1f}")
    print(f"[{ts}]   CartPole->Acrobot fine-tuned: {results['cartpole_to_acrobot']['final_reward']:.1f}")
    print(f"[{ts}]   LunarLander->Acrobot zero-shot: {results['lunarlander_to_acrobot']['zero_shot_reward']:.1f}")
    print(f"[{ts}]   LunarLander->Acrobot fine-tuned: {results['lunarlander_to_acrobot']['final_reward']:.1f}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Okabe-Ito colorblind-friendly palette
        OI_BLUE, OI_ORANGE = "#0072B2", "#E69F00"
        OI_BLACK = "#000000"

        ensure_dir("results/figures")
        plt.rcParams.update({"font.size": 11})
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot learning curves
        scratch = results["acrobot_scratch"]["learning_curve"]
        cp_ft = results["cartpole_to_acrobot"]["fine_tune_curve"]
        ll_ft = results["lunarlander_to_acrobot"]["fine_tune_curve"]

        if scratch:
            ax.plot([s["generation"] for s in scratch],
                    [s["mean_reward"] for s in scratch],
                    "-s", color=OI_BLACK, label="From scratch",
                    markersize=5, linewidth=2)
        if cp_ft:
            ax.plot([s["generation"] for s in cp_ft],
                    [s["mean_reward"] for s in cp_ft],
                    "-o", color=OI_BLUE, label="CartPole transfer",
                    markersize=5, linewidth=2)
        if ll_ft:
            ax.plot([s["generation"] for s in ll_ft],
                    [s["mean_reward"] for s in ll_ft],
                    "-^", color=OI_ORANGE, label="LunarLander transfer",
                    markersize=5, linewidth=2)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Mean Reward", fontsize=12)
        ax.set_title("Cross-Domain Transfer: Acrobot Learning Curves", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

        fig_path = "results/figures/transfer_cross_domain.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return results


def run_multi_seed_cross_domain(seeds=None, n_generations=50):
    """
    Run cross-domain transfer with multiple seeds for statistical significance.

    Also builds the full 3x3 transfer matrix across CartPole, Acrobot, and
    LunarLander (including same-domain sanity checks).

    Args:
        seeds: List of random seeds. Defaults to [42, 123, 456].
        n_generations: Training generations per condition.

    Returns:
        Dict with per-seed results, aggregate metrics, and transfer matrix.
    """
    from scipy import stats

    if seeds is None:
        seeds = [42, 123, 456]

    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Cross-Domain Transfer Matrix ({len(seeds)} seeds)")
    print(f"[{timestamp}] ============================================================")

    # Define all environments and their configurations
    DOMAINS = {
        "CartPole": {
            "env_class": CartPoleConceptEnv,
            "encoder_path": "models/simple/ppo_cartpole_encoder.pt",
            "concepts_path": "models/simple/concepts_cartpole_k32.pkl",
            "policy_path": "models/simple/ppo_cartpole_bottleneck.pt",
            "n_actions": 2,
            "n_concepts": 32,
        },
        "LunarLander": {
            "env_class": LunarLanderConceptEnv,
            "encoder_path": "models/lunar_lander/encoder.pt",
            "concepts_path": "models/lunar_lander/concepts_k32.pkl",
            "policy_path": "models/lunar_lander/bottleneck_final.pt",
            "n_actions": 4,
            "n_concepts": 32,
        },
        "Acrobot": {
            "env_class": AcrobotConceptEnv,
            "encoder_path": "models/acrobot/acrobot_encoder.pt",
            "concepts_path": "models/acrobot/concepts_k32.pkl",
            "policy_path": "models/acrobot/bottleneck_scratch.pt",
            "n_actions": 3,
            "n_concepts": 32,
        },
    }

    # Load all encoders, concepts, and policies
    print(f"\n--- Loading all domain agents ---")
    agents = {}
    for name, config in DOMAINS.items():
        print(f"  Loading {name}...")
        env = config["env_class"]()
        encoder = SimpleMLPEncoder(env.observation_space, features_dim=128)

        # Handle potential architecture mismatch (LunarLander)
        sd = torch.load(config["encoder_path"], map_location=device, weights_only=True)
        try:
            encoder.load_state_dict(sd)
        except RuntimeError:
            first_hidden = sd["net.0.weight"].shape[0]
            obs_dim = sd["net.0.weight"].shape[1]
            import torch.nn as tnn
            encoder.net = tnn.Sequential(
                tnn.Linear(obs_dim, first_hidden),
                tnn.ReLU(),
                tnn.Linear(first_hidden, 128),
                tnn.ReLU(),
                tnn.Linear(128, 128),
                tnn.ReLU(),
            )
            encoder.load_state_dict(sd)
        encoder.to(device)
        encoder.eval()

        cm = ConceptManager(n_concepts=config["n_concepts"])
        cm.load(config["concepts_path"])

        policy = ConceptBottleneckPolicy(
            n_concepts=config["n_concepts"],
            embed_dim=32, hidden_dim=64,
            n_actions=config["n_actions"],
        )
        if os.path.exists(config["policy_path"]):
            policy.load_state_dict(
                torch.load(config["policy_path"], map_location=device, weights_only=True)
            )
        policy.to(device)
        policy.eval()

        agents[name] = {
            "encoder": encoder, "cm": cm, "policy": policy,
            "env_class": config["env_class"],
            "n_actions": config["n_actions"],
            "n_concepts": config["n_concepts"],
        }
        env.close()

    # Build transfer matrix
    domain_names = list(DOMAINS.keys())
    transfer_matrix = {}

    for src_name in domain_names:
        for tgt_name in domain_names:
            pair_key = f"{src_name}_to_{tgt_name}"
            print(f"\n--- {src_name} -> {tgt_name} ---")

            src = agents[src_name]
            tgt = agents[tgt_name]

            # Align concepts
            aligner = ConceptAligner(src["cm"], tgt["cm"])
            mapping = aligner.hungarian_alignment()
            quality = aligner.alignment_quality(mapping)
            print(f"  Alignment: mean_sim={quality['mean_similarity']:.4f}")

            # Transfer policy
            transferred = aligner.transfer_policy(
                src["policy"], mapping,
                target_n_concepts=tgt["n_concepts"],
                target_n_actions=tgt["n_actions"],
            )
            transferred.to(device)

            # Zero-shot evaluation
            eval_env = tgt["env_class"]()

            def make_agent_fn(policy_ref, cm_ref, enc_ref):
                def agent_fn(obs):
                    c = cm_ref.assign_concept_from_obs(enc_ref, obs, device)
                    return policy_ref.get_action(c, deterministic=True)
                return agent_fn

            zero_shot = evaluate_simple_agent(
                make_agent_fn(transferred, tgt["cm"], tgt["encoder"]),
                eval_env, n_episodes=50,
            )
            eval_env.close()
            print(f"  Zero-shot: reward={zero_shot['mean_reward']:.1f}")

            # Multi-seed fine-tuning
            seed_results = []
            for seed in seeds:
                set_seed(seed)
                ft_trainer = SimplePPOBottleneckTrainer(
                    encoder=tgt["encoder"], concept_manager=tgt["cm"],
                    n_actions=tgt["n_actions"], n_concepts=tgt["n_concepts"],
                    lr=1e-4, device=device,
                )
                ft_trainer.policy.load_state_dict(transferred.state_dict())
                ft_trainer.policy.to(device)

                ft_env = tgt["env_class"]()
                ft_eval_env = tgt["env_class"]()
                ft_curve = []

                for gen in range(n_generations):
                    rollout = ft_trainer.collect_rollout(ft_env, n_steps=4096)
                    ft_trainer.update(rollout)

                    if gen % 10 == 0 or gen == n_generations - 1:
                        def agent_fn_eval(obs, _tr=ft_trainer):
                            c = _tr.get_concept(obs)
                            return _tr.policy.get_action(c, deterministic=True)

                        ev = evaluate_simple_agent(agent_fn_eval, ft_eval_env, n_episodes=20)
                        ft_curve.append({"generation": gen, "mean_reward": ev["mean_reward"]})

                ft_env.close()
                ft_eval_env.close()
                final_r = ft_curve[-1]["mean_reward"] if ft_curve else -500.0
                seed_results.append({"seed": seed, "final_reward": final_r, "curve": ft_curve})
                print(f"    Seed {seed}: final_reward={final_r:.1f}")

            final_rewards = [r["final_reward"] for r in seed_results]

            transfer_matrix[pair_key] = {
                "source": src_name,
                "target": tgt_name,
                "alignment_sim": quality["mean_similarity"],
                "zero_shot_reward": zero_shot["mean_reward"],
                "fine_tune_mean": float(np.mean(final_rewards)),
                "fine_tune_std": float(np.std(final_rewards)),
                "seed_results": seed_results,
            }

    # Compute from-scratch baselines for each target (multi-seed)
    print(f"\n--- From-Scratch Baselines ---")
    scratch_baselines = {}
    for tgt_name in domain_names:
        tgt = agents[tgt_name]
        scratch_rewards = []
        for seed in seeds:
            set_seed(seed)
            _, scratch_curve = train_bottleneck_simple(
                tgt["encoder"], tgt["cm"], tgt["env_class"](),
                n_actions=tgt["n_actions"], n_concepts=tgt["n_concepts"],
                n_generations=n_generations, steps_per_gen=4096,
                lr=3e-4, device=device,
            )
            final_r = scratch_curve[-1]["mean_reward"] if scratch_curve else -500.0
            scratch_rewards.append(final_r)
            print(f"  {tgt_name} seed {seed}: {final_r:.1f}")

        scratch_baselines[tgt_name] = {
            "mean": float(np.mean(scratch_rewards)),
            "std": float(np.std(scratch_rewards)),
            "values": scratch_rewards,
        }

    # Compute improvement over scratch for each transfer pair
    for pair_key, pair_data in transfer_matrix.items():
        tgt = pair_data["target"]
        scratch_mean = scratch_baselines[tgt]["mean"]
        ft_mean = pair_data["fine_tune_mean"]

        # For Acrobot/LunarLander, improvement = less negative reward
        if scratch_mean != 0:
            improvement_pct = ((ft_mean - scratch_mean) / abs(scratch_mean)) * 100
        else:
            improvement_pct = 0.0
        pair_data["improvement_pct"] = improvement_pct
        pair_data["scratch_baseline"] = scratch_baselines[tgt]

    # Correlation: alignment quality vs transfer improvement
    sims = [v["alignment_sim"] for v in transfer_matrix.values()]
    improvements = [v["improvement_pct"] for v in transfer_matrix.values()]
    if len(sims) > 2:
        corr, corr_p = stats.pearsonr(sims, improvements)
    else:
        corr, corr_p = 0.0, 1.0

    multi_results = {
        "transfer_matrix": transfer_matrix,
        "scratch_baselines": scratch_baselines,
        "correlation": {
            "alignment_vs_improvement": {"r": float(corr), "p": float(corr_p)},
        },
        "seeds": seeds,
        "n_generations": n_generations,
    }

    # Save
    ensure_dir("results")
    output_path = "results/transfer_matrix.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(multi_results, f, indent=2, default=convert)
    print(f"\nTransfer matrix saved to {output_path}")

    # Summary table
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Transfer Matrix Summary")
    print(f"[{ts}] {'Source':<12} {'Target':<12} {'Align':>6} {'Zero-Shot':>10} {'Fine-Tune':>10} {'Improv':>8}")
    print(f"[{ts}] {'-'*60}")
    for pair_key, data in transfer_matrix.items():
        print(f"[{ts}] {data['source']:<12} {data['target']:<12} "
              f"{data['alignment_sim']:>6.3f} "
              f"{data['zero_shot_reward']:>10.1f} "
              f"{data['fine_tune_mean']:>8.1f}+/-{data['fine_tune_std']:.1f} "
              f"{data['improvement_pct']:>7.1f}%")
    print(f"[{ts}] Correlation (alignment vs improvement): r={corr:.3f}, p={corr_p:.4f}")

    # Visualization: transfer matrix heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_dir("results/figures")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Alignment similarity matrix
        ax = axes[0]
        n = len(domain_names)
        sim_matrix = np.zeros((n, n))
        for i, src in enumerate(domain_names):
            for j, tgt in enumerate(domain_names):
                key = f"{src}_to_{tgt}"
                if key in transfer_matrix:
                    sim_matrix[i, j] = transfer_matrix[key]["alignment_sim"]

        im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(domain_names, fontsize=10)
        ax.set_yticklabels(domain_names, fontsize=10)
        ax.set_title("Concept Alignment Similarity", fontsize=12)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{sim_matrix[i,j]:.3f}",
                        ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Improvement heatmap
        ax = axes[1]
        imp_matrix = np.zeros((n, n))
        for i, src in enumerate(domain_names):
            for j, tgt in enumerate(domain_names):
                key = f"{src}_to_{tgt}"
                if key in transfer_matrix:
                    imp_matrix[i, j] = transfer_matrix[key]["improvement_pct"]

        im2 = ax.imshow(imp_matrix, cmap="RdYlGn", vmin=-50, vmax=50)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(domain_names, fontsize=10)
        ax.set_yticklabels(domain_names, fontsize=10)
        ax.set_title("Transfer Improvement (%)", fontsize=12)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{imp_matrix[i,j]:.1f}%",
                        ha="center", va="center", fontsize=9)
        plt.colorbar(im2, ax=ax, shrink=0.8)

        plt.tight_layout()
        fig_path = "results/figures/transfer_matrix.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return multi_results


def run_tuned_cross_domain(n_seeds=5, n_generations=100):
    """
    Run cross-domain transfer with tuned hyperparameters and full 3x3 matrix.

    Improvements over default:
        - LR sweep: [5e-5, 1e-4, 3e-4] per pair, pick best
        - More generations: 100 (up from 50)
        - Higher entropy coefficient: 0.02 (encourages exploration during adaptation)
        - Full 5 seeds per configuration
        - Reports mean +/- std, improvement over scratch, p-values

    Args:
        n_seeds: Number of random seeds (default: 5).
        n_generations: Training generations per condition (default: 100).
    """
    from scipy import stats as sp_stats

    seeds = list(range(42, 42 + n_seeds * 10, 10))  # [42, 52, 62, 72, 82]
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")
    lr_options = [5e-5, 1e-4, 3e-4]

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] Tuned Cross-Domain Transfer ({n_seeds} seeds, {n_generations} gens)")
    print(f"[{timestamp}]   LR sweep: {lr_options}")
    print(f"[{timestamp}] ============================================================")

    # Load all domain agents
    DOMAINS = {
        "CartPole": {
            "env_class": CartPoleConceptEnv,
            "encoder_path": "models/simple/ppo_cartpole_encoder.pt",
            "concepts_path": "models/simple/concepts_cartpole_k32.pkl",
            "policy_path": "models/simple/ppo_cartpole_bottleneck.pt",
            "n_actions": 2, "n_concepts": 32,
        },
        "LunarLander": {
            "env_class": LunarLanderConceptEnv,
            "encoder_path": "models/lunar_lander/encoder.pt",
            "concepts_path": "models/lunar_lander/concepts_k32.pkl",
            "policy_path": "models/lunar_lander/bottleneck_final.pt",
            "n_actions": 4, "n_concepts": 32,
        },
        "Acrobot": {
            "env_class": AcrobotConceptEnv,
            "encoder_path": "models/acrobot/acrobot_encoder.pt",
            "concepts_path": "models/acrobot/concepts_k32.pkl",
            "policy_path": "models/acrobot/bottleneck_scratch.pt",
            "n_actions": 3, "n_concepts": 32,
        },
    }

    print(f"\n--- Loading all domain agents ---")
    agents = {}
    for name, config in DOMAINS.items():
        print(f"  Loading {name}...")
        env = config["env_class"]()
        encoder = SimpleMLPEncoder(env.observation_space, features_dim=128)
        sd = torch.load(config["encoder_path"], map_location=device, weights_only=True)
        try:
            encoder.load_state_dict(sd)
        except RuntimeError:
            first_hidden = sd["net.0.weight"].shape[0]
            obs_dim = sd["net.0.weight"].shape[1]
            import torch.nn as tnn
            encoder.net = tnn.Sequential(
                tnn.Linear(obs_dim, first_hidden), tnn.ReLU(),
                tnn.Linear(first_hidden, 128), tnn.ReLU(),
                tnn.Linear(128, 128), tnn.ReLU(),
            )
            encoder.load_state_dict(sd)
        encoder.to(device)
        encoder.eval()

        cm = ConceptManager(n_concepts=config["n_concepts"])
        cm.load(config["concepts_path"])

        policy = ConceptBottleneckPolicy(
            n_concepts=config["n_concepts"],
            embed_dim=32, hidden_dim=64,
            n_actions=config["n_actions"],
        )
        if os.path.exists(config["policy_path"]):
            policy.load_state_dict(
                torch.load(config["policy_path"], map_location=device, weights_only=True)
            )
        policy.to(device)
        policy.eval()

        agents[name] = {
            "encoder": encoder, "cm": cm, "policy": policy,
            "env_class": config["env_class"],
            "n_actions": config["n_actions"],
            "n_concepts": config["n_concepts"],
        }
        env.close()

    domain_names = list(DOMAINS.keys())
    transfer_matrix = {}

    for src_name in domain_names:
        for tgt_name in domain_names:
            pair_key = f"{src_name}_to_{tgt_name}"
            print(f"\n--- {src_name} -> {tgt_name} ---")

            src = agents[src_name]
            tgt = agents[tgt_name]

            # Align concepts
            aligner = ConceptAligner(src["cm"], tgt["cm"])
            mapping = aligner.hungarian_alignment()
            quality = aligner.alignment_quality(mapping)
            print(f"  Alignment: mean_sim={quality['mean_similarity']:.4f}")

            # Transfer policy
            transferred = aligner.transfer_policy(
                src["policy"], mapping,
                target_n_concepts=tgt["n_concepts"],
                target_n_actions=tgt["n_actions"],
            )
            transferred.to(device)

            # Zero-shot evaluation
            eval_env = tgt["env_class"]()
            def make_agent_fn(pol, cm_ref, enc_ref):
                def fn(obs):
                    c = cm_ref.assign_concept_from_obs(enc_ref, obs, device)
                    return pol.get_action(c, deterministic=True)
                return fn

            zero_shot = evaluate_simple_agent(
                make_agent_fn(transferred, tgt["cm"], tgt["encoder"]),
                eval_env, n_episodes=50,
            )
            eval_env.close()
            print(f"  Zero-shot: reward={zero_shot['mean_reward']:.1f}")

            # LR sweep: pick best LR on first seed
            best_lr = lr_options[1]  # default
            best_lr_reward = -float("inf")

            print(f"  LR sweep on seed {seeds[0]}...")
            for lr_candidate in lr_options:
                set_seed(seeds[0])
                ft_trainer = SimplePPOBottleneckTrainer(
                    encoder=tgt["encoder"], concept_manager=tgt["cm"],
                    n_actions=tgt["n_actions"], n_concepts=tgt["n_concepts"],
                    lr=lr_candidate, ent_coef=0.02, device=device,
                )
                ft_trainer.policy.load_state_dict(transferred.state_dict())
                ft_trainer.policy.to(device)

                ft_env = tgt["env_class"]()
                # Quick eval after 30 generations
                for gen in range(30):
                    rollout = ft_trainer.collect_rollout(ft_env, n_steps=4096)
                    ft_trainer.update(rollout)
                ft_env.close()

                lr_eval_env = tgt["env_class"]()
                def agent_fn_lr(obs, _tr=ft_trainer):
                    c = _tr.get_concept(obs)
                    return _tr.policy.get_action(c, deterministic=True)
                ev = evaluate_simple_agent(agent_fn_lr, lr_eval_env, n_episodes=20)
                lr_eval_env.close()

                print(f"    LR={lr_candidate:.0e}: reward={ev['mean_reward']:.1f}")
                if ev["mean_reward"] > best_lr_reward:
                    best_lr_reward = ev["mean_reward"]
                    best_lr = lr_candidate

            print(f"  Best LR: {best_lr:.0e}")

            # Full multi-seed fine-tuning with best LR
            seed_results = []
            for seed in seeds:
                set_seed(seed)
                ft_trainer = SimplePPOBottleneckTrainer(
                    encoder=tgt["encoder"], concept_manager=tgt["cm"],
                    n_actions=tgt["n_actions"], n_concepts=tgt["n_concepts"],
                    lr=best_lr, ent_coef=0.02, device=device,
                )
                ft_trainer.policy.load_state_dict(transferred.state_dict())
                ft_trainer.policy.to(device)

                ft_env = tgt["env_class"]()
                ft_eval_env = tgt["env_class"]()
                ft_curve = []

                for gen in range(n_generations):
                    rollout = ft_trainer.collect_rollout(ft_env, n_steps=4096)
                    ft_trainer.update(rollout)

                    if gen % 10 == 0 or gen == n_generations - 1:
                        def agent_fn_eval(obs, _tr=ft_trainer):
                            c = _tr.get_concept(obs)
                            return _tr.policy.get_action(c, deterministic=True)
                        ev = evaluate_simple_agent(agent_fn_eval, ft_eval_env, n_episodes=20)
                        ft_curve.append({"generation": gen, "mean_reward": ev["mean_reward"]})

                ft_env.close()
                ft_eval_env.close()
                final_r = ft_curve[-1]["mean_reward"] if ft_curve else -500.0
                seed_results.append({"seed": seed, "final_reward": final_r, "curve": ft_curve})
                print(f"    Seed {seed}: final_reward={final_r:.1f}")

            final_rewards = [r["final_reward"] for r in seed_results]

            transfer_matrix[pair_key] = {
                "source": src_name, "target": tgt_name,
                "alignment_sim": quality["mean_similarity"],
                "zero_shot_reward": zero_shot["mean_reward"],
                "best_lr": best_lr,
                "fine_tune_mean": float(np.mean(final_rewards)),
                "fine_tune_std": float(np.std(final_rewards)),
                "seed_results": seed_results,
            }

    # From-scratch baselines (multi-seed)
    print(f"\n--- From-Scratch Baselines ---")
    scratch_baselines = {}
    for tgt_name in domain_names:
        tgt = agents[tgt_name]
        scratch_rewards = []
        for seed in seeds:
            set_seed(seed)
            _, scratch_curve = train_bottleneck_simple(
                tgt["encoder"], tgt["cm"], tgt["env_class"](),
                n_actions=tgt["n_actions"], n_concepts=tgt["n_concepts"],
                n_generations=n_generations, steps_per_gen=4096,
                lr=3e-4, device=device,
            )
            final_r = scratch_curve[-1]["mean_reward"] if scratch_curve else -500.0
            scratch_rewards.append(final_r)
            print(f"  {tgt_name} seed {seed}: {final_r:.1f}")

        scratch_baselines[tgt_name] = {
            "mean": float(np.mean(scratch_rewards)),
            "std": float(np.std(scratch_rewards)),
            "values": scratch_rewards,
        }

    # Compute improvement + significance for each pair
    for pair_key, pair_data in transfer_matrix.items():
        tgt = pair_data["target"]
        scratch = scratch_baselines[tgt]
        ft_vals = [r["final_reward"] for r in pair_data["seed_results"]]
        scratch_vals = scratch["values"]

        if len(ft_vals) >= 2 and len(scratch_vals) >= 2:
            t_stat, p_val = sp_stats.ttest_ind(ft_vals, scratch_vals)
        else:
            t_stat, p_val = 0.0, 1.0

        if abs(scratch["mean"]) > 0.01:
            improvement_pct = ((pair_data["fine_tune_mean"] - scratch["mean"])
                              / abs(scratch["mean"])) * 100
        else:
            improvement_pct = 0.0

        pair_data["improvement_pct"] = improvement_pct
        pair_data["scratch_baseline"] = scratch
        pair_data["significance"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        }

    # Save
    ensure_dir("results")
    output_path = "results/transfer_matrix_tuned.json"
    tuned_results = {
        "transfer_matrix": transfer_matrix,
        "scratch_baselines": scratch_baselines,
        "config": {
            "n_seeds": n_seeds,
            "n_generations": n_generations,
            "lr_options": lr_options,
            "ent_coef": 0.02,
        },
        "seeds": seeds,
    }

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(tuned_results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    # Summary with significance stars
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] Tuned Transfer Matrix Summary")
    print(f"[{ts}] {'Source':<12} {'Target':<12} {'Fine-Tune':>12} {'Scratch':>10} {'Improv':>8} {'p':>7}")
    print(f"[{ts}] {'-'*65}")
    for pair_key, data in transfer_matrix.items():
        sig = "*" if data.get("significance", {}).get("significant", False) else ""
        print(f"[{ts}] {data['source']:<12} {data['target']:<12} "
              f"{data['fine_tune_mean']:>8.1f}+/-{data['fine_tune_std']:.1f} "
              f"{data['scratch_baseline']['mean']:>8.1f}+/-{data['scratch_baseline']['std']:.1f} "
              f"{data['improvement_pct']:>7.1f}% "
              f"{data.get('significance', {}).get('p_value', 1.0):>6.4f}{sig}")

    # Visualization: 3x3 heatmap with significance stars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_dir("results/figures")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n = len(domain_names)

        # Improvement heatmap
        ax = axes[0]
        imp_matrix = np.zeros((n, n))
        for i, src in enumerate(domain_names):
            for j, tgt in enumerate(domain_names):
                key = f"{src}_to_{tgt}"
                if key in transfer_matrix:
                    imp_matrix[i, j] = transfer_matrix[key]["improvement_pct"]

        im = ax.imshow(imp_matrix, cmap="RdYlGn", vmin=-50, vmax=50)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(domain_names, fontsize=10)
        ax.set_yticklabels(domain_names, fontsize=10)
        ax.set_title("Transfer Improvement (%) - Tuned", fontsize=12)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        for i in range(n):
            for j in range(n):
                key = f"{domain_names[i]}_to_{domain_names[j]}"
                sig = ""
                if key in transfer_matrix:
                    if transfer_matrix[key].get("significance", {}).get("significant"):
                        sig = "*"
                ax.text(j, i, f"{imp_matrix[i,j]:.1f}%{sig}",
                        ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Alignment heatmap
        ax = axes[1]
        sim_matrix = np.zeros((n, n))
        for i, src in enumerate(domain_names):
            for j, tgt in enumerate(domain_names):
                key = f"{src}_to_{tgt}"
                if key in transfer_matrix:
                    sim_matrix[i, j] = transfer_matrix[key]["alignment_sim"]

        im2 = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(domain_names, fontsize=10)
        ax.set_yticklabels(domain_names, fontsize=10)
        ax.set_title("Alignment Similarity", fontsize=12)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{sim_matrix[i,j]:.3f}",
                        ha="center", va="center", fontsize=9)
        plt.colorbar(im2, ax=ax, shrink=0.8)

        plt.tight_layout()
        fig_path = "results/figures/transfer_matrix_tuned.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    print(f"\nDone!")
    return tuned_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Domain Concept Transfer")
    parser.add_argument("--matrix", action="store_true",
                        help="Run full 3x3 transfer matrix with multi-seed")
    parser.add_argument("--tuned", action="store_true",
                        help="Run tuned transfer with LR sweep and 5 seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds to use")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for tuned mode (default: 5)")
    parser.add_argument("--n-generations", type=int, default=100,
                        help="Generations for tuned mode (default: 100)")
    args = parser.parse_args()

    if args.tuned:
        run_tuned_cross_domain(n_seeds=args.n_seeds, n_generations=args.n_generations)
    elif args.matrix:
        run_multi_seed_cross_domain(seeds=args.seeds)
    else:
        run_cross_domain_transfer()
