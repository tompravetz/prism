"""
Cross-algorithm concept transfer on Breakout (ALE).

Tests PPO->DQN and DQN->PPO zero-shot transfer via Hungarian alignment.
Compares to: random agent, no-alignment (identity), own bottleneck.

Saves: results/atari_transfer.json

Usage:
    python experiments/atari_transfer.py
    python experiments/atari_transfer.py --n-seeds 10 --n-eval 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.preprocessing import preprocess_obs as sb3_preprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, ensure_dir
from train_atari import make_env, SAVE_DIR

N_CONCEPTS = 64
N_ACTIONS = 4      # Breakout minimal: NOOP, FIRE, RIGHT, LEFT
FEATURES_DIM = 512  # NatureCNN output


def get_encoder(model, algo):
    if algo == "ppo":
        return model.policy.features_extractor
    else:
        return model.q_net.features_extractor


def preprocess(model, obs_tensor):
    return sb3_preprocess(obs_tensor, model.observation_space,
                          normalize_images=model.policy.normalize_images)


def load_all(device):
    """Load both baselines, concept managers, and bottleneck policies."""
    agents = {}
    for algo, cls in [("ppo", PPO), ("dqn", DQN)]:
        model = cls.load(f"{SAVE_DIR}/{algo}_breakout", device=device)
        model.policy.set_training_mode(False)

        cm = ConceptManager(n_concepts=N_CONCEPTS)
        cm.load(f"{SAVE_DIR}/concepts_{algo}_k{N_CONCEPTS}.pkl")

        policy = ConceptBottleneckPolicy(
            n_concepts=N_CONCEPTS, embed_dim=64,
            hidden_dim=128, n_actions=N_ACTIONS,
        )
        policy.load_state_dict(
            torch.load(f"{SAVE_DIR}/{algo}_bottleneck.pt",
                       map_location=device, weights_only=True)
        )
        policy.to(device)
        policy.eval()

        agents[algo] = {
            "model": model,
            "encoder": get_encoder(model, algo),
            "cm": cm,
            "policy": policy,
        }
    return agents


def build_transferred_policy(src_cm, tgt_cm, source_policy, mapping, device):
    """Create the transferred policy once; reuse across evaluation seeds."""
    aligner = ConceptAligner(src_cm, tgt_cm)
    transferred = aligner.transfer_policy(
        source_policy, mapping,
        target_n_concepts=N_CONCEPTS,
        target_n_actions=N_ACTIONS,
    )
    transferred.to(device)
    transferred.eval()
    return transferred


def evaluate_transferred(transferred_policy, target_model, target_encoder,
                         target_cm, n_episodes, seed, device):
    """Evaluate a (pre-built) transferred policy on Breakout."""
    env = make_env(n_envs=1, seed=seed + 8888)
    obs = env.reset()
    ep_rewards, ep_reward = [], 0.0

    while len(ep_rewards) < n_episodes:
        with torch.no_grad():
            obs_tensor, _ = target_model.policy.obs_to_tensor(obs)
            features = target_encoder(preprocess(target_model, obs_tensor)).cpu().numpy()
            concept_id = target_cm.assign_concept(features)[0]
            action = transferred_policy.get_action(
                concept_id, action_mask=None, deterministic=True
            )
        obs, reward, done, _ = env.step([action])
        ep_reward += reward[0]
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0

    env.close()
    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "win_rate": float(np.mean([r > 0 for r in ep_rewards])),
        "rewards": [float(r) for r in ep_rewards],
    }


def evaluate_own_bottleneck(model, encoder, cm, policy,
                             n_episodes, seed, device):
    """Evaluate an agent using its own bottleneck (no transfer)."""
    policy.eval()
    env = make_env(n_envs=1, seed=seed + 6666)
    obs = env.reset()
    ep_rewards, ep_reward = [], 0.0

    while len(ep_rewards) < n_episodes:
        with torch.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            features = encoder(preprocess(model, obs_tensor)).cpu().numpy()
            concept_id = cm.assign_concept(features)[0]
            action = policy.get_action(concept_id, action_mask=None, deterministic=True)
        obs, reward, done, _ = env.step([action])
        ep_reward += reward[0]
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0

    env.close()
    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "win_rate": float(np.mean([r > 0 for r in ep_rewards])),
    }


def evaluate_random(n_episodes, seed):
    """Evaluate a random agent on Breakout."""
    env = make_env(n_envs=1, seed=seed + 3333)
    obs = env.reset()
    ep_rewards, ep_reward = [], 0.0
    rng = np.random.RandomState(seed)

    while len(ep_rewards) < n_episodes:
        action = rng.randint(0, N_ACTIONS, size=(1,))
        obs, reward, done, _ = env.step(action)
        ep_reward += reward[0]
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0

    env.close()
    return {"mean_reward": float(np.mean(ep_rewards))}


def run_transfer(n_seeds=5, n_eval=50):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] Atari (Breakout) Cross-Algorithm Transfer")
    print(f"       Seeds: {n_seeds}, Eval episodes per seed: {n_eval}")
    print(f"       Device: {device_str}")

    print("\nLoading models...")
    agents = load_all(device)
    ppo, dqn = agents["ppo"], agents["dqn"]

    pairs = [
        ("ppo", "dqn"),
        ("dqn", "ppo"),
    ]

    results = {}

    # Random baseline
    print("\nEvaluating random agent...")
    rnd = evaluate_random(n_episodes=n_eval * n_seeds, seed=0)
    results["random"] = rnd
    print(f"  Random: mean_reward={rnd['mean_reward']:.1f}")

    # Own bottleneck performance
    for algo in ("ppo", "dqn"):
        ag = agents[algo]
        print(f"\nEvaluating {algo.upper()} own bottleneck...")
        seed_rewards, seed_wrs = [], []
        for s in range(n_seeds):
            set_seed(s * 100)
            r = evaluate_own_bottleneck(
                ag["model"], ag["encoder"], ag["cm"], ag["policy"],
                n_episodes=n_eval, seed=s, device=device,
            )
            seed_rewards.append(r["mean_reward"])
            seed_wrs.append(r["win_rate"])
            print(f"    seed {s}: reward={r['mean_reward']:.1f}, wr={r['win_rate']:.0%}")
        results[f"{algo}_own"] = {
            "mean_reward": float(np.mean(seed_rewards)),
            "std_reward": float(np.std(seed_rewards)),
            "mean_wr": float(np.mean(seed_wrs)),
            "std_wr": float(np.std(seed_wrs)),
        }

    # Transfer experiments
    for src_key, tgt_key in pairs:
        src = agents[src_key]
        tgt = agents[tgt_key]
        aligner = ConceptAligner(src["cm"], tgt["cm"])

        print(f"\n--- {src_key.upper()} -> {tgt_key.upper()} transfer ---")

        # Hungarian alignment
        hungarian_mapping = aligner.hungarian_alignment()
        align_quality = aligner.alignment_quality(hungarian_mapping)
        print(f"  Alignment similarity: {align_quality['mean_similarity']:.4f}")

        # Identity mapping (no alignment)
        identity_mapping = {i: i for i in range(N_CONCEPTS)}

        # Build transferred policies once (not per seed)
        transferred_h = build_transferred_policy(
            src["cm"], tgt["cm"], src["policy"], hungarian_mapping, device
        )
        transferred_i = build_transferred_policy(
            src["cm"], tgt["cm"], src["policy"], identity_mapping, device
        )

        seed_wrs_h, seed_rewards_h = [], []
        seed_wrs_i, seed_rewards_i = [], []

        for s in range(n_seeds):
            set_seed(s * 100)

            # Hungarian
            r_h = evaluate_transferred(
                transferred_h, tgt["model"], tgt["encoder"], tgt["cm"],
                n_episodes=n_eval, seed=s, device=device,
            )
            seed_wrs_h.append(r_h["win_rate"])
            seed_rewards_h.append(r_h["mean_reward"])
            print(f"    seed {s} [Hungarian]: reward={r_h['mean_reward']:.1f}, "
                  f"wr={r_h['win_rate']:.0%}")

            # Identity (no alignment)
            r_i = evaluate_transferred(
                transferred_i, tgt["model"], tgt["encoder"], tgt["cm"],
                n_episodes=n_eval, seed=s, device=device,
            )
            seed_wrs_i.append(r_i["win_rate"])
            seed_rewards_i.append(r_i["mean_reward"])

        pair_key = f"{src_key}_to_{tgt_key}"
        results[pair_key] = {
            "alignment_sim": float(align_quality["mean_similarity"]),
            "hungarian": {
                "mean_reward": float(np.mean(seed_rewards_h)),
                "std_reward": float(np.std(seed_rewards_h)),
                "mean_wr": float(np.mean(seed_wrs_h)),
                "std_wr": float(np.std(seed_wrs_h)),
                "seed_rewards": [float(x) for x in seed_rewards_h],
            },
            "identity": {
                "mean_reward": float(np.mean(seed_rewards_i)),
                "std_reward": float(np.std(seed_rewards_i)),
                "mean_wr": float(np.mean(seed_wrs_i)),
                "std_wr": float(np.std(seed_wrs_i)),
            },
        }

    # Statistical tests (Hungarian vs identity)
    from scipy import stats
    for pair_key in [f"{s}_to_{t}" for s, t in pairs]:
        h_rewards = results[pair_key]["hungarian"]["seed_rewards"]
        # One-sample t-test: does transfer beat random?
        random_mean = results["random"]["mean_reward"]
        if n_seeds >= 2:
            t, p = stats.ttest_1samp(h_rewards, random_mean)
            results[pair_key]["hungarian"]["t_vs_random"] = float(t)
            results[pair_key]["hungarian"]["p_vs_random"] = float(p)

    ensure_dir("results")
    output_path = "results/atari_transfer.json"
    with open(output_path, "w") as f:
        json.dump({"n_seeds": n_seeds, "n_eval": n_eval,
                   "env": "ALE/Breakout-v5", "results": results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============ SUMMARY ============")
    print(f"  Random agent:        reward={results['random']['mean_reward']:.1f}")
    for algo in ("ppo", "dqn"):
        r = results[f"{algo}_own"]
        print(f"  {algo.upper()} own bottleneck:  "
              f"reward={r['mean_reward']:.1f} ± {r['std_reward']:.1f}")
    for src, tgt in pairs:
        key = f"{src}_to_{tgt}"
        h = results[key]["hungarian"]
        i = results[key]["identity"]
        print(f"  {src.upper()}->{tgt.upper()} Hungarian: "
              f"reward={h['mean_reward']:.1f} ± {h['std_reward']:.1f}")
        print(f"  {src.upper()}->{tgt.upper()} Identity:  "
              f"reward={i['mean_reward']:.1f} ± {i['std_reward']:.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari Breakout transfer experiment")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()
    run_transfer(n_seeds=args.n_seeds, n_eval=args.n_eval)
