"""
Experiment: K Ablation — Concept Count vs Transfer Performance.

Sweeps K ∈ {8, 16, 32, 64, 128} for the concept bottleneck and measures:
  - Transfer win rate: PPO→DQN zero-shot transfer vs GnuGo Level 3.
  - Concept compactness: mean within-cluster cosine variance (lower = tighter concepts).

The tradeoff curve (interpretability proxy vs performance) is Fig 4 in the paper.

Design:
  For each K:
    1. Re-cluster the PPO encoder features with K centres (fast, <30 s).
    2. Train a PPO bottleneck policy for a SHORT budget (no_curriculum=True,
       300 k steps) — enough to rank K values, not for final results.
    3. Evaluate zero-shot PPO→DQN transfer vs GnuGo Level 3 (50 games).
    4. Compute mean within-cluster cosine variance over a held-out feature set.

  The DQN encoder and concept manager are also re-clustered at each K so that
  the alignment step is meaningful.

Output:
  results/k_ablation.json
  results/figures/fig_k_tradeoff.png

Usage:
  python experiments/k_ablation.py [--quick] [--seed 42]

  --quick: 50 k steps per K (for smoke-testing; not for paper results)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir
from train_bottleneck import (
    PPOBottleneckTrainer,
    discover_concepts, evaluate_agent,
)
from experiments.eval_strong import eval_agent_vs_gnugo


K_VALUES = [8, 16, 32, 64, 128]

# Training budget per K for the sweep.  Increased in non-quick mode.
QUICK_STEPS  = 50_000      # smoke-test only
NORMAL_STEPS = 300_000     # paper sweep (no curriculum, random opponent)

EVAL_LEVEL   = 3           # GnuGo level used to rank K values
EVAL_GAMES   = 50          # games per K per direction (PPO & DQN)
FEATURES_DIM = 128


# ---------------------------------------------------------------------------
# Within-cluster cosine variance
# ---------------------------------------------------------------------------

def within_cluster_cosine_variance(features: np.ndarray, labels: np.ndarray,
                                   n_concepts: int) -> float:
    """
    Compute mean within-cluster cosine variance.

    For each cluster c:
        1. Normalize all member vectors to unit length.
        2. Compute pairwise cosine similarities.
        3. Variance = 1 - mean(pairwise_cosine_sim).
    Return mean over all non-empty clusters.

    Lower values → tighter, more interpretable concepts.
    """
    variances = []
    for c in range(n_concepts):
        members = features[labels == c]
        if len(members) < 2:
            continue
        norms = np.linalg.norm(members, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normed = members / norms
        # Mean pairwise cosine similarity via dot-product mean trick
        centroid = normed.mean(axis=0)
        mean_sim = float(np.dot(normed, centroid).mean())
        variances.append(1.0 - mean_sim)
    return float(np.mean(variances)) if variances else 1.0


def collect_holdout_features(encoder, device, n_states=2000, seed=777):
    """Collect feature vectors for evaluating compactness (independent of training)."""
    set_seed(seed)
    env = GoEnv(board_size=7)
    feats = []
    while len(feats) < n_states:
        obs, info = env.reset()
        done = False
        n_moves = np.random.randint(5, 30)
        for _ in range(n_moves):
            if done:
                break
            mask = info.get("action_mask", np.ones(50, dtype=np.int8))
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                break
            action = np.random.choice(legal)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(obs_t).cpu().numpy()[0]
            feats.append(feat)
    env.close()
    return np.array(feats[:n_states], dtype=np.float32)


# ---------------------------------------------------------------------------
# Train bottleneck for a given K without curriculum (quick sweep mode)
# ---------------------------------------------------------------------------

def train_ppo_bottleneck_for_k(encoder, cm_ppo, K, total_steps, device):
    """
    Train a PPO bottleneck policy for K concepts with a fixed step budget.

    Returns the trained PPOBottleneckTrainer (policy accessible as trainer.policy).
    """
    n_actions = 50  # Go 7x7
    trainer = PPOBottleneckTrainer(
        encoder=encoder,
        concept_manager=cm_ppo,
        n_actions=n_actions,
        n_concepts=K,
        lr=3e-4,
        gamma=0.99,
        device=device,
    )

    env = GoEnv(board_size=7)
    steps_done = 0
    steps_per_gen = min(4096, total_steps)

    while steps_done < total_steps:
        rollout = trainer.collect_rollout(env, n_steps=steps_per_gen)
        trainer.update(rollout)
        steps_done += steps_per_gen

    env.close()
    return trainer



# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_k_ablation(quick=False, seed=42, gnugo_level=EVAL_LEVEL,
                   n_eval_games=EVAL_GAMES, n_holdout_states=2000):
    set_seed(seed)
    device = get_device()
    total_steps = QUICK_STEPS if quick else NORMAL_STEPS

    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] K-Ablation sweep: K={K_VALUES}, steps={total_steps:,}, "
          f"eval=GnuGo L{gnugo_level} x{n_eval_games}")

    # --- Load frozen encoders (same for all K) ---
    env_tmp = GoEnv(board_size=7)
    ppo_encoder = GoCNNEncoder(env_tmp.observation_space, features_dim=FEATURES_DIM)
    dqn_encoder = GoCNNEncoder(env_tmp.observation_space, features_dim=FEATURES_DIM)
    env_tmp.close()

    ppo_enc_path = "models/baseline/ppo_go_encoder.pt"
    dqn_enc_path = "models/baseline/dqn_go_encoder.pt"

    if not os.path.exists(ppo_enc_path) or not os.path.exists(dqn_enc_path):
        raise FileNotFoundError(
            f"Baseline encoders not found. Run train_baseline.py first.\n"
            f"  Expected: {ppo_enc_path}\n"
            f"  Expected: {dqn_enc_path}"
        )

    ppo_encoder.load_state_dict(
        torch.load(ppo_enc_path, map_location=device, weights_only=True)
    )
    dqn_encoder.load_state_dict(
        torch.load(dqn_enc_path, map_location=device, weights_only=True)
    )
    ppo_encoder.to(device).eval()
    dqn_encoder.to(device).eval()

    # --- Collect holdout features once (shared across all K) ---
    print(f"\nCollecting {n_holdout_states} holdout features for compactness metric...")
    holdout_feats = collect_holdout_features(ppo_encoder, device,
                                             n_states=n_holdout_states)
    print(f"Holdout features: {holdout_feats.shape}")

    # --- Collect training features once, re-cluster for each K ---
    print(f"\nCollecting training features for concept discovery...")
    env_feat = GoEnv(board_size=7)
    tmp_cm = ConceptManager(n_concepts=64, features_dim=FEATURES_DIM)
    tmp_cm.collect_features(ppo_encoder, env_feat, n_episodes=300, device=device)
    dqn_tmp_cm = ConceptManager(n_concepts=64, features_dim=FEATURES_DIM)
    dqn_tmp_cm.collect_features(dqn_encoder, env_feat, n_episodes=300, device=device)
    env_feat.close()

    ppo_raw_feats = np.concatenate(tmp_cm._collected_features, axis=0)
    dqn_raw_feats = np.concatenate(dqn_tmp_cm._collected_features, axis=0)
    print(f"Collected {len(ppo_raw_feats)} PPO features, "
          f"{len(dqn_raw_feats)} DQN features")

    # --- Sweep ---
    results = {}
    ensure_dir("results/figures")

    for K in K_VALUES:
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] ====== K={K} ======")

        # 1. Cluster at this K
        cm_ppo = ConceptManager(n_concepts=K, features_dim=FEATURES_DIM)
        cm_ppo._collected_features = [ppo_raw_feats]
        cm_ppo.n_samples_collected = len(ppo_raw_feats)
        cm_ppo.fit()

        cm_dqn = ConceptManager(n_concepts=K, features_dim=FEATURES_DIM)
        cm_dqn._collected_features = [dqn_raw_feats]
        cm_dqn.n_samples_collected = len(dqn_raw_feats)
        cm_dqn.fit()

        # 2. Compactness on holdout features
        holdout_labels = cm_ppo.assign_concept(holdout_feats)
        compactness = within_cluster_cosine_variance(holdout_feats, holdout_labels, K)
        print(f"  Compactness (within-cluster cosine var): {compactness:.4f}")

        # 3. Train PPO bottleneck (the source policy for transfer)
        print(f"  Training PPO bottleneck ({total_steps:,} steps)...")
        t0 = time.time()
        ppo_trainer = train_ppo_bottleneck_for_k(
            ppo_encoder, cm_ppo, K, total_steps, device
        )
        print(f"  PPO trained in {time.time()-t0:.0f}s")

        # 5. PPO direct win rate vs GnuGo (own encoder + concepts)
        ppo_policy = ppo_trainer.policy
        ppo_policy.eval()

        def ppo_direct_fn(obs, mask):
            c = cm_ppo.assign_concept_from_obs(ppo_encoder, obs, device)
            return ppo_policy.get_action(c, mask, deterministic=True)

        ppo_direct = eval_agent_vs_gnugo(
            ppo_direct_fn, ppo_encoder, cm_ppo,
            gnugo_level=gnugo_level, n_games=n_eval_games, device=device,
        )
        print(f"  PPO direct WR vs L{gnugo_level}: {ppo_direct['win_rate']:.1%}")

        # 6. PPO→DQN zero-shot transfer
        aligner = ConceptAligner(cm_ppo, cm_dqn)
        mapping = aligner.hungarian_alignment()
        transferred = aligner.transfer_policy(
            ppo_policy, mapping,
            target_n_concepts=K, target_n_actions=50,
        )
        transferred.to(device).eval()

        def transfer_fn(obs, mask):
            c = cm_dqn.assign_concept_from_obs(dqn_encoder, obs, device)
            return transferred.get_action(c, mask, deterministic=True)

        transfer_result = eval_agent_vs_gnugo(
            transfer_fn, dqn_encoder, cm_dqn,
            gnugo_level=gnugo_level, n_games=n_eval_games, device=device,
        )
        print(f"  PPO→DQN transfer WR vs L{gnugo_level}: "
              f"{transfer_result['win_rate']:.1%}")

        results[K] = {
            "K": K,
            "compactness": round(compactness, 4),
            "ppo_direct_win_rate": round(ppo_direct["win_rate"], 4),
            "transfer_win_rate": round(transfer_result["win_rate"], 4),
            "ppo_direct_wins": ppo_direct["wins"],
            "transfer_wins": transfer_result["wins"],
            "n_eval_games": n_eval_games,
            "gnugo_level": gnugo_level,
            "training_steps": total_steps,
        }

    # --- Save results ---
    out_path = "results/k_ablation.json"
    with open(out_path, "w") as f:
        json.dump({
            "sweep": list(results.values()),
            "K_values": K_VALUES,
            "gnugo_level": gnugo_level,
            "n_eval_games": n_eval_games,
            "training_steps": total_steps,
            "seed": seed,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Plot ---
    _plot_k_tradeoff(results, gnugo_level)

    return results


def _plot_k_tradeoff(results, gnugo_level):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OI_BLUE   = "#0072B2"
        OI_ORANGE = "#E69F00"

        ks           = [results[K]["K"]                 for K in K_VALUES if K in results]
        compactness  = [results[K]["compactness"]        for K in K_VALUES if K in results]
        transfer_wrs = [results[K]["transfer_win_rate"]  for K in K_VALUES if K in results]
        ppo_wrs      = [results[K]["ppo_direct_win_rate"] for K in K_VALUES if K in results]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()

        l1, = ax1.plot(ks, compactness, "o-", color=OI_ORANGE, linewidth=2,
                       markersize=7, label="Compactness (within-cluster var)")
        l2, = ax2.plot(ks, transfer_wrs, "s-", color=OI_BLUE, linewidth=2,
                       markersize=7, label=f"PPO→DQN transfer WR (GnuGo L{gnugo_level})")
        ax2.plot(ks, ppo_wrs, "^--", color=OI_BLUE, linewidth=1.5, markersize=6,
                 alpha=0.5, label="PPO direct WR")

        ax1.set_xlabel("Number of concepts (K)", fontsize=12)
        ax1.set_ylabel("Within-cluster cosine variance\n(lower = tighter concepts)",
                       color=OI_ORANGE, fontsize=11)
        ax2.set_ylabel(f"Win rate vs GnuGo L{gnugo_level}", color=OI_BLUE, fontsize=11)
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(ks)
        ax1.set_xticklabels([str(k) for k in ks])
        ax1.tick_params(axis="y", labelcolor=OI_ORANGE)
        ax2.tick_params(axis="y", labelcolor=OI_BLUE)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)
        ax1.set_title("K tradeoff: Interpretability vs Transfer Performance",
                      fontsize=12)
        plt.tight_layout()

        fig_path = "results/figures/fig_k_tradeoff.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {fig_path}")
    except Exception as e:
        print(f"Warning: Could not generate K-tradeoff figure: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K ablation: concept count sweep")
    parser.add_argument("--quick", action="store_true",
                        help=f"Smoke-test mode ({QUICK_STEPS:,} steps/K instead of "
                             f"{NORMAL_STEPS:,})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=int, default=EVAL_LEVEL,
                        help=f"GnuGo level for evaluation (default: {EVAL_LEVEL})")
    parser.add_argument("--games", type=int, default=EVAL_GAMES,
                        help=f"Games per K evaluation (default: {EVAL_GAMES})")
    args = parser.parse_args()

    run_k_ablation(
        quick=args.quick,
        seed=args.seed,
        gnugo_level=args.level,
        n_eval_games=args.games,
    )
