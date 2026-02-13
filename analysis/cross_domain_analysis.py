"""
Cross-Domain Transfer Analysis: Why Zero-Shot Fails.

Provides deeper analysis of cross-domain concept transfer to explain:
    - Why alignment similarity is moderate (~0.41) but zero-shot fails (-500)
    - What structural differences exist between concept spaces
    - t-SNE visualization of features colored by domain

This generates a figure for the paper showing:
    Panel A: Feature space overlap (t-SNE of CartPole + Acrobot features)
    Panel B: Per-concept alignment quality (sorted by similarity)
    Panel C: Action distribution comparison for aligned concept pairs

Usage:
    python analysis/cross_domain_analysis.py
"""

import os
import sys
import json
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.simple_env import CartPoleConceptEnv, LunarLanderConceptEnv
from src.environments.acrobot_env import AcrobotConceptEnv
from src.networks import SimpleMLPEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir


def collect_features(encoder, env, n_episodes=200, device=None):
    """
    Collect encoder features from environment rollouts.

    Args:
        encoder: Trained encoder.
        env: Environment.
        n_episodes: Number of episodes to collect.
        device: Torch device.

    Returns:
        (features, obs_raw) — features shape (N, 128), obs_raw shape (N, obs_dim).
    """
    device = device or get_device()
    features = []
    obs_raw = []

    encoder.eval()
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs_raw.append(obs.copy())

            # Get encoder features
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                feat = encoder(obs_t).cpu().numpy()[0]
            features.append(feat)

            # Random action to explore state space
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return np.array(features), np.array(obs_raw)


def analyze_alignment_details(source_cm, target_cm, source_policy, target_policy):
    """
    Analyze per-concept alignment details between two concept spaces.

    For each aligned concept pair, compare:
        - Centroid similarity
        - Action distribution divergence

    Args:
        source_cm: Source concept manager.
        target_cm: Target concept manager.
        source_policy: Source bottleneck policy.
        target_policy: Target bottleneck policy.

    Returns:
        List of per-pair analysis dicts.
    """
    aligner = ConceptAligner(source_cm, target_cm)
    mapping = aligner.hungarian_alignment()
    details = aligner.get_concept_mapping_details(mapping)

    # Add action distribution analysis
    source_policy.eval()
    target_policy.eval()

    for detail in details:
        src_id = detail["source_id"]
        tgt_id = detail["target_id"]

        # Get action distributions from each policy for their respective concepts
        with torch.no_grad():
            src_cid = torch.LongTensor([src_id])
            tgt_cid = torch.LongTensor([tgt_id])

            src_logits, _ = source_policy(src_cid)
            src_probs = torch.softmax(src_logits[0], dim=-1).numpy()

            tgt_logits, _ = target_policy(tgt_cid)
            tgt_probs = torch.softmax(tgt_logits[0], dim=-1).numpy()

        detail["source_action_probs"] = src_probs.tolist()
        detail["target_action_probs"] = tgt_probs.tolist()

        # Entropy of each distribution
        detail["source_entropy"] = float(-np.sum(src_probs * np.log(src_probs + 1e-8)))
        detail["target_entropy"] = float(-np.sum(tgt_probs * np.log(tgt_probs + 1e-8)))

    return details


def run_cross_domain_analysis():
    """Run the full cross-domain transfer analysis."""
    set_seed(42)
    device = get_device()

    print("============================================================")
    print("Cross-Domain Transfer Analysis")
    print("============================================================")

    # Load CartPole agent
    print("\n--- Loading CartPole ---")
    cp_env = CartPoleConceptEnv()
    cp_encoder = SimpleMLPEncoder(cp_env.observation_space, features_dim=128)
    cp_encoder.load_state_dict(
        torch.load("models/simple/ppo_cartpole_encoder.pt",
                    map_location=device, weights_only=True)
    )
    cp_encoder.to(device)
    cp_encoder.eval()

    cp_cm = ConceptManager(n_concepts=32)
    cp_cm.load("models/simple/concepts_cartpole_k32.pkl")

    cp_policy = ConceptBottleneckPolicy(
        n_concepts=32, embed_dim=32, hidden_dim=64, n_actions=2,
    )
    cp_policy.load_state_dict(
        torch.load("models/simple/ppo_cartpole_bottleneck.pt",
                    map_location=device, weights_only=True)
    )

    # Load Acrobot agent
    print("--- Loading Acrobot ---")
    ac_env = AcrobotConceptEnv()
    ac_encoder = SimpleMLPEncoder(ac_env.observation_space, features_dim=128)
    ac_encoder_path = "models/acrobot/acrobot_encoder.pt"
    if os.path.exists(ac_encoder_path):
        sd = torch.load(ac_encoder_path, map_location=device, weights_only=True)
        try:
            ac_encoder.load_state_dict(sd)
        except RuntimeError:
            first_hidden = sd["net.0.weight"].shape[0]
            obs_dim = sd["net.0.weight"].shape[1]
            import torch.nn as tnn
            ac_encoder.net = tnn.Sequential(
                tnn.Linear(obs_dim, first_hidden),
                tnn.ReLU(),
                tnn.Linear(first_hidden, 128),
                tnn.ReLU(),
                tnn.Linear(128, 128),
                tnn.ReLU(),
            )
            ac_encoder.load_state_dict(sd)
    ac_encoder.to(device)
    ac_encoder.eval()

    ac_cm = ConceptManager(n_concepts=32)
    ac_cm_path = "models/acrobot/concepts_k32.pkl"
    if os.path.exists(ac_cm_path):
        ac_cm.load(ac_cm_path)

    ac_policy = ConceptBottleneckPolicy(
        n_concepts=32, embed_dim=32, hidden_dim=64, n_actions=3,
    )
    ac_policy_path = "models/acrobot/bottleneck_scratch.pt"
    if os.path.exists(ac_policy_path):
        ac_policy.load_state_dict(
            torch.load(ac_policy_path, map_location=device, weights_only=True)
        )

    # Collect features from both domains
    print("\n--- Collecting features ---")
    cp_features, cp_obs = collect_features(cp_encoder, cp_env, n_episodes=200, device=device)
    ac_features, ac_obs = collect_features(ac_encoder, ac_env, n_episodes=200, device=device)
    print(f"  CartPole: {len(cp_features)} samples, obs_dim={cp_obs.shape[1]}")
    print(f"  Acrobot:  {len(ac_features)} samples, obs_dim={ac_obs.shape[1]}")
    cp_env.close()
    ac_env.close()

    # Feature space statistics
    print("\n--- Feature Space Statistics ---")
    cp_norms = np.linalg.norm(cp_features, axis=1)
    ac_norms = np.linalg.norm(ac_features, axis=1)
    print(f"  CartPole feature norms: {cp_norms.mean():.2f} +/- {cp_norms.std():.2f}")
    print(f"  Acrobot  feature norms: {ac_norms.mean():.2f} +/- {ac_norms.std():.2f}")

    # Per-dimension comparison
    cp_mean = cp_features.mean(axis=0)
    ac_mean = ac_features.mean(axis=0)
    cp_std = cp_features.std(axis=0)
    ac_std = ac_features.std(axis=0)

    # Cosine similarity of mean feature vectors
    cos_sim = np.dot(cp_mean, ac_mean) / (np.linalg.norm(cp_mean) * np.linalg.norm(ac_mean) + 1e-8)
    print(f"  Mean feature cosine similarity: {cos_sim:.4f}")

    # Active dimensions (>0.01 mean activation)
    cp_active = np.sum(cp_mean > 0.01)
    ac_active = np.sum(ac_mean > 0.01)
    overlap = np.sum((cp_mean > 0.01) & (ac_mean > 0.01))
    print(f"  Active dims: CartPole={cp_active}/128, Acrobot={ac_active}/128, overlap={overlap}")

    # Alignment analysis
    print("\n--- Alignment Analysis ---")
    aligner = ConceptAligner(cp_cm, ac_cm)
    mapping = aligner.hungarian_alignment()
    quality = aligner.alignment_quality(mapping)
    print(f"  Mean alignment similarity: {quality['mean_similarity']:.4f}")
    print(f"  Min alignment similarity: {quality['min_similarity']:.4f}")
    print(f"  Max alignment similarity: {quality['max_similarity']:.4f}")

    # Per-concept alignment details
    details = analyze_alignment_details(cp_cm, ac_cm, cp_policy, ac_policy)

    results = {
        "feature_stats": {
            "cartpole_norm_mean": float(cp_norms.mean()),
            "cartpole_norm_std": float(cp_norms.std()),
            "acrobot_norm_mean": float(ac_norms.mean()),
            "acrobot_norm_std": float(ac_norms.std()),
            "mean_cosine_sim": float(cos_sim),
            "cartpole_active_dims": int(cp_active),
            "acrobot_active_dims": int(ac_active),
            "dim_overlap": int(overlap),
        },
        "alignment": {
            "mean_similarity": quality["mean_similarity"],
            "per_pair_similarities": quality["similarity_distribution"],
        },
    }

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        ensure_dir("results/figures")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel A: t-SNE of combined feature spaces
        ax = axes[0]
        # Subsample for speed
        n_sub = min(2000, len(cp_features), len(ac_features))
        cp_sub = cp_features[np.random.choice(len(cp_features), n_sub, replace=False)]
        ac_sub = ac_features[np.random.choice(len(ac_features), n_sub, replace=False)]
        combined = np.vstack([cp_sub, ac_sub])
        labels = ["CartPole"] * n_sub + ["Acrobot"] * n_sub

        print("\n--- Running t-SNE (this may take a minute) ---")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embedded = tsne.fit_transform(combined)

        cp_emb = embedded[:n_sub]
        ac_emb = embedded[n_sub:]

        ax.scatter(cp_emb[:, 0], cp_emb[:, 1], c="#2196F3", alpha=0.3, s=5, label="CartPole")
        ax.scatter(ac_emb[:, 0], ac_emb[:, 1], c="#F44336", alpha=0.3, s=5, label="Acrobot")

        # Plot centroids
        cp_centroids_2d = []
        ac_centroids_2d = []
        for i in range(cp_cm.n_concepts):
            # Find nearest sample to each centroid
            dists = np.linalg.norm(cp_sub - cp_cm.cluster_centers[i], axis=1)
            nearest = np.argmin(dists)
            cp_centroids_2d.append(cp_emb[nearest])
        for i in range(ac_cm.n_concepts):
            dists = np.linalg.norm(ac_sub - ac_cm.cluster_centers[i], axis=1)
            nearest = np.argmin(dists)
            ac_centroids_2d.append(ac_emb[nearest])

        cp_c2d = np.array(cp_centroids_2d)
        ac_c2d = np.array(ac_centroids_2d)
        ax.scatter(cp_c2d[:, 0], cp_c2d[:, 1], c="#1565C0", marker="^", s=50, edgecolors="black", linewidths=0.5, zorder=5)
        ax.scatter(ac_c2d[:, 0], ac_c2d[:, 1], c="#C62828", marker="v", s=50, edgecolors="black", linewidths=0.5, zorder=5)

        ax.set_title("Feature Space Overlap (t-SNE)", fontsize=12)
        ax.legend(fontsize=10, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel B: Per-concept alignment quality (sorted)
        ax = axes[1]
        sims = sorted(quality["similarity_distribution"], reverse=True)
        ax.bar(range(len(sims)), sims, color="#4CAF50", edgecolor="black", linewidth=0.3)
        ax.axhline(y=quality["mean_similarity"], color="red", linestyle="--",
                   label=f"Mean = {quality['mean_similarity']:.3f}")
        ax.set_xlabel("Concept Pair (sorted)", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title("Per-Concept Alignment Quality", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)

        # Panel C: Action space mismatch illustration
        ax = axes[2]
        # Show entropy comparison for aligned pairs
        src_entropies = [d["source_entropy"] for d in details]
        tgt_entropies = [d["target_entropy"] for d in details]
        pair_sims = [d["similarity"] for d in details]

        ax.scatter(src_entropies, tgt_entropies, c=pair_sims, cmap="RdYlGn",
                  s=50, edgecolors="black", linewidths=0.3, vmin=0, vmax=1)
        ax.set_xlabel("Source Action Entropy (CartPole, 2 actions)", fontsize=10)
        ax.set_ylabel("Target Action Entropy (Acrobot, 3 actions)", fontsize=10)
        ax.set_title("Action Entropy: Aligned Concept Pairs", fontsize=12)
        cb = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
        cb.set_label("Alignment Similarity")

        # Add diagonal reference
        max_ent = max(max(src_entropies), max(tgt_entropies))
        ax.plot([0, max_ent], [0, max_ent], "k--", alpha=0.3)

        plt.tight_layout()
        fig_path = "results/figures/cross_domain_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved analysis figure to {fig_path}")

    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    ensure_dir("results")
    output_path = "results/cross_domain_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")

    print("\nDone!")
    return results


if __name__ == "__main__":
    run_cross_domain_analysis()
