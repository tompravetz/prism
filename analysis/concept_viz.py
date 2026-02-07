"""
Concept Visualization — The "wow factor" figures.

Generates visually impressive visualizations of the learned concepts:

1. t-SNE Cluster Map: 128D encoder features projected to 2D, colored by
   concept assignment. Clear clusters = concepts capture structure.

2. Example Boards Per Concept: Shows 4 representative Go boards for each
   of the top 10 concepts. Lets humans see what each concept "means."

3. Concept Transition Graph: Directed graph of (concept, action) → next_concept.
   Edge thickness = frequency, color = win rate. Shows the agent's "strategy flow."

4. Concept Action Heatmap: For each concept, what action does the policy prefer?
   Shows the full concept→action mapping in one image.

Usage:
    python analysis/concept_viz.py --algo ppo
    python analysis/concept_viz.py --algo both
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.utils import get_device, ensure_dir, set_seed, render_go_board_ascii


def collect_features_and_concepts(encoder, concept_manager, env,
                                   n_episodes=300, device=None):
    """
    Collect encoder features, concept assignments, and board states.

    Runs many episodes, recording the 128D feature vector, assigned concept,
    and raw observation at each step. Used for t-SNE and board examples.

    Returns:
        features: (N, 128) array of encoder features
        concepts: (N,) array of concept IDs
        observations: list of (7,7,3) arrays (board states)
    """
    device = device or get_device()
    encoder.eval()

    features_list = []
    concepts_list = []
    observations_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # Get features from encoder
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                feat = encoder(obs_t).cpu().numpy()[0]

            # Get concept
            concept_id = concept_manager.assign_concept_from_obs(
                encoder, obs, device
            )

            features_list.append(feat)
            concepts_list.append(concept_id)
            observations_list.append(obs.copy())

            # Take random action to explore diverse states
            mask = info.get("action_mask", np.ones(env.action_count, dtype=np.int8))
            legal = np.where(mask == 1)[0]
            action = int(np.random.choice(legal))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return (np.array(features_list), np.array(concepts_list), observations_list)


def plot_tsne_clusters(features, concepts, algo="ppo", save_dir="results/figures"):
    """
    Plot t-SNE visualization of concept clusters.

    Projects 128D encoder features to 2D using t-SNE, then colors each
    point by its concept assignment. Well-separated clusters indicate
    that concepts capture meaningful structure in the feature space.

    This is one of the most visually striking figures for the paper:
    it shows the discrete concept space in a way humans can understand.
    """
    ensure_dir(save_dir)

    print(f"  Computing t-SNE (n={len(features)})...")
    # Subsample if too many points (t-SNE is O(n²))
    max_points = 5000
    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features_sub = features[idx]
        concepts_sub = concepts[idx]
    else:
        features_sub = features
        concepts_sub = concepts

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(features_sub)

    # Get unique concepts and assign colors
    unique_concepts = np.unique(concepts_sub)
    n_unique = len(unique_concepts)

    # Use a colormap with enough distinct colors
    if n_unique <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.nipy_spectral

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i, c in enumerate(unique_concepts):
        mask = concepts_sub == c
        color = cmap(i / max(n_unique - 1, 1))
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color], s=8, alpha=0.6, label=f"C{c}" if n_unique <= 20 else "")

    ax.set_title(f"t-SNE of Encoder Features — {algo.upper()} ({n_unique} concepts)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)

    if n_unique <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3, fontsize=8)

    # Add annotation
    ax.text(0.02, 0.02,
            f"n={len(features_sub)} states, {n_unique} concepts\n"
            f"Clear clusters = concepts capture meaningful structure",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(save_dir, f"tsne_concepts_{algo}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_example_boards(concepts, observations, concept_manager,
                        algo="ppo", top_k=10, examples_per_concept=4,
                        save_dir="results/figures"):
    """
    Show example Go boards for the most common concepts.

    For each of the top-K most frequent concepts, displays 4 example
    board states. This lets humans inspect what each concept "looks like."

    Concepts should group strategically similar positions:
        - Opening concepts: few stones, scattered placement
        - Middle game: groups forming, territories emerging
        - Endgame: full board, territories defined
    """
    ensure_dir(save_dir)

    # Find most frequent concepts
    from collections import Counter
    concept_counts = Counter(concepts)
    top_concepts = [c for c, _ in concept_counts.most_common(top_k)]

    fig, axes = plt.subplots(top_k, examples_per_concept,
                              figsize=(examples_per_concept * 2.5, top_k * 2.5))

    for row, concept_id in enumerate(top_concepts):
        # Find all observations with this concept
        indices = np.where(concepts == concept_id)[0]
        # Sample examples
        n_examples = min(examples_per_concept, len(indices))
        sample_idx = np.random.choice(indices, n_examples, replace=False)

        for col in range(examples_per_concept):
            ax = axes[row, col] if top_k > 1 else axes[col]

            if col < n_examples:
                obs = observations[sample_idx[col]]
                board_size = obs.shape[0]

                # Render board as image
                board_img = np.ones((board_size, board_size, 3))  # White background
                for r in range(board_size):
                    for c in range(board_size):
                        if obs[r, c, 0] > 0.5:  # Black stone
                            board_img[r, c] = [0.1, 0.1, 0.1]
                        elif obs[r, c, 1] > 0.5:  # White stone
                            board_img[r, c] = [0.8, 0.8, 0.8]
                        else:  # Empty
                            board_img[r, c] = [0.9, 0.75, 0.5]  # Board color

                ax.imshow(board_img, interpolation='nearest')

                # Add grid lines
                for i in range(board_size):
                    ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
                    ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            else:
                ax.axis('off')
                continue

            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                count = concept_counts[concept_id]
                ax.set_ylabel(f"C{concept_id}\n(n={count})",
                             fontsize=9, fontweight='bold', rotation=0,
                             labelpad=40, va='center')

    fig.suptitle(f"Example Board States Per Concept — {algo.upper()}",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, f"concept_examples_{algo}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_concept_action_heatmap(policy, n_concepts=64, n_actions=50,
                                 algo="ppo", board_size=7,
                                 save_dir="results/figures", device=None):
    """
    Heatmap of policy action probabilities for each concept.

    X-axis: actions (0-49), Y-axis: concepts (0-63).
    Color intensity = probability of taking that action given that concept.

    A good bottleneck policy should show:
        - Different concepts → different action distributions (horizontal variation)
        - Each concept has a few preferred actions (not uniform across all)
        - Clear block structure if concepts group into strategy types
    """
    ensure_dir(save_dir)
    device = device or get_device()
    policy.eval()

    # Build the action probability matrix
    action_probs = np.zeros((n_concepts, n_actions))

    with torch.no_grad():
        for c in range(n_concepts):
            cid = torch.LongTensor([c]).to(device)
            logits, _ = policy(cid)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()
            action_probs[c] = probs

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Use log scale for better visibility (many near-zero probabilities)
    log_probs = np.log10(action_probs + 1e-10)

    sns.heatmap(log_probs, ax=ax, cmap='YlOrRd', vmin=-4, vmax=0,
                cbar_kws={'label': 'log10(probability)'})

    ax.set_xlabel("Action (0-48: board positions, 49: pass)", fontsize=11)
    ax.set_ylabel("Concept ID", fontsize=11)
    ax.set_title(f"Policy Action Probabilities by Concept — {algo.upper()}",
                 fontsize=13, fontweight='bold')

    # Mark the pass action
    ax.axvline(x=49, color='blue', linewidth=1, alpha=0.5, linestyle='--')
    ax.text(49.5, -1.5, "pass", fontsize=8, color='blue', ha='center')

    plt.tight_layout()
    path = os.path.join(save_dir, f"concept_action_heatmap_{algo}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_concept_transition_graph(encoder, concept_manager, policy, env,
                                   algo="ppo", n_episodes=200,
                                   save_dir="results/figures", device=None):
    """
    Visualize concept transitions as a directed graph.

    Collects (concept_t, action_t) → concept_{t+1} transitions from gameplay,
    then draws a graph where:
        - Nodes = concepts (sized by frequency)
        - Edges = transitions (thickness = frequency, color = associated win rate)

    This shows the "strategy flow" — how the agent moves between concepts
    during a game, revealing temporal structure in its decision-making.
    """
    ensure_dir(save_dir)
    device = device or get_device()

    # Collect transitions
    transitions = defaultdict(int)
    concept_freq = defaultdict(int)
    total_transitions = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        prev_concept = None

        while not done:
            concept_id = concept_manager.assign_concept_from_obs(
                encoder, obs, device
            )
            concept_freq[concept_id] += 1

            if prev_concept is not None:
                transitions[(prev_concept, concept_id)] += 1
                total_transitions += 1

            prev_concept = concept_id
            mask = info.get("action_mask", np.ones(env.action_count, dtype=np.int8))

            if algo == "ppo":
                action = policy.get_action(concept_id, mask, deterministic=True)
            else:
                action = policy.get_action(concept_id, mask, epsilon=0.0)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    # Find top concepts and transitions for visualization
    top_concepts = sorted(concept_freq.keys(), key=lambda c: concept_freq[c], reverse=True)[:15]
    top_concept_set = set(top_concepts)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Position concepts in a circle
    n_nodes = len(top_concepts)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = {c: (3 * np.cos(a), 3 * np.sin(a)) for c, a in zip(top_concepts, angles)}

    # Draw edges (transitions between top concepts)
    max_trans = max(transitions.values()) if transitions else 1
    for (src, dst), count in transitions.items():
        if src in top_concept_set and dst in top_concept_set and src != dst:
            freq = count / total_transitions
            if freq < 0.005:  # Skip very rare transitions
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            width = 0.5 + 4 * (count / max_trans)
            alpha = min(0.8, 0.2 + freq * 20)
            ax.annotate("",
                        xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>",
                                       connectionstyle="arc3,rad=0.15",
                                       lw=width, alpha=alpha,
                                       color='steelblue'))

    # Draw nodes
    max_freq = max(concept_freq.values())
    for c in top_concepts:
        x, y = positions[c]
        size = 300 + 1500 * (concept_freq[c] / max_freq)
        ax.scatter(x, y, s=size, c='coral', edgecolors='black',
                   linewidths=1.5, zorder=5, alpha=0.9)
        ax.text(x, y, f"C{c}", ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=6)
        # Show frequency below
        freq_pct = concept_freq[c] / sum(concept_freq.values()) * 100
        ax.text(x, y - 0.4, f"{freq_pct:.1f}%",
                ha='center', va='top', fontsize=7, color='gray')

    ax.set_title(f"Concept Transition Graph — {algo.upper()} (Top 15 Concepts)",
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    ax.text(-4.5, -4.5,
            f"Node size = concept frequency\n"
            f"Arrow thickness = transition frequency\n"
            f"Total transitions: {total_transitions:,}",
            fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, f"concept_transitions_{algo}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate concept visualizations")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    parser.add_argument("--save-dir", type=str, default="results/figures")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Generating Concept Visualizations — {algo.upper()}")
        print(f"{'='*60}")

        set_seed(args.seed)
        device = get_device()

        # Load models
        env = GoEnv(board_size=7)
        encoder = GoCNNEncoder(env.observation_space, features_dim=128)

        encoder_path = os.path.join(args.baseline_dir, f"{algo}_go_encoder.pt")
        if not os.path.exists(encoder_path):
            print(f"ERROR: Encoder not found at {encoder_path}")
            continue
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        encoder.to(device)
        encoder.eval()

        cm = ConceptManager(n_concepts=args.n_concepts)
        concept_path = os.path.join(args.model_dir, f"concepts_{algo}_k{args.n_concepts}.pkl")
        if not os.path.exists(concept_path):
            print(f"ERROR: Concepts not found at {concept_path}")
            continue
        cm.load(concept_path)

        if algo == "ppo":
            policy = ConceptBottleneckPolicy(
                n_concepts=args.n_concepts, embed_dim=64, hidden_dim=128, n_actions=50
            ).to(device)
        else:
            policy = ConceptDQNPolicy(
                n_concepts=args.n_concepts, embed_dim=64, hidden_dim=128, n_actions=50
            ).to(device)

        policy_path = os.path.join(args.model_dir, f"{algo}_bottleneck_final.pt")
        if not os.path.exists(policy_path):
            print(f"ERROR: Policy not found at {policy_path}")
            continue
        policy.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=True)
        )
        policy.eval()

        # Collect data
        print("Collecting features and concepts from gameplay...")
        features, concepts, observations = collect_features_and_concepts(
            encoder, cm, env, n_episodes=300, device=device
        )
        print(f"  Collected {len(features)} states")

        # Generate all visualizations
        print("\nGenerating t-SNE cluster plot...")
        plot_tsne_clusters(features, concepts, algo=algo, save_dir=args.save_dir)

        print("Generating example board states...")
        plot_example_boards(concepts, observations, cm,
                           algo=algo, save_dir=args.save_dir)

        print("Generating concept-action heatmap...")
        plot_concept_action_heatmap(policy, n_concepts=args.n_concepts,
                                     algo=algo, save_dir=args.save_dir, device=device)

        print("Generating concept transition graph...")
        plot_concept_transition_graph(encoder, cm, policy, env,
                                      algo=algo, save_dir=args.save_dir, device=device)

        env.close()

    print(f"\nAll visualizations saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
