"""
Mutual Information Analysis Between Agent Concept Spaces.

Computes empirical mutual information I(C^A; C^B) between all pairs of concept
managers in the PRISM project. This quantifies how much knowing one agent's
concept assignment tells you about another agent's concept assignment for the
SAME observation.

The key idea: if two agents discover similar concept structures, then their
concept assignments should be highly correlated, yielding high MI. If their
concept spaces are unrelated, MI will be low. We then correlate MI with
actual transfer success (zero-shot win rate) to test whether MI predicts
transferability.

Methodology:
    1. Collect a shared pool of ~5000 Go observations via random play.
    2. For each agent pair (A, B), encode the SAME observations through both
       pipelines: obs -> encoder_A -> features_A -> km_A.predict -> concept_A
                  obs -> encoder_B -> features_B -> km_B.predict -> concept_B
    3. Compute the joint distribution P(C^A, C^B) from paired concept labels.
    4. Calculate MI = H(C^A) + H(C^B) - H(C^A, C^B) in bits.
    5. Also compute NMI (normalized MI) which is scale-invariant.
    6. Scatter plot MI vs transfer win rate, with Pearson correlation + p-value.

Usage:
    python analysis/mutual_information.py
"""

import os
import sys
import json
import time
import numpy as np
import torch

# ============================================================
# Add project root to Python path so we can import src modules.
# This is needed because scripts in analysis/ are one level down.
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.environments.go_env import MaskedGoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.utils import set_seed, get_device, ensure_dir


# ============================================================
# Agent configurations: encoder + concept manager paths.
# These match the agents used in transfer_same_task experiments.
# ============================================================

AGENTS = {
    "PPO": {
        "encoder_path": os.path.join("models", "baseline", "ppo_go_encoder.pt"),
        "concepts_path": os.path.join("models", "bottleneck", "concepts_ppo_k64.pkl"),
        "n_concepts": 64,
    },
    "DQN": {
        "encoder_path": os.path.join("models", "baseline", "dqn_go_encoder.pt"),
        "concepts_path": os.path.join("models", "bottleneck", "concepts_dqn_k64.pkl"),
        "n_concepts": 64,
    },
    "DAgger": {
        "encoder_path": os.path.join("models", "cloned_dagger", "ppo_go_encoder.pt"),
        "concepts_path": os.path.join("models", "bottleneck_dagger", "concepts_ppo_k64.pkl"),
        "n_concepts": 64,
    },
}


def collect_observations(env, n_target=5000, seed=42):
    """
    Collect observations from the Go environment by playing random games.

    We use random play to get a diverse set of board states across all game
    phases (opening, middlegame, endgame). Random play tends to produce varied
    board configurations, which is desirable for measuring concept agreement
    across the full state space.

    Args:
        env: MaskedGoEnv instance.
        n_target: Approximate number of observations to collect.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (N, 7, 7, 3) containing the collected observations.
    """
    np.random.seed(seed)
    observations = []
    n_episodes = 0

    print(f"  Collecting ~{n_target} observations via random play...")
    while len(observations) < n_target:
        obs, info = env.reset()
        done = False
        while not done:
            observations.append(obs.copy())
            if len(observations) >= n_target:
                break

            # Pick a random legal action
            mask = info.get("action_mask", None)
            if mask is not None:
                legal = np.where(mask == 1)[0]
                action = np.random.choice(legal) if len(legal) > 0 else 0
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        n_episodes += 1

    observations = np.array(observations, dtype=np.float32)
    print(f"  Collected {len(observations)} observations from {n_episodes} episodes")
    return observations


def load_encoder(agent_name, config, device):
    """
    Load a trained GoCNNEncoder from disk.

    The encoder transforms (B, 7, 7, 3) observations into (B, 128) feature
    vectors. GoCNNEncoder internally handles the channels-last to channels-first
    permutation, so we pass observations as (N, 7, 7, 3) directly.

    Args:
        agent_name: Name for logging (e.g., "PPO").
        config: Dict with "encoder_path" key.
        device: torch.device.

    Returns:
        Loaded GoCNNEncoder in eval mode.
    """
    # Create encoder with default 7x7 Go observation space
    from gymnasium import spaces
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(7, 7, 3), dtype=np.float32)
    encoder = GoCNNEncoder(obs_space, features_dim=128)

    # Load trained weights
    encoder_path = config["encoder_path"]
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    state_dict = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()

    print(f"  Loaded {agent_name} encoder from {encoder_path}")
    return encoder


def load_concept_manager(agent_name, config):
    """
    Load a fitted ConceptManager (K-Means) from disk.

    The concept manager might be saved as:
        (a) A ConceptManager object (with .kmeans attribute) -- loaded via .load()
        (b) A raw sklearn KMeans object -- loaded via joblib/pickle

    We handle both cases gracefully.

    Args:
        agent_name: Name for logging.
        config: Dict with "concepts_path" and "n_concepts" keys.

    Returns:
        Loaded ConceptManager with a working .assign_concept() method.
    """
    concepts_path = config["concepts_path"]
    if not os.path.exists(concepts_path):
        raise FileNotFoundError(f"Concept manager not found: {concepts_path}")

    n_concepts = config["n_concepts"]

    # Try loading as a ConceptManager first (the standard format in this project)
    cm = ConceptManager(n_concepts=n_concepts)
    try:
        cm.load(concepts_path)
        print(f"  Loaded {agent_name} concept manager from {concepts_path} "
              f"({cm.n_concepts} concepts)")
        return cm
    except Exception:
        pass

    # Fallback: try loading as a raw KMeans object via pickle/joblib
    import pickle
    try:
        import joblib
        raw = joblib.load(concepts_path)
    except Exception:
        with open(concepts_path, "rb") as f:
            raw = pickle.load(f)

    # If it's a raw KMeans, wrap it in a ConceptManager
    if hasattr(raw, "predict") and hasattr(raw, "cluster_centers_"):
        cm.kmeans = raw
        cm.cluster_centers = raw.cluster_centers_.copy()
        cm.is_fitted = True
        cm.n_concepts = raw.n_clusters
        cm.features_dim = raw.cluster_centers_.shape[1]
        print(f"  Loaded {agent_name} concept manager (raw KMeans) from {concepts_path} "
              f"({cm.n_concepts} concepts)")
        return cm

    # If it's a dict-like ConceptManager save, extract manually
    if isinstance(raw, dict) and "cluster_centers" in raw:
        cm.n_concepts = raw["n_concepts"]
        cm.features_dim = raw["features_dim"]
        cm.cluster_centers = raw["cluster_centers"]
        cm.is_fitted = raw.get("is_fitted", True)
        # Re-initialize KMeans internal state
        from sklearn.cluster import MiniBatchKMeans
        cm.kmeans = MiniBatchKMeans(n_clusters=cm.n_concepts, n_init=1,
                                    init=cm.cluster_centers)
        cm.kmeans.cluster_centers_ = cm.cluster_centers
        cm.kmeans.n_features_in_ = cm.features_dim
        cm.kmeans._n_threads = 1
        print(f"  Loaded {agent_name} concept manager (dict) from {concepts_path} "
              f"({cm.n_concepts} concepts)")
        return cm

    raise ValueError(f"Could not load concept manager from {concepts_path}: "
                     f"unknown format {type(raw)}")


def encode_observations(encoder, observations, device, batch_size=256):
    """
    Encode a batch of observations through an encoder to get feature vectors.

    Processes observations in mini-batches to avoid GPU/CPU memory issues
    for large observation sets.

    IMPORTANT: GoCNNEncoder.forward() internally permutes (B,H,W,C)->(B,C,H,W),
    so we pass observations as (N,7,7,3) directly -- do NOT pre-transpose.

    Args:
        encoder: Trained encoder in eval mode.
        observations: np.ndarray of shape (N, 7, 7, 3).
        device: torch.device.
        batch_size: Mini-batch size for encoding.

    Returns:
        np.ndarray of shape (N, 128) feature vectors.
    """
    encoder.eval()
    all_features = []

    n = len(observations)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.FloatTensor(observations[start:end]).to(device)
        with torch.no_grad():
            features = encoder(batch).cpu().numpy()
        all_features.append(features)

    return np.concatenate(all_features, axis=0)


def compute_concept_assignments(features, concept_manager):
    """
    Assign concept IDs to a set of feature vectors using the concept manager.

    Args:
        features: np.ndarray of shape (N, 128) feature vectors.
        concept_manager: Fitted ConceptManager with working predict.

    Returns:
        np.ndarray of shape (N,) integer concept IDs.
    """
    # Match the dtype expected by the KMeans centroids to avoid sklearn warnings
    centroid_dtype = concept_manager.cluster_centers.dtype
    features_cast = features.astype(centroid_dtype)
    return concept_manager.assign_concept(features_cast)


def compute_entropy(labels, n_classes=None):
    """
    Compute Shannon entropy H(X) of a discrete label distribution.

    H(X) = -sum_x P(x) * log2(P(x))

    This measures the "uncertainty" or "information content" of the label
    distribution. Maximum entropy occurs when all classes are equally likely.

    Args:
        labels: 1D array of integer labels.
        n_classes: Total number of possible classes (for binning).
                   If None, inferred from the data.

    Returns:
        Float entropy in bits.
    """
    if n_classes is None:
        n_classes = int(labels.max()) + 1

    # Count occurrences of each label
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    # Normalize to probabilities
    probs = counts / counts.sum()
    # Remove zero entries to avoid log(0)
    probs = probs[probs > 0]
    # Shannon entropy in bits
    return float(-np.sum(probs * np.log2(probs)))


def compute_joint_entropy(labels_a, labels_b, n_classes_a=None, n_classes_b=None):
    """
    Compute joint Shannon entropy H(X, Y) of two discrete label distributions.

    H(X, Y) = -sum_{x,y} P(x, y) * log2(P(x, y))

    This measures the total uncertainty in the joint distribution. The mutual
    information can be computed as I(X; Y) = H(X) + H(Y) - H(X, Y).

    Args:
        labels_a: 1D array of integer labels for variable A.
        labels_b: 1D array of integer labels for variable B (same length).
        n_classes_a: Number of classes for A.
        n_classes_b: Number of classes for B.

    Returns:
        Float joint entropy in bits.
    """
    assert len(labels_a) == len(labels_b), "Label arrays must have same length"

    if n_classes_a is None:
        n_classes_a = int(labels_a.max()) + 1
    if n_classes_b is None:
        n_classes_b = int(labels_b.max()) + 1

    # Build joint histogram: count co-occurrences of (concept_a, concept_b)
    joint_counts = np.zeros((n_classes_a, n_classes_b), dtype=np.float64)
    for a, b in zip(labels_a, labels_b):
        joint_counts[a, b] += 1

    # Normalize to joint probability distribution
    joint_probs = joint_counts / joint_counts.sum()
    # Remove zero entries
    joint_probs = joint_probs[joint_probs > 0]
    # Joint entropy in bits
    return float(-np.sum(joint_probs * np.log2(joint_probs)))


def compute_mutual_information(labels_a, labels_b, n_classes_a=None, n_classes_b=None):
    """
    Compute mutual information I(C^A; C^B) between two concept assignments.

    MI measures the amount of information that knowing one agent's concept
    assignment provides about the other agent's concept assignment. It's
    defined as:
        I(C^A; C^B) = H(C^A) + H(C^B) - H(C^A, C^B)

    Properties:
        - I >= 0 always (knowing one variable can't hurt prediction of another)
        - I = 0 iff C^A and C^B are independent
        - I = H(C^A) = H(C^B) when they are deterministically related

    We also validate against sklearn's mutual_info_score for correctness.

    Args:
        labels_a: 1D integer array of concept assignments from agent A.
        labels_b: 1D integer array of concept assignments from agent B.
        n_classes_a: Number of concepts for agent A.
        n_classes_b: Number of concepts for agent B.

    Returns:
        dict with:
            - mi_bits: mutual information in bits
            - mi_nats: mutual information in nats (natural log)
            - h_a: marginal entropy of A (bits)
            - h_b: marginal entropy of B (bits)
            - h_ab: joint entropy (bits)
            - nmi: normalized mutual information (0-1 scale)
    """
    h_a = compute_entropy(labels_a, n_classes_a)
    h_b = compute_entropy(labels_b, n_classes_b)
    h_ab = compute_joint_entropy(labels_a, labels_b, n_classes_a, n_classes_b)

    # MI = H(A) + H(B) - H(A, B) in bits
    mi_bits = h_a + h_b - h_ab

    # Clamp to non-negative (numerical rounding can make it slightly < 0)
    mi_bits = max(0.0, mi_bits)

    # Convert to nats for comparison with sklearn (which uses natural log)
    mi_nats = mi_bits * np.log(2)

    # Normalized Mutual Information (NMI):
    # NMI = 2 * I(A; B) / (H(A) + H(B))
    # Ranges from 0 (independent) to 1 (perfect agreement up to relabeling)
    denom = h_a + h_b
    nmi = (2.0 * mi_bits / denom) if denom > 0 else 0.0

    # Cross-validate with sklearn for correctness
    from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
    sklearn_mi_nats = mutual_info_score(labels_a, labels_b)
    sklearn_nmi = normalized_mutual_info_score(labels_a, labels_b)

    # Log any significant discrepancy (should be negligible)
    mi_diff = abs(mi_nats - sklearn_mi_nats)
    if mi_diff > 0.01:
        print(f"    WARNING: MI discrepancy: ours={mi_nats:.4f} nats, "
              f"sklearn={sklearn_mi_nats:.4f} nats")

    return {
        "mi_bits": float(mi_bits),
        "mi_nats": float(mi_nats),
        "h_a_bits": float(h_a),
        "h_b_bits": float(h_b),
        "h_ab_bits": float(h_ab),
        "nmi": float(nmi),
        "sklearn_mi_nats": float(sklearn_mi_nats),
        "sklearn_nmi": float(sklearn_nmi),
    }


def load_transfer_results(results_path):
    """
    Load zero-shot transfer win rates from the 5-seed experiment results.

    The results file contains win rates for each source->target pair.
    We extract the mean win rate for each pair to correlate with MI.

    Args:
        results_path: Path to transfer_same_task_5seed.json.

    Returns:
        Dict mapping (source, target) tuples to mean win rates.
    """
    if not os.path.exists(results_path):
        print(f"  WARNING: Transfer results not found at {results_path}")
        print(f"  Will skip correlation analysis.")
        return {}

    with open(results_path, "r") as f:
        data = json.load(f)

    win_rates = {}
    for pair in data["pairs"]:
        key = (pair["source"], pair["target"])
        win_rates[key] = float(pair["mean_wr"])

    print(f"  Loaded {len(win_rates)} transfer win rates from {results_path}")
    return win_rates


def create_figure(mi_results, win_rates, output_path):
    """
    Create a two-panel scatter plot: MI vs win rate, NMI vs win rate.

    Panel 1 (left):  I(C^A; C^B) [bits] vs Zero-Shot Win Rate
    Panel 2 (right): NMI vs Zero-Shot Win Rate

    Each point is one source->target transfer pair. Points are colored by
    source type: RL-trained (PPO, DQN) = blue, BC-trained (DAgger) = orange.
    A linear regression line with R^2 and p-value annotation is overlaid.

    Uses the Okabe-Ito colorblind-friendly palette as required by the project
    style guidelines (300 DPI, 11pt+ fonts).

    Args:
        mi_results: List of dicts, each with 'source', 'target', 'mi_bits', 'nmi'.
        win_rates: Dict mapping (source, target) to mean win rate.
        output_path: Path to save the figure PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    # Okabe-Ito colorblind palette
    # Blue (#0072B2) for RL-trained sources, Orange (#E69F00) for BC-trained
    OI_BLUE = "#0072B2"
    OI_ORANGE = "#E69F00"

    # Collect matched (MI, win_rate) pairs with color info
    mi_vals = []
    nmi_vals = []
    wr_vals = []
    colors = []
    labels_text = []

    for r in mi_results:
        key = (r["source"], r["target"])
        if key in win_rates:
            mi_vals.append(r["mi_bits"])
            nmi_vals.append(r["nmi"])
            wr_vals.append(win_rates[key])
            # Color by source type: RL (PPO, DQN) or BC (DAgger)
            if r["source"] in ("PPO", "DQN"):
                colors.append(OI_BLUE)
            else:
                colors.append(OI_ORANGE)
            labels_text.append(f"{r['source']}->{r['target']}")

    if len(mi_vals) < 2:
        print("  Not enough matched data points for scatter plot (need >= 2).")
        print("  Skipping figure generation.")
        return None

    mi_vals = np.array(mi_vals)
    nmi_vals = np.array(nmi_vals)
    wr_vals = np.array(wr_vals)

    # Set global font size for 11pt+ requirement
    plt.rcParams.update({"font.size": 12})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel 1: MI (bits) vs Win Rate ----
    ax = axes[0]
    ax.scatter(mi_vals, wr_vals, c=colors, s=90, edgecolors="black",
               linewidths=0.6, zorder=5)

    # Add point labels
    for i, txt in enumerate(labels_text):
        ax.annotate(txt, (mi_vals[i], wr_vals[i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, alpha=0.8)

    # Linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(mi_vals, wr_vals)
    x_line = np.linspace(mi_vals.min() - 0.1, mi_vals.max() + 0.1, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#CC79A7", linewidth=2, linestyle="--",
            alpha=0.8, label="Regression")

    # Annotate with R^2 and p-value
    r_squared = r_value ** 2
    ax.text(0.05, 0.95,
            f"$R^2$ = {r_squared:.3f}\n$r$ = {r_value:.3f}\np = {p_value:.4f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Mutual Information $I(C^A; C^B)$ [bits]", fontsize=12)
    ax.set_ylabel("Zero-Shot Win Rate", fontsize=12)
    ax.set_title("MI vs Transfer Success", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)

    # Custom legend for source type
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OI_BLUE,
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="RL source (PPO/DQN)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OI_ORANGE,
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="BC source (DAgger)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    # ---- Panel 2: NMI vs Win Rate ----
    ax = axes[1]
    ax.scatter(nmi_vals, wr_vals, c=colors, s=90, edgecolors="black",
               linewidths=0.6, zorder=5)

    # Add point labels
    for i, txt in enumerate(labels_text):
        ax.annotate(txt, (nmi_vals[i], wr_vals[i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, alpha=0.8)

    # Linear regression on NMI
    slope_n, intercept_n, r_value_n, p_value_n, std_err_n = stats.linregress(
        nmi_vals, wr_vals
    )
    x_line_n = np.linspace(nmi_vals.min() - 0.01, nmi_vals.max() + 0.01, 100)
    y_line_n = slope_n * x_line_n + intercept_n
    ax.plot(x_line_n, y_line_n, color="#CC79A7", linewidth=2, linestyle="--",
            alpha=0.8, label="Regression")

    r_squared_n = r_value_n ** 2
    ax.text(0.05, 0.95,
            f"$R^2$ = {r_squared_n:.3f}\n$r$ = {r_value_n:.3f}\np = {p_value_n:.4f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Normalized Mutual Information (NMI)", fontsize=12)
    ax.set_ylabel("Zero-Shot Win Rate", fontsize=12)
    ax.set_title("NMI vs Transfer Success", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure to {output_path}")

    return {
        "mi_vs_wr": {
            "pearson_r": float(r_value),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "nmi_vs_wr": {
            "pearson_r": float(r_value_n),
            "r_squared": float(r_squared_n),
            "p_value": float(p_value_n),
            "slope": float(slope_n),
            "intercept": float(intercept_n),
        },
    }


def run_mutual_information_analysis():
    """
    Main function: compute MI between all agent pairs and correlate with transfer.

    Steps:
        1. Collect shared observation pool from Go 7x7 via random play.
        2. Load all agent encoders and concept managers.
        3. Encode observations through each agent's encoder.
        4. Assign concepts using each agent's concept manager.
        5. Compute pairwise MI for all 6 directed pairs.
        6. Load transfer win rates and compute Pearson correlation.
        7. Save results (JSON) and figure (PNG).
    """
    set_seed(42)
    device = get_device()
    start_time = time.time()

    print("=" * 64)
    print("Mutual Information Analysis: I(C^A; C^B) Between Agent Pairs")
    print("=" * 64)

    # ---- Step 1: Collect shared observations ----
    print("\n--- Step 1: Collecting shared observations ---")
    env = MaskedGoEnv(board_size=7)
    observations = collect_observations(env, n_target=5000, seed=42)
    env.close()
    print(f"  Observation shape: {observations.shape}")

    # ---- Step 2: Load all agents ----
    print("\n--- Step 2: Loading agents ---")
    encoders = {}
    concept_managers = {}
    available_agents = []

    for agent_name, config in AGENTS.items():
        try:
            encoders[agent_name] = load_encoder(agent_name, config, device)
            concept_managers[agent_name] = load_concept_manager(agent_name, config)
            available_agents.append(agent_name)
        except FileNotFoundError as e:
            print(f"  SKIPPING {agent_name}: {e}")
        except Exception as e:
            print(f"  SKIPPING {agent_name}: unexpected error: {e}")
            import traceback
            traceback.print_exc()

    if len(available_agents) < 2:
        print(f"\nERROR: Need at least 2 agents for pairwise MI, "
              f"only found {len(available_agents)}: {available_agents}")
        return

    print(f"\n  Available agents: {available_agents}")

    # ---- Step 3: Encode observations through each agent ----
    print("\n--- Step 3: Encoding observations ---")
    all_features = {}
    for agent_name in available_agents:
        print(f"  Encoding through {agent_name}...")
        features = encode_observations(encoders[agent_name], observations, device)
        all_features[agent_name] = features
        print(f"    Features shape: {features.shape}, "
              f"norm mean: {np.linalg.norm(features, axis=1).mean():.2f}")

    # ---- Step 4: Assign concepts ----
    print("\n--- Step 4: Assigning concepts ---")
    all_concepts = {}
    for agent_name in available_agents:
        concepts = compute_concept_assignments(
            all_features[agent_name], concept_managers[agent_name]
        )
        all_concepts[agent_name] = concepts
        n_unique = len(np.unique(concepts))
        print(f"  {agent_name}: {n_unique}/{concept_managers[agent_name].n_concepts} "
              f"active concepts")

    # ---- Step 5: Compute pairwise MI ----
    print("\n--- Step 5: Computing pairwise mutual information ---")
    mi_results = []

    for source in available_agents:
        for target in available_agents:
            if source == target:
                continue

            concepts_a = all_concepts[source]
            concepts_b = all_concepts[target]
            n_classes_a = concept_managers[source].n_concepts
            n_classes_b = concept_managers[target].n_concepts

            mi_data = compute_mutual_information(
                concepts_a, concepts_b, n_classes_a, n_classes_b
            )

            result = {
                "source": source,
                "target": target,
                **mi_data,
            }
            mi_results.append(result)

            print(f"  {source} -> {target}: "
                  f"MI = {mi_data['mi_bits']:.4f} bits, "
                  f"NMI = {mi_data['nmi']:.4f}, "
                  f"H(A) = {mi_data['h_a_bits']:.2f}, "
                  f"H(B) = {mi_data['h_b_bits']:.2f}")

    # ---- Step 6: Load transfer results and correlate ----
    print("\n--- Step 6: Correlation with transfer success ---")
    results_path = os.path.join("results", "transfer_same_task_5seed.json")
    win_rates = load_transfer_results(results_path)

    # Print comparison table if win rates available
    if win_rates:
        print("\n  Pair              | MI (bits) | NMI   | Win Rate")
        print("  " + "-" * 55)
        for r in mi_results:
            key = (r["source"], r["target"])
            wr = win_rates.get(key, None)
            wr_str = f"{wr:.3f}" if wr is not None else "N/A"
            print(f"  {r['source']:>6} -> {r['target']:<6} | "
                  f"{r['mi_bits']:.4f}    | {r['nmi']:.4f} | {wr_str}")

    # ---- Step 7: Compute Pearson correlation ----
    correlation_results = None
    if win_rates:
        # Collect matched MI and win rate pairs
        matched_mi = []
        matched_nmi = []
        matched_wr = []
        for r in mi_results:
            key = (r["source"], r["target"])
            if key in win_rates:
                matched_mi.append(r["mi_bits"])
                matched_nmi.append(r["nmi"])
                matched_wr.append(win_rates[key])

        if len(matched_mi) >= 3:
            from scipy.stats import pearsonr

            # MI vs win rate
            r_mi, p_mi = pearsonr(matched_mi, matched_wr)
            print(f"\n  Pearson correlation (MI vs WR):  r = {r_mi:.4f}, "
                  f"p = {p_mi:.4f}")

            # NMI vs win rate
            r_nmi, p_nmi = pearsonr(matched_nmi, matched_wr)
            print(f"  Pearson correlation (NMI vs WR): r = {r_nmi:.4f}, "
                  f"p = {p_nmi:.4f}")

            correlation_results = {
                "mi_vs_wr": {
                    "pearson_r": float(r_mi),
                    "p_value": float(p_mi),
                    "n_pairs": len(matched_mi),
                },
                "nmi_vs_wr": {
                    "pearson_r": float(r_nmi),
                    "p_value": float(p_nmi),
                    "n_pairs": len(matched_nmi),
                },
            }
        else:
            print(f"\n  Not enough matched pairs for correlation "
                  f"(found {len(matched_mi)}, need >= 3)")

    # ---- Step 8: Create figure ----
    print("\n--- Step 7: Creating figure ---")
    ensure_dir(os.path.join("results", "figures"))
    fig_path = os.path.join("results", "figures", "mutual_information.png")

    figure_stats = None
    if win_rates:
        try:
            figure_stats = create_figure(mi_results, win_rates, fig_path)
        except Exception as e:
            print(f"  WARNING: Could not create figure: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipping figure (no transfer results available)")

    # ---- Step 9: Save results to JSON ----
    print("\n--- Step 8: Saving results ---")
    ensure_dir("results")
    output = {
        "n_observations": int(len(observations)),
        "agents": available_agents,
        "pairwise_mi": mi_results,
        "correlation": correlation_results,
        "figure_regression": figure_stats,
        "elapsed_seconds": float(time.time() - start_time),
    }

    output_path = os.path.join("results", "mutual_information.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {output_path}")

    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\n{'=' * 64}")
    print(f"Analysis complete in {elapsed:.1f}s")
    print(f"{'=' * 64}")

    # Print key findings
    if mi_results:
        max_mi = max(mi_results, key=lambda x: x["mi_bits"])
        min_mi = min(mi_results, key=lambda x: x["mi_bits"])
        print(f"\n  Highest MI:  {max_mi['source']} -> {max_mi['target']} = "
              f"{max_mi['mi_bits']:.4f} bits (NMI = {max_mi['nmi']:.4f})")
        print(f"  Lowest MI:   {min_mi['source']} -> {min_mi['target']} = "
              f"{min_mi['mi_bits']:.4f} bits (NMI = {min_mi['nmi']:.4f})")

    if correlation_results:
        r_val = correlation_results["mi_vs_wr"]["pearson_r"]
        p_val = correlation_results["mi_vs_wr"]["p_value"]
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"\n  MI-WR Pearson r = {r_val:.4f} (p = {p_val:.4f}, {sig})")
        print(f"  Interpretation: {'Higher MI predicts better transfer' if r_val > 0 else 'Negative relationship'}")

    return output


if __name__ == "__main__":
    run_mutual_information_analysis()
