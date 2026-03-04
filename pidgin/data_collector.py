"""
PIDGIN Data Collector.

Gathers all evidence needed to describe each of PRISM's 64 concepts:
  - Gameplay examples: board states (obs array + ASCII) and the agent's
    chosen action, for states where each concept fired.
  - Action distribution: softmax probabilities over 50 actions, heatmap
    over the 7x7 board. Computed directly from the policy, not from gameplay.
  - Frequency: how often each concept fires during gameplay.
  - KL divergence from uniform: how action-specific each concept is.
  - Ablation impact: win rate drop when ablated (from results/ablation_ppo.json).
  - Intervention sensitivity (from results/intervention_ppo.json).
  - Centroid statistics: mean activation, sparsity, neighbor similarities.

All collected data is stored in ConceptEvidence dataclasses — one per concept.
These feed directly into concept_prompter.py which formats them for the LLM.

Usage:
    collector = DataCollector()
    evidence = collector.collect_all(n_games=500)
    # evidence[k] is a ConceptEvidence for concept k
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Allow running from either the project root or the pidgin/ subdirectory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.utils import get_device, set_seed


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class BoardExample:
    """One board state where a concept fired, plus the agent's action."""
    obs: np.ndarray              # Raw observation (7,7,3)
    action: int                  # Action chosen by the bottleneck policy
    ascii_board: str             # Human-readable board representation
    dist_to_centroid: float      # L2 distance from the concept's centroid
    move_number: int             # Approximate game move number (step in episode)


@dataclass
class ConceptEvidence:
    """All evidence for one concept. Fed into concept_prompter.py."""
    concept_id: int

    # Board state examples (split into optimization set and held-out eval set)
    examples: List[BoardExample]       # Used during TextGrad optimization
    holdout_examples: List[BoardExample]  # Held out for BDM-10 / DBM-64 eval

    # Action distribution (from policy, not gameplay)
    action_probs: np.ndarray           # (50,) softmax probabilities
    action_heatmap: np.ndarray         # (7,7) positional frequency; last entry is pass
    top_actions: List[Tuple[int, float]]  # Top-10 (action_id, probability) pairs

    # Gameplay statistics
    frequency: float                   # Fraction of game steps this concept fires
    kl_from_uniform: float             # KL(policy | uniform) — how specific is this concept
    entropy: float                     # Entropy of the action distribution

    # Centroid statistics
    centroid: np.ndarray               # (128,) raw centroid vector
    centroid_mean_activation: float
    centroid_sparsity: float           # Fraction of dims with |activation| < 0.1
    neighbor_ids: List[int]            # Top-5 nearest neighbors by cosine similarity
    neighbor_similarities: List[float]

    # From existing PRISM experiment results (may be None if files absent)
    ablation_win_rate_drop: Optional[float]  # Win rate drop when ablated
    ablation_rank: Optional[int]             # Rank by impact (1 = most impactful)
    intervention_change_rate: Optional[float]
    concept_specificity: Optional[float]     # Most common action's frequency


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class DataCollector:
    """
    Loads PRISM artifacts and collects per-concept evidence.

    Artifacts loaded (all read-only):
        models/baseline/ppo_go_encoder.pt
        models/bottleneck/concepts_ppo_k64.pkl
        models/bottleneck/ppo_bottleneck_final.pt
        results/ablation_ppo.json
        results/intervention_ppo.json
    """

    def __init__(
        self,
        n_concepts: int = 64,
        n_examples: int = 20,       # Per concept, used during optimization
        n_holdout: int = 20,        # Per concept, held out for evaluation
        seed: int = 42,
        device: Optional[torch.device] = None,
        root_dir: Optional[str] = None,
    ):
        self.n_concepts = n_concepts
        self.n_examples = n_examples
        self.n_holdout = n_holdout
        self.seed = seed
        self.device = device or get_device()
        self.root_dir = root_dir or _ROOT

        self.encoder: Optional[GoCNNEncoder] = None
        self.concept_manager: Optional[ConceptManager] = None
        self.policy: Optional[ConceptBottleneckPolicy] = None
        self._ablation_data: Optional[Dict] = None
        self._intervention_data: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_artifacts(self) -> None:
        """Load all PRISM artifacts from disk. Call before collect_all()."""
        root = self.root_dir
        env = GoEnv(board_size=7)

        # Encoder
        encoder_path = os.path.join(root, "models", "baseline", "ppo_go_encoder.pt")
        self.encoder = GoCNNEncoder(env.observation_space, features_dim=128)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device, weights_only=True)
        )
        self.encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f"Loaded encoder from {encoder_path}")

        # Concept manager
        cm_path = os.path.join(
            root, "models", "bottleneck", f"concepts_ppo_k{self.n_concepts}.pkl"
        )
        self.concept_manager = ConceptManager(n_concepts=self.n_concepts)
        self.concept_manager.load(cm_path)
        print(f"Loaded {self.n_concepts} concepts from {cm_path}")

        # Bottleneck policy
        policy_path = os.path.join(
            root, "models", "bottleneck", "ppo_bottleneck_final.pt"
        )
        self.policy = ConceptBottleneckPolicy(
            n_concepts=self.n_concepts, embed_dim=64, hidden_dim=128, n_actions=50
        ).to(self.device)
        self.policy.load_state_dict(
            torch.load(policy_path, map_location=self.device, weights_only=True)
        )
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")

        # Ablation results
        ablation_path = os.path.join(root, "results", "ablation_ppo.json")
        if os.path.exists(ablation_path):
            with open(ablation_path) as f:
                self._ablation_data = json.load(f)
            print(f"Loaded ablation data ({ablation_path})")
        else:
            print(f"WARNING: ablation data not found at {ablation_path}")

        # Intervention results
        intervention_path = os.path.join(root, "results", "intervention_ppo.json")
        if os.path.exists(intervention_path):
            with open(intervention_path) as f:
                self._intervention_data = json.load(f)
            print(f"Loaded intervention data ({intervention_path})")
        else:
            print(f"WARNING: intervention data not found at {intervention_path}")

        env.close()

    # ------------------------------------------------------------------
    # Gameplay collection
    # ------------------------------------------------------------------

    def _play_games(self, n_games: int) -> List[Dict]:
        """
        Play n_games using the bottleneck agent. Record every step.

        Returns a list of step records:
            {obs, action, concept_id, dist_to_centroid, move_number, ascii_board}
        """
        assert self.encoder is not None, "Call load_artifacts() first."
        set_seed(self.seed)

        env = GoEnv(board_size=7)
        records = []

        for game_idx in range(n_games):
            obs, info = env.reset()
            done = False
            move_number = 0

            while not done:
                # Encode observation → features → concept ID
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.encoder(obs_t).cpu().numpy()[0]

                concept_id = int(self.concept_manager.assign_concept(features))
                dist = float(np.linalg.norm(
                    features - self.concept_manager.cluster_centers[concept_id]
                ))

                # Get the policy's action
                mask = info.get("action_mask", None)
                action = self.policy.get_action(concept_id, mask, deterministic=True)

                # Render the board
                ascii_board = _render_board(obs, action)

                records.append({
                    "obs": obs.copy(),
                    "action": action,
                    "concept_id": concept_id,
                    "dist_to_centroid": dist,
                    "move_number": move_number,
                    "ascii_board": ascii_board,
                })

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                move_number += 1

            if (game_idx + 1) % 100 == 0:
                print(f"  Played {game_idx + 1}/{n_games} games "
                      f"({len(records)} total steps)")

        env.close()
        return records

    # ------------------------------------------------------------------
    # Action distribution (from policy, not gameplay)
    # ------------------------------------------------------------------

    def _get_action_distribution(self, concept_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the policy's action probability distribution for concept_id.

        Returns:
            action_probs: (50,) softmax probabilities over all 50 actions
            heatmap: (7,7) spatial frequency (pass action excluded)
        """
        with torch.no_grad():
            cid_t = torch.LongTensor([concept_id]).to(self.device)
            logits, _ = self.policy(cid_t)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()  # (50,)

        # Map to 7x7 heatmap (actions 0-48 = board positions, 49 = pass)
        heatmap = np.zeros((7, 7), dtype=np.float32)
        for action_id in range(49):
            row = action_id // 7
            col = action_id % 7
            heatmap[row, col] = probs[action_id]

        return probs, heatmap

    # ------------------------------------------------------------------
    # Centroid statistics
    # ------------------------------------------------------------------

    def _compute_centroid_stats(self) -> Dict[int, Dict]:
        """
        Compute per-concept centroid statistics and nearest neighbors.

        Returns dict: concept_id -> {
            mean_activation, sparsity,
            neighbor_ids (top-5), neighbor_similarities (top-5)
        }
        """
        centroids = self.concept_manager.cluster_centers  # (K, 128)
        sim_matrix = cosine_similarity(centroids, centroids)  # (K, K)

        stats = {}
        for k in range(self.n_concepts):
            c = centroids[k]
            mean_act = float(np.mean(c))
            sparsity = float(np.mean(np.abs(c) < 0.1))

            # Top-6 most similar (index 0 is self), take indices 1-6
            sorted_idx = np.argsort(-sim_matrix[k])
            neighbor_ids = [int(i) for i in sorted_idx[1:6]]
            neighbor_sims = [float(sim_matrix[k, i]) for i in sorted_idx[1:6]]

            stats[k] = {
                "mean_activation": mean_act,
                "sparsity": sparsity,
                "neighbor_ids": neighbor_ids,
                "neighbor_similarities": neighbor_sims,
            }

        return stats

    # ------------------------------------------------------------------
    # Existing experiment results
    # ------------------------------------------------------------------

    def _parse_ablation(self) -> Dict[int, Dict]:
        """Extract per-concept ablation results keyed by concept_id."""
        if self._ablation_data is None:
            return {}

        result = {}
        ablation_results = self._ablation_data.get("ablation_results", [])
        baseline_wr = self._ablation_data.get("baseline_win_rate", None)

        # Build rank: sort by win_rate_drop descending, assign rank 1 = most impactful
        sorted_results = sorted(
            ablation_results, key=lambda r: r.get("win_rate_drop", 0), reverse=True
        )
        for rank, r in enumerate(sorted_results, start=1):
            cid = int(r.get("concept_id", -1))
            if cid < 0:
                continue
            result[cid] = {
                "win_rate_drop": r.get("win_rate_drop"),
                "ablation_rank": rank,
                "trigger_rate": r.get("trigger_rate"),
                "baseline_win_rate": baseline_wr,
            }

        return result

    def _parse_intervention(self) -> Dict[str, float]:
        """Extract overall intervention statistics."""
        if self._intervention_data is None:
            return {}
        return {
            "overall_change_rate": self._intervention_data.get("overall_change_rate"),
            "mean_concept_specificity": self._intervention_data.get(
                "mean_concept_specificity"
            ),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def collect_all(self, n_games: int = 500) -> Dict[int, ConceptEvidence]:
        """
        Collect all evidence for all concepts.

        Plays n_games, groups steps by concept, builds ConceptEvidence for each.

        Args:
            n_games: Number of complete games to play for data collection.
                     500 games ≈ 15,000 steps, giving ~235 steps per concept
                     on average (with high variance for rare concepts).

        Returns:
            Dict mapping concept_id (int) → ConceptEvidence.
        """
        print(f"Collecting evidence for {self.n_concepts} concepts "
              f"over {n_games} games...")

        # 1. Compute centroid statistics (no gameplay needed)
        centroid_stats = self._compute_centroid_stats()

        # 2. Parse existing experiment results
        ablation_by_concept = self._parse_ablation()
        intervention_stats = self._parse_intervention()

        # 3. Compute action distributions (no gameplay needed)
        print("Computing action distributions...")
        action_distributions = {}
        for k in range(self.n_concepts):
            probs, heatmap = self._get_action_distribution(k)
            action_distributions[k] = (probs, heatmap)

        # 4. Play games and collect per-step records
        print(f"Playing {n_games} games...")
        records = self._play_games(n_games)
        print(f"Collected {len(records)} steps total.")

        # 5. Group records by concept
        rng = np.random.default_rng(self.seed)
        by_concept: Dict[int, List[Dict]] = defaultdict(list)
        for r in records:
            by_concept[r["concept_id"]].append(r)

        total_steps = len(records)

        # 6. Build ConceptEvidence for each concept
        evidence: Dict[int, ConceptEvidence] = {}
        n_actions = 50

        for k in range(self.n_concepts):
            concept_records = by_concept.get(k, [])
            freq = len(concept_records) / total_steps if total_steps > 0 else 0.0

            # Sort records by distance to centroid (closest first = most representative)
            concept_records.sort(key=lambda r: r["dist_to_centroid"])

            # Split into optimization and holdout sets
            # Take examples from throughout the sorted list for diversity:
            # close-to-centroid examples for optimization, boundary examples for holdout
            n_available = len(concept_records)
            n_opt = min(self.n_examples, n_available)
            n_hld = min(self.n_holdout, max(0, n_available - n_opt))

            # For optimization: sample from the closer half (more representative)
            close_half_n = max(1, n_available // 2)
            if close_half_n >= n_opt:
                opt_indices = set(rng.choice(close_half_n, size=n_opt, replace=False).tolist())
                opt_records = [concept_records[i] for i in sorted(opt_indices)]
            else:
                opt_indices = set(range(min(n_opt, close_half_n)))
                opt_records = [concept_records[i] for i in sorted(opt_indices)]

            # For holdout: sample from the remaining records (boundary + varied)
            remaining = [concept_records[i] for i in range(n_available) if i not in opt_indices]
            if len(remaining) >= n_hld:
                hld_indices = rng.choice(
                    len(remaining), size=n_hld, replace=False
                )
                hld_records = [remaining[i] for i in sorted(hld_indices)]
            else:
                hld_records = remaining[:n_hld]

            def make_examples(recs: List[Dict]) -> List[BoardExample]:
                return [
                    BoardExample(
                        obs=r["obs"],
                        action=r["action"],
                        ascii_board=r["ascii_board"],
                        dist_to_centroid=r["dist_to_centroid"],
                        move_number=r["move_number"],
                    )
                    for r in recs
                ]

            # Action distribution
            probs, heatmap = action_distributions[k]
            uniform = np.ones(n_actions) / n_actions
            # KL(policy || uniform) = sum(p * log(p/u)); skip near-zero probs
            safe_probs = np.clip(probs, 1e-10, 1.0)
            kl = float(np.sum(safe_probs * np.log(safe_probs / uniform)))
            entropy = float(-np.sum(safe_probs * np.log(safe_probs)))

            top_actions = sorted(
                enumerate(probs), key=lambda x: x[1], reverse=True
            )[:10]

            # Centroid stats
            cs = centroid_stats[k]

            # Ablation stats
            ab = ablation_by_concept.get(k, {})

            evidence[k] = ConceptEvidence(
                concept_id=k,
                examples=make_examples(opt_records),
                holdout_examples=make_examples(hld_records),
                action_probs=probs,
                action_heatmap=heatmap,
                top_actions=top_actions,
                frequency=freq,
                kl_from_uniform=kl,
                entropy=entropy,
                centroid=self.concept_manager.cluster_centers[k].copy(),
                centroid_mean_activation=cs["mean_activation"],
                centroid_sparsity=cs["sparsity"],
                neighbor_ids=cs["neighbor_ids"],
                neighbor_similarities=cs["neighbor_similarities"],
                ablation_win_rate_drop=ab.get("win_rate_drop"),
                ablation_rank=ab.get("ablation_rank"),
                intervention_change_rate=intervention_stats.get("overall_change_rate"),
                concept_specificity=float(np.max(probs)) if len(probs) > 0 else None,
            )

        # Report coverage
        n_well_covered = sum(
            1 for ev in evidence.values() if len(ev.examples) >= 10
        )
        n_sparse = sum(
            1 for ev in evidence.values() if len(ev.examples) < 5
        )
        print(f"Coverage: {n_well_covered}/{self.n_concepts} concepts have ≥10 examples; "
              f"{n_sparse} have <5 examples (may produce weaker descriptions).")

        return evidence


# ---------------------------------------------------------------------------
# Board rendering helper
# ---------------------------------------------------------------------------

def _render_board(obs: np.ndarray, action: int) -> str:
    """
    Render a 7×7 Go board as ASCII.

    obs shape: (7, 7, 3)

    IMPORTANT — channel inversion:
    Due to PettingZoo go_v5's AlphaZero-style rolling board history, the
    observation the agent (black) receives has:
        Channel 0 = WHITE's (opponent's) stones
        Channel 1 = BLACK's (agent's) stones
    This is the opposite of the documented convention, but is consistent
    throughout training. The visualizer applies the same correction.

    We render channel 0 as "W" and channel 1 as "B" so the LLM sees
    physically correct stone colors: B = the agent's black stones,
    W = the opponent's white stones.

    action: 0-48 = board position (row * 7 + col), 49 = pass

    Returns a multiline string like:
        A B C D E F G
      7 . . . B . . .
      6 . W . . . . .
      ...
    """
    cols = "ABCDEFG"
    lines = ["  " + " ".join(cols)]
    for row in range(7):
        display_row = 7 - row  # Go convention: row 0 = rank 7 (top)
        cells = []
        for col in range(7):
            if obs[row, col, 1] > 0.5:   # channel 1 = agent's (black) stones
                cells.append("B")
            elif obs[row, col, 0] > 0.5:  # channel 0 = opponent's (white) stones
                cells.append("W")
            else:
                cells.append(".")
        lines.append(f"{display_row:2d} " + " ".join(cells))

    if action == 49:
        action_str = "Pass"
    else:
        r, c = divmod(action, 7)
        action_str = f"{cols[c]}{7 - r}"

    lines.append(f"Agent played: {action_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: save / load evidence to avoid replaying 500 games each run
# ---------------------------------------------------------------------------

def save_evidence(evidence: Dict[int, ConceptEvidence], path: str) -> None:
    """Pickle the evidence dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(evidence, f)
    print(f"Evidence saved to {path}")


def load_evidence(path: str) -> Dict[int, ConceptEvidence]:
    """Load pickled evidence dict from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
