"""
Strategy Memory: tracks concept → action → outcome mappings.

Records which actions are taken under which concepts and their outcomes,
enabling strategy extraction, ablation experiments, and interpretability analysis.
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.utils import ensure_dir


class StrategyMemory:
    """
    Tracks {concept_id: {action: {rewards: [], count: int, win_rate: float}}}.

    Used for:
        - Extracting top strategies (concept-action pairs with high win rates)
        - Ablation experiments (disable specific strategies)
        - Interpretability (which concepts lead to which actions)
    """

    def __init__(self, n_concepts: int = 64, n_actions: int = 50):
        self.n_concepts = n_concepts
        self.n_actions = n_actions

        # {concept_id: {action: {"rewards": [], "count": 0}}}
        self.memory: Dict[int, Dict[int, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {"rewards": [], "count": 0})
        )

        # Episode-level tracking
        self._current_episode: List[Tuple[int, int]] = []  # [(concept, action), ...]
        self._total_episodes = 0

    def record_step(self, concept_id: int, action: int):
        """Record a (concept, action) pair during an episode."""
        self._current_episode.append((concept_id, action))

    def end_episode(self, final_reward: float):
        """
        Finalize an episode: assign the final reward to all concept-action pairs.

        For Go: reward is typically {-1, 0, 1} for loss/draw/win.
        """
        for concept_id, action in self._current_episode:
            entry = self.memory[concept_id][action]
            entry["rewards"].append(final_reward)
            entry["count"] += 1

        self._current_episode = []
        self._total_episodes += 1

    def get_strategy_stats(self, concept_id: int, action: int) -> Dict:
        """Get statistics for a specific concept-action pair."""
        entry = self.memory[concept_id][action]
        rewards = entry["rewards"]
        if not rewards:
            return {"count": 0, "mean_reward": 0.0, "win_rate": 0.0}

        return {
            "count": entry["count"],
            "mean_reward": float(np.mean(rewards)),
            "win_rate": float(np.mean([r > 0 for r in rewards])),
            "std_reward": float(np.std(rewards)),
        }

    def get_top_strategies(self, min_count: int = 50, min_win_rate: float = 0.6,
                           top_k: int = 20) -> List[Dict]:
        """
        Extract top strategies ranked by win rate.

        A strategy is a (concept_id, action) pair with sufficient observations
        and high win rate.

        Args:
            min_count: Minimum number of observations required.
            min_win_rate: Minimum win rate threshold.
            top_k: Number of top strategies to return.

        Returns:
            List of strategy dicts with keys: concept_id, action, count,
            win_rate, mean_reward.
        """
        strategies = []
        for concept_id in self.memory:
            for action in self.memory[concept_id]:
                stats = self.get_strategy_stats(concept_id, action)
                if stats["count"] >= min_count and stats["win_rate"] >= min_win_rate:
                    strategies.append({
                        "concept_id": concept_id,
                        "action": action,
                        "count": stats["count"],
                        "win_rate": stats["win_rate"],
                        "mean_reward": stats["mean_reward"],
                    })

        # Sort by win rate descending, then count descending
        strategies.sort(key=lambda s: (s["win_rate"], s["count"]), reverse=True)
        return strategies[:top_k]

    def get_concept_distribution(self) -> Dict[int, int]:
        """Get total usage count per concept."""
        dist = {}
        for concept_id in self.memory:
            total = sum(e["count"] for e in self.memory[concept_id].values())
            dist[concept_id] = total
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def get_concept_action_preference(self, concept_id: int) -> Dict[int, float]:
        """
        Get preferred action distribution for a concept.

        Returns: {action: proportion_of_times_chosen}
        """
        if concept_id not in self.memory:
            return {}

        total = sum(e["count"] for e in self.memory[concept_id].values())
        if total == 0:
            return {}

        return {
            action: entry["count"] / total
            for action, entry in self.memory[concept_id].items()
            if entry["count"] > 0
        }

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        n_active_concepts = len(self.memory)
        total_records = sum(
            entry["count"]
            for c in self.memory.values()
            for entry in c.values()
        )
        strategies = self.get_top_strategies(min_count=10, min_win_rate=0.5, top_k=100)

        return {
            "total_episodes": self._total_episodes,
            "total_records": total_records,
            "active_concepts": n_active_concepts,
            "strategies_found": len(strategies),
            "top_3_strategies": strategies[:3],
        }

    def save(self, path: str):
        """Save strategy memory to disk."""
        ensure_dir(os.path.dirname(path))
        # Convert defaultdict to regular dict for pickling
        data = {
            "n_concepts": self.n_concepts,
            "n_actions": self.n_actions,
            "memory": {k: dict(v) for k, v in self.memory.items()},
            "total_episodes": self._total_episodes,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"StrategyMemory saved to {path}")

    def load(self, path: str):
        """Load strategy memory from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_concepts = data["n_concepts"]
        self.n_actions = data["n_actions"]
        self._total_episodes = data["total_episodes"]

        # Restore as defaultdict
        self.memory = defaultdict(
            lambda: defaultdict(lambda: {"rewards": [], "count": 0})
        )
        for k, v in data["memory"].items():
            for a, entry in v.items():
                self.memory[int(k)][int(a)] = entry

        print(f"StrategyMemory loaded from {path} "
              f"({self._total_episodes} episodes)")
        return self

    def export_to_json(self, path: str, top_k: int = 50):
        """Export top strategies to JSON for analysis."""
        ensure_dir(os.path.dirname(path))
        strategies = self.get_top_strategies(
            min_count=10, min_win_rate=0.4, top_k=top_k
        )
        with open(path, "w") as f:
            json.dump({
                "summary": self.get_summary(),
                "strategies": strategies,
                "concept_distribution": self.get_concept_distribution(),
            }, f, indent=2, default=str)
        print(f"Strategies exported to {path}")
