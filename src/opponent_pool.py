"""
Opponent Pool for self-play training.

Maintains a pool of past agent checkpoints. During training, opponents
are sampled from this pool to provide curriculum-style difficulty.
"""

import os
import copy
import random
import numpy as np
import torch
from typing import Optional, List, Callable

from src.utils import ensure_dir


class OpponentPool:
    """
    Pool of past agent versions for self-play.

    Saves checkpoints every N generations and samples from them as opponents.
    Supports both PPO-style policies (action distribution) and DQN-style
    policies (Q-value argmax).
    """

    def __init__(self, save_dir: str = "models/opponents", max_pool_size: int = 50):
        self.save_dir = save_dir
        self.max_pool_size = max_pool_size
        self.pool: List[dict] = []  # List of {"path": str, "gen": int, "win_rate": float}
        ensure_dir(save_dir)

    def add_checkpoint(self, model_state_dict: dict, generation: int,
                       win_rate: float = 0.0, prefix: str = "opponent"):
        """Save a model checkpoint to the pool."""
        path = os.path.join(self.save_dir, f"{prefix}_gen{generation:04d}.pt")
        torch.save(model_state_dict, path)

        self.pool.append({
            "path": path,
            "gen": generation,
            "win_rate": win_rate,
        })

        # Trim oldest if pool exceeds max size
        if len(self.pool) > self.max_pool_size:
            removed = self.pool.pop(0)
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])

        return path

    def sample_opponent_path(self) -> Optional[str]:
        """
        Sample a random opponent checkpoint path.

        Weighted towards more recent checkpoints.
        """
        if not self.pool:
            return None

        # Linear weight: more recent = higher probability
        n = len(self.pool)
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights /= weights.sum()

        idx = np.random.choice(n, p=weights)
        return self.pool[idx]["path"]

    def load_opponent_state_dict(self) -> Optional[dict]:
        """Sample and load an opponent's state dict."""
        path = self.sample_opponent_path()
        if path is None:
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def make_opponent_fn(self, policy_class, encoder, concept_manager=None,
                         device=torch.device("cpu"), n_actions=50):
        """
        Create an opponent function from a sampled checkpoint.

        For baseline agents: loads encoder+policy state dict.
        For concept bottleneck agents: loads concept policy state dict.

        Returns:
            Callable(obs, action_mask) -> action, or None if pool is empty.
        """
        state_dict = self.load_opponent_state_dict()
        if state_dict is None:
            return None

        # Instantiate policy and load weights
        try:
            policy = policy_class
            if isinstance(policy, type):
                # It's a class, need to instantiate — caller should pass instance
                return None
            policy.load_state_dict(state_dict)
            policy.to(device)
            policy.eval()
        except Exception as e:
            print(f"Warning: Could not load opponent: {e}")
            return None

        if concept_manager is not None:
            # Concept bottleneck opponent
            def opponent_fn(obs, action_mask):
                concept_id = concept_manager.assign_concept_from_obs(
                    encoder, obs, device
                )
                cid = torch.LongTensor([concept_id]).to(device)
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = policy(cid, mask_t)
                return logits[0].argmax().item()
        else:
            # Full-info baseline opponent
            def opponent_fn(obs, action_mask):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = encoder(obs_t)
                    # Assume policy has a forward that takes features
                    q_values = policy(features)
                    if action_mask is not None:
                        mask_t = torch.FloatTensor(action_mask).to(device)
                        q_values = q_values.masked_fill(mask_t == 0, float('-inf'))
                    return q_values[0].argmax().item()

        return opponent_fn

    @property
    def size(self):
        return len(self.pool)

    def get_summary(self) -> dict:
        return {
            "pool_size": len(self.pool),
            "generations": [p["gen"] for p in self.pool],
            "win_rates": [p["win_rate"] for p in self.pool],
        }
