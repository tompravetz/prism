"""
Concept Manager: K-Means concept discovery and assignment.

Collects encoder features from trained agents, discovers concept clusters
via MiniBatchKMeans, and provides concept assignment for new observations.
"""

import os
import pickle
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from typing import Optional

from src.utils import ensure_dir


class ConceptManager:
    """
    Manages concept discovery and assignment via K-Means clustering.

    Workflow:
        1. Collect features from a trained encoder over many game states.
        2. Fit K-Means to discover concept clusters.
        3. Assign concepts to new observations by encoding + nearest cluster.
    """

    def __init__(self, n_concepts: int = 64, features_dim: int = 128,
                 random_state: int = 42):
        self.n_concepts = n_concepts
        self.features_dim = features_dim
        self.random_state = random_state

        self.kmeans = MiniBatchKMeans(
            n_clusters=n_concepts,
            random_state=random_state,
            batch_size=1024,
            n_init=3,
        )
        self.is_fitted = False
        self.cluster_centers = None

        # Stats
        self.n_samples_collected = 0
        self._collected_features = []

    def collect_features(self, encoder: torch.nn.Module, env, n_episodes: int = 500,
                         device: torch.device = torch.device("cpu")):
        """
        Collect encoder features by running episodes in the environment.

        Args:
            encoder: Frozen encoder network.
            env: Gymnasium environment.
            n_episodes: Number of episodes to collect from.
            device: Torch device.
        """
        encoder.eval()
        features_list = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = encoder(obs_tensor).cpu().numpy()
                features_list.append(features[0])

                # Random action for data collection
                mask = info.get("action_mask", None)
                if mask is not None:
                    legal = np.where(mask == 1)[0]
                    action = np.random.choice(legal) if len(legal) > 0 else 0
                else:
                    action = env.action_space.sample()

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        all_features = np.array(features_list, dtype=np.float32)
        self._collected_features.append(all_features)
        self.n_samples_collected += len(all_features)
        print(f"Collected {len(all_features)} features from {n_episodes} episodes "
              f"(total: {self.n_samples_collected})")

    def collect_features_from_policy(self, encoder: torch.nn.Module, policy_fn,
                                     env, n_episodes: int = 500,
                                     device: torch.device = torch.device("cpu")):
        """
        Collect features using a trained policy (not random).

        Args:
            encoder: Frozen encoder network.
            policy_fn: Callable(obs, action_mask) -> action.
            env: Gymnasium environment.
            n_episodes: Number of episodes to collect from.
            device: Torch device.
        """
        encoder.eval()
        features_list = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = encoder(obs_tensor).cpu().numpy()
                features_list.append(features[0])

                mask = info.get("action_mask", None)
                action = policy_fn(obs, mask)

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        all_features = np.array(features_list, dtype=np.float32)
        self._collected_features.append(all_features)
        self.n_samples_collected += len(all_features)
        print(f"Collected {len(all_features)} features from {n_episodes} policy episodes "
              f"(total: {self.n_samples_collected})")

    def fit(self):
        """Fit K-Means on all collected features."""
        if not self._collected_features:
            raise ValueError("No features collected. Call collect_features() first.")

        all_features = np.concatenate(self._collected_features, axis=0)
        print(f"Fitting K-Means (K={self.n_concepts}) on {len(all_features)} samples...")

        self.kmeans.fit(all_features)
        self.cluster_centers = self.kmeans.cluster_centers_.copy()
        self.is_fitted = True

        # Report cluster distribution
        labels = self.kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Cluster distribution: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}, std={counts.std():.1f}")
        print(f"Active clusters: {len(unique)}/{self.n_concepts}")

        return self

    def assign_concept(self, features: np.ndarray) -> np.ndarray:
        """
        Assign concept IDs to feature vectors.

        Args:
            features: (N, features_dim) or (features_dim,) array.

        Returns:
            Concept IDs as integer array.
        """
        if not self.is_fitted:
            raise ValueError("ConceptManager not fitted. Call fit() first.")

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        labels = self.kmeans.predict(features)
        return labels[0] if single else labels

    def assign_concept_from_obs(self, encoder: torch.nn.Module,
                                 obs: np.ndarray,
                                 device: torch.device = torch.device("cpu")) -> int:
        """
        End-to-end: observation → encoder → concept ID.

        Args:
            encoder: Frozen encoder network.
            obs: Single observation (not batched).
            device: Torch device.

        Returns:
            Integer concept ID.
        """
        encoder.eval()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(obs_tensor).cpu().numpy()[0]
        return int(self.assign_concept(features))

    def get_concept_distances(self, features: np.ndarray) -> np.ndarray:
        """
        Get distances to all concept centers (for soft concepts).

        Args:
            features: (features_dim,) array.

        Returns:
            (n_concepts,) distance array.
        """
        if not self.is_fitted:
            raise ValueError("ConceptManager not fitted.")
        diffs = self.cluster_centers - features.reshape(1, -1)
        return np.linalg.norm(diffs, axis=1)

    def save(self, path: str):
        """Save fitted concept manager to disk."""
        ensure_dir(os.path.dirname(path))
        data = {
            "n_concepts": self.n_concepts,
            "features_dim": self.features_dim,
            "cluster_centers": self.cluster_centers,
            "is_fitted": self.is_fitted,
            "n_samples_collected": self.n_samples_collected,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"ConceptManager saved to {path}")

    def load(self, path: str):
        """Load fitted concept manager from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_concepts = data["n_concepts"]
        self.features_dim = data["features_dim"]
        self.cluster_centers = data["cluster_centers"]
        self.is_fitted = data["is_fitted"]
        self.n_samples_collected = data["n_samples_collected"]

        # Re-initialize KMeans with loaded centers
        if self.is_fitted:
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_concepts,
                random_state=self.random_state,
                batch_size=1024,
                n_init=1,
                init=self.cluster_centers,
            )
            # Fake-fit to set internal state
            self.kmeans.cluster_centers_ = self.cluster_centers
            self.kmeans._check_params = lambda X: None
            # Mark as fitted
            self.kmeans.n_features_in_ = self.features_dim
            self.kmeans._n_threads = 1

        print(f"ConceptManager loaded from {path} ({self.n_concepts} concepts)")
        return self
