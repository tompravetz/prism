"""
Strategy Library: Registry of reusable strategies with search and compose capabilities.

A strategy in PRISM is a (concepts, policy) pair — the discrete concept centroids
(K x 128D) plus the bottleneck policy weights that map concept IDs to actions.
Strategies are independent of the encoder, which is the whole point: the concept
layer is the universal interface between observation space and action selection.

The StrategyLibrary collects strategies from multiple agents across domains and
provides operations for:
    1. Search: Given a new ConceptManager, find the most similar existing strategy
       (by centroid cosine similarity). This enables zero-shot transfer.
    2. Compose: Combine multiple strategies by averaging aligned embeddings,
       creating a "generalist" from "specialists".

Usage:
    library = StrategyLibrary()
    library.add_strategy("Go-PPO", concepts_path, policy_path,
                         metadata={"domain": "go", "algo": "ppo"})
    library.add_strategy("CartPole", concepts_path, policy_path,
                         metadata={"domain": "cartpole", "algo": "ppo"})

    # Find most similar strategy for a new agent
    matches = library.find_similar(new_concept_manager, top_k=3)

    # Compute pairwise similarity matrix across all strategies
    sim_matrix = library.compute_cross_similarity()
"""

import os
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.concept_aligner import ConceptAligner


@dataclass
class StrategyEntry:
    """
    A single strategy in the library: concepts + policy + metadata.

    Attributes:
        name: Human-readable identifier (e.g., "Go-PPO-K64", "CartPole-PPO-K32").
        concept_manager: Fitted ConceptManager with K-means centroids.
        policy: Trained bottleneck policy (PPO or DQN variant).
        metadata: Arbitrary dict with extra info (domain, algorithm, training steps,
                  win rate, number of actions, etc.)
    """
    name: str
    concept_manager: ConceptManager
    policy: Optional[torch.nn.Module] = None
    metadata: Dict = field(default_factory=dict)

    def centroid_summary(self) -> dict:
        """Quick summary of this strategy's concept space."""
        centroids = self.concept_manager.cluster_centers
        return {
            "n_concepts": len(centroids),
            "feature_dim": centroids.shape[1],
            "centroid_norms_mean": float(np.linalg.norm(centroids, axis=1).mean()),
            "centroid_norms_std": float(np.linalg.norm(centroids, axis=1).std()),
        }


class StrategyLibrary:
    """
    Registry of reusable strategies with similarity search and composition.

    Strategies are stored in-memory and can be saved/loaded to disk.
    The library supports cross-strategy similarity computation for
    visualization (heatmaps) and transfer recommendations.
    """

    def __init__(self):
        """Initialize an empty strategy library."""
        self.strategies: Dict[str, StrategyEntry] = {}

    def add_strategy(
        self,
        name: str,
        concept_manager: ConceptManager,
        policy: Optional[torch.nn.Module] = None,
        metadata: Optional[Dict] = None,
    ) -> StrategyEntry:
        """
        Register a strategy in the library.

        Args:
            name: Unique identifier for this strategy.
            concept_manager: Fitted ConceptManager with cluster_centers.
            policy: Trained bottleneck policy (optional — can add later).
            metadata: Extra information (domain, algorithm, performance, etc.)

        Returns:
            The created StrategyEntry.
        """
        if metadata is None:
            metadata = {}

        entry = StrategyEntry(
            name=name,
            concept_manager=concept_manager,
            policy=policy,
            metadata=metadata,
        )
        self.strategies[name] = entry
        print(f"  Added strategy '{name}' (K={concept_manager.n_concepts})")
        return entry

    def add_strategy_from_paths(
        self,
        name: str,
        concept_path: str,
        policy_path: Optional[str] = None,
        policy_class: str = "ppo",
        n_actions: int = 50,
        metadata: Optional[Dict] = None,
    ) -> StrategyEntry:
        """
        Convenience method: load a strategy from file paths.

        Args:
            name: Strategy identifier.
            concept_path: Path to saved ConceptManager .pkl file.
            policy_path: Path to saved policy .pt file (optional).
            policy_class: "ppo" or "dqn" — which policy class to instantiate.
            n_actions: Number of actions for the policy.
            metadata: Extra information.

        Returns:
            The created StrategyEntry.
        """
        # Load concept manager
        cm = ConceptManager()
        cm.load(concept_path)

        # Load policy if path provided
        # Infer embed_dim and hidden_dim from the checkpoint to handle
        # different architectures (Go uses 64/128, CartPole uses 32/64)
        policy = None
        if policy_path and os.path.exists(policy_path):
            sd = torch.load(policy_path, map_location="cpu")

            if policy_class == "dqn":
                # Infer dims from DQN checkpoint
                embed_dim = sd["embedding.weight"].shape[1] if "embedding.weight" in sd else 64
                hidden_dim = sd["q_net.0.weight"].shape[0] if "q_net.0.weight" in sd else 128
                policy = ConceptDQNPolicy(
                    n_concepts=cm.n_concepts, n_actions=n_actions,
                    embed_dim=embed_dim, hidden_dim=hidden_dim,
                )
            else:
                # Infer dims from PPO checkpoint
                embed_dim = sd["embedding.weight"].shape[1] if "embedding.weight" in sd else 64
                hidden_dim = sd["policy_head.0.weight"].shape[0] if "policy_head.0.weight" in sd else 128
                policy = ConceptBottleneckPolicy(
                    n_concepts=cm.n_concepts, n_actions=n_actions,
                    embed_dim=embed_dim, hidden_dim=hidden_dim,
                )
            policy.load_state_dict(sd)
            policy.eval()

        return self.add_strategy(name, cm, policy, metadata)

    def find_similar(
        self,
        query_cm: ConceptManager,
        top_k: int = 3,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find the most similar strategies to a query ConceptManager.

        Similarity is computed as the mean cosine similarity between aligned
        centroid pairs (using greedy alignment to handle different K values).

        Args:
            query_cm: ConceptManager to match against the library.
            top_k: Number of top matches to return.

        Returns:
            List of (strategy_name, mean_similarity, alignment_quality) tuples,
            sorted by similarity descending.
        """
        results = []

        for name, entry in self.strategies.items():
            # Align query concepts to this strategy's concepts
            aligner = ConceptAligner(query_cm, entry.concept_manager)
            mapping = aligner.greedy_alignment()
            quality = aligner.alignment_quality(mapping)

            results.append((name, quality["mean_similarity"], quality))

        # Sort by mean similarity, descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def compute_cross_similarity(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise similarity matrix across all strategies in the library.

        Uses greedy alignment + mean cosine similarity for each pair.
        Useful for visualization (heatmaps) and understanding which strategies
        share similar concept spaces.

        Returns:
            (similarity_matrix, strategy_names) where similarity_matrix[i,j]
            is the mean aligned cosine similarity between strategy i and j.
        """
        names = sorted(self.strategies.keys())
        n = len(names)
        sim_matrix = np.zeros((n, n))

        for i, name_i in enumerate(names):
            sim_matrix[i, i] = 1.0  # Self-similarity is perfect
            for j in range(i + 1, n):
                name_j = names[j]
                cm_i = self.strategies[name_i].concept_manager
                cm_j = self.strategies[name_j].concept_manager

                aligner = ConceptAligner(cm_i, cm_j)
                mapping = aligner.greedy_alignment()
                quality = aligner.alignment_quality(mapping)

                sim_matrix[i, j] = quality["mean_similarity"]
                sim_matrix[j, i] = quality["mean_similarity"]

        return sim_matrix, names

    def compose_strategies(
        self,
        strategy_names: List[str],
        weights: Optional[List[float]] = None,
        reference_name: Optional[str] = None,
    ) -> ConceptBottleneckPolicy:
        """
        Compose multiple strategies by averaging aligned policy embeddings.

        Takes N specialist strategies, aligns all their concept spaces to a
        reference strategy, then creates a new policy whose embedding layer is
        a weighted average of the aligned specialists' embeddings.

        This produces a "generalist" that combines knowledge from multiple
        specialists without retraining.

        Args:
            strategy_names: List of strategy names to compose.
            weights: Optional weight for each strategy (default: equal weights).
                     Must sum to 1.0.
            reference_name: Strategy whose concept space is the reference frame.
                            If None, uses the first strategy in the list.

        Returns:
            New ConceptBottleneckPolicy with composed embedding weights.
        """
        if len(strategy_names) < 2:
            raise ValueError("Need at least 2 strategies to compose.")

        if weights is None:
            weights = [1.0 / len(strategy_names)] * len(strategy_names)

        if reference_name is None:
            reference_name = strategy_names[0]

        ref_entry = self.strategies[reference_name]
        ref_policy = ref_entry.policy

        if ref_policy is None:
            raise ValueError(f"Reference strategy '{reference_name}' has no policy.")

        # Start with zero embeddings, accumulate weighted contributions
        embed_dim = ref_policy.embedding.embedding_dim
        n_concepts = ref_entry.concept_manager.n_concepts
        n_actions = ref_policy.n_actions
        composed_embed = torch.zeros(n_concepts, embed_dim)

        for name, weight in zip(strategy_names, weights):
            entry = self.strategies[name]
            if entry.policy is None:
                print(f"  Warning: strategy '{name}' has no policy, skipping.")
                continue

            if name == reference_name:
                # No alignment needed — use embeddings directly
                composed_embed += weight * entry.policy.embedding.weight.data
            else:
                # Align this strategy's concepts to the reference
                aligner = ConceptAligner(entry.concept_manager, ref_entry.concept_manager)
                mapping = aligner.greedy_alignment()

                # For each reference concept, find which source concepts map to it
                # and average their embeddings
                src_embed = entry.policy.embedding.weight.data
                for src_id, tgt_id in mapping.items():
                    if tgt_id < n_concepts:
                        composed_embed[tgt_id] += weight * src_embed[src_id]

        # Create the composed policy (same architecture as reference)
        composed_policy = ConceptBottleneckPolicy(
            n_concepts=n_concepts,
            embed_dim=embed_dim,
            hidden_dim=128,
            n_actions=n_actions,
        )

        # Set the composed embeddings
        composed_policy.embedding.weight.data = composed_embed

        # Copy the reference policy's head weights as a starting point
        composed_policy.policy_head.load_state_dict(ref_policy.policy_head.state_dict())
        composed_policy.value_head.load_state_dict(ref_policy.value_head.state_dict())

        return composed_policy

    def list_strategies(self) -> List[dict]:
        """List all strategies with summary info."""
        summaries = []
        for name, entry in sorted(self.strategies.items()):
            summary = {
                "name": name,
                "n_concepts": entry.concept_manager.n_concepts,
                "has_policy": entry.policy is not None,
                **entry.metadata,
            }
            summaries.append(summary)
        return summaries

    def __len__(self) -> int:
        return len(self.strategies)

    def __repr__(self) -> str:
        return f"StrategyLibrary({len(self.strategies)} strategies)"
