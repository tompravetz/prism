"""
Concept Aligner: Align concept spaces between ConceptManagers for strategy transfer.

This is the foundation for ALL transfer experiments in PRISM. Given two ConceptManagers
(each with K-means centroids in 128D feature space), this module computes optimal
concept-to-concept mappings via two alignment methods:

    1. Hungarian algorithm: Optimal 1:1 bipartite matching on raw cosine similarity.
       Best for same-task transfers where feature spaces are already similar.

    2. Procrustes analysis: Finds the optimal orthogonal rotation aligning source
       centroids to target centroids, THEN applies Hungarian matching in the rotated
       space. Inspired by cross-lingual word embedding alignment (Conneau et al., 2018).
       Best for cross-domain transfers where feature spaces have systematic rotational
       differences.

The key insight: since ALL encoders (CNN for Go, MLP for CartPole/LunarLander/Acrobot)
produce 128D feature vectors, K-means centroids from ANY encoder live in the same
dimensionality. Cosine similarity between centroids measures how "similar" two concepts
are — even if they came from different algorithms (PPO vs DQN) or different domains
(Go vs CartPole).

Transfer pipeline:
    Source: encoder_A -> concepts_A -> policy_A
    Target: encoder_B -> concepts_B -> ???

    1. Align: concepts_A <-> concepts_B via centroid similarity (Hungarian or Procrustes)
    2. Transfer: remap policy_A's embedding weights using alignment mapping
    3. Result: policy_B initialized with transferred knowledge from policy_A

Usage:
    aligner = ConceptAligner(source_cm, target_cm)

    # Method 1: Direct Hungarian matching
    mapping = aligner.hungarian_alignment()

    # Method 2: Procrustes rotation + Hungarian matching
    mapping, R = aligner.procrustes_alignment()

    quality = aligner.alignment_quality(mapping)
    new_policy = aligner.transfer_policy(source_policy, mapping, target_n_actions=50)
"""

import numpy as np
import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes

from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy


class ConceptAligner:
    """
    Aligns concept spaces between two ConceptManagers using centroid similarity.

    Supports two alignment strategies:
        1. Hungarian (optimal 1:1): For same-K transfers. Finds the mapping that
           maximizes total similarity across all concept pairs. O(K^3) but K<=256
           so it's fast.
        2. Greedy (nearest-neighbor): For different-K transfers. Each source concept
           maps to its most similar target concept. Multiple sources can map to the
           same target. O(Ks * Kt).
    """

    def __init__(self, source_cm: ConceptManager, target_cm: ConceptManager):
        """
        Args:
            source_cm: ConceptManager from the source agent (the one we're transferring FROM).
            target_cm: ConceptManager from the target agent (the one we're transferring TO).
        """
        if not source_cm.is_fitted or not target_cm.is_fitted:
            raise ValueError("Both ConceptManagers must be fitted before alignment.")

        self.source_cm = source_cm
        self.target_cm = target_cm

        # Source and target centroid arrays: (K, 128) each
        self.source_centroids = source_cm.cluster_centers.astype(np.float32)
        self.target_centroids = target_cm.cluster_centers.astype(np.float32)

        self.Ks = len(self.source_centroids)  # Number of source concepts
        self.Kt = len(self.target_centroids)  # Number of target concepts

        # Compute similarity matrix once (used by all alignment methods)
        self._sim_matrix = None

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute cosine similarity between all source and target centroid pairs.

        Cosine similarity measures the angle between two vectors in 128D space:
            cos(A, B) = (A . B) / (|A| * |B|)

        Range: [-1, 1], where 1 = identical direction, 0 = orthogonal, -1 = opposite.
        In practice, our centroids are all in the positive quadrant (ReLU features),
        so similarities are typically in [0, 1].

        Returns:
            (Ks, Kt) similarity matrix. Entry [i,j] = similarity between
            source concept i and target concept j.
        """
        if self._sim_matrix is None:
            self._sim_matrix = cosine_similarity(
                self.source_centroids, self.target_centroids
            )
        return self._sim_matrix

    def hungarian_alignment(self) -> Dict[int, int]:
        """
        Optimal 1:1 concept alignment using the Hungarian algorithm.

        The Hungarian algorithm solves the linear assignment problem: given a cost
        matrix, find the assignment of rows to columns that minimizes total cost.
        We use negative similarity as cost (to maximize total similarity).

        For same-K: produces a perfect matching (every concept mapped).
        For different-K: produces min(Ks, Kt) mappings (some concepts unmapped).

        Returns:
            Dict mapping source concept ID -> target concept ID.
            Example: {0: 5, 1: 12, 2: 3, ...} means source concept 0 aligns
            with target concept 5, source concept 1 with target concept 12, etc.
        """
        sim_matrix = self.compute_similarity_matrix()

        # Hungarian algorithm minimizes cost, so negate similarity to maximize it
        cost_matrix = -sim_matrix

        # linear_sum_assignment returns (row_indices, col_indices) for optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build mapping: source_id -> target_id
        mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

        return mapping

    def greedy_alignment(self) -> Dict[int, int]:
        """
        Greedy nearest-neighbor alignment (for different-K or quick alignment).

        Each source concept maps to its most similar target concept. This is simpler
        than Hungarian but allows many-to-one mappings (multiple source concepts can
        map to the same target concept).

        This is useful when:
            - Ks != Kt (can't do perfect 1:1 matching)
            - Speed matters more than optimality
            - You want every source concept to have a mapping

        Returns:
            Dict mapping source concept ID -> target concept ID.
        """
        sim_matrix = self.compute_similarity_matrix()

        # For each source concept, find the most similar target concept
        mapping = {}
        for src_id in range(self.Ks):
            best_target = int(np.argmax(sim_matrix[src_id]))
            mapping[src_id] = best_target

        return mapping

    def procrustes_alignment(self) -> Tuple[Dict[int, int], np.ndarray]:
        """
        Align concept spaces using Procrustes analysis followed by Hungarian matching.

        Procrustes finds an orthogonal rotation R that minimizes ||A @ R - B||_F,
        aligning the entire source feature space to the target. Then Hungarian
        matching is applied in the rotated space for the final 1:1 mapping.

        This is more powerful than direct Hungarian because it accounts for
        systematic rotations between feature spaces — common when encoders are
        trained with different algorithms or on different domains. The rotation
        preserves distances and angles within each concept space while bringing
        the two spaces into better correspondence.

        Inspired by cross-lingual word embedding alignment (Conneau et al., 2018;
        Schonemann, 1966).

        Returns:
            Tuple of (mapping, rotation_matrix):
                - mapping: Dict[int, int] source->target concept mapping
                - rotation_matrix: (D, D) orthogonal rotation matrix where D is
                  feature dimensionality (128)
        """
        A = self.source_centroids.copy()  # (Ks, D)
        B = self.target_centroids.copy()  # (Kt, D)

        # For Procrustes, we need the same number of points.
        # If Ks != Kt, use a preliminary greedy matching to select anchor pairs.
        if self.Ks != self.Kt:
            # Use greedy matching to find anchor pairs for Procrustes
            sim = cosine_similarity(A, B)
            n_anchors = min(self.Ks, self.Kt)
            if self.Ks <= self.Kt:
                # Each source maps to nearest target
                anchors_src = list(range(self.Ks))
                anchors_tgt = [int(np.argmax(sim[i])) for i in range(self.Ks)]
            else:
                # Each target maps to nearest source
                anchors_tgt = list(range(self.Kt))
                anchors_src = [int(np.argmax(sim[:, j])) for j in range(self.Kt)]
            A_anchor = A[anchors_src]
            B_anchor = B[anchors_tgt]
        else:
            # Same K: use all centroids as anchors (preliminary 1:1 via Hungarian)
            prelim_sim = cosine_similarity(A, B)
            row_ind, col_ind = linear_sum_assignment(-prelim_sim)
            A_anchor = A[row_ind]
            B_anchor = B[col_ind]

        # Center both anchor sets
        A_mean = A_anchor.mean(axis=0)
        B_mean = B_anchor.mean(axis=0)
        A_centered = A_anchor - A_mean
        B_centered = B_anchor - B_mean

        # Find optimal orthogonal rotation R minimizing ||A_centered @ R - B_centered||_F
        R, scale = orthogonal_procrustes(A_centered, B_centered)

        # Apply rotation to ALL source centroids (center, rotate, uncenter to target space)
        A_global_mean = A.mean(axis=0)
        B_global_mean = B.mean(axis=0)
        A_rotated = (A - A_global_mean) @ R + B_global_mean

        # Store rotated centroids for quality computation
        self._procrustes_rotated = A_rotated
        self._procrustes_R = R

        # Compute similarity in the rotated space
        aligned_sim = cosine_similarity(A_rotated, B)

        # Final Hungarian matching on the rotated similarity matrix
        cost_matrix = -aligned_sim
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

        # Compute Procrustes distance (residual after alignment)
        if self.Ks == self.Kt:
            A_matched = A_rotated[row_ind]
            B_matched = B[col_ind]
            procrustes_distance = float(np.linalg.norm(A_matched - B_matched, 'fro'))
        else:
            procrustes_distance = float(np.linalg.norm(
                (A_anchor - A_mean) @ R - B_centered, 'fro'
            ))

        self._procrustes_distance = procrustes_distance

        return mapping, R

    def alignment_quality(self, mapping: Dict[int, int]) -> dict:
        """
        Evaluate the quality of a concept alignment.

        Metrics:
            - mean_similarity: Average cosine similarity of mapped pairs.
              Higher = concepts are more similar. >0.8 is excellent, >0.5 is decent.
            - min_similarity: Worst-case pair similarity. Low values indicate
              some concepts don't have good matches.
            - max_similarity: Best-case pair similarity.
            - coverage_source: Fraction of source concepts that are mapped.
            - coverage_target: Fraction of target concepts that are mapped to.
            - similarity_distribution: Full list of per-pair similarities.

        Args:
            mapping: Dict of source_id -> target_id from an alignment method.

        Returns:
            Dict of quality metrics.
        """
        sim_matrix = self.compute_similarity_matrix()

        # Collect similarities for all mapped pairs
        similarities = []
        for src_id, tgt_id in mapping.items():
            similarities.append(sim_matrix[src_id, tgt_id])
        similarities = np.array(similarities)

        # How many unique target concepts are used?
        unique_targets = len(set(mapping.values()))

        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "median_similarity": float(np.median(similarities)),
            "coverage_source": len(mapping) / self.Ks,
            "coverage_target": unique_targets / self.Kt,
            "n_mapped_pairs": len(mapping),
            "n_unique_targets": unique_targets,
            "similarity_distribution": similarities.tolist(),
        }

    def transfer_policy(
        self,
        source_policy: nn.Module,
        mapping: Dict[int, int],
        target_n_concepts: Optional[int] = None,
        target_n_actions: Optional[int] = None,
    ) -> nn.Module:
        """
        Transfer a bottleneck policy from source to target using concept alignment.

        The bottleneck policy has two learnable components:
            1. Embedding layer: maps concept_id -> embedding vector (64D)
            2. Policy/Q head: maps embedding -> action logits

        Transfer strategy:
            - Embedding: For each target concept j, copy the embedding weights from
              the aligned source concept mapping[i] -> j. Unmapped target concepts
              get the mean of all source embeddings (neutral initialization).
            - Policy head: Copy directly if n_actions matches. If n_actions differs,
              copy shared action weights and randomly initialize new actions.

        Args:
            source_policy: Trained ConceptBottleneckPolicy or ConceptDQNPolicy.
            mapping: Alignment mapping from source to target concept IDs.
            target_n_concepts: Number of concepts in target (defaults to target_cm.n_concepts).
            target_n_actions: Number of actions in target (defaults to source's n_actions).

        Returns:
            New policy instance with transferred weights.
        """
        if target_n_concepts is None:
            target_n_concepts = self.Kt
        if target_n_actions is None:
            target_n_actions = source_policy.n_actions

        # Determine policy type and create new instance with target dimensions
        if isinstance(source_policy, ConceptDQNPolicy):
            # Infer hidden_dim from source policy's Q network first layer
            source_hidden = source_policy.q_net[0].out_features
            new_policy = ConceptDQNPolicy(
                n_concepts=target_n_concepts,
                embed_dim=source_policy.embed_dim,
                hidden_dim=source_hidden,
                n_actions=target_n_actions,
            )
        else:
            # ConceptBottleneckPolicy (PPO variant)
            # Infer embed_dim and hidden_dim from source policy's architecture
            embed_dim = source_policy.embedding.embedding_dim
            source_hidden = source_policy.policy_head[0].out_features
            new_policy = ConceptBottleneckPolicy(
                n_concepts=target_n_concepts,
                embed_dim=embed_dim,
                hidden_dim=source_hidden,
                n_actions=target_n_actions,
            )

        # --- Transfer embedding weights ---
        # Source embedding: (Ks, embed_dim)
        source_embed = source_policy.embedding.weight.data.clone()
        # Compute mean embedding for unmapped concepts (neutral initialization)
        mean_embed = source_embed.mean(dim=0)

        # Initialize target embedding with mean (safe default for unmapped concepts)
        new_embed = mean_embed.unsqueeze(0).expand(target_n_concepts, -1).clone()

        # Build reverse mapping: target_id -> list of source_ids that map to it
        reverse_map = {}
        for src_id, tgt_id in mapping.items():
            if tgt_id not in reverse_map:
                reverse_map[tgt_id] = []
            reverse_map[tgt_id].append(src_id)

        # For each target concept, average the embeddings of all source concepts
        # that map to it (handles both 1:1 and many:1 mappings)
        for tgt_id, src_ids in reverse_map.items():
            if tgt_id < target_n_concepts:
                src_embeds = source_embed[src_ids]
                new_embed[tgt_id] = src_embeds.mean(dim=0)

        new_policy.embedding.weight.data = new_embed

        # --- Transfer policy/Q head weights ---
        # Identify the head module in source and target
        if isinstance(source_policy, ConceptDQNPolicy):
            src_head = source_policy.q_net
            tgt_head = new_policy.q_net
        else:
            src_head = source_policy.policy_head
            tgt_head = new_policy.policy_head

        src_state = src_head.state_dict()
        tgt_state = tgt_head.state_dict()

        # Copy all layers that have matching shapes
        for key in src_state:
            if key in tgt_state:
                if src_state[key].shape == tgt_state[key].shape:
                    # Exact match: copy directly
                    tgt_state[key] = src_state[key].clone()
                elif len(src_state[key].shape) >= 1:
                    # Shape mismatch (likely final layer with different n_actions)
                    # Copy the overlapping portion, leave the rest as random init
                    src_shape = src_state[key].shape
                    tgt_shape = tgt_state[key].shape
                    # Copy min(src, tgt) along each dimension
                    slices = tuple(
                        slice(0, min(s, t)) for s, t in zip(src_shape, tgt_shape)
                    )
                    tgt_state[key][slices] = src_state[key][slices].clone()

        tgt_head.load_state_dict(tgt_state)

        # --- Transfer value head if PPO policy ---
        if isinstance(source_policy, ConceptBottleneckPolicy) and \
           isinstance(new_policy, ConceptBottleneckPolicy):
            src_val_state = source_policy.value_head.state_dict()
            tgt_val_state = new_policy.value_head.state_dict()
            for key in src_val_state:
                if key in tgt_val_state and src_val_state[key].shape == tgt_val_state[key].shape:
                    tgt_val_state[key] = src_val_state[key].clone()
            new_policy.value_head.load_state_dict(tgt_val_state)

        return new_policy

    def get_concept_mapping_details(self, mapping: Dict[int, int]) -> list:
        """
        Get detailed per-concept alignment information for analysis/visualization.

        Returns:
            List of dicts, one per mapping pair, with:
                - source_id, target_id, similarity
                - source_centroid, target_centroid (128D vectors)
        """
        sim_matrix = self.compute_similarity_matrix()
        details = []
        for src_id, tgt_id in sorted(mapping.items()):
            details.append({
                "source_id": src_id,
                "target_id": tgt_id,
                "similarity": float(sim_matrix[src_id, tgt_id]),
                "source_centroid": self.source_centroids[src_id].tolist(),
                "target_centroid": self.target_centroids[tgt_id].tolist(),
            })
        return details

    def __repr__(self) -> str:
        return (f"ConceptAligner(source_K={self.Ks}, target_K={self.Kt}, "
                f"feature_dim={self.source_centroids.shape[1]})")


def compose_alignments(mapping_ab: Dict[int, int], mapping_bc: Dict[int, int]) -> tuple:
    """
    Compose two alignment mappings to create a transitive mapping A -> C.

    Given:
        mapping_ab: A's concept i -> B's concept j
        mapping_bc: B's concept j -> C's concept k

    Produces mapping_ac: A's concept i -> C's concept k by chaining through B.
    If a concept in B doesn't have a mapping in mapping_bc, it's "lost in
    translation" -- that source concept gets no mapping in the output.

    This is the key operation for transitive transfer: if concepts truly form
    a universal interface, then chained transfer (A->B->C) should produce
    results comparable to direct transfer (A->C).

    Args:
        mapping_ab: Source->intermediate alignment (e.g., PPO->DQN).
        mapping_bc: Intermediate->target alignment (e.g., DQN->DAgger).

    Returns:
        Tuple of (composed_mapping, n_lost):
            - composed_mapping: Dict[int, int] mapping source -> target concepts
            - n_lost: Number of source concepts that couldn't be mapped through
              the chain (intermediate concept not found in mapping_bc)
    """
    composed = {}
    n_lost = 0

    for src_a, tgt_b in mapping_ab.items():
        if tgt_b in mapping_bc:
            composed[src_a] = mapping_bc[tgt_b]
        else:
            n_lost += 1

    return composed, n_lost
