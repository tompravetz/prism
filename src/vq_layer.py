"""
Vector Quantization Layer for End-to-End Concept Learning.

Phase 2 of ASTRIA: instead of the two-stage K-Means approach, this module
implements differentiable discrete concepts using Vector Quantization (VQ),
as introduced in VQ-VAE (van den Oord et al., 2017).

Key idea: maintain a learnable CODEBOOK of K concept embeddings. For each
encoder output, find the nearest codebook entry and use it as the concept
representation. The encoder learns to produce outputs that map well to the
codebook, and the codebook learns to represent useful concepts.

The main challenge: argmin (nearest neighbor lookup) is non-differentiable.
Solution: the Straight-Through Estimator (STE) — during the forward pass,
we use the quantized codebook entry; during the backward pass, we copy
gradients directly from the output to the input, bypassing the quantization.

Loss components:
    1. Commitment loss: Encourages encoder outputs to stay close to codebook
       entries. Prevents the encoder from "drifting" away from the codebook.
       L_commit = ||sg[encoder_output] - codebook_entry||^2

    2. Codebook loss: Moves codebook entries towards encoder outputs.
       L_codebook = ||encoder_output - sg[codebook_entry]||^2
       (where sg = stop_gradient)

Known issue: CODEBOOK COLLAPSE — some codebook entries may never get used,
wasting capacity. Mitigations implemented:
    - EMA (Exponential Moving Average) updates for the codebook
    - Codebook reset for dead entries
    - Usage tracking to detect collapse early
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with Straight-Through Estimator.

    Maps continuous encoder features to discrete concept embeddings.

    Forward pass:
        1. Compute distances between encoder output and all codebook entries
        2. Select nearest codebook entry (the "concept")
        3. Return the codebook entry (but copy gradients straight through)

    Args:
        n_concepts: Number of codebook entries (K). Each entry = one concept.
        embedding_dim: Dimensionality of each codebook entry.
        commitment_cost: Weight for the commitment loss (beta in VQ-VAE paper).
                         Higher = encoder outputs stay closer to codebook entries.
        use_ema: Whether to use Exponential Moving Average updates for codebook.
                 EMA is often more stable than gradient-based codebook updates.
        ema_decay: Decay rate for EMA (0.99 = slow update, 0.9 = fast update).
        reset_threshold: If a codebook entry is used less than this fraction
                         of the batch, consider resetting it.
    """

    def __init__(self, n_concepts: int = 64, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, use_ema: bool = True,
                 ema_decay: float = 0.99, reset_threshold: float = 0.01):
        super().__init__()

        self.n_concepts = n_concepts
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold

        # The codebook: a matrix of K concept embeddings, each of dimension D
        # Shape: (n_concepts, embedding_dim)
        self.embedding = nn.Embedding(n_concepts, embedding_dim)
        # Initialize uniformly in [-1/K, 1/K] (standard VQ-VAE initialization)
        self.embedding.weight.data.uniform_(-1.0 / n_concepts, 1.0 / n_concepts)

        if use_ema:
            # EMA tracking variables (not trainable parameters)
            # ema_cluster_size: running count of how many inputs map to each entry
            self.register_buffer("ema_cluster_size", torch.zeros(n_concepts))
            # ema_embedding_sum: running sum of encoder outputs mapped to each entry
            self.register_buffer("ema_embedding_sum",
                                 self.embedding.weight.data.clone())

        # Usage tracking: count how often each codebook entry is used
        self.register_buffer("usage_count", torch.zeros(n_concepts, dtype=torch.long))
        self.total_updates = 0

    def forward(self, z: torch.Tensor):
        """
        Quantize encoder features to nearest codebook entries.

        Args:
            z: (batch_size, embedding_dim) — continuous encoder features.

        Returns:
            quantized: (batch_size, embedding_dim) — quantized features
                       (with straight-through gradient).
            vq_loss: Scalar — combined commitment + codebook loss.
            concept_ids: (batch_size,) — integer concept IDs (argmin indices).
            perplexity: Scalar — measure of codebook utilization.
                        Higher perplexity = more uniform usage = healthier codebook.
        """
        # Step 1: Compute distances between z and all codebook entries
        # d(z, e_j) = ||z||^2 + ||e_j||^2 - 2 * z · e_j
        # We use this expanded form because it's faster than computing pairwise L2
        distances = (
            z.pow(2).sum(1, keepdim=True)                      # ||z||^2
            + self.embedding.weight.pow(2).sum(1)               # ||e_j||^2
            - 2 * z @ self.embedding.weight.t()                 # -2 * z · e_j
        )
        # Shape: (batch_size, n_concepts)

        # Step 2: Find nearest codebook entry for each input
        concept_ids = distances.argmin(dim=1)  # (batch_size,)

        # Step 3: Look up the quantized vectors from the codebook
        quantized = self.embedding(concept_ids)  # (batch_size, embedding_dim)

        # Step 4: Compute VQ losses
        if self.use_ema and self.training:
            # ---- EMA codebook update ----
            # Instead of using gradients to update the codebook, we use
            # exponential moving averages. This is often more stable.

            # One-hot encoding of which codebook entry each input mapped to
            encodings = F.one_hot(concept_ids, self.n_concepts).float()

            # Update cluster sizes (how many inputs map to each entry)
            self.ema_cluster_size = (
                self.ema_decay * self.ema_cluster_size
                + (1 - self.ema_decay) * encodings.sum(0)
            )

            # Update embedding sums (sum of encoder outputs per entry)
            self.ema_embedding_sum = (
                self.ema_decay * self.ema_embedding_sum
                + (1 - self.ema_decay) * encodings.t() @ z
            )

            # Laplace smoothing to avoid division by zero
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.n_concepts * 1e-5) * n
            )

            # Update codebook entries
            self.embedding.weight.data = self.ema_embedding_sum / cluster_size.unsqueeze(1)

            # VQ loss = commitment loss only (codebook updated via EMA, not gradients)
            vq_loss = self.commitment_cost * F.mse_loss(z.detach(), quantized)
        else:
            # ---- Gradient-based codebook update ----
            # Codebook loss: move entries toward encoder outputs
            codebook_loss = F.mse_loss(z.detach(), quantized)
            # Commitment loss: move encoder outputs toward entries
            commitment_loss = F.mse_loss(z, quantized.detach())
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Step 5: Straight-Through Estimator
        # During forward: use quantized (discrete)
        # During backward: gradients flow to z (continuous)
        # The magic: quantized = z + (quantized - z).detach()
        # The .detach() means gradients don't flow through the quantization step,
        # so they pass straight through from quantized back to z.
        quantized = z + (quantized - z).detach()

        # Step 6: Compute perplexity (codebook utilization metric)
        # Perplexity = exp(entropy of the usage distribution)
        # Maximum perplexity = n_concepts (all entries used equally)
        # Perplexity ≈ 1 means only one entry is used → codebook collapse
        avg_probs = F.one_hot(concept_ids, self.n_concepts).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Track usage
        self.usage_count.scatter_add_(0, concept_ids, torch.ones_like(concept_ids, dtype=torch.long))
        self.total_updates += 1

        return quantized, vq_loss, concept_ids, perplexity

    def reset_dead_entries(self, z_batch: torch.Tensor):
        """
        Reset codebook entries that are rarely used (codebook collapse mitigation).

        If an entry has been used less than reset_threshold of the time,
        reinitialize it to a random encoder output from the current batch.

        This prevents "dead" codebook entries that waste capacity.

        Args:
            z_batch: Current batch of encoder outputs to sample replacements from.
        """
        if self.total_updates < 100:
            # Don't reset too early
            return 0

        # Compute usage fraction
        total_usage = self.usage_count.sum().float()
        if total_usage == 0:
            return 0

        usage_frac = self.usage_count.float() / total_usage
        dead_mask = usage_frac < self.reset_threshold

        n_dead = dead_mask.sum().item()
        if n_dead > 0 and len(z_batch) > 0:
            # Sample random encoder outputs as replacements
            dead_indices = torch.where(dead_mask)[0]
            for idx in dead_indices:
                # Pick a random encoder output + small noise
                rand_idx = torch.randint(0, len(z_batch), (1,)).item()
                noise = torch.randn_like(z_batch[rand_idx]) * 0.01
                self.embedding.weight.data[idx] = z_batch[rand_idx] + noise

            # Reset usage counts
            self.usage_count.zero_()
            self.total_updates = 0

        return n_dead

    def get_codebook_utilization(self):
        """
        Report codebook health statistics.

        Returns dict with:
            - active_entries: number of codebook entries that have been used
            - dead_entries: entries never used
            - usage_entropy: entropy of usage distribution (higher = healthier)
        """
        total = self.usage_count.sum().float()
        if total == 0:
            return {"active_entries": 0, "dead_entries": self.n_concepts,
                    "usage_entropy": 0.0}

        usage_frac = self.usage_count.float() / total
        active = (self.usage_count > 0).sum().item()
        dead = self.n_concepts - active

        # Entropy: higher = more uniform usage
        entropy = -torch.sum(
            usage_frac * torch.log(usage_frac + 1e-10)
        ).item()

        return {
            "active_entries": active,
            "dead_entries": dead,
            "usage_entropy": entropy,
            "max_entropy": np.log(self.n_concepts),
        }


class VQConceptEncoder(nn.Module):
    """
    End-to-end encoder with VQ bottleneck.

    Combines: CNN/MLP encoder → VQ layer → concept policy.
    This is the full differentiable pipeline for Phase 2.

    The forward pass produces both:
        - A discrete concept ID (for interpretability)
        - A quantized embedding (for the policy to use)
    """

    def __init__(self, encoder: nn.Module, n_concepts: int = 64,
                 embedding_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.encoder = encoder
        self.vq = VectorQuantizer(
            n_concepts=n_concepts,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            use_ema=True,
        )

    def forward(self, obs: torch.Tensor):
        """
        Full forward pass: obs → features → VQ → quantized + concept_id.

        Returns:
            quantized: (batch, embedding_dim) — quantized concept embedding.
            vq_loss: Scalar — VQ auxiliary loss (add to main RL loss).
            concept_ids: (batch,) — discrete concept IDs.
            perplexity: Scalar — codebook utilization metric.
        """
        features = self.encoder(obs)
        quantized, vq_loss, concept_ids, perplexity = self.vq(features)
        return quantized, vq_loss, concept_ids, perplexity

    def get_concept_ids(self, obs: torch.Tensor) -> torch.Tensor:
        """Get concept IDs without the full VQ overhead (for inference)."""
        with torch.no_grad():
            features = self.encoder(obs)
            distances = (
                features.pow(2).sum(1, keepdim=True)
                + self.vq.embedding.weight.pow(2).sum(1)
                - 2 * features @ self.vq.embedding.weight.t()
            )
            return distances.argmin(dim=1)
