"""
Concept Bottleneck Policy.

Policy that receives ONLY a concept ID (single integer) as input.
This is the core bottleneck — the policy must learn to act using only
the discrete concept representation, not the full feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from src.concept_manager import ConceptManager


class ConceptBottleneckPolicy(nn.Module):
    """
    Bottleneck policy that maps concept_id → action logits.

    Architecture: Embedding(n_concepts, embed_dim) → MLP → action logits.
    The concept ID is the ONLY input — this enforces the information bottleneck.
    """

    def __init__(self, n_concepts: int = 64, embed_dim: int = 64,
                 hidden_dim: int = 128, n_actions: int = 50):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_actions = n_actions

        self.embedding = nn.Embedding(n_concepts, embed_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, concept_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            concept_ids: (batch,) integer tensor of concept IDs.
            action_mask: (batch, n_actions) binary mask of legal actions.

        Returns:
            action_logits: (batch, n_actions) — masked if mask provided.
            state_values: (batch, 1).
        """
        embed = self.embedding(concept_ids)  # (batch, embed_dim)
        logits = self.policy_head(embed)     # (batch, n_actions)
        values = self.value_head(embed)      # (batch, 1)

        if action_mask is not None:
            # Mask illegal actions with very negative logits
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        return logits, values

    def get_action(self, concept_id: int,
                   action_mask: Optional[np.ndarray] = None,
                   deterministic: bool = False) -> int:
        """
        Select an action given a concept ID.

        Args:
            concept_id: Integer concept ID.
            action_mask: Binary mask of legal actions.
            deterministic: If True, pick argmax; else sample from distribution.

        Returns:
            Selected action (int).
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            cid = torch.LongTensor([concept_id]).to(device)
            mask = None
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).unsqueeze(0).to(device)

            logits, _ = self.forward(cid, mask)
            logits = logits[0]

            if deterministic:
                action = logits.argmax().item()
            else:
                probs = F.softmax(logits, dim=-1)
                # Handle all-zero probs (all actions masked)
                if probs.sum() < 1e-8:
                    if action_mask is not None:
                        legal = np.where(action_mask == 1)[0]
                        return int(np.random.choice(legal)) if len(legal) > 0 else 0
                    return 0
                action = torch.multinomial(probs, 1).item()

        return action

    def get_q_values(self, concept_ids: torch.Tensor,
                     action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get Q-values (using policy head as Q-head for DQN variant).

        Args:
            concept_ids: (batch,) integer tensor.
            action_mask: (batch, n_actions) binary mask.

        Returns:
            Q-values: (batch, n_actions).
        """
        embed = self.embedding(concept_ids)
        q_values = self.policy_head(embed)

        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))

        return q_values


class ConceptBottleneckAgent:
    """
    Complete concept bottleneck agent: encoder + concept_manager + bottleneck_policy.

    This is the full inference pipeline:
        obs → encoder → features → concept_manager → concept_id → policy → action
    """

    def __init__(self, encoder: nn.Module, concept_manager: ConceptManager,
                 policy: ConceptBottleneckPolicy,
                 device: torch.device = torch.device("cpu")):
        self.encoder = encoder.to(device)
        self.concept_manager = concept_manager
        self.policy = policy.to(device)
        self.device = device

        # Freeze encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_concept(self, obs: np.ndarray) -> int:
        """Get concept ID for an observation."""
        return self.concept_manager.assign_concept_from_obs(
            self.encoder, obs, self.device
        )

    def get_action(self, obs: np.ndarray,
                   action_mask: Optional[np.ndarray] = None,
                   deterministic: bool = False) -> Tuple[int, int]:
        """
        Full pipeline: obs → concept → action.

        Returns:
            (action, concept_id)
        """
        concept_id = self.get_concept(obs)
        action = self.policy.get_action(concept_id, action_mask, deterministic)
        return action, concept_id

    def get_action_with_override(self, obs: np.ndarray,
                                  override_concept: int,
                                  action_mask: Optional[np.ndarray] = None,
                                  deterministic: bool = True) -> int:
        """
        Get action with a forced concept override (for intervention experiments).

        Args:
            obs: Observation (unused except for mask).
            override_concept: Forced concept ID.
            action_mask: Legal action mask.
            deterministic: Whether to pick argmax.

        Returns:
            Action selected under the overridden concept.
        """
        return self.policy.get_action(override_concept, action_mask, deterministic)


class ConceptDQNPolicy(nn.Module):
    """
    DQN variant of the concept bottleneck policy.

    Separate from ConceptBottleneckPolicy to cleanly handle DQN-specific
    features: target network updates, epsilon-greedy, etc.
    """

    def __init__(self, n_concepts: int = 64, embed_dim: int = 64,
                 hidden_dim: int = 128, n_actions: int = 50):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_actions = n_actions
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(n_concepts, embed_dim)

        self.q_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, concept_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: concept_id → Q-values.

        Args:
            concept_ids: (batch,) integer tensor.
            action_mask: (batch, n_actions) binary mask.

        Returns:
            Q-values: (batch, n_actions).
        """
        embed = self.embedding(concept_ids)
        q_values = self.q_net(embed)

        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))

        return q_values

    def get_action(self, concept_id: int,
                   action_mask: Optional[np.ndarray] = None,
                   epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            concept_id: Integer concept ID.
            action_mask: Binary mask of legal actions.
            epsilon: Exploration rate.

        Returns:
            Selected action (int).
        """
        if np.random.random() < epsilon:
            if action_mask is not None:
                legal = np.where(action_mask == 1)[0]
                return int(np.random.choice(legal)) if len(legal) > 0 else 0
            return np.random.randint(0, self.n_actions)

        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            cid = torch.LongTensor([concept_id]).to(device)
            mask = None
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
            q_values = self.forward(cid, mask)[0]
            return q_values.argmax().item()
