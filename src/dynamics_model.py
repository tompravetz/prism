"""
Concept Dynamics Model: Predicts next concept given current concept + action.

Phase 2B of ASTRIA. If concepts are meaningful, there should be predictable
dynamics: taking certain actions in certain concept states should lead to
predictable next concept states.

This is analogous to a "world model" but operating in concept space rather
than observation space. If accurate, it enables:
    1. Planning: look ahead in concept space to choose better actions
    2. Interpretation: understand cause-and-effect at the concept level
    3. Transfer: concept dynamics might transfer between similar domains

The model: P(concept_{t+1} | concept_t, action_t)
This is a classification problem with n_concepts output classes.

If accuracy > 40%: concept dynamics are structured and partially predictable.
If accuracy < 40%: concept transitions are too noisy/stochastic for prediction.
Both are informative results.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Optional

from src.utils import ensure_dir


class ConceptDynamicsModel(nn.Module):
    """
    MLP that predicts next concept given current concept and action.

    Architecture:
        concat(concept_embedding, action_one_hot) → MLP → logits over next concepts

    The model learns the "physics" of the concept space: how actions cause
    transitions between concepts.
    """

    def __init__(self, n_concepts: int = 64, n_actions: int = 50,
                 embed_dim: int = 32, hidden_dim: int = 128):
        """
        Args:
            n_concepts: Number of concept categories.
            n_actions: Number of possible actions.
            embed_dim: Dimension of concept embedding.
            hidden_dim: Width of hidden layers.
        """
        super().__init__()

        self.n_concepts = n_concepts
        self.n_actions = n_actions

        # Embed the current concept into a dense vector
        self.concept_embedding = nn.Embedding(n_concepts, embed_dim)

        # Action is one-hot encoded, so input to MLP is embed_dim + n_actions
        input_dim = embed_dim + n_actions

        # MLP that predicts distribution over next concepts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_concepts),  # Output: logits over next concept
        )

    def forward(self, concept_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict distribution over next concepts.

        Args:
            concept_ids: (batch,) integer tensor of current concept IDs.
            actions: (batch,) integer tensor of actions taken.

        Returns:
            logits: (batch, n_concepts) — unnormalized log-probabilities of next concept.
        """
        # Embed current concept
        concept_emb = self.concept_embedding(concept_ids)  # (batch, embed_dim)

        # One-hot encode action
        action_onehot = F.one_hot(actions, self.n_actions).float()  # (batch, n_actions)

        # Concatenate and predict
        x = torch.cat([concept_emb, action_onehot], dim=-1)  # (batch, embed_dim + n_actions)
        logits = self.net(x)  # (batch, n_concepts)

        return logits

    def predict(self, concept_id: int, action: int) -> np.ndarray:
        """
        Predict probability distribution over next concepts (single input).

        Args:
            concept_id: Current concept ID.
            action: Action taken.

        Returns:
            (n_concepts,) probability array.
        """
        self.eval()
        with torch.no_grad():
            c = torch.LongTensor([concept_id])
            a = torch.LongTensor([action])
            logits = self.forward(c, a)
            probs = F.softmax(logits, dim=-1)
            return probs[0].numpy()

    def predict_top_k(self, concept_id: int, action: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict top-K most likely next concepts.

        Returns:
            List of (concept_id, probability) tuples, sorted by probability.
        """
        probs = self.predict(concept_id, action)
        top_indices = np.argsort(probs)[::-1][:k]
        return [(int(idx), float(probs[idx])) for idx in top_indices]


class TransitionCollector:
    """
    Collects concept transitions during training/evaluation for the dynamics model.

    Records: (concept_t, action_t, concept_{t+1}) triples.
    """

    def __init__(self, max_size: int = 100_000):
        self.transitions = deque(maxlen=max_size)

    def add(self, concept_t: int, action: int, concept_t1: int):
        """Record a single transition."""
        self.transitions.append((concept_t, action, concept_t1))

    def get_dataset(self, train_frac: float = 0.8):
        """
        Split collected transitions into train/test sets.

        Returns:
            (train_concepts, train_actions, train_next_concepts,
             test_concepts, test_actions, test_next_concepts)
            Each is a numpy array.
        """
        data = list(self.transitions)
        np.random.shuffle(data)

        n = len(data)
        split = int(n * train_frac)

        concepts = np.array([d[0] for d in data])
        actions = np.array([d[1] for d in data])
        next_concepts = np.array([d[2] for d in data])

        return (
            concepts[:split], actions[:split], next_concepts[:split],
            concepts[split:], actions[split:], next_concepts[split:],
        )

    def __len__(self):
        return len(self.transitions)


def train_dynamics_model(transitions: TransitionCollector,
                         n_concepts: int = 64, n_actions: int = 50,
                         n_epochs: int = 50, lr: float = 1e-3,
                         batch_size: int = 256, device: str = "cpu"):
    """
    Train the concept dynamics model on collected transitions.

    Uses cross-entropy loss: the model predicts a distribution over next
    concepts, and we minimize the negative log-likelihood of the true next concept.

    Args:
        transitions: TransitionCollector with collected data.
        n_concepts: Number of concept categories.
        n_actions: Number of possible actions.
        n_epochs: Training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        device: Torch device.

    Returns:
        model: Trained ConceptDynamicsModel.
        results: Dictionary with training/test accuracy.
    """
    device = torch.device(device)

    # Split data
    (train_c, train_a, train_nc,
     test_c, test_a, test_nc) = transitions.get_dataset(train_frac=0.8)

    print(f"Dynamics model: {len(train_c)} train, {len(test_c)} test transitions")

    # Create model
    model = ConceptDynamicsModel(n_concepts, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    n_train = len(train_c)
    best_test_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        # Shuffle training data
        perm = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            c_batch = torch.LongTensor(train_c[idx]).to(device)
            a_batch = torch.LongTensor(train_a[idx]).to(device)
            nc_batch = torch.LongTensor(train_nc[idx]).to(device)

            # Forward pass: predict next concept distribution
            logits = model(c_batch, a_batch)

            # Cross-entropy loss
            loss = F.cross_entropy(logits, nc_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == nc_batch).sum().item()

        train_acc = correct / n_train
        train_loss = total_loss / n_train

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_c_t = torch.LongTensor(test_c).to(device)
            test_a_t = torch.LongTensor(test_a).to(device)
            test_nc_t = torch.LongTensor(test_nc).to(device)

            test_logits = model(test_c_t, test_a_t)
            test_acc = (test_logits.argmax(1) == test_nc_t).float().mean().item()
            test_loss = F.cross_entropy(test_logits, test_nc_t).item()

        best_test_acc = max(best_test_acc, test_acc)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.2%} | "
                  f"Test: loss={test_loss:.4f} acc={test_acc:.2%} "
                  f"(best={best_test_acc:.2%})")

    # Compute top-5 test accuracy (is true next concept in top 5 predictions?)
    model.eval()
    with torch.no_grad():
        test_c_t = torch.LongTensor(test_c).to(device)
        test_a_t = torch.LongTensor(test_a).to(device)
        test_nc_t = torch.LongTensor(test_nc).to(device)
        test_logits = model(test_c_t, test_a_t)
        top5_preds = test_logits.topk(5, dim=1).indices
        top5_correct = (top5_preds == test_nc_t.unsqueeze(1)).any(1).float().mean().item()

    results = {
        "n_train": len(train_c),
        "n_test": len(test_c),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "best_test_accuracy": float(best_test_acc),
        "top5_test_accuracy": float(top5_correct),
        "test_loss": float(test_loss),
    }

    print(f"\nDynamics Model Results:")
    print(f"  Top-1 accuracy: {test_acc:.2%}")
    print(f"  Top-5 accuracy: {top5_correct:.2%}")
    print(f"  {'GOOD' if test_acc > 0.40 else 'INFORMATIVE'}: "
          f"{'Concept dynamics are structured' if test_acc > 0.40 else 'Transitions are stochastic'}")

    return model, results
