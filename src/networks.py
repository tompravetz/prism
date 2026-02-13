"""
Neural network architectures for PRISM.

This module defines the encoder networks that transform raw observations into
128-dimensional feature vectors, plus policy/value heads that map features to
actions and state values.

Architecture decisions:
    - CNN for Go (7x7x3 spatial input): Conv layers preserve spatial structure,
      which is critical for board games where position matters.
    - MLP for CartPole (4D vector input): No spatial structure to exploit,
      so a simple feedforward network suffices.

Both encoder types inherit from SB3's BaseFeaturesExtractor so they plug
directly into Stable Baselines 3's PPO/DQN policies. They can also be used
standalone for the concept bottleneck pipeline.

The encoder output (128D feature vector) serves two purposes:
    1. Direct input to standard RL policies (baseline agents)
    2. Input to K-Means clustering to discover concepts (bottleneck agents)
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

# SB3's base class for custom feature extractors.
# By inheriting from this, our encoders work seamlessly with SB3's PPO/DQN.
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GoCNNEncoder(BaseFeaturesExtractor):
    """
    Convolutional Neural Network encoder for Go board observations.

    Transforms the (7, 7, 3) board representation into a 128-dimensional
    feature vector. The CNN architecture is designed for small board games:
    all convolutions use 3x3 kernels with padding=1, which preserves the
    spatial dimensions (7x7) through each layer.

    Architecture:
        Conv2d(3→32, 3x3, pad=1) → BatchNorm → ReLU
        Conv2d(32→64, 3x3, pad=1) → BatchNorm → ReLU
        Conv2d(64→64, 3x3, pad=1) → BatchNorm → ReLU
        Flatten → Linear(64*7*7 → 128) → ReLU

    Why this architecture:
        - 3 conv layers with 3x3 kernels gives a receptive field of 7x7
          (the entire board), so the network can "see" the full board
        - BatchNorm helps with training stability, especially for RL where
          the data distribution shifts as the policy changes
        - No pooling layers: the board is already small (7x7), and we want
          to preserve spatial resolution for precise move selection
        - 128D output is a good balance: large enough to encode board
          complexity, small enough for efficient K-Means clustering

    Input: (batch, 7, 7, 3) — channels-last format from the Go environment
           Automatically converted to (batch, 3, 7, 7) channels-first for PyTorch
    Output: (batch, 128) — feature vector
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        """
        Args:
            observation_space: Gymnasium Box space describing the observation.
                               Expected shape: (7, 7, 3) for Go.
            features_dim: Size of the output feature vector (default 128).
        """
        super().__init__(observation_space, features_dim)

        # Determine number of input channels
        # Our Go env uses channels-last (H, W, C), but PyTorch CNNs expect
        # channels-first (C, H, W). We detect the format and convert if needed.
        n_input_channels = observation_space.shape[-1]  # 3 for (H, W, C)
        if observation_space.shape[0] == 3:
            n_input_channels = 3  # Already channels-first

        # The convolutional trunk: extracts spatial features from the board
        self.cnn = nn.Sequential(
            # Layer 1: 3→32 filters, detects basic patterns (single stones, edges)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Layer 2: 32→64 filters, detects compound patterns (groups, connections)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 3: 64→64 filters, detects higher-level patterns (influence, territory)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Compute the size of the flattened CNN output dynamically
        # For 7x7 input with no pooling: 64 channels * 7 * 7 = 3136
        # For 5x5 input: 64 * 5 * 5 = 1600 (used in curriculum learning)
        if observation_space.shape[0] == 3:
            # Channels-first: (C, H, W)
            h, w = observation_space.shape[1], observation_space.shape[2]
        else:
            # Channels-last: (H, W, C)
            h, w = observation_space.shape[0], observation_space.shape[1]
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, h, w)
            n_flatten = self.cnn(sample).flatten(1).shape[1]

        # The fully-connected head: compresses spatial features into a compact vector
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),  # 3136 → 128
            nn.ReLU(),
        )

        # Track whether we need to permute channels (most environments use channels-last)
        self._obs_is_channels_last = (observation_space.shape[-1] == 3
                                       and observation_space.shape[0] != 3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: board observation → 128D feature vector.

        Args:
            observations: (batch, 7, 7, 3) or (batch, 3, 7, 7) tensor.

        Returns:
            (batch, 128) feature tensor.
        """
        # Convert channels-last (B, H, W, C) to channels-first (B, C, H, W)
        # PyTorch's Conv2d expects channels-first format
        if self._obs_is_channels_last:
            observations = observations.permute(0, 3, 1, 2)
        return self.fc(self.cnn(observations))


class SimpleMLPEncoder(BaseFeaturesExtractor):
    """
    Multi-Layer Perceptron encoder for simple environments.

    Used for CartPole (4D input) and LunarLander (8D input) where the
    observation is a flat vector with no spatial structure.

    Architecture:
        Linear(obs_dim → 128) → ReLU
        Linear(128 → 128) → ReLU
        Linear(128 → features_dim) → ReLU

    Three layers is sufficient for these simple environments. The network
    learns to transform the raw physics state (positions, velocities, angles)
    into a feature space that's useful for both policy learning and concept
    clustering.

    Input: (batch, obs_dim) — e.g., (batch, 4) for CartPole
    Output: (batch, 128) — feature vector (same size as Go encoder for consistency)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        """
        Args:
            observation_space: Gymnasium Box space (e.g., shape=(4,) for CartPole).
            features_dim: Output feature vector size (default 128).
        """
        super().__init__(observation_space, features_dim)
        # Flatten the observation in case it's multi-dimensional
        obs_dim = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: flat observation → 128D feature vector.

        Args:
            observations: (batch, obs_dim) tensor.

        Returns:
            (batch, features_dim) feature tensor.
        """
        # Flatten just in case the input has extra dimensions
        return self.net(observations.flatten(1))


class QNetwork(nn.Module):
    """
    Q-Network head for DQN.

    Maps encoder features → Q-values for all actions. The agent picks
    the action with the highest Q-value (in exploitation mode).

    Q(s, a) estimates the expected total future reward from taking action a
    in state s and then following the optimal policy. Higher Q = better action.

    Architecture:
        Linear(features_dim → 256) → ReLU
        Linear(256 → 128) → ReLU
        Linear(128 → n_actions)

    Supports action masking: illegal moves get Q-value = -inf so they're
    never selected by argmax.
    """

    def __init__(self, features_dim: int, n_actions: int):
        """
        Args:
            features_dim: Size of input feature vector (128 from encoder).
            n_actions: Number of possible actions (50 for Go 7x7).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Q-values for all actions.

        Args:
            features: (batch, features_dim) from the encoder.
            action_mask: (batch, n_actions) binary mask. 1=legal, 0=illegal.
                         If provided, illegal actions get Q=-inf.

        Returns:
            Q-values: (batch, n_actions) tensor.
        """
        q_values = self.net(features)
        if action_mask is not None:
            # Set illegal action Q-values to -infinity so argmax never picks them.
            # masked_fill replaces values where mask==0 with -inf.
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
        return q_values


class PolicyNetwork(nn.Module):
    """
    Actor-Critic policy head for PPO.

    Two outputs:
        - Action logits: unnormalized probabilities for each action.
          Apply softmax to get a proper probability distribution.
        - State value: V(s), the estimated expected return from this state.

    PPO uses the policy for action selection and the value function for
    advantage estimation (how much better was this action than average?).
    """

    def __init__(self, features_dim: int, n_actions: int):
        """
        Args:
            features_dim: Size of input feature vector (128).
            n_actions: Number of possible actions.
        """
        super().__init__()
        # Policy head (actor): features → action logits
        self.policy = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        # Value head (critic): features → scalar state value
        self.value = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor):
        """
        Compute action logits and state value.

        Args:
            features: (batch, features_dim) from the encoder.

        Returns:
            (action_logits, state_value) tuple.
            action_logits: (batch, n_actions)
            state_value: (batch, 1)
        """
        return self.policy(features), self.value(features)


class DQNPolicy(nn.Module):
    """
    Complete DQN policy: encoder + Q-head combined into one module.

    This standalone module is used for custom DQN training when we need
    more control than SB3 provides (e.g., for bottleneck variants).

    The get_features() method is particularly useful: it lets us extract
    the intermediate 128D features for concept discovery without running
    the full Q-network.
    """

    def __init__(self, encoder: nn.Module, features_dim: int, n_actions: int):
        """
        Args:
            encoder: Feature extractor (GoCNNEncoder or SimpleMLPEncoder).
            features_dim: Output size of the encoder (128).
            n_actions: Number of possible actions.
        """
        super().__init__()
        self.encoder = encoder
        self.q_head = QNetwork(features_dim, n_actions)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Full forward pass: observation → Q-values.

        Args:
            obs: Raw observation tensor.
            action_mask: Optional legal action mask.

        Returns:
            Q-values: (batch, n_actions).
        """
        features = self.encoder(obs)
        return self.q_head(features, action_mask)

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract encoder features without computing Q-values.

        This is used during concept discovery: we run many observations
        through the encoder to collect feature vectors, then cluster them
        with K-Means to discover concepts.

        Args:
            obs: Raw observation tensor.

        Returns:
            (batch, features_dim) feature tensor.
        """
        return self.encoder(obs)


def get_encoder_for_env(env_name: str, observation_space: spaces.Box,
                        features_dim: int = 128) -> BaseFeaturesExtractor:
    """
    Factory function: get the appropriate encoder for an environment.

    Matches the encoder architecture to the observation structure:
        - Go (spatial 2D board) → CNN encoder
        - CartPole/LunarLander (1D vector) → MLP encoder

    Args:
        env_name: Environment identifier ("go", "cartpole", "lunarlander").
        observation_space: Gymnasium Box space of the environment.
        features_dim: Size of output feature vector.

    Returns:
        Encoder instance (GoCNNEncoder or SimpleMLPEncoder).
    """
    if env_name in ("go", "go_7x7", "go_5x5"):
        return GoCNNEncoder(observation_space, features_dim)
    elif env_name in ("cartpole", "lunarlander", "acrobot", "mountaincar"):
        return SimpleMLPEncoder(observation_space, features_dim)
    else:
        # Default to MLP for unknown environments
        return SimpleMLPEncoder(observation_space, features_dim)
