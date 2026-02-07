"""
Simple Environment Wrappers for task-agnosticism testing.

These wrappers add a consistent interface around standard Gymnasium environments
(CartPole, LunarLander) so they match our GoEnv's API. This lets us use the
EXACT same concept bottleneck code on these environments.

Why this matters: If the concept bottleneck architecture only works on Go,
it might just be an artifact of that specific domain. By showing it works on
CartPole too, we prove the approach is general-purpose (task-agnostic).

Key additions over raw Gymnasium:
    - action_masks() method (always returns all-ones since these envs don't
      have illegal actions, but required for API compatibility)
    - action_mask in the info dict (same reason)
    - float32 observation dtype (some Gymnasium envs return float64)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CartPoleConceptEnv(gym.Wrapper):
    """
    CartPole wrapper with interface matching GoEnv.

    CartPole-v1 (Classic Control):
        - A pole is attached to a cart on a frictionless track
        - Goal: keep the pole balanced by pushing the cart left or right
        - Episode ends when pole tilts >12 degrees or cart moves >2.4 units
        - Maximum episode length: 500 steps

    Observation (4D continuous vector):
        [0] Cart position: float in [-4.8, 4.8]
        [1] Cart velocity: float in [-inf, inf]
        [2] Pole angle: float in [-0.42 rad, 0.42 rad] (~24 degrees)
        [3] Pole angular velocity: float in [-inf, inf]

    Actions (2 discrete):
        0 = Push cart left
        1 = Push cart right

    Reward: +1 for every step the pole stays balanced.
    Good performance: consistently reaching 400-500 steps.

    Why CartPole for concept testing:
        The 4D observation space is simple enough that K=32 concepts should
        capture the key states (e.g., "tilting left", "balanced", "tilting right").
        This makes it easy to verify concepts are meaningful.
    """

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: "human" for visual rendering, None for headless.
        """
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        self._action_count = 2  # Left or right

    def action_masks(self):
        """
        Return action mask (all actions always legal in CartPole).

        CartPole has no illegal actions — you can always push left or right.
        This method exists for API compatibility with our Go environment
        and SB3's MaskablePPO.

        Returns:
            numpy array of ones with shape (2,).
        """
        return np.ones(self._action_count, dtype=np.int8)

    def reset(self, seed=None, options=None):
        """
        Reset CartPole and add action_mask to info dict.

        Returns:
            obs: (4,) float32 array — [cart_pos, cart_vel, pole_angle, pole_vel].
            info: Dict with "action_mask" key.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info["action_mask"] = self.action_masks()
        # Ensure float32 dtype (some Gymnasium versions return float64)
        return obs.astype(np.float32), info

    def step(self, action):
        """
        Take a step in CartPole.

        Args:
            action: 0 (push left) or 1 (push right).

        Returns:
            obs: (4,) float32 observation.
            reward: +1.0 for each step the pole stays balanced.
            terminated: True if pole fell or cart moved too far.
            truncated: True if 500 steps reached.
            info: Dict with "action_mask".
        """
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        info["action_mask"] = self.action_masks()
        return obs.astype(np.float32), reward, terminated, truncated, info


class LunarLanderConceptEnv(gym.Wrapper):
    """
    LunarLander wrapper with interface matching GoEnv.

    LunarLander-v3 (Box2D):
        - A spacecraft must land safely on a landing pad
        - Goal: land softly between the flags without crashing
        - Episode ends on landing (good or crash) or going off-screen

    Observation (8D continuous vector):
        [0] x position: horizontal position
        [1] y position: vertical position
        [2] x velocity: horizontal speed
        [3] y velocity: vertical speed (negative = falling)
        [4] angle: tilt of the lander
        [5] angular velocity: rotation speed
        [6] left leg contact: 1.0 if left leg touching ground, else 0.0
        [7] right leg contact: 1.0 if right leg touching ground, else 0.0

    Actions (4 discrete):
        0 = Do nothing (coast)
        1 = Fire left thruster (push right)
        2 = Fire main thruster (push up)
        3 = Fire right thruster (push left)

    Reward: complex shaped reward:
        +100/-100 for landing/crashing
        +10 per leg contact
        -0.3 per main thruster fire (fuel cost)
        -0.03 per side thruster fire
    """

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: "human" for visual rendering, None for headless.
        """
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        super().__init__(env)
        self._action_count = 4  # Nothing, left, main, right

    def action_masks(self):
        """All actions always legal in LunarLander."""
        return np.ones(self._action_count, dtype=np.int8)

    def reset(self, seed=None, options=None):
        """Reset LunarLander and add action_mask to info."""
        obs, info = self.env.reset(seed=seed, options=options)
        info["action_mask"] = self.action_masks()
        return obs.astype(np.float32), info

    def step(self, action):
        """Take a step in LunarLander."""
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        info["action_mask"] = self.action_masks()
        return obs.astype(np.float32), reward, terminated, truncated, info
