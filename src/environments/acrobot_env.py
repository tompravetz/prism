"""
Acrobot Environment Wrapper for PRISM concept bottleneck testing.

Follows the same pattern as CartPoleConceptEnv and LunarLanderConceptEnv:
adds action_mask to the info dict and ensures float32 observations for
API compatibility with the GoEnv and the concept bottleneck pipeline.

Acrobot-v1 (Classic Control):
    - A two-link pendulum that must swing up to reach a target height
    - Goal: swing the free end above the target line
    - Episode ends when the tip reaches the target or after 500 steps

Observation (6D continuous vector):
    [0] cos(theta1): cosine of angle of first link
    [1] sin(theta1): sine of angle of first link
    [2] cos(theta2): cosine of angle of second link
    [3] sin(theta2): sine of angle of second link
    [4] angular velocity of first link
    [5] angular velocity of second link

Actions (3 discrete):
    0 = Apply -1 torque to joint
    1 = Apply 0 torque (do nothing)
    2 = Apply +1 torque to joint

Reward: -1 for each step until reaching the goal (maximally -500).
Good performance: reaching goal in ~80-100 steps (reward ~-80 to -100).
"""

import gymnasium as gym
import numpy as np


class AcrobotConceptEnv(gym.Wrapper):
    """
    Acrobot-v1 wrapper with interface matching GoEnv.

    All 3 actions are always legal, so action_mask is always all-ones.
    This exists purely for API compatibility with the concept bottleneck
    pipeline which expects action_mask in the info dict.
    """

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: "human" for visual rendering, None for headless.
        """
        env = gym.make("Acrobot-v1", render_mode=render_mode)
        super().__init__(env)
        self._action_count = 3  # -1 torque, 0 torque, +1 torque

    def action_masks(self):
        """All actions always legal in Acrobot."""
        return np.ones(self._action_count, dtype=np.int8)

    def reset(self, seed=None, options=None):
        """Reset Acrobot and add action_mask to info."""
        obs, info = self.env.reset(seed=seed, options=options)
        info["action_mask"] = self.action_masks()
        return obs.astype(np.float32), info

    def step(self, action):
        """Take a step in Acrobot."""
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        info["action_mask"] = self.action_masks()
        return obs.astype(np.float32), reward, terminated, truncated, info
