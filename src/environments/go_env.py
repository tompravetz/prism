"""
Go 7x7 Environment Wrapper.

Wraps PettingZoo's Go AEC (Agent Environment Cycle) environment into a
single-agent Gymnasium interface. PettingZoo environments are multi-agent
by default (two players alternate turns), but for RL training we need a
standard single-agent interface where:
    - Our agent (black) takes an action
    - The opponent (white) automatically responds
    - The environment returns the next observation

This wrapper handles:
    1. Converting PettingZoo's AEC turn-based protocol to standard Gym step/reset
    2. Converting the 17-plane PettingZoo observation to a simpler 3-plane format
       (black stones, white stones, empty intersections)
    3. Managing the opponent's turns transparently
    4. Providing action masks for legal move enforcement
    5. Extracting rewards (Go uses terminal rewards: +1 win, -1 loss)

The agent plays as black (goes first). The opponent defaults to random legal moves.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Try to import PettingZoo Go; provide a clear error message if unavailable.
# PettingZoo is a library for multi-agent RL environments.
# The [classic] extra includes board games like Go, Chess, etc.
try:
    from pettingzoo.classic import go_v5
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False


class GoEnv(gym.Env):
    """
    Single-agent Gym wrapper around PettingZoo Go (7x7).

    This is the main environment used for training and evaluation.

    Observation space: Box(0, 1, shape=(7, 7, 3), dtype=float32)
        - Plane 0: Black stones (1.0 where black has a stone, 0.0 elsewhere)
        - Plane 1: White stones (1.0 where white has a stone, 0.0 elsewhere)
        - Plane 2: Empty intersections (1.0 where no stone, 0.0 elsewhere)
        Note: planes 0 + 1 + 2 always sum to 1.0 at every position.

    Action space: Discrete(50)
        - Actions 0-48: place a stone at position (action // 7, action % 7)
        - Action 49: pass (decline to place a stone)

    Rewards: sparse, only at game end
        - +1.0 for winning
        - -1.0 for losing
        - 0.0 during the game

    The agent plays as black_0; white_0 uses `opponent_fn` (default: random).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, board_size=7, opponent_fn=None, render_mode=None,
                 reward_shaping=False, capture_reward=0.05):
        """
        Initialize the Go environment wrapper.

        Args:
            board_size: Size of the Go board (default 7 for 7x7).
                        Standard Go is 19x19, but 7x7 is much faster to train.
            opponent_fn: Callable(obs, action_mask) -> action for the white player.
                         If None, uses random legal moves. Can be replaced with a
                         trained model for self-play or stronger opponents.
            render_mode: "human" prints to console, "ansi" returns string.
            reward_shaping: If True, add small intermediate rewards for captures.
                           This gives DQN denser learning signal — instead of only
                           +1/-1 at game end, it gets small bonuses throughout.
                           Default False to keep PPO training unchanged.
            capture_reward: Reward per captured stone when reward_shaping=True.
                           0.05 means capturing 3 opponent stones = +0.15 bonus.
                           Kept small so terminal win/loss (+1/-1) still dominates.
        """
        super().__init__()
        if not PETTINGZOO_AVAILABLE:
            raise ImportError(
                "PettingZoo Go is not available. Install with: "
                "pip install 'pettingzoo[classic]'"
            )

        self.board_size = board_size
        # Total board positions: 7*7 = 49
        self.num_positions = board_size * board_size
        # Total actions: 49 board positions + 1 pass action = 50
        self.action_count = self.num_positions + 1
        self.render_mode = render_mode

        # Reward shaping configuration
        # When enabled, the agent receives small intermediate rewards for
        # capturing opponent stones, making the reward signal denser.
        # This is critical for DQN which struggles with sparse terminal rewards
        # because it relies on one-step bootstrapping (Q-learning target:
        # r + gamma * max Q(s', a')) — with only terminal rewards, Q-values
        # must propagate backwards through ~30-50 moves, which is very slow.
        self.reward_shaping = reward_shaping
        self.capture_reward = capture_reward

        # Track previous stone counts for reward shaping.
        # _prev_black_count and _prev_white_count store how many stones
        # of each color were on the board at the end of the last step.
        # By comparing with current counts, we detect captures:
        #   - If white stones decreased: we captured some → positive reward
        #   - If black stones decreased: opponent captured ours → negative reward
        self._prev_black_count = 0
        self._prev_white_count = 0

        # Opponent policy: defaults to random legal moves.
        # This can be swapped during training for self-play opponents.
        self.opponent_fn = opponent_fn or self._random_opponent

        # Define Gymnasium spaces for SB3 compatibility
        # Observation: 3 binary planes stacked as a (7, 7, 3) tensor
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(board_size, board_size, 3),
            dtype=np.float32,
        )
        # Action: single integer from 0 to 49
        self.action_space = spaces.Discrete(self.action_count)

        # Internal PettingZoo environment (created on reset)
        self._env = None
        # PettingZoo agent names (fixed by the Go environment)
        self._agent_name = "black_0"      # Our agent (goes first in Go)
        self._opponent_name = "white_0"    # The opponent

    def _create_env(self):
        """
        Create a fresh PettingZoo Go environment.

        komi=5.5 is the standard compensation given to white for going second.
        In Go, white receives 5.5 extra points when counting the final score
        to offset black's first-move advantage.
        """
        env = go_v5.env(board_size=self.board_size, komi=5.5)
        return env

    def _extract_obs(self, raw_obs):
        """
        Convert PettingZoo's multi-plane observation to our simpler 3-plane format.

        PettingZoo Go provides a (board_size, board_size, 17) observation with:
            - Plane 0: Current player's stones (binary)
            - Plane 1: Opponent's stones (binary)
            - Planes 2-16: Move history and other metadata

        We simplify to (board_size, board_size, 3):
            - Plane 0: Black stones
            - Plane 1: White stones
            - Plane 2: Empty intersections (computed as 1 - black - white)

        This is sufficient for basic play and much simpler for the CNN encoder.

        Args:
            raw_obs: Raw PettingZoo observation array.

        Returns:
            numpy array of shape (board_size, board_size, 3).
        """
        if raw_obs is None:
            # Return empty board if no observation available
            return np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)

        obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)

        if len(raw_obs.shape) == 3 and raw_obs.shape[2] >= 2:
            # Standard case: extract first two planes
            obs[:, :, 0] = raw_obs[:, :, 0]  # Current player's stones (black when it's our turn)
            obs[:, :, 1] = raw_obs[:, :, 1]  # Opponent's stones (white)
            # Compute empty intersections: positions that have neither black nor white stones
            obs[:, :, 2] = 1.0 - obs[:, :, 0] - obs[:, :, 1]
        else:
            # Fallback for unexpected observation formats
            obs[:, :, 0] = raw_obs[:, :, 0] if len(raw_obs.shape) == 3 else raw_obs

        return obs

    def _get_action_mask(self):
        """
        Get binary mask of legal actions for the current agent.

        In Go, many moves are illegal:
            - Occupied positions (can't place on existing stone)
            - Ko rule violations (can't recreate the previous board state)
            - Suicide moves (can't place where your stone would be immediately captured,
              unless it also captures opponent stones)

        The mask is a binary array where 1 = legal and 0 = illegal.
        This is used by MaskablePPO and our custom DQN to avoid illegal moves.

        Returns:
            numpy array of shape (action_count,) with dtype int8.
        """
        if self._env is None:
            return np.ones(self.action_count, dtype=np.int8)

        agent = self._env.agent_selection
        # Try to get mask from the info dict first (fastest)
        mask = self._env.infos.get(agent, {}).get("action_mask", None)
        if mask is None:
            # Fallback: get mask from observe() (some PettingZoo versions)
            obs_dict = self._env.observe(agent)
            if isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                mask = obs_dict["action_mask"]
            else:
                # Last resort: allow all actions (not ideal but prevents crash)
                mask = np.ones(self.action_count, dtype=np.int8)

        # Handle size mismatches between PettingZoo's mask and our action count.
        # PettingZoo might include extra actions or have a different size.
        if len(mask) > self.action_count:
            mask = mask[:self.action_count]
        elif len(mask) < self.action_count:
            padded = np.zeros(self.action_count, dtype=np.int8)
            padded[:len(mask)] = mask
            mask = padded

        return np.asarray(mask, dtype=np.int8)

    def _random_opponent(self, obs, action_mask):
        """
        Default opponent policy: pick a random legal move.

        This is the simplest possible opponent. Good for initial training,
        but agents trained only against random opponents may develop exploitable
        strategies. Use opponent pool for stronger training later.

        Args:
            obs: Board observation (unused by random opponent).
            action_mask: Binary mask of legal actions.

        Returns:
            Integer action.
        """
        legal = np.where(action_mask == 1)[0]
        if len(legal) == 0:
            return self.num_positions  # pass if no legal moves
        return np.random.choice(legal)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new game.

        Creates a fresh PettingZoo Go environment and handles any initial
        opponent turns (shouldn't happen since black goes first, but we
        handle it for robustness).

        Args:
            seed: Random seed for reproducibility.
            options: Unused, included for Gymnasium API compatibility.

        Returns:
            obs: Initial observation (empty board), shape (7, 7, 3).
            info: Dict with "action_mask" key for legal actions.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Create a brand new game
        self._env = self._create_env()
        self._env.reset(seed=seed)

        # Notify opponent of new game (e.g. GnuGo needs clear_board between games)
        if hasattr(self.opponent_fn, 'reset'):
            self.opponent_fn.reset()

        # Handle edge case: if opponent somehow goes first
        self._handle_opponent_turns()

        # Get initial observation (should be empty board)
        obs = self._get_agent_obs()
        info = {"action_mask": self._get_action_mask()}

        # Initialize stone counts for reward shaping.
        # On an empty board both counts are 0.
        if self.reward_shaping:
            self._prev_black_count = np.sum(obs[:, :, 0] > 0.5)
            self._prev_white_count = np.sum(obs[:, :, 1] > 0.5)

        return obs, info

    def _get_agent_obs(self):
        """
        Get the current observation from our agent's perspective.

        Only valid when it's our agent's turn to move.

        Returns:
            numpy array of shape (board_size, board_size, 3).
        """
        if self._env.agent_selection == self._agent_name:
            obs_dict = self._env.observe(self._agent_name)
            if isinstance(obs_dict, dict):
                # PettingZoo returns {"observation": ..., "action_mask": ...}
                raw = obs_dict.get("observation", obs_dict)
            else:
                raw = obs_dict
            return self._extract_obs(raw)
        # If it's not our turn, return zeros (shouldn't happen in normal flow)
        return np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)

    def _is_agent_dead(self, agent_name):
        """
        Check if an agent is dead (terminated/truncated) in PettingZoo.

        In PettingZoo's AEC protocol, after a game ends, agents may still
        appear in the turn cycle but are "dead". Dead agents MUST be stepped
        with action=None, or PettingZoo raises a ValueError.

        Args:
            agent_name: The agent to check (e.g., "black_0" or "white_0").

        Returns:
            True if the agent is dead and should receive None as action.
        """
        # Check terminations dict (present in newer PettingZoo versions)
        if hasattr(self._env, 'terminations'):
            if self._env.terminations.get(agent_name, False):
                return True
        if hasattr(self._env, 'truncations'):
            if self._env.truncations.get(agent_name, False):
                return True

        # Alternative: use env.last() to check current agent's status
        # This is the most reliable method across PettingZoo versions
        if self._env.agent_selection == agent_name:
            try:
                _, _, termination, truncation, _ = self._env.last()
                if termination or truncation:
                    return True
            except Exception:
                pass

        return False

    def _handle_opponent_turns(self):
        """
        Let the opponent play until it's our agent's turn or the game ends.

        In PettingZoo's AEC (Agent Environment Cycle) protocol, agents take
        turns. After our agent plays, the opponent may need to play one or
        more turns before it's our turn again. This method handles all
        opponent turns transparently.

        IMPORTANT: If an agent is "dead" (game already ended for them),
        PettingZoo requires stepping them with action=None. Passing a real
        action to a dead agent raises ValueError. We check for this using
        env.last() or the terminations/truncations dicts.

        The max_steps safety limit prevents infinite loops if the game
        enters an unexpected state.
        """
        max_steps = 10  # Safety limit to prevent infinite loops
        steps = 0
        while (
            self._env.agents                                    # Game not over
            and self._env.agent_selection == self._opponent_name # It's opponent's turn
            and steps < max_steps                                # Safety limit
        ):
            # Check if this agent is dead — if so, step with None
            if self._is_agent_dead(self._opponent_name):
                self._env.step(None)
                steps += 1
                continue

            # Get opponent's view of the board
            obs_dict = self._env.observe(self._opponent_name)
            if isinstance(obs_dict, dict):
                mask = obs_dict.get("action_mask", np.ones(self.action_count, dtype=np.int8))
                raw_obs = obs_dict.get("observation", None)
            else:
                mask = np.ones(self.action_count, dtype=np.int8)
                raw_obs = obs_dict

            # Ask the opponent policy for an action
            opp_obs = self._extract_obs(raw_obs)
            opp_action = self.opponent_fn(opp_obs, mask)

            # Validate the opponent's action (ensure it's legal)
            opp_action = int(np.clip(opp_action, 0, self.action_count - 1))
            if mask[opp_action] != 1:
                # If the opponent's chosen action is illegal, fall back to random
                legal = np.where(mask == 1)[0]
                opp_action = int(np.random.choice(legal)) if len(legal) > 0 else self.num_positions

            # Execute opponent's move
            self._env.step(opp_action)
            steps += 1

    def _compute_shaped_reward(self, obs):
        """
        Compute intermediate reward based on stone captures.

        Compares the current board with the previous board to detect captures:
            - If opponent (white) stones decreased → we captured them → positive reward
            - If our (black) stones decreased → opponent captured ours → negative reward

        The shaped reward is kept small (default 0.05 per stone) so the terminal
        win/loss reward (+1/-1) still dominates the value function. This prevents
        the agent from optimizing for captures over winning.

        Example: If we capture 2 opponent stones and lose 0 of ours:
            shaped_reward = 2 * 0.05 - 0 * 0.05 = +0.10

        Args:
            obs: Current observation (7, 7, 3) — used to count stones.

        Returns:
            Float shaped reward (positive = net captures in our favor).
        """
        if not self.reward_shaping:
            return 0.0

        # Count current stones on the board
        # obs[:, :, 0] > 0.5 gives a boolean mask of black stone positions
        curr_black = np.sum(obs[:, :, 0] > 0.5)
        curr_white = np.sum(obs[:, :, 1] > 0.5)

        # Detect captures by comparing with previous counts.
        # Note: between steps, BOTH players place a stone, so counts naturally
        # increase by 1 each (ignoring captures). We only care about decreases:
        #   - If white stones decreased from prev, we captured some
        #   - If black stones decreased from prev, opponent captured ours
        # But since both players also PLACE stones, we need to account for that.
        # Simpler approach: just look at net stone count changes.
        #
        # Actually the clearest signal is: did opponent's stones go DOWN?
        # If prev_white=5 and curr_white=3, we captured 2 (even though opponent
        # also placed one, the net decrease means captures happened).
        white_captured = max(0, self._prev_white_count - curr_white)
        black_captured = max(0, self._prev_black_count - curr_black)

        # Update stored counts for next step
        self._prev_black_count = curr_black
        self._prev_white_count = curr_white

        # Net shaped reward: bonus for capturing opponent, penalty for losing ours
        shaped = (white_captured - black_captured) * self.capture_reward
        return shaped

    def step(self, action):
        """
        Take an action as our agent (black).

        This is the main interaction method. The flow is:
            1. Validate and execute our action
            2. Let opponent respond (via _handle_opponent_turns)
            3. Check if the game is over
            4. Return new observation, reward, and termination status

        When reward_shaping=True, small intermediate rewards are added for
        capturing opponent stones. This makes the reward signal denser, which
        is particularly helpful for DQN (which struggles with sparse rewards).

        Args:
            action: Integer action (0-48 = board position, 49 = pass).

        Returns:
            obs: New observation after both sides have moved, shape (7, 7, 3).
            reward: Float reward (+1 win, -1 loss, 0 during game; plus shaped
                    reward if reward_shaping=True).
            terminated: True if game is over (someone won or both passed).
            truncated: Always False (Go games end naturally, not by time limit).
            info: Dict with "action_mask" for the next legal actions.
        """
        action = int(action)

        # Check if game already ended (shouldn't happen but prevents crash)
        if not self._env.agents:
            obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
            return obs, 0.0, True, False, {"action_mask": np.zeros(self.action_count, dtype=np.int8)}

        # Check if our agent is dead — if so, step with None and end.
        # PettingZoo delivers the reward via env.last() when the agent is
        # terminated, so we capture it before stepping with None.
        if self._is_agent_dead(self._agent_name):
            _, reward, _, _, _ = self._env.last()
            self._env.step(None)
            obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
            return obs, float(reward), True, False, {"action_mask": np.zeros(self.action_count, dtype=np.int8)}

        # Validate action against the legal move mask
        mask = self._get_action_mask()
        if mask[action] != 1:
            # If the agent tries an illegal move, force it to pass instead.
            # This prevents crashes but the agent should learn to avoid this.
            action = self.num_positions  # pass

        # Execute our agent's move in the PettingZoo environment
        self._env.step(action)

        # Let the opponent respond (plays until it's our turn again)
        self._handle_opponent_turns()

        # Now check the game state. There are three possibilities:
        #   1. Game continues → it's our turn, agent is alive → return obs
        #   2. Game ended → it's "our turn" but we're dead → capture reward
        #   3. No agents left → game fully over → return terminal

        if not self._env.agents:
            # Case 3: Game fully over, agents list is empty.
            # Rewards were already consumed. Return 0 reward.
            obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
            return obs, 0.0, True, False, {"action_mask": np.zeros(self.action_count, dtype=np.int8)}

        if self._env.agent_selection == self._agent_name:
            # It's our turn — check if we're dead (game ended) or alive
            _, reward, termination, truncation, _ = self._env.last()

            if termination or truncation:
                # Case 2: Game ended, PettingZoo delivered our reward via last().
                # Step with None to acknowledge the terminal state.
                self._env.step(None)
                # Also drain the opponent's dead turn if needed
                while self._env.agents and self._is_agent_dead(self._env.agent_selection):
                    self._env.step(None)
                obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
                return obs, float(reward), True, False, {"action_mask": np.zeros(self.action_count, dtype=np.int8)}
            else:
                # Case 1: Game continues normally. Return observation.
                obs = self._get_agent_obs()
                info = {"action_mask": self._get_action_mask()}
                # Add shaped reward for captures (0.0 if shaping disabled)
                shaped_reward = self._compute_shaped_reward(obs)
                return obs, shaped_reward, False, False, info

        # Edge case: it's the opponent's turn but they're not dead.
        # This shouldn't happen after _handle_opponent_turns, but handle it.
        obs = self._get_agent_obs()
        info = {"action_mask": self._get_action_mask()}
        shaped_reward = self._compute_shaped_reward(obs)
        return obs, shaped_reward, False, False, info

    def render(self):
        """Render the board (either print to console or return string)."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())

    def _render_ansi(self):
        """
        Create an ASCII art representation of the board.

        Example output for a 7x7 board:
            0 1 2 3 4 5 6
          0 . . . . . . .
          1 . . B . . . .
          2 . . . W . . .
          ...
        """
        if self._env is None:
            return "No game in progress"

        obs = self._get_agent_obs()
        lines = []
        # Column headers
        lines.append("  " + " ".join(str(i) for i in range(self.board_size)))
        for r in range(self.board_size):
            row = f"{r} "
            for c in range(self.board_size):
                if obs[r, c, 0] > 0.5:
                    row += "B "  # Black stone
                elif obs[r, c, 1] > 0.5:
                    row += "W "  # White stone
                else:
                    row += ". "  # Empty intersection
            lines.append(row)
        return "\n".join(lines)

    def close(self):
        """Clean up the PettingZoo environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def unwrapped_env(self):
        """Access the underlying PettingZoo environment (for debugging)."""
        return self._env


class MaskedGoEnv(GoEnv):
    """
    GoEnv with SB3-compatible action masking interface.

    This subclass adds the `action_masks()` method required by
    sb3-contrib's MaskablePPO and the ActionMasker wrapper.

    SB3's MaskablePPO calls `env.action_masks()` during training to get
    the current legal actions. Without this, the PPO agent would try
    illegal moves and crash.

    Inherits reward_shaping support from GoEnv — pass reward_shaping=True
    to enable intermediate capture rewards (useful for DQN).

    Usage with SB3:
        from sb3_contrib.common.wrappers import ActionMasker
        env = MaskedGoEnv(board_size=7)
        env = ActionMasker(env, lambda env: env.action_masks())
        model = MaskablePPO("MlpPolicy", env)
    """

    def action_masks(self):
        """
        Return current action mask.

        Required by sb3-contrib's MaskablePPO. Returns a binary array where
        1 = legal action and 0 = illegal action.

        Returns:
            numpy array of shape (action_count,) with dtype int8.
        """
        return self._get_action_mask()

    def reset(self, seed=None, options=None):
        """Reset and cache the action mask for SB3 compatibility."""
        obs, info = super().reset(seed=seed, options=options)
        # Cache the mask so action_masks() can return it even before step()
        self._last_mask = info.get("action_mask", np.ones(self.action_count, dtype=np.int8))
        return obs, info

    def step(self, action):
        """Step and cache the new action mask."""
        obs, reward, terminated, truncated, info = super().step(action)
        self._last_mask = info.get("action_mask", np.zeros(self.action_count, dtype=np.int8))
        return obs, reward, terminated, truncated, info


# ============================================================
# Symmetry Augmentation for Rotation-Invariant Feature Learning
# ============================================================

def _build_symmetry_tables(board_size=7):
    """
    Precompute action mapping tables for all 8 board symmetries.

    Go boards have 8-fold symmetry (the dihedral group D4):
        - 4 rotations: 0°, 90°, 180°, 270°
        - Each rotation × 2 reflections: original, horizontally flipped

    When we rotate/flip the observation, we must also rotate/flip the
    actions (board positions) to match. For example, if the observation
    is rotated 90° clockwise, then action (0,0) should map to (0,6),
    action (0,1) to (1,6), etc.

    The pass action (board_size²) is unchanged by any symmetry.

    Returns:
        List of 8 dicts, each mapping original_action → transformed_action.
        Also returns inverse tables for mapping back.
    """
    n = board_size
    forward_tables = []   # obs_transform_index → {old_action: new_action}
    inverse_tables = []   # obs_transform_index → {new_action: old_action}

    for k in range(4):       # 4 rotations
        for flip in [False, True]:  # with/without horizontal flip
            fwd = {}
            inv = {}
            for r in range(n):
                for c in range(n):
                    old_action = r * n + c

                    # Apply rotation: np.rot90 with k rotations (counterclockwise)
                    # For k=1 (90° CCW): (r, c) → (c, n-1-r)
                    # For k=2 (180°):    (r, c) → (n-1-r, n-1-c)
                    # For k=3 (270° CCW): (r, c) → (n-1-c, r)
                    if k == 0:
                        nr, nc = r, c
                    elif k == 1:
                        nr, nc = c, n - 1 - r
                    elif k == 2:
                        nr, nc = n - 1 - r, n - 1 - c
                    elif k == 3:
                        nr, nc = n - 1 - c, r

                    # Apply horizontal flip if needed
                    if flip:
                        nc = n - 1 - nc

                    new_action = nr * n + nc
                    fwd[old_action] = new_action
                    inv[new_action] = old_action

            # Pass action maps to itself
            pass_action = n * n
            fwd[pass_action] = pass_action
            inv[pass_action] = pass_action

            forward_tables.append(fwd)
            inverse_tables.append(inv)

    return forward_tables, inverse_tables


class AugmentedGoEnv(MaskedGoEnv):
    """
    Go environment with random symmetry augmentation during training.

    On each reset(), a random symmetry (out of 8) is selected and applied
    to ALL observations returned during that episode. Actions from the
    agent are inverse-transformed before being passed to the real env.

    This teaches the encoder that rotated/reflected board positions are
    strategically equivalent, producing rotation-invariant features.
    This is standard practice in Go AI — AlphaGo used this same technique.

    Why it matters for concept bottleneck research:
        Without augmentation, the encoder treats (0,0) corner and (6,6) corner
        as completely different → concepts become position-dependent artifacts.
        With augmentation, the encoder learns that corners are corners regardless
        of orientation → concepts capture genuine strategic structure.

    The augmentation is ONLY applied during training. Evaluation environments
    should use plain MaskedGoEnv for consistent measurement.
    """

    def __init__(self, board_size=7, opponent_fn=None, render_mode=None,
                 reward_shaping=False, capture_reward=0.05):
        super().__init__(board_size=board_size, opponent_fn=opponent_fn,
                         render_mode=render_mode, reward_shaping=reward_shaping,
                         capture_reward=capture_reward)

        # Precompute symmetry tables once (avoid recomputing every step)
        self._fwd_tables, self._inv_tables = _build_symmetry_tables(board_size)
        self._n_symmetries = 8
        # Current symmetry index for this episode (randomized on reset)
        self._sym_idx = 0

    def _apply_obs_symmetry(self, obs, sym_idx):
        """
        Apply symmetry transformation to an observation.

        Uses np.rot90 for rotation and np.flip for reflection, matching
        the same transformation encoded in the action mapping tables.

        Args:
            obs: (board_size, board_size, 3) observation.
            sym_idx: Symmetry index 0-7 (k*2 + flip).

        Returns:
            Transformed observation.
        """
        k = sym_idx // 2      # Rotation count (0-3)
        flip = sym_idx % 2     # Whether to flip horizontally

        result = np.rot90(obs, k=k, axes=(0, 1))
        if flip:
            result = np.flip(result, axis=1)
        return result.copy()  # .copy() to ensure contiguous memory

    def _apply_mask_symmetry(self, mask, sym_idx):
        """
        Apply symmetry transformation to an action mask.

        Reorders the mask entries so mask[new_action] corresponds to the
        legality of the transformed position.

        Args:
            mask: (action_count,) binary mask.
            sym_idx: Symmetry index 0-7.

        Returns:
            Transformed mask.
        """
        fwd = self._fwd_tables[sym_idx]
        new_mask = np.zeros_like(mask)
        for old_a, new_a in fwd.items():
            if old_a < len(mask) and new_a < len(mask):
                new_mask[new_a] = mask[old_a]
        return new_mask

    def reset(self, seed=None, options=None):
        """
        Reset and randomly select a symmetry for this episode.

        Each episode gets a fresh random symmetry. This means the agent
        sees the same game from a different orientation each time,
        forcing the encoder to learn orientation-invariant features.
        """
        obs, info = super().reset(seed=seed, options=options)

        # Pick random symmetry for this episode
        self._sym_idx = np.random.randint(0, self._n_symmetries)

        # Transform observation and action mask
        obs = self._apply_obs_symmetry(obs, self._sym_idx)
        if "action_mask" in info:
            info["action_mask"] = self._apply_mask_symmetry(
                info["action_mask"], self._sym_idx
            )

        # Update cached mask for action_masks() method
        self._last_mask = info.get("action_mask",
                                    np.ones(self.action_count, dtype=np.int8))
        return obs, info

    def step(self, action):
        """
        Inverse-transform the agent's action, step the real env,
        then forward-transform the resulting observation.

        Flow:
            1. Agent picks action in the augmented (rotated) coordinate frame
            2. We inverse-transform it to the original frame
            3. Step the real environment with the original-frame action
            4. Forward-transform the new observation back to augmented frame
        """
        # Map agent's action from augmented frame → original frame
        inv = self._inv_tables[self._sym_idx]
        real_action = inv.get(int(action), int(action))

        # Step the real environment
        obs, reward, terminated, truncated, info = super().step(real_action)

        # Transform observation and mask to augmented frame
        if not terminated and not truncated:
            obs = self._apply_obs_symmetry(obs, self._sym_idx)
            if "action_mask" in info:
                info["action_mask"] = self._apply_mask_symmetry(
                    info["action_mask"], self._sym_idx
                )

        self._last_mask = info.get("action_mask",
                                    np.zeros(self.action_count, dtype=np.int8))
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """Return the augmented (transformed) action mask."""
        return self._last_mask
