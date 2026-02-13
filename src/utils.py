"""
Shared utilities for PRISM.

Board rendering, seeding, device selection, logging helpers.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def render_go_board_ascii(obs, board_size=7):
    """
    Render a Go board observation as ASCII text.

    Args:
        obs: numpy array of shape (board_size, board_size, 3)
             Planes: [black, white, empty]
        board_size: size of the board

    Returns:
        str: ASCII board representation
    """
    lines = []
    col_labels = "  " + " ".join(chr(ord('A') + i) for i in range(board_size))
    lines.append(col_labels)

    for r in range(board_size):
        row = f"{r + 1} "
        for c in range(board_size):
            if obs[r, c, 0] > 0.5:
                row += "X "  # Black
            elif obs[r, c, 1] > 0.5:
                row += "O "  # White
            else:
                row += ". "  # Empty
        lines.append(row)

    return "\n".join(lines)


def compute_board_symmetries(obs):
    """
    Compute all 8 symmetries (rotations + reflections) of a Go board observation.

    Args:
        obs: numpy array of shape (board_size, board_size, 3)

    Returns:
        list of 8 numpy arrays, each of shape (board_size, board_size, 3)
    """
    symmetries = []
    for k in range(4):
        rotated = np.rot90(obs, k=k, axes=(0, 1))
        symmetries.append(rotated.copy())
        symmetries.append(np.flip(rotated, axis=1).copy())
    return symmetries


def action_to_coord(action, board_size=7):
    """Convert flat action index to (row, col) or 'pass'."""
    if action >= board_size * board_size:
        return "pass"
    row = action // board_size
    col = action % board_size
    return (row, col)


def coord_to_action(row, col, board_size=7):
    """Convert (row, col) to flat action index."""
    return row * board_size + col


class RunningStats:
    """Online computation of mean and variance for reward normalization."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        if self.n < 2:
            return 1.0
        return self.M2 / (self.n - 1)

    @property
    def std(self):
        return max(np.sqrt(self.variance), 1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std
