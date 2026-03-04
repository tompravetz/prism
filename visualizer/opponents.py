"""
PRISM Visualizer — Opponent Policies.

Three opponent types for use in visualizer games:

    RandomOpponent     — uniform random legal moves (default, fastest)
    SelfPlayOpponent   — uses the same trained PRISM encoder + policy as white
    GnuGoOpponent      — GnuGo via GTP (requires gnugo in PATH)

All expose a __call__(obs, action_mask) -> int interface matching GoEnv's
opponent_fn convention, plus optional reset() / close() lifecycle hooks.

GnuGo coordinate convention (GTP):
    columns  A-G  (7x7 board; no 'I' skip needed for N≤8)
    rows     1-7  where 1 = bottom row (row index 6 in 0-indexed array)
    e.g. action (r=0, c=0)  →  A7
         action (r=6, c=6)  →  G1
"""

import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GTP coordinate helpers
# ---------------------------------------------------------------------------

# Standard GTP columns (skip 'I' per GTP spec; irrelevant for N≤8 but kept)
_GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def _action_to_gtp(action: int, board_size: int = 7) -> str:
    """Convert action index to GTP coordinate string."""
    if action >= board_size * board_size:
        return "pass"
    r, c = action // board_size, action % board_size
    return f"{_GTP_COLS[c]}{board_size - r}"


def _gtp_to_action(gtp_coord: str, board_size: int = 7) -> int:
    """Convert GTP coordinate string to action index.  Returns pass on failure."""
    gtp_coord = gtp_coord.strip().upper()
    if gtp_coord in ("PASS", "RESIGN", ""):
        return board_size * board_size
    try:
        c = _GTP_COLS.index(gtp_coord[0])
        row_num = int(gtp_coord[1:])
        r = board_size - row_num
        return r * board_size + c
    except (ValueError, IndexError):
        return board_size * board_size  # fall back to pass


# ---------------------------------------------------------------------------
# Random opponent
# ---------------------------------------------------------------------------

class RandomOpponent:
    """
    Uniform random opponent — picks a random legal move each turn.

    This is equivalent to GoEnv's built-in _random_opponent, exposed here
    as a named object for symmetry with the other opponent types.
    """

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.where(mask == 1)[0]
        return int(np.random.choice(legal)) if len(legal) > 0 else 49

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __str__(self) -> str:
        return "RandomOpponent"


# ---------------------------------------------------------------------------
# Self-play opponent
# ---------------------------------------------------------------------------

class SelfPlayOpponent:
    """
    Self-play opponent — runs the same trained PRISM policy as white.

    The observation white receives from GoEnv._handle_opponent_turns has:
        plane 0  =  black's stones  (the last agent to step)
        plane 1  =  white's stones

    The policy was trained with plane 0 = opponent and plane 1 = own stones,
    which aligns correctly for white's perspective (black is white's opponent).
    No channel swap is needed.

    Args:
        encoder:         Loaded GoCNNEncoder (frozen, eval mode).
        concept_manager: Loaded ConceptManager with cluster centres.
        policy:          Loaded ConceptBottleneckPolicy (frozen, eval mode).
        device:          torch.device to run inference on.
        n_actions:       Action space size (default 50 for 7×7 Go).
    """

    def __init__(self, encoder, concept_manager, policy, device,
                 n_actions: int = 50):
        self.encoder = encoder
        self.concept_manager = concept_manager
        self.policy = policy
        self.device = device
        self.n_actions = n_actions

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(obs_t).cpu().numpy()[0]

        concept_id = int(self.concept_manager.assign_concept(features))
        cid_t   = torch.LongTensor([concept_id]).to(self.device)
        mask_t  = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.policy(cid_t, mask_t)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()

        return int(np.argmax(probs))

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __str__(self) -> str:
        return "SelfPlayOpponent"


# ---------------------------------------------------------------------------
# GnuGo opponent (GTP)
# ---------------------------------------------------------------------------

class GnuGoOpponent:
    """
    GnuGo opponent via the Go Text Protocol (GTP).

    Requires gnugo to be installed and in PATH:
        Windows:  winget install gnugo      (or download from gnu.org/software/gnugo)
        Linux:    sudo apt install gnugo
        macOS:    brew install gnugo

    The GnuGo process is launched once on construction and reused for
    multiple games via reset(). Call close() when done.

    Args:
        level:      GnuGo strength level 0–10 (default 1; 10 is strongest).
        board_size: Board size (default 7).
        komi:       Komi value (default 5.5, matching GoEnv).
    """

    def __init__(self, level: int = 1, board_size: int = 7, komi: float = 5.5,
                 exe_path: Optional[str] = None):
        self.level = level
        self.board_size = board_size
        self.komi = komi
        self._exe_hint = exe_path
        self._proc: Optional[subprocess.Popen] = None
        self._prev_black: Optional[np.ndarray] = None
        self._stdout_queue: queue.Queue = queue.Queue()
        self._launch()

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    @staticmethod
    def find_exe(hint: Optional[str] = None) -> Optional[str]:
        """
        Locate the gnugo executable.

        Search order:
          1. The path given by `hint` (if provided and exists).
          2. gnugo / gnugo.exe in PATH.
          3. gnugo-3.8/gnugo.exe relative to the project root (C:/prism).

        Returns the resolved path string, or None if not found.
        """
        import shutil
        import os

        candidates = []
        if hint:
            candidates.append(hint)
        candidates.append("gnugo")  # PATH lookup

        # Auto-detect bundled gnugo next to the project root
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(_root, "gnugo-3.8", "gnugo.exe"))
        candidates.append(os.path.join(_root, "gnugo-3.8", "gnugo"))

        for path in candidates:
            resolved = shutil.which(path) or (path if os.path.isfile(path) else None)
            if resolved:
                return resolved
        return None

    @staticmethod
    def is_available(hint: Optional[str] = None) -> bool:
        """Return True if a gnugo executable can be located."""
        return GnuGoOpponent.find_exe(hint) is not None

    def _launch(self) -> None:
        """Start the gnugo subprocess and send initial config commands."""
        exe = GnuGoOpponent.find_exe(self._exe_hint)
        if exe is None:
            raise RuntimeError(
                "gnugo not found.\n"
                "  It was not in PATH and not at gnugo-3.8/gnugo.exe.\n"
                "  Pass the path with --gnugo-exe, or install gnugo:\n"
                "    winget install gnugo          (Windows)\n"
                "    sudo apt install gnugo         (Linux)\n"
                "    brew install gnugo             (macOS)\n"
            )
        try:
            self._proc = subprocess.Popen(
                [exe, "--mode", "gtp", f"--level={self.level}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to launch gnugo at '{exe}': {e}")

        self._start_reader_thread()
        self._send(f"boardsize {self.board_size}")
        self._send(f"komi {self.komi}")

    # ------------------------------------------------------------------
    # GTP I/O
    # ------------------------------------------------------------------

    def _start_reader_thread(self) -> None:
        """Pump GnuGo stdout into self._stdout_queue from a daemon thread.

        Captures proc/queue by value so that _restart() can replace
        self._proc / self._stdout_queue without a race condition.
        """
        proc = self._proc
        q = self._stdout_queue

        def _reader():
            try:
                for line in proc.stdout:
                    q.put(line)
            except (ValueError, OSError):
                pass
            finally:
                q.put(None)  # EOF / process-death sentinel

        t = threading.Thread(target=_reader, daemon=True, name="gnugo-reader")
        t.start()

    def _restart(self) -> None:
        """Kill the current GnuGo process and start a fresh one."""
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        # Fresh queue — the old reader thread will drain into the orphaned old queue
        self._stdout_queue = queue.Queue()
        self._launch()

    def _send(self, cmd: str) -> str:
        """Send one GTP command; return the response text (without = prefix)."""
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        return self._read_response()

    def _read_response(self, timeout: float = 60.0) -> str:
        """Read until a blank line (GTP response terminator), with timeout.

        If GnuGo doesn't respond within `timeout` seconds (process dead or
        hung), the process is killed and automatically restarted.
        """
        lines = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"\n[GnuGoOpponent] GTP timeout ({timeout}s) — restarting GnuGo")
                self._restart()
                return ""
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except queue.Empty:
                print(f"\n[GnuGoOpponent] GTP timeout ({timeout}s) — restarting GnuGo")
                self._restart()
                return ""
            if line is None:  # Process died
                print("\n[GnuGoOpponent] GnuGo process died — restarting")
                self._restart()
                return ""
            if line in ("", "\n"):
                break
            lines.append(line.rstrip("\n"))
        # Response is "= value" or "? error"
        response = " ".join(lines).strip()
        if response.startswith("= "):
            return response[2:].strip()
        elif response.startswith("?"):
            return ""   # error — treat as pass
        return response.strip()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the board for a new game."""
        self._prev_black = None
        self._send("clear_board")

    def close(self) -> None:
        """Send quit and terminate the subprocess."""
        if self._proc and self._proc.poll() is None:
            try:
                self._send("quit")
            except Exception:
                pass
            self._proc.terminate()
        self._proc = None

    # ------------------------------------------------------------------
    # Opponent interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Given white's observation after black has just moved, return white's
        action (as an integer 0-49).

        Detects black's last move by comparing the current black-stone plane
        with the previously seen one, then tells GnuGo via `play black`.
        """
        # plane 0 of white's obs = black's stones (last to step)
        black_now = obs[:, :, 0] > 0.5

        if self._prev_black is not None:
            diff = black_now & ~self._prev_black
            new_pos = np.argwhere(diff)
            if len(new_pos) == 1:
                r, c = int(new_pos[0][0]), int(new_pos[0][1])
                black_gtp = _action_to_gtp(r * self.board_size + c, self.board_size)
                self._send(f"play black {black_gtp}")
            else:
                # Black passed or multiple new stones (shouldn't happen)
                self._send("play black pass")
        # else: first call, board is empty from GnuGo's view already

        self._prev_black = black_now.copy()

        # Ask GnuGo for white's move
        resp = self._send("genmove white")
        action = _gtp_to_action(resp, self.board_size)

        # Validate against the legal move mask
        if action < len(mask) and mask[action] == 1:
            return action
        # GnuGo chose something illegal — fall back to random
        legal = np.where(mask == 1)[0]
        return int(np.random.choice(legal)) if len(legal) > 0 else self.board_size ** 2

    def __str__(self) -> str:
        return f"GnuGoOpponent(level={self.level}, exe={GnuGoOpponent.find_exe(self._exe_hint)})"
