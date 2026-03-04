"""
Train a DAgger (Dataset Aggregation) agent on Go 7x7.

Uses GnuGo Level 5 as the expert oracle. Three phases:

  Phase 0 — BC pretrain:
      Collect N_BC_SAMPLES from GnuGo self-play (GnuGo plays both colors).
      Build (observation, black_action) pairs for every black move.
      Train encoder + policy head via cross-entropy loss.

  Phases 1..N — DAgger rounds:
      Run current policy as black vs GnuGo white.
      At each state, label with expert's reg_genmove (no board advance).
      Add to aggregate dataset and retrain.

Saves:
    {out_dir}/ppo_go_encoder.pt       — GoCNNEncoder state dict
    {out_dir}/cloned_policy_head.pt   — PolicyHead state dict  (net.0/2/4 keys)
    {out_dir}/dagger_progress.json    — per-round training log

To run the full bottleneck pipeline after this:
    python train_bottleneck.py --algo ppo \\
        --baseline-dir models/cloned_dagger \\
        --save-dir models/bottleneck_dagger

Usage:
    python train_dagger.py
    python train_dagger.py --level 5 --bc-samples 285000 --dagger-rounds 2
    python train_dagger.py --bc-samples 10000 --bc-epochs 10  # quick smoke test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.utils import set_seed, get_device, ensure_dir


# ---------------------------------------------------------------------------
# GTP coordinate helpers  (same convention as visualizer/opponents.py)
# ---------------------------------------------------------------------------

_GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def _action_to_gtp(action: int, board_size: int = 7) -> str:
    if action >= board_size * board_size:
        return "pass"
    r, c = action // board_size, action % board_size
    return f"{_GTP_COLS[c]}{board_size - r}"


def _gtp_to_action(gtp_coord: str, board_size: int = 7) -> int:
    gtp_coord = gtp_coord.strip().upper()
    if gtp_coord in ("PASS", "RESIGN", ""):
        return board_size * board_size
    try:
        c = _GTP_COLS.index(gtp_coord[0])
        row_num = int(gtp_coord[1:])
        r = board_size - row_num
        return r * board_size + c
    except (ValueError, IndexError):
        return board_size * board_size


def build_obs_from_stones(
    black_positions: List[str],
    white_positions: List[str],
    board_size: int = 7,
) -> np.ndarray:
    """
    Build a (board_size, board_size, 3) float32 observation from GTP stone lists.

    Plane 0 = black, plane 1 = white, plane 2 = empty.
    Matches GoEnv's observation convention exactly.
    """
    obs = np.zeros((board_size, board_size, 3), dtype=np.float32)
    for gtp in black_positions:
        a = _gtp_to_action(gtp.strip(), board_size)
        if a < board_size * board_size:
            obs[a // board_size, a % board_size, 0] = 1.0
    for gtp in white_positions:
        a = _gtp_to_action(gtp.strip(), board_size)
        if a < board_size * board_size:
            obs[a // board_size, a % board_size, 1] = 1.0
    obs[:, :, 2] = 1.0 - obs[:, :, 0] - obs[:, :, 1]
    return obs


def build_mask_from_obs(obs: np.ndarray, board_size: int = 7) -> np.ndarray:
    """
    Build a legal-move mask from observation (occupied = illegal, pass = always legal).

    This is an approximation (no ko detection) used during BC data collection.
    Only the EXPERT's actions are recorded — it never plays illegal moves — so
    the mask is only needed at inference time, not for the loss computation.
    """
    mask = np.ones(board_size * board_size + 1, dtype=np.int8)
    for r in range(board_size):
        for c in range(board_size):
            if obs[r, c, 0] > 0.5 or obs[r, c, 1] > 0.5:
                mask[r * board_size + c] = 0
    return mask


def detect_white_action(
    obs_before: np.ndarray,
    obs_after: np.ndarray,
    board_size: int = 7,
) -> int:
    """
    Detect white's action from the observation before and after env.step().

    obs_before = state before black's move (and before white's response).
    obs_after  = state after both black's move AND white's response.

    We look for net-new white stones (those in obs_after[:,:,1] but not in
    obs_before[:,:,1]).  This handles captures correctly: captured white stones
    disappear, but the one newly placed white stone still appears as a net-new
    position.  If no new white stone is found, white passed.
    """
    white_before = obs_before[:, :, 1] > 0.5
    white_after  = obs_after[:, :, 1] > 0.5
    new_white = np.argwhere(white_after & ~white_before)
    if len(new_white) == 1:
        r, c = int(new_white[0][0]), int(new_white[0][1])
        return r * board_size + c
    return board_size * board_size  # pass


# ---------------------------------------------------------------------------
# Low-level GnuGo subprocess wrapper
# ---------------------------------------------------------------------------

class GnuGoProcess:
    """
    Thin wrapper around a GnuGo GTP subprocess.

    Provides _send() for command I/O and standard lifecycle methods.
    Subclassed by GnuGoSelfPlayCollector and DAggerExpert.
    """

    def __init__(
        self,
        level: int = 5,
        board_size: int = 7,
        komi: float = 5.5,
        exe_path: Optional[str] = None,
    ):
        self.level = level
        self.board_size = board_size
        self.komi = komi
        self._exe_hint = exe_path
        self._proc: Optional[subprocess.Popen] = None
        self._launch()

    # -- Subprocess management -----------------------------------------------

    @staticmethod
    def _find_exe(hint: Optional[str] = None) -> Optional[str]:
        import shutil
        candidates = []
        if hint:
            candidates.append(hint)
        candidates.append("gnugo")
        _root = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(_root, "gnugo-3.8", "gnugo.exe"))
        candidates.append(os.path.join(_root, "gnugo-3.8", "gnugo"))
        for path in candidates:
            resolved = shutil.which(path) or (path if os.path.isfile(path) else None)
            if resolved:
                return resolved
        return None

    def _launch(self) -> None:
        exe = self._find_exe(self._exe_hint)
        if exe is None:
            raise RuntimeError(
                "gnugo not found. Install it (winget install gnugo / "
                "sudo apt install gnugo / brew install gnugo) or pass --gnugo-exe."
            )
        self._proc = subprocess.Popen(
            [exe, "--mode", "gtp", f"--level={self.level}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send(f"boardsize {self.board_size}")
        self._send(f"komi {self.komi}")

    # -- GTP I/O -------------------------------------------------------------

    def _send(self, cmd: str) -> str:
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if line in ("", "\n"):
                break
            lines.append(line.rstrip("\n"))
        response = " ".join(lines).strip()
        if response.startswith("= "):
            return response[2:].strip()
        elif response.startswith("?"):
            return ""
        return response.strip()

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._send("quit")
            except Exception:
                pass
            self._proc.terminate()
        self._proc = None


# ---------------------------------------------------------------------------
# BC data collector: GnuGo self-play
# ---------------------------------------------------------------------------

class GnuGoSelfPlayCollector(GnuGoProcess):
    """
    Collect behavioral-cloning training data from GnuGo self-play.

    One GnuGo process plays both colors.  Before each black move we capture the
    board state as a (7,7,3) observation; after genmove black we record the
    (observation, action) pair.  Games reset when both sides pass or GnuGo
    resigns.
    """

    def collect(
        self,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect n_samples (obs, action) pairs from GnuGo self-play.

        Returns:
            obs_arr:     float32 array, shape (n_samples, board_size, board_size, 3)
            action_arr:  int64 array,   shape (n_samples,)
        """
        obs_list: List[np.ndarray] = []
        act_list: List[int] = []

        games = 0
        t0 = time.time()

        while len(obs_list) < n_samples:
            self._send("clear_board")
            game_over = False
            prev_black_passed = False

            while not game_over and len(obs_list) < n_samples:
                # --- capture pre-move observation ---
                raw_b = self._send("list_stones black")
                raw_w = self._send("list_stones white")
                b_pos = raw_b.split() if raw_b else []
                w_pos = raw_w.split() if raw_w else []
                obs = build_obs_from_stones(b_pos, w_pos, self.board_size)

                # --- black's move ---
                resp_b = self._send("genmove black")
                if resp_b.upper() == "RESIGN":
                    game_over = True
                    break
                action_b = _gtp_to_action(resp_b, self.board_size)
                obs_list.append(obs)
                act_list.append(action_b)
                black_passed = (action_b == self.board_size * self.board_size)

                # --- white's move ---
                resp_w = self._send("genmove white")
                if resp_w.upper() == "RESIGN":
                    game_over = True
                    break
                white_passed = _gtp_to_action(resp_w, self.board_size) == self.board_size * self.board_size

                if black_passed and white_passed:
                    game_over = True
                prev_black_passed = black_passed

            games += 1
            if games % 100 == 0:
                rate = len(obs_list) / (time.time() - t0 + 1e-6)
                eta = (n_samples - len(obs_list)) / (rate + 1e-6)
                print(
                    f"  BC collect: {len(obs_list):>7,}/{n_samples:,} samples  "
                    f"({games} games,  {rate:.0f} samp/s,  ETA {eta/60:.1f}m)"
                )

        obs_arr    = np.array(obs_list[:n_samples], dtype=np.float32)
        action_arr = np.array(act_list[:n_samples], dtype=np.int64)
        return obs_arr, action_arr


# ---------------------------------------------------------------------------
# DAgger expert: labels states with reg_genmove (no board advance)
# ---------------------------------------------------------------------------

class DAggerExpert(GnuGoProcess):
    """
    GnuGo expert used during DAgger rollouts.

    Tracks the full game state by receiving play commands for BOTH colors.
    Labels each black-to-play state via reg_genmove black — which returns
    GnuGo's recommended move WITHOUT advancing its internal board state.

    Usage per game:
        expert.reset_game()
        # For each step:
        label = expert.get_expert_action()   # BEFORE env.step
        env.step(learner_action)
        expert.play_black(learner_action)    # sync: what learner played
        white_action = detect_white_action(obs_before, obs_after)
        expert.play_white(white_action)      # sync: what white played
    """

    def reset_game(self) -> None:
        self._send("clear_board")

    def play_black(self, action: int) -> None:
        gtp = _action_to_gtp(action, self.board_size)
        self._send(f"play black {gtp}")

    def play_white(self, action: int) -> None:
        gtp = _action_to_gtp(action, self.board_size)
        self._send(f"play white {gtp}")

    def get_expert_action(self) -> int:
        """Return GnuGo's recommendation for black without advancing the board."""
        resp = self._send("reg_genmove black")
        return _gtp_to_action(resp, self.board_size)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """
    Three-layer MLP policy head.

    Architecture matches the existing cloned_policy_head.pt checkpoint:
        self.net = Sequential(
            Linear(128→256),  # net.0
            ReLU,             # net.1
            Linear(256→128),  # net.2
            ReLU,             # net.3
            Linear(128→50),   # net.4
        )
    """

    def __init__(self, in_dim: int = 128, n_actions: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DAggerPolicy(nn.Module):
    """
    GoCNNEncoder + PolicyHead trained end-to-end via cross-entropy imitation.

    The encoder is NOT frozen during DAgger training — both components are
    updated jointly on (observation, expert_action) pairs.
    """

    def __init__(
        self,
        board_size: int = 7,
        features_dim: int = 128,
        n_actions: int = 50,
    ):
        super().__init__()
        env = GoEnv(board_size=board_size)
        self.encoder = GoCNNEncoder(env.observation_space, features_dim=features_dim)
        env.close()
        self.head = PolicyHead(in_dim=features_dim, n_actions=n_actions)

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs_tensor)
        return self.head(features)

    def predict(
        self,
        obs_np: np.ndarray,
        mask_np: Optional[np.ndarray],
        device: torch.device,
    ) -> int:
        """Greedy action with illegal moves masked out."""
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(obs_t)[0]
        if mask_np is not None:
            mask_t = torch.FloatTensor(mask_np).to(device)
            logits = logits.masked_fill(mask_t == 0, float("-inf"))
        return int(logits.argmax().item())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    policy: DAggerPolicy,
    optimizer: optim.Optimizer,
    obs_arr: np.ndarray,
    action_arr: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    One epoch of cross-entropy imitation learning over the full dataset.

    Returns mean loss across all batches.
    """
    policy.train()
    n = len(obs_arr)
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        obs_b   = torch.FloatTensor(obs_arr[idx]).to(device)
        act_b   = torch.LongTensor(action_arr[idx]).to(device)

        logits = policy(obs_b)
        loss   = F.cross_entropy(logits, act_b)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# DAgger rollout
# ---------------------------------------------------------------------------

def collect_dagger_rollout(
    policy: DAggerPolicy,
    expert: DAggerExpert,
    n_steps: int,
    level: int,
    board_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect n_steps (observation, expert_action) pairs for DAgger training.

    The LEARNER plays as black (its own actions advance the game).
    The EXPERT labels each state via reg_genmove (without advancing GnuGo's state).
    A GnuGoOpponent handles white independently via GoEnv's opponent_fn.
    The expert tracks full game state via explicit play commands.
    """
    from visualizer.opponents import GnuGoOpponent

    policy.eval()

    gnugo_white = GnuGoOpponent(level=level, board_size=board_size)
    env = GoEnv(board_size=board_size, opponent_fn=gnugo_white)

    obs_list: List[np.ndarray] = []
    act_list: List[int]        = []

    gnugo_white.reset()
    expert.reset_game()
    obs, info = env.reset()
    done = False

    t0 = time.time()

    while len(obs_list) < n_steps:
        # 1. Get expert label for current state (no board advance)
        expert_action = expert.get_expert_action()

        # 2. Get learner's action (masked greedy)
        mask = info.get("action_mask", None)
        learner_action = policy.predict(obs, mask, device)

        # 3. Record (current_obs, expert_label)
        obs_list.append(obs.copy())
        act_list.append(expert_action)

        obs_before = obs.copy()

        # 4. Advance environment with LEARNER's action
        obs, reward, terminated, truncated, info = env.step(learner_action)
        done = terminated or truncated

        # 5. Sync expert: tell it what black and white played
        expert.play_black(learner_action)
        white_action = detect_white_action(obs_before, obs, board_size)
        expert.play_white(white_action)

        if done:
            gnugo_white.reset()
            expert.reset_game()
            env.close()
            env = GoEnv(board_size=board_size, opponent_fn=gnugo_white)
            obs, info = env.reset()
            done = False

        if len(obs_list) % 1000 == 0:
            rate = len(obs_list) / (time.time() - t0 + 1e-6)
            print(
                f"  DAgger collect: {len(obs_list):>6,}/{n_steps:,}  "
                f"({rate:.0f} samp/s)"
            )

    env.close()
    gnugo_white.close()

    return (
        np.array(obs_list[:n_steps], dtype=np.float32),
        np.array(act_list[:n_steps], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy: DAggerPolicy,
    gnugo_level: int,
    n_games: int,
    board_size: int,
    device: torch.device,
) -> float:
    """
    Evaluate the DAgger policy win rate against GnuGo.

    Uses the same GnuGoOpponent-as-opponent_fn pattern as the fixed
    eval_agent_vs_gnugo() to avoid the two-state double-step bug.
    """
    from visualizer.opponents import GnuGoOpponent

    policy.eval()
    wins = losses = draws = 0

    gnugo = GnuGoOpponent(level=gnugo_level, board_size=board_size)

    for _ in range(n_games):
        gnugo.reset()
        env = GoEnv(board_size=board_size, opponent_fn=gnugo)
        obs, info = env.reset()
        done = False
        move_count = 0
        last_reward = 0.0

        while not done and move_count < 200:
            mask = info.get("action_mask", None)
            action = policy.predict(obs, mask, device)
            obs, reward, terminated, truncated, info = env.step(action)
            move_count += 1
            done = terminated or truncated
            if done:
                last_reward = float(reward)

        if last_reward > 0.5:
            wins += 1
        elif last_reward < -0.5:
            losses += 1
        else:
            draws += 1
        env.close()

    gnugo.close()
    total = wins + losses + draws
    return wins / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a DAgger agent on Go 7x7 with GnuGo as expert"
    )
    parser.add_argument(
        "--level", type=int, default=5, metavar="N",
        help="GnuGo expert level (default: 5)",
    )
    parser.add_argument(
        "--bc-samples", type=int, default=285_000, metavar="N",
        help="BC pretraining samples from GnuGo self-play (default: 285000)",
    )
    parser.add_argument(
        "--dagger-rounds", type=int, default=2, metavar="N",
        help="Number of DAgger refinement rounds (default: 2)",
    )
    parser.add_argument(
        "--dagger-steps", type=int, default=25_000, metavar="N",
        help="New labeled steps per DAgger round (default: 25000)",
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=50, metavar="N",
        help="Training epochs for BC pretrain phase (default: 50)",
    )
    parser.add_argument(
        "--dagger-epochs", type=int, default=20, metavar="N",
        help="Training epochs per DAgger round (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, metavar="N",
        help="Mini-batch size (default: 512)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=30, metavar="N",
        help="Games per evaluation (default: 30)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="models/cloned_dagger",
        help="Directory to save encoder + policy head (default: models/cloned_dagger)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gnugo-exe", type=str, default=None,
        help="Path to gnugo executable (auto-detected if omitted)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)

    print(f"\n{'='*60}")
    print(f"DAgger Training  |  Go 7x7  |  Expert: GnuGo Level {args.level}")
    print(f"BC samples: {args.bc_samples:,}  |  DAgger rounds: {args.dagger_rounds}")
    print(f"BC epochs: {args.bc_epochs}  |  DAgger epochs/round: {args.dagger_epochs}")
    print(f"Batch: {args.batch_size}  |  Device: {device}")
    print(f"{'='*60}\n")

    # Build policy
    policy = DAggerPolicy(board_size=7, features_dim=128, n_actions=50)
    policy.to(device)

    progress_log = []

    # =========================================================
    # Phase 0: BC pretraining
    # =========================================================
    print(f"Phase 0: BC pretrain — collecting {args.bc_samples:,} samples from "
          f"GnuGo Level {args.level} self-play...")
    t_bc0 = time.time()

    collector = GnuGoSelfPlayCollector(
        level=args.level, board_size=7, exe_path=args.gnugo_exe
    )
    obs_arr, action_arr = collector.collect(args.bc_samples)
    collector.close()

    print(f"  Collected {len(obs_arr):,} samples in {time.time()-t_bc0:.0f}s\n")

    # Aggregate dataset (starts as just BC data)
    agg_obs    = obs_arr
    agg_acts   = action_arr

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    print(f"  Training BC ({args.bc_epochs} epochs, {len(agg_obs):,} samples)...")
    for epoch in range(args.bc_epochs):
        loss = train_epoch(policy, optimizer, agg_obs, agg_acts,
                           args.batch_size, device)
        if (epoch + 1) % 10 == 0 or epoch == args.bc_epochs - 1:
            print(f"    Epoch {epoch+1:3d}/{args.bc_epochs}  loss={loss:.4f}")

    print(f"\n  Evaluating after BC pretrain...")
    eval_results = {}
    for lvl in [1, 5]:
        wr = evaluate_policy(policy, lvl, args.eval_games, 7, device)
        eval_results[str(lvl)] = round(wr, 4)
        print(f"    vs GnuGo Level {lvl}: {wr:.1%}")

    progress_log.append({
        "round": 0,
        "n_samples": int(len(agg_obs)),
        "eval": eval_results,
    })
    _save_progress(args.out_dir, progress_log)

    # =========================================================
    # DAgger rounds
    # =========================================================
    expert = DAggerExpert(
        level=args.level, board_size=7, exe_path=args.gnugo_exe
    )

    for round_idx in range(1, args.dagger_rounds + 1):
        print(f"\n{'='*60}")
        print(f"DAgger Round {round_idx}/{args.dagger_rounds}  —  "
              f"collecting {args.dagger_steps:,} new steps...")
        t_round = time.time()

        new_obs, new_acts = collect_dagger_rollout(
            policy, expert, args.dagger_steps, args.level, 7, device
        )

        print(f"  Collected {len(new_obs):,} samples in "
              f"{time.time()-t_round:.0f}s")

        # Aggregate
        agg_obs  = np.concatenate([agg_obs, new_obs],  axis=0)
        agg_acts = np.concatenate([agg_acts, new_acts], axis=0)

        # Retrain on full aggregated dataset
        optimizer = optim.Adam(policy.parameters(), lr=5e-4)  # lower LR
        print(f"  Retraining on {len(agg_obs):,} samples "
              f"({args.dagger_epochs} epochs)...")
        for epoch in range(args.dagger_epochs):
            loss = train_epoch(policy, optimizer, agg_obs, agg_acts,
                               args.batch_size, device)
            if (epoch + 1) % 5 == 0 or epoch == args.dagger_epochs - 1:
                print(f"    Epoch {epoch+1:3d}/{args.dagger_epochs}  loss={loss:.4f}")

        print(f"\n  Evaluating after DAgger round {round_idx}...")
        eval_results = {}
        for lvl in [1, 5]:
            wr = evaluate_policy(policy, lvl, args.eval_games, 7, device)
            eval_results[str(lvl)] = round(wr, 4)
            print(f"    vs GnuGo Level {lvl}: {wr:.1%}")

        progress_log.append({
            "round": round_idx,
            "n_samples": int(len(agg_obs)),
            "dagger_samples_this_round": int(len(new_obs)),
            "eval": eval_results,
        })
        _save_progress(args.out_dir, progress_log)

    expert.close()

    # =========================================================
    # Save models
    # =========================================================
    encoder_path = os.path.join(args.out_dir, "ppo_go_encoder.pt")
    head_path    = os.path.join(args.out_dir, "cloned_policy_head.pt")

    torch.save(policy.encoder.state_dict(), encoder_path)
    torch.save(policy.head.state_dict(),    head_path)

    print(f"\n{'='*60}")
    print(f"DAgger training complete.")
    print(f"  Encoder:     {encoder_path}")
    print(f"  Policy head: {head_path}")
    print(f"  Progress:    {os.path.join(args.out_dir, 'dagger_progress.json')}")
    print(f"\nNext steps:")
    print(f"  python train_bottleneck.py --algo ppo \\")
    print(f"      --baseline-dir {args.out_dir} \\")
    print(f"      --save-dir models/bottleneck_dagger")
    print(f"{'='*60}\n")


def _save_progress(out_dir: str, log: list) -> None:
    path = os.path.join(out_dir, "dagger_progress.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
