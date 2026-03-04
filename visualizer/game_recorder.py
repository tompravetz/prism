"""
PRISM Game Recorder.

Plays a deterministic game using a trained PRISM concept-bottleneck agent
and captures the full interpretability state at each of the agent's moves:
    - Board observation (before the agent acts)
    - Concept assignment and distance to centroid
    - Full action probability distribution (masked softmax)
    - Chosen action
    - Cumulative move log and concept history

Two entry points:
    record_game(seed, max_moves)         → List[FrameData] for one game
    record_best_game(n_candidates, ...)  → highest concept-diversity game
                                           from n_candidates seeds
"""

import os
import sys
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy
from src.utils import get_device, set_seed


# ---------------------------------------------------------------------------
# FrameData — one entry per agent (black) move
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """
    All information needed to render one visualization frame.

    Each FrameData corresponds to one agent (black) move. The board state
    `obs` is captured BEFORE the agent's move is applied, so the heatmap
    shows what the agent is considering. White's moves are silently
    incorporated into the board state between agent turns.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    agent_move_number: int      # 1-indexed count of agent's moves this game
    total_half_moves: int       # Total plies (black + white) so far

    # ── Board state (BEFORE the agent's move is applied) ──────────────────
    obs: np.ndarray             # (7, 7, 3) float32 — [black, white, empty]
    action_mask: np.ndarray     # (50,) int8 — 1 = legal, 0 = illegal

    # ── Concept assignment ─────────────────────────────────────────────────
    concept_id: int             # 0–63
    dist_to_centroid: float     # L2 distance from features to cluster centre
    features: np.ndarray        # (128,) float32 — encoder output

    # ── Action distribution (post-masking softmax) ─────────────────────────
    action_probs: np.ndarray    # (50,) — 0 at illegal positions
    action_heatmap: np.ndarray  # (7, 7) — action_probs[:49] reshaped
    chosen_action: int          # 0–49

    # ── History (cumulative up to this frame) ─────────────────────────────
    # Each entry: {"half_move": int, "player": "black"|"white", "action": int}
    move_log: List[Dict] = field(default_factory=list)
    # Concept ID for each of the agent's moves, including the current one
    concept_history: List[int] = field(default_factory=list)

    # ── Game metadata ─────────────────────────────────────────────────────
    game_seed: int = 0
    algo: str = "ppo"


# ---------------------------------------------------------------------------
# GameResult — returned alongside frames from record_game
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """
    End-of-game summary produced by record_game().

    Captures the outcome, final score, and aggregate concept statistics
    so the renderer can produce a meaningful summary frame.
    """

    # ── Outcome ───────────────────────────────────────────────────────────
    winner: str               # "black" | "white" | "draw" | "incomplete"
    reward: float             # terminal reward: +1, -1, or 0
    score_margin: Optional[float]   # Go score from black's view (+ = black ahead);
                                    # None if game ended by max-moves truncation

    # ── Move counts ───────────────────────────────────────────────────────
    n_agent_moves: int        # Number of black (agent) moves recorded
    n_total_half_moves: int   # Total plies including white's responses

    # ── Game identity ─────────────────────────────────────────────────────
    game_seed: int
    algo: str
    opponent_name: str        # Human-readable label, e.g. "RandomOpponent"

    # ── Final board state ─────────────────────────────────────────────────
    # Captured directly from the Go engine after the game ends.
    # Same channel convention as training obs: plane0=white, plane1=black.
    # Falls back to the last FrameData obs if the engine is unreachable.
    final_obs: np.ndarray           # (7, 7, 3) float32

    # ── Concept statistics (across the full game) ─────────────────────────
    concept_history: List[int]      # concept_id per agent move, in order
    concept_counts: Dict[int, int]  # concept_id → occurrence count
    n_unique_concepts: int
    diversity_score: float


# ---------------------------------------------------------------------------
# Scoring — used by record_best_game
# ---------------------------------------------------------------------------

def concept_diversity_score(frames: List[FrameData]) -> float:
    """
    Score a game by its concept diversity and dynamism.

        score = n_unique × (1 + 0.5 × dynamism)
        dynamism = n_concept_transitions / max(n_agent_moves − 1, 1)

    This prioritises visiting many distinct concepts (primary) while
    rewarding frequent transitions as a secondary multiplier.
    """
    if not frames:
        return 0.0
    history = [f.concept_id for f in frames]
    n_unique = len(set(history))
    n_moves = len(history)
    n_transitions = sum(
        1 for i in range(1, n_moves) if history[i] != history[i - 1]
    )
    dynamism = n_transitions / max(n_moves - 1, 1)
    return n_unique * (1.0 + 0.5 * dynamism)


# ---------------------------------------------------------------------------
# GameRecorder
# ---------------------------------------------------------------------------

class GameRecorder:
    """
    Loads PRISM artifacts and records games for visualization.

    Usage:
        recorder = GameRecorder(algo="ppo")
        recorder.load_artifacts()
        frames = recorder.record_game(seed=0)
        # or:
        frames, stats = recorder.record_best_game(n_candidates=20)
    """

    def __init__(self, algo: str = "ppo", root_dir: Optional[str] = None):
        self.algo = algo
        self.root_dir = root_dir or _ROOT
        self.device = get_device()

        self.encoder: Optional[GoCNNEncoder] = None
        self.concept_manager: Optional[ConceptManager] = None
        self.policy: Optional[ConceptBottleneckPolicy] = None

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def load_artifacts(self) -> None:
        """Load encoder, concept manager, and bottleneck policy from disk."""
        root = self.root_dir
        env = GoEnv(board_size=7)

        # Encoder
        encoder_path = os.path.join(
            root, "models", "baseline", f"{self.algo}_go_encoder.pt"
        )
        self.encoder = GoCNNEncoder(env.observation_space, features_dim=128)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device, weights_only=True)
        )
        self.encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f"Loaded encoder from {encoder_path}")

        # Concept manager
        cm_path = os.path.join(
            root, "models", "bottleneck", f"concepts_{self.algo}_k64.pkl"
        )
        self.concept_manager = ConceptManager(n_concepts=64)
        self.concept_manager.load(cm_path)
        print(f"Loaded concept manager from {cm_path}")

        # Bottleneck policy
        policy_path = os.path.join(
            root, "models", "bottleneck", f"{self.algo}_bottleneck_final.pt"
        )
        self.policy = ConceptBottleneckPolicy(
            n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50
        ).to(self.device)
        self.policy.load_state_dict(
            torch.load(policy_path, map_location=self.device, weights_only=True)
        )
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")

        env.close()

    # ------------------------------------------------------------------
    # Single-game recording
    # ------------------------------------------------------------------

    def record_game(
        self,
        seed: int = 0,
        max_moves: int = 60,
        opponent_fn=None,
    ) -> Tuple[List[FrameData], "GameResult"]:
        """
        Play one deterministic game and return a FrameData per agent move.

        Args:
            seed: RNG seed for the environment (opponent's random moves).
            max_moves: Maximum number of agent moves to record. The game
                       may end earlier if both players pass or one wins.
            opponent_fn: Optional callable(obs, mask) -> action for white.
                         If None, GoEnv defaults to random legal moves.
                         Objects with a reset() method will have it called
                         at the start of each game.

        Returns:
            (frames, game_result) where frames is one FrameData per agent
            move and game_result holds the outcome and aggregate statistics.
        """
        assert self.encoder is not None, "Call load_artifacts() first."
        set_seed(seed)

        # Reset stateful opponents (e.g. GnuGo needs clear_board per game)
        if opponent_fn is not None and hasattr(opponent_fn, "reset"):
            opponent_fn.reset()

        opponent_name = str(opponent_fn) if opponent_fn is not None else "RandomOpponent"

        env = GoEnv(board_size=7, opponent_fn=opponent_fn)
        obs, info = env.reset()

        frames: List[FrameData] = []
        move_log: List[Dict] = []
        concept_history: List[int] = []
        half_move = 0
        agent_move = 0
        done = False
        last_reward = 0.0

        while not done and agent_move < max_moves:
            mask = info.get("action_mask", np.ones(50, dtype=np.int8))

            # ── Encode observation ──────────────────────────────────────
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.encoder(obs_t).cpu().numpy()[0]  # (128,)

            # ── Assign concept ──────────────────────────────────────────
            concept_id = int(self.concept_manager.assign_concept(features))
            dist = float(np.linalg.norm(
                features - self.concept_manager.cluster_centers[concept_id]
            ))

            # ── Get action distribution (masked softmax) ───────────────
            with torch.no_grad():
                cid_t = torch.LongTensor([concept_id]).to(self.device)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                logits, _ = self.policy(cid_t, mask_t)
                probs = F.softmax(logits[0], dim=-1).cpu().numpy()  # (50,)

            # ── Heatmap: board positions only ──────────────────────────
            heatmap = np.zeros((7, 7), dtype=np.float32)
            for a in range(49):
                heatmap[a // 7, a % 7] = probs[a]

            # ── Choose action (deterministic) ──────────────────────────
            chosen = int(np.argmax(probs))

            # ── Build cumulative histories ─────────────────────────────
            agent_move += 1
            half_move += 1
            concept_history = concept_history + [concept_id]
            move_log = move_log + [
                {"half_move": half_move, "player": "black", "action": chosen}
            ]

            frames.append(FrameData(
                agent_move_number=agent_move,
                total_half_moves=half_move,
                obs=obs.copy(),
                action_mask=mask.copy(),
                concept_id=concept_id,
                dist_to_centroid=dist,
                features=features.copy(),
                action_probs=probs.copy(),
                action_heatmap=heatmap.copy(),
                chosen_action=chosen,
                move_log=list(move_log),
                concept_history=list(concept_history),
                game_seed=seed,
                algo=self.algo,
            ))

            # ── Step environment (applies black's move, then white moves) ─
            obs, step_reward, terminated, truncated, info = env.step(chosen)
            done = terminated or truncated
            if done:
                last_reward = float(step_reward)

            if not done:
                # Record white's response in the move log for display
                half_move += 1
                move_log = move_log + [
                    {"half_move": half_move, "player": "white", "action": None}
                ]

        # ── Extract final board + Go score before closing ─────────────
        score_margin: Optional[float] = None
        final_obs: np.ndarray = (
            frames[-1].obs.copy() if frames
            else np.zeros((7, 7, 3), dtype=np.float32)
        )
        if done:
            try:
                # Unwrap PettingZoo's 3-layer wrapper to reach raw_env._go
                raw_env = env._env.env.env.env
                score_margin = float(raw_env._go.score())
                # Build final obs from raw board (go_base: BLACK=1, WHITE=-1)
                # Use same channel convention as training: plane0=white, plane1=black
                board = raw_env._go.board
                fo = np.zeros((7, 7, 3), dtype=np.float32)
                fo[:, :, 1] = (board == 1).astype(np.float32)   # black
                fo[:, :, 0] = (board == -1).astype(np.float32)  # white
                fo[:, :, 2] = (board == 0).astype(np.float32)   # empty
                final_obs = fo
            except Exception:
                pass  # fall back to last frame's obs already set above

        env.close()

        # ── Determine winner ───────────────────────────────────────────
        if not done:
            winner = "incomplete"
        elif last_reward > 0.5:
            winner = "black"
        elif last_reward < -0.5:
            winner = "white"
        else:
            winner = "draw"

        concept_counts = dict(Counter(concept_history))
        game_result = GameResult(
            winner=winner,
            reward=last_reward,
            score_margin=score_margin,
            n_agent_moves=agent_move,
            n_total_half_moves=half_move,
            game_seed=seed,
            algo=self.algo,
            opponent_name=opponent_name,
            final_obs=final_obs,
            concept_history=concept_history,
            concept_counts=concept_counts,
            n_unique_concepts=len(set(concept_history)),
            diversity_score=concept_diversity_score(frames),
        )
        return frames, game_result

    # ------------------------------------------------------------------
    # Best-game selection
    # ------------------------------------------------------------------

    def record_best_game(
        self,
        n_candidates: int = 20,
        max_moves: int = 60,
        opponent_fn=None,
    ) -> Tuple[List[FrameData], Dict]:
        """
        Play n_candidates games and return the one with the highest
        concept diversity score.

        Args:
            n_candidates: Number of seeds to try (seeds 0 .. n_candidates-1).
            max_moves: Maximum agent moves per game.
            opponent_fn: Optional opponent callable; passed to record_game.

        Returns:
            (best_frames, stats, best_result) where stats contains per-game
            scores and the winning seed (for game_selection_log.json), and
            best_result is the GameResult for the chosen game.
        """
        best_frames: List[FrameData] = []
        best_result: Optional["GameResult"] = None
        best_score = -1.0
        best_seed = 0
        all_stats = []

        print(f"Selecting best game from {n_candidates} candidates...")
        for seed in range(n_candidates):
            frames, game_result = self.record_game(seed=seed, max_moves=max_moves,
                                                   opponent_fn=opponent_fn)
            score = concept_diversity_score(frames)
            history = [f.concept_id for f in frames]
            n_moves = len(history)
            n_unique = len(set(history))
            n_transitions = sum(
                1 for i in range(1, n_moves) if history[i] != history[i - 1]
            )
            all_stats.append({
                "seed": seed,
                "n_agent_moves": n_moves,
                "n_unique_concepts": n_unique,
                "n_transitions": n_transitions,
                "dynamism": n_transitions / max(n_moves - 1, 1),
                "diversity_score": round(score, 3),
                "winner": game_result.winner,
            })
            print(
                f"  seed {seed:3d}: {n_moves} moves, "
                f"{n_unique} concepts, score={score:.2f}, {game_result.winner}"
            )
            if score > best_score:
                best_score = score
                best_seed = seed
                best_frames = frames
                best_result = game_result

        stats = {
            "n_candidates": n_candidates,
            "best_seed": best_seed,
            "best_score": round(best_score, 3),
            "algo": self.algo,
            "all_games": all_stats,
        }
        print(
            f"Best game: seed={best_seed} "
            f"(score={best_score:.2f}, "
            f"{len(best_frames)} frames, "
            f"{len(set(f.concept_id for f in best_frames))} unique concepts)"
        )
        return best_frames, stats, best_result


# ---------------------------------------------------------------------------
# Concept frequency loading (for colour palette construction)
# ---------------------------------------------------------------------------

def load_concept_frequencies(root_dir: Optional[str] = None) -> Optional[Dict[int, float]]:
    """
    Load concept frequencies from the PIDGIN evidence cache.

    Returns a dict {concept_id: frequency} or None if the cache is absent.
    Frequencies are the fraction of game steps where each concept fired.
    """
    root = root_dir or _ROOT
    cache_path = os.path.join(root, "pidgin", "results", "evidence_cache.pkl")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            evidence = pickle.load(f)
        return {k: ev.frequency for k, ev in evidence.items()}
    except Exception as e:
        print(f"Warning: could not load evidence cache ({e}). Using ID-order colours.")
        return None
