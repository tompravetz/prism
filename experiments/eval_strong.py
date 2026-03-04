"""
Evaluate transferred agents against GnuGo (strong opponent).

Tests whether concept-mediated transfer preserves performance against strong
opponents, not just random ones. Takes the best transferred agents from
transfer_same_task and curriculum experiments, evaluates them vs GnuGo L1-L3.

Protocol:
    1. Load best transferred agents (PPO->DQN, curriculum 5x5->7x7)
    2. Load baseline agents (PPO bottleneck, DQN bottleneck)
    3. Play each vs GnuGo L1, L2, L3 (50 games each)
    4. Report: whether transfer preserves relative performance

Requires: GnuGo binary at gnugo-3.8/gnugo.exe (or gnugo-3.8/gnugo on Linux)

Usage:
    python experiments/eval_strong.py
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.concept_aligner import ConceptAligner
from src.utils import set_seed, get_device, ensure_dir


# ============================================================
# GTP interface for GnuGo evaluation
# ============================================================

class GTPInterface:
    """
    GTP (Go Text Protocol) interface for communicating with GnuGo.

    GTP is a text-based protocol for Go engines. Commands are sent as
    plain text, responses start with '=' (success) or '?' (error).

    Key commands:
        boardsize N        — set board size
        clear_board        — reset board
        play COLOR COORD   — place a stone (e.g., "play black D4")
        genmove COLOR      — ask GnuGo to generate a move
        final_score        — get game score
    """

    def __init__(self, gnugo_path="gnugo-3.8/gnugo.exe", level=1, board_size=7):
        """
        Start a GnuGo subprocess.

        Args:
            gnugo_path: Path to GnuGo binary.
            level: GnuGo playing strength (1-10).
            board_size: Board size.
        """
        # Find GnuGo binary
        if not os.path.exists(gnugo_path):
            # Try alternative paths
            alternatives = [
                "gnugo-3.8/gnugo",
                "gnugo.exe",
                "gnugo",
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    gnugo_path = alt
                    break
            else:
                raise FileNotFoundError(
                    f"GnuGo binary not found. Tried: {gnugo_path}, {alternatives}"
                )

        self.process = subprocess.Popen(
            [gnugo_path, "--mode", "gtp", "--level", str(level),
             "--boardsize", str(board_size)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.board_size = board_size

    def send_command(self, cmd):
        """Send a GTP command and return the response."""
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

        response_lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == "":
                if response_lines:
                    break
                continue
            response_lines.append(line)

        response = "\n".join(response_lines)
        if response.startswith("="):
            return response[1:].strip()
        elif response.startswith("?"):
            raise RuntimeError(f"GnuGo error: {response}")
        return response

    def play(self, color, row, col):
        """Place a stone on the board."""
        coord = self._rc_to_gtp(row, col)
        self.send_command(f"play {color} {coord}")

    def play_pass(self, color):
        """Pass."""
        self.send_command(f"play {color} pass")

    def genmove(self, color):
        """Ask GnuGo to generate a move. Returns (row, col) or 'pass' or 'resign'."""
        response = self.send_command(f"genmove {color}")
        if response.lower() in ("pass", "resign"):
            return response.lower()
        return self._gtp_to_rc(response)

    def clear_board(self):
        """Clear the board for a new game."""
        self.send_command("clear_board")

    def final_score(self):
        """Get the final score. Returns string like 'B+5.5' or 'W+3.5'."""
        return self.send_command("final_score")

    def close(self):
        """Terminate the GnuGo process."""
        try:
            self.send_command("quit")
        except Exception:
            pass
        self.process.terminate()
        self.process.wait()

    def _rc_to_gtp(self, row, col):
        """Convert (row, col) to GTP coordinate (e.g., 'D4')."""
        # GTP uses letters A-T (skipping I) for columns, 1-19 for rows (bottom-up)
        col_letter = chr(ord('A') + col + (1 if col >= 8 else 0))  # Skip 'I'
        row_number = self.board_size - row  # GTP rows are bottom-up
        return f"{col_letter}{row_number}"

    def _gtp_to_rc(self, coord):
        """Convert GTP coordinate to (row, col)."""
        col_letter = coord[0].upper()
        row_number = int(coord[1:])

        col = ord(col_letter) - ord('A')
        if col > 8:
            col -= 1  # Adjust for skipped 'I'

        row = self.board_size - row_number
        return (row, col)


def eval_agent_vs_gnugo(agent_fn, encoder, cm, gnugo_level=1,
                        n_games=50, board_size=7, device=None):
    """
    Evaluate a concept bottleneck agent against GnuGo.

    The agent plays as Black (first mover). GnuGo plays as White via
    GoEnv's opponent_fn interface — the same pattern used by the visualizer.
    This ensures the agent always receives correct board observations.

    Args:
        agent_fn: Function(obs, action_mask) -> action_int.
        encoder: Agent's encoder (unused here; baked into agent_fn).
        cm: Agent's concept manager (unused here; baked into agent_fn).
        gnugo_level: GnuGo playing strength (0-10).
        n_games: Number of games to play.
        board_size: Board size.
        device: Torch device.

    Returns:
        Dict with win_rate, wins, losses, draws, game_lengths.
    """
    from visualizer.opponents import GnuGoOpponent

    device = device or get_device()
    wins, losses, draws = 0, 0, 0
    game_lengths = []

    # Create the GnuGo opponent once; reset its board state between games.
    try:
        gnugo = GnuGoOpponent(level=gnugo_level, board_size=board_size)
    except Exception as e:
        print(f"  GnuGo not found! Skipping evaluation. ({e})")
        return {
            "win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0,
            "game_lengths": [], "error": str(e),
        }

    for game_idx in range(n_games):
        gnugo.reset()
        # Pass GnuGo as opponent_fn so GoEnv drives white's turns correctly —
        # identical to how the visualizer records games.
        env = GoEnv(board_size=board_size, opponent_fn=gnugo)
        obs, info = env.reset()

        done = False
        move_count = 0
        last_reward = 0.0

        while not done and move_count < 200:
            action_mask = info.get("action_mask", None)
            action = agent_fn(obs, action_mask)
            move_count += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                last_reward = float(reward)

        # Winner determined by GoEnv's terminal reward (+1 black, -1 white).
        if last_reward > 0.5:
            wins += 1
        elif last_reward < -0.5:
            losses += 1
        else:
            draws += 1

        game_lengths.append(move_count)
        env.close()

        if (game_idx + 1) % 10 == 0:
            wr = wins / (game_idx + 1)
            print(f"    Game {game_idx+1}/{n_games}: W={wins} L={losses} D={draws} WR={wr:.2%}")

    gnugo.close()
    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0

    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_games": total,
        "mean_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
    }


def run_gnugo_evaluation():
    """Run GnuGo evaluation for baseline and transferred agents."""
    set_seed(42)
    device = get_device()
    timestamp = time.strftime("%H:%M:%S")

    print(f"[{timestamp}] ============================================================")
    print(f"[{timestamp}] GnuGo Evaluation: Baseline vs Transferred Agents")
    print(f"[{timestamp}] ============================================================")

    results = {}

    # Load PPO baseline bottleneck
    print(f"\n--- Loading agents ---")
    env = GoEnv(board_size=7)

    ppo_encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    ppo_encoder.load_state_dict(
        torch.load("models/baseline/ppo_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    ppo_encoder.to(device)
    ppo_encoder.eval()

    ppo_cm = ConceptManager(n_concepts=64)
    ppo_cm.load("models/bottleneck/concepts_ppo_k64.pkl")

    ppo_policy = ConceptBottleneckPolicy(
        n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50,
    )
    ppo_policy.load_state_dict(
        torch.load("models/bottleneck/ppo_bottleneck_final.pt",
                    map_location=device, weights_only=True)
    )
    ppo_policy.to(device)
    ppo_policy.eval()

    # Load DQN encoder + concepts for transferred policy evaluation
    dqn_encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    dqn_encoder.load_state_dict(
        torch.load("models/baseline/dqn_go_encoder.pt",
                    map_location=device, weights_only=True)
    )
    dqn_encoder.to(device)
    dqn_encoder.eval()

    dqn_cm = ConceptManager(n_concepts=64)
    dqn_cm.load("models/bottleneck/concepts_dqn_k64.pkl")
    env.close()

    # Create PPO->DQN transferred policy
    aligner = ConceptAligner(ppo_cm, dqn_cm)
    mapping = aligner.hungarian_alignment()
    transferred = aligner.transfer_policy(
        ppo_policy, mapping, target_n_concepts=64, target_n_actions=50,
    )
    transferred.to(device)
    transferred.eval()

    # Define agent functions
    def ppo_agent_fn(obs, mask):
        c = ppo_cm.assign_concept_from_obs(ppo_encoder, obs, device)
        return ppo_policy.get_action(c, mask, deterministic=True)

    def transferred_agent_fn(obs, mask):
        c = dqn_cm.assign_concept_from_obs(dqn_encoder, obs, device)
        return transferred.get_action(c, mask, deterministic=True)

    # Evaluate vs GnuGo at different levels
    agents_to_eval = {
        "PPO Bottleneck (baseline)": (ppo_agent_fn, ppo_encoder, ppo_cm),
        "PPO->DQN Transfer": (transferred_agent_fn, dqn_encoder, dqn_cm),
    }

    for agent_name, (agent_fn, enc, cm) in agents_to_eval.items():
        results[agent_name] = {}
        for level in [1, 2, 3]:
            print(f"\n--- {agent_name} vs GnuGo Level {level} ---")
            level_result = eval_agent_vs_gnugo(
                agent_fn, enc, cm,
                gnugo_level=level, n_games=50, device=device,
            )
            results[agent_name][f"level_{level}"] = level_result
            print(f"  Win rate: {level_result['win_rate']:.2%} "
                  f"(W={level_result['wins']} L={level_result['losses']})")

    # Save results
    ensure_dir("results")
    output_path = "results/gnugo_evaluation.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    # Summary
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ============================================================")
    print(f"[{ts}] GnuGo Evaluation Summary")
    print(f"[{ts}] {'Agent':<30} {'L1':>6} {'L2':>6} {'L3':>6}")
    print(f"[{ts}] {'-'*52}")
    for agent_name, levels in results.items():
        l1 = levels.get("level_1", {}).get("win_rate", 0)
        l2 = levels.get("level_2", {}).get("win_rate", 0)
        l3 = levels.get("level_3", {}).get("win_rate", 0)
        print(f"[{ts}] {agent_name:<30} {l1:>5.0%} {l2:>5.0%} {l3:>5.0%}")

    print(f"\nDone!")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate PRISM bottleneck agent against GnuGo"
    )
    parser.add_argument(
        "--algo", choices=["ppo", "dqn"], default="ppo",
        help="Which bottleneck agent to evaluate (default: ppo)"
    )
    parser.add_argument(
        "--level", type=int, default=1, metavar="N",
        help="GnuGo strength level 0-10 (default: 1)"
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Number of games to play (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the full multi-agent, multi-level evaluation suite"
    )
    parser.add_argument(
        "--bottleneck-dir", type=str, default="models/bottleneck",
        help="Directory containing bottleneck model and concepts "
             "(default: models/bottleneck). Use models/bottleneck_dagger "
             "to evaluate the DAgger-encoder bottleneck."
    )
    parser.add_argument(
        "--encoder-dir", type=str, default="models/baseline",
        help="Directory containing the encoder .pt file "
             "(default: models/baseline)."
    )
    args = parser.parse_args()

    if args.full:
        run_gnugo_evaluation()
    else:
        # Quick single-agent evaluation
        set_seed(args.seed)
        device = get_device()

        bottleneck_dir = args.bottleneck_dir
        encoder_dir = args.encoder_dir
        print(f"\nLoading {args.algo.upper()} bottleneck agent "
              f"(bottleneck_dir={bottleneck_dir})...")
        env = GoEnv(board_size=7)
        encoder = GoCNNEncoder(env.observation_space, features_dim=128)
        encoder.load_state_dict(
            torch.load(f"{encoder_dir}/{args.algo}_go_encoder.pt",
                       map_location=device, weights_only=True)
        )
        encoder.to(device).eval()

        cm = ConceptManager(n_concepts=64)
        cm.load(f"{bottleneck_dir}/concepts_{args.algo}_k64.pkl")

        if args.algo == "dqn":
            policy = ConceptDQNPolicy(
                n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50,
            )
            policy.load_state_dict(
                torch.load(f"{bottleneck_dir}/{args.algo}_bottleneck_final.pt",
                           map_location=device, weights_only=True)
            )
            policy.to(device).eval()

            def agent_fn(obs, mask):
                c = cm.assign_concept_from_obs(encoder, obs, device)
                return policy.get_action(c, mask, epsilon=0.0)
        else:
            policy = ConceptBottleneckPolicy(
                n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50,
            )
            policy.load_state_dict(
                torch.load(f"{bottleneck_dir}/{args.algo}_bottleneck_final.pt",
                           map_location=device, weights_only=True)
            )
            policy.to(device).eval()

            def agent_fn(obs, mask):
                c = cm.assign_concept_from_obs(encoder, obs, device)
                return policy.get_action(c, mask, deterministic=True)
        env.close()

        print(f"Evaluating {args.algo.upper()} vs GnuGo Level {args.level} "
              f"({args.games} games, seed={args.seed})...\n")
        result = eval_agent_vs_gnugo(
            agent_fn, encoder, cm,
            gnugo_level=args.level, n_games=args.games, device=device,
        )
        print(f"\n{'='*50}")
        print(f"  {args.algo.upper()} vs GnuGo Level {args.level}")
        print(f"  W={result['wins']}  L={result['losses']}  D={result['draws']}"
              f"  ({result['total_games']} games)")
        print(f"  Win rate: {result['win_rate']:.1%}")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print(f"{'='*50}")

        # Save single-run results so downstream scripts can aggregate them
        ensure_dir("results")
        tag = "" if bottleneck_dir == "models/bottleneck" else "_dagger"
        out_path = f"results/eval_strong_{args.algo}{tag}_L{args.level}.json"
        save_data = {
            "algo": args.algo,
            "level": args.level,
            "n_games": args.games,
            "seed": args.seed,
            **result,
        }

        def _convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(out_path, "w") as f:
            import json as _json
            _json.dump(save_data, f, indent=2, default=_convert)
        print(f"Results saved to {out_path}")
