"""
PRISM Visualizer — CLI Entry Point.

Plays a PRISM agent game (or selects the best of N candidates), renders
each agent move as a visualization frame, and exports the result as a GIF,
MP4, and/or individual PNG frames.

Usage:
    # Best game from 20 candidates, GIF output (default)
    python -m visualizer.run_visualizer

    # Specific seed, also export MP4 and per-frame PNGs for paper
    python -m visualizer.run_visualizer --seed 7 --mp4 --export-frames

    # Quick test: 10 moves, 2 fps
    python -m visualizer.run_visualizer --max-moves 10 --fps 2 --seed 0

    # Use DQN model, portrait layout for paper frames
    python -m visualizer.run_visualizer --algo dqn --paper --export-frames

    # Select best of 30 candidates
    python -m visualizer.run_visualizer --best-of 30

    # Self-play opponent (PRISM policy plays both sides)
    python -m visualizer.run_visualizer --opponent self

    # GnuGo opponent at level 3 (auto-detects gnugo-3.8/gnugo.exe)
    python -m visualizer.run_visualizer --opponent gnugo --gnugo-level 3

    # GnuGo with explicit executable path
    python -m visualizer.run_visualizer --opponent gnugo --gnugo-exe C:/prism/gnugo-3.8/gnugo.exe
"""

import argparse
import json
import os
import sys
import time

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from visualizer.game_recorder import GameRecorder, load_concept_frequencies, GameResult
from visualizer.frame_renderer import FrameRenderer, build_concept_colors, load_descriptions
from visualizer.exporter import (
    export_gif, export_mp4, export_frames,
    build_frame_metadata, build_summary_metadata,
)
from visualizer.opponents import RandomOpponent, SelfPlayOpponent, GnuGoOpponent


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

PIDGIN_RESULTS = os.path.join(_ROOT, "pidgin", "results")
DESC_PATH      = os.path.join(PIDGIN_RESULTS, "concept_descriptions.json")
GENERIC_PATH   = os.path.join(PIDGIN_RESULTS, "descriptions_generic.json")
OUTPUTS_DIR    = os.path.join(_ROOT, "visualizer", "outputs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_opponent(args, recorder: GameRecorder):
    """Construct the white opponent from parsed CLI args."""
    if args.opponent == "random":
        print("Opponent: random")
        return RandomOpponent()

    if args.opponent == "self":
        print("Opponent: self-play (PRISM policy)")
        return SelfPlayOpponent(
            encoder=recorder.encoder,
            concept_manager=recorder.concept_manager,
            policy=recorder.policy,
            device=recorder.device,
        )

    if args.opponent == "gnugo":
        if not GnuGoOpponent.is_available(hint=args.gnugo_exe):
            print(
                "ERROR: gnugo not found.\n"
                "  Pass the executable path with --gnugo-exe, or install gnugo:\n"
                "    winget install gnugo          (Windows)\n"
                "    sudo apt install gnugo         (Linux)\n"
                "    brew install gnugo             (macOS)\n"
                "  Or place gnugo.exe in C:/prism/gnugo-3.8/gnugo.exe"
            )
            sys.exit(1)
        opp = GnuGoOpponent(
            level=args.gnugo_level,
            board_size=7,
            komi=5.5,
            exe_path=args.gnugo_exe,
        )
        print(f"Opponent: {opp}")
        return opp

    # Fallback
    return RandomOpponent()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM game visualizer — renders interpretability panels for a Go game"
    )

    # Game selection
    game_group = parser.add_mutually_exclusive_group()
    game_group.add_argument(
        "--seed", type=int, default=None,
        help="Use a specific RNG seed (skips best-game selection)"
    )
    game_group.add_argument(
        "--best-of", type=int, default=20,
        metavar="N",
        help="Select the highest concept-diversity game from N candidates (default: 20)"
    )

    # Agent
    parser.add_argument(
        "--algo", choices=["ppo", "dqn"], default="ppo",
        help="Which trained agent to use (default: ppo)"
    )
    parser.add_argument(
        "--max-moves", type=int, default=60,
        help="Maximum agent moves to record per game (default: 60)"
    )

    # Opponent
    parser.add_argument(
        "--opponent", choices=["random", "self", "gnugo"], default="random",
        help="White opponent policy: random (default), self (PRISM self-play), "
             "or gnugo"
    )
    parser.add_argument(
        "--gnugo-level", type=int, default=1, metavar="N",
        help="GnuGo strength level 0–10 (default: 1; only used with --opponent gnugo)"
    )
    parser.add_argument(
        "--gnugo-exe", default=None, metavar="PATH",
        help="Path to gnugo executable (auto-detected if omitted)"
    )

    # Output
    parser.add_argument(
        "--out-dir", default=OUTPUTS_DIR,
        help="Directory for output files"
    )
    parser.add_argument(
        "--fps", type=float, default=1.5,
        help="Frames per second for GIF / MP4 (default: 1.5)"
    )
    parser.add_argument(
        "--no-gif", action="store_true",
        help="Skip GIF export"
    )
    parser.add_argument(
        "--mp4", action="store_true",
        help="Also export MP4 (requires imageio-ffmpeg)"
    )
    parser.add_argument(
        "--export-frames", action="store_true",
        help="Save every frame as a PNG for paper figure selection"
    )
    parser.add_argument(
        "--frame-dpi", type=int, default=300,
        help="DPI for PNG frame export (default: 300)"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Use portrait layout for frame export (default: video/landscape)"
    )
    parser.add_argument(
        "--summary-hold", type=float, default=6.0, metavar="SEC",
        help="Seconds to hold the summary frame in GIF/MP4 (default: 6)"
    )

    # Descriptions
    parser.add_argument(
        "--descriptions", default=DESC_PATH,
        help="Path to PIDGIN concept_descriptions.json"
    )
    parser.add_argument(
        "--no-descriptions", action="store_true",
        help="Skip PIDGIN panel; show concept ID badge only"
    )

    args = parser.parse_args()
    layout = "paper" if args.paper else "video"

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # ── Stage 1: Load artifacts ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PRISM Visualizer  |  algo={args.algo}  |  layout={layout}")
    print(f"{'='*60}")

    recorder = GameRecorder(algo=args.algo, root_dir=_ROOT)
    recorder.load_artifacts()

    # ── Build opponent ─────────────────────────────────────────────────
    opponent = _build_opponent(args, recorder)

    # ── Stage 2: Record game ───────────────────────────────────────────
    print()
    try:
        if args.seed is not None:
            print(f"Recording game (seed={args.seed}, max_moves={args.max_moves}, "
                  f"opponent={opponent})...")
            frames, game_result = recorder.record_game(
                seed=args.seed, max_moves=args.max_moves, opponent_fn=opponent,
            )
            selection_stats = {
                "mode": "fixed_seed",
                "seed": args.seed,
                "n_agent_moves": len(frames),
                "opponent": str(opponent),
                "winner": game_result.winner,
            }
        else:
            frames, selection_stats, game_result = recorder.record_best_game(
                n_candidates=args.best_of,
                max_moves=args.max_moves,
                opponent_fn=opponent,
            )
            selection_stats["opponent"] = str(opponent)
            # Save game selection log for reproducibility
            log_path = os.path.join(args.out_dir, "game_selection_log.json")
            with open(log_path, "w") as f:
                json.dump(selection_stats, f, indent=2)
            print(f"Selection log saved: {log_path}")
    finally:
        # Always clean up GnuGo subprocess, even if recording failed
        if hasattr(opponent, "close"):
            opponent.close()

    if not frames:
        print("ERROR: No frames recorded. Check that the model files exist.")
        sys.exit(1)

    print(
        f"\nRecorded {len(frames)} agent moves  "
        f"({len(set(f.concept_id for f in frames))} unique concepts)  "
        f"→  {game_result.winner.upper()}"
        + (f"  ({game_result.score_margin:+.1f} pts)" if game_result.score_margin is not None else "")
    )

    # ── Stage 3: Build renderer ────────────────────────────────────────
    print("\nBuilding renderer...")

    # Load descriptions (with fallback to generic for un-described concepts)
    if args.no_descriptions:
        descriptions = {}
    else:
        descriptions = load_descriptions(
            desc_path=args.descriptions,
            generic_path=GENERIC_PATH,
        )
        n_textgrad = sum(
            1 for d in descriptions.values() if d.get("_source") == "textgrad"
        )
        print(
            f"Loaded descriptions: {n_textgrad}/64 TextGrad-optimized, "
            f"{64 - n_textgrad} generic fallback"
        )

    # Build concept colour map (frequency-ranked, bit-reversal hue order)
    frequencies = load_concept_frequencies(root_dir=_ROOT)
    concept_colors = build_concept_colors(frequencies=frequencies)
    if frequencies:
        print("Concept colours: frequency-ranked (bit-reversal hue permutation)")
    else:
        print("Concept colours: ID-order (evidence cache not found)")

    renderer = FrameRenderer(
        descriptions=descriptions,
        concept_colors=concept_colors,
        layout=layout,
    )

    # ── Stage 4: Render frames ─────────────────────────────────────────
    print(f"\nRendering {len(frames)} frames...")
    rendered: list = []
    for i, fd in enumerate(frames, start=1):
        rgb = renderer.render_frame(fd)
        rendered.append(rgb)
        if i % 5 == 0 or i == len(frames):
            elapsed = time.time() - t_start
            print(f"  {i}/{len(frames)} frames  ({elapsed:.1f}s elapsed)")

    print(f"All frames rendered. Shape: {rendered[0].shape}")

    # ── Summary frame ──────────────────────────────────────────────────
    print("Rendering summary frame...")
    summary_rgb = renderer.render_summary_frame(frames, game_result)
    rendered.append(summary_rgb)
    print(f"Summary frame appended  ({game_result.winner.upper()})")

    # ── Stage 5: Export ────────────────────────────────────────────────
    algo_tag = args.algo
    seed_tag = (
        f"seed{args.seed}" if args.seed is not None
        else f"best{args.best_of}"
    )
    opp_tag = {
        "random": "rnd",
        "self":   "self",
        "gnugo":  f"gnugo{args.gnugo_level}",
    }.get(args.opponent, "rnd")
    base_name = f"prism_{algo_tag}_{seed_tag}_{opp_tag}"

    print()
    if not args.no_gif:
        gif_path = os.path.join(args.out_dir, f"{base_name}.gif")
        export_gif(rendered, gif_path, fps=args.fps, summary_hold_sec=args.summary_hold)

    if args.mp4:
        mp4_path = os.path.join(args.out_dir, f"{base_name}.mp4")
        export_mp4(rendered, mp4_path, fps=args.fps, summary_hold_sec=args.summary_hold)

    if args.export_frames:
        frames_dir = os.path.join(args.out_dir, f"{base_name}_frames")
        metadata = build_frame_metadata(frames) + [build_summary_metadata(game_result)]
        export_frames(rendered, metadata, frames_dir, dpi=args.frame_dpi)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
