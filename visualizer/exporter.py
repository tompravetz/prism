"""
PRISM Visualizer Exporter.

Assembles a list of rendered RGB frames (numpy uint8 arrays) into:
    - GIF   via Pillow (no extra deps)
    - MP4   via imageio + imageio-ffmpeg (optional; skips gracefully if absent)
    - PNG   individual frames at user-specified DPI, plus frame_index.json
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------

def export_gif(
    frames: List[np.ndarray],
    out_path: str,
    fps: float = 1.5,
    summary_hold_sec: float = 6.0,
) -> None:
    """
    Save frames as an animated GIF using Pillow.

    The last frame (expected to be the summary frame) is held for
    summary_hold_sec seconds so there is time to read it.  All other
    frames use the normal per-fps duration.

    Args:
        frames: List of H×W×3 uint8 RGB arrays.
        out_path: Destination file path (should end in .gif).
        fps: Playback speed in frames per second for regular frames.
        summary_hold_sec: How long to hold the last (summary) frame.
    """
    if not frames:
        print("export_gif: no frames to export.")
        return

    frame_ms   = int(1000.0 / fps)
    summary_ms = int(summary_hold_sec * 1000)

    # Per-frame duration list: regular speed for all but the last frame
    durations = [frame_ms] * (len(frames) - 1) + [summary_ms]

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=durations,
        optimize=False,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"GIF saved: {out_path}  ({len(frames)} frames, {size_kb:.0f} KB, "
          f"summary held {summary_hold_sec:.0f}s)")


# ---------------------------------------------------------------------------
# MP4
# ---------------------------------------------------------------------------

def export_mp4(
    frames: List[np.ndarray],
    out_path: str,
    fps: float = 1.5,
    summary_hold_sec: float = 6.0,
) -> bool:
    """
    Save frames as an MP4 video using imageio + imageio-ffmpeg.

    Returns True on success, False if imageio is unavailable.

    The last frame (expected to be the summary frame) is repeated enough
    times to fill summary_hold_sec seconds at the given fps.

    Args:
        frames: List of H×W×3 uint8 RGB arrays.
        out_path: Destination file path (should end in .mp4).
        fps: Playback speed in frames per second.
        summary_hold_sec: How long to hold the last (summary) frame.
    """
    try:
        import imageio
    except ImportError:
        print(
            "MP4 export skipped: imageio is not installed.\n"
            "  Install with: venv/Scripts/python.exe -m pip install imageio imageio-ffmpeg"
        )
        return False

    if not frames:
        print("export_mp4: no frames to export.")
        return False

    summary_repeats = max(1, int(round(summary_hold_sec * fps)))

    with imageio.get_writer(out_path, fps=fps, codec="h264", quality=8) as writer:
        for frame in frames[:-1]:
            writer.append_data(frame)
        for _ in range(summary_repeats):
            writer.append_data(frames[-1])

    total = len(frames) - 1 + summary_repeats
    size_kb = os.path.getsize(out_path) / 1024
    print(f"MP4 saved: {out_path}  ({total} frames, {size_kb:.0f} KB, "
          f"summary held {summary_hold_sec:.0f}s)")
    return True


# ---------------------------------------------------------------------------
# Individual PNG frames
# ---------------------------------------------------------------------------

def export_frames(
    frames: List[np.ndarray],
    frame_metadata: List[Dict],
    out_dir: str,
    dpi: int = 300,
) -> None:
    """
    Save each frame as a high-DPI PNG for paper figure selection.

    Also writes frame_index.json alongside the PNGs for quick browsing.

    Args:
        frames: List of H×W×3 uint8 RGB arrays.
        frame_metadata: Per-frame metadata dicts (concept_id, action_notation, …)
                        from build_frame_metadata().
        out_dir: Directory to write PNGs and the index file into.
        dpi: DPI hint written into the PNG metadata (actual pixels are fixed
             by the array dimensions; this affects cm/inch metadata).
    """
    os.makedirs(out_dir, exist_ok=True)

    index = []
    for i, (frame_rgb, meta) in enumerate(zip(frames, frame_metadata)):
        filename = f"frame_{i + 1:03d}.png"
        path = os.path.join(out_dir, filename)

        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(path, format="PNG", dpi=(dpi, dpi))

        index.append({"file": filename, **meta})

    index_path = os.path.join(out_dir, "frame_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(
        f"Frames saved: {out_dir}/  "
        f"({len(frames)} PNGs @ {dpi} dpi, index → frame_index.json)"
    )


def build_summary_metadata(game_result) -> Dict:
    """
    Build the metadata dict for the summary frame entry in frame_index.json.

    Args:
        game_result: GameResult from record_game().

    Returns:
        A dict suitable for appending to the frame index.
    """
    return {
        "type": "summary",
        "winner": game_result.winner,
        "score_margin": game_result.score_margin,
        "reward": game_result.reward,
        "n_agent_moves": game_result.n_agent_moves,
        "n_total_half_moves": game_result.n_total_half_moves,
        "game_seed": game_result.game_seed,
        "algo": game_result.algo,
        "opponent": game_result.opponent_name,
        "n_unique_concepts": game_result.n_unique_concepts,
        "diversity_score": round(game_result.diversity_score, 3),
        "concept_counts": game_result.concept_counts,
    }


def build_frame_metadata(frames) -> List[Dict]:
    """
    Build per-frame metadata dicts from a list of FrameData objects.

    Used as the sidecar data written to frame_index.json.
    """
    from visualizer.frame_renderer import action_to_notation

    index = []
    for fd in frames:
        index.append({
            "frame": fd.agent_move_number,
            "agent_move": fd.agent_move_number,
            "game_seed": fd.game_seed,
            "algo": fd.algo,
            "concept_id": fd.concept_id,
            "dist_to_centroid": round(float(fd.dist_to_centroid), 3),
            "chosen_action": fd.chosen_action,
            "chosen_action_notation": action_to_notation(fd.chosen_action),
            "chosen_action_prob": round(float(fd.action_probs[fd.chosen_action]), 4),
            "top_action_notation": action_to_notation(
                int(np.argmax(fd.action_probs))
            ),
            "top_action_prob": round(float(np.max(fd.action_probs)), 4),
            "n_unique_concepts_so_far": len(set(fd.concept_history)),
            "concept_transition": (
                fd.concept_history[-2] != fd.concept_history[-1]
                if len(fd.concept_history) >= 2 else False
            ),
        })
    return index
