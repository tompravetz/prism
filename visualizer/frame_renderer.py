"""
PRISM Frame Renderer.

Converts a FrameData + PIDGIN descriptions into a rendered RGB frame
(numpy uint8 array) using matplotlib's non-interactive Agg backend.

Each frame is rendered independently — no state is shared between frames.
The renderer is safe to call in a loop; each call creates and closes its
own matplotlib figure.

Key rendering layers (board, bottom to top):
    1. Board background (tan)
    2. Grid lines
    3. Star points (hoshi)
    4. Action probability heatmap (imshow, alpha=0.55)
    5. Illegal-position markers (faint x at empty illegal cells)
    6. Stone circles (black and white)
    7. Chosen-move ring (lime green)

Right panel:
    - Concept description (PIDGIN Name / Description / Key Actions / Strategic Importance)
    - Top-moves bar chart

Bottom strip:
    - Concept history timeline (coloured squares, one per agent move)
    - Move log (algebraic notation)
"""

import colorsys
import re
import sys
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np

# Must set backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from visualizer.game_recorder import FrameData


# ---------------------------------------------------------------------------
# Layout presets
# ---------------------------------------------------------------------------

LAYOUTS = {
    "video": {
        "figsize": (16.0, 9.0),   # inches at 80 dpi → 1280×720 px
        "dpi": 80,
        "board_font": 9,
        "desc_font": 8,
        "title_font": 10,
        "log_font": 7,
    },
    "paper": {
        "figsize": (8.0, 10.67),  # inches at 100 dpi → 800×1067 px (portrait)
        "dpi": 100,
        "board_font": 10,
        "desc_font": 9,
        "title_font": 12,
        "log_font": 8,
    },
}

# ---------------------------------------------------------------------------
# Concept colour palette
# ---------------------------------------------------------------------------

def _bit_reversed_hues(n: int = 64) -> List[float]:
    """Return n hue values in bit-reversal order for maximal separation."""
    n_bits = int(np.ceil(np.log2(n)))
    hues = []
    for i in range(n):
        bits = format(i, f"0{n_bits}b")
        rev_idx = int(bits[::-1], 2)
        hues.append(rev_idx / n)
    return hues


def build_concept_colors(
    frequencies: Optional[Dict[int, float]] = None,
    n_concepts: int = 64,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Map concept_id → RGB colour.

    Concepts are ranked by descending frequency, then assigned palette slots
    in bit-reversal hue order so the most frequent concepts get maximally
    separated colours.

    If frequencies is None, falls back to assignment by concept ID order.
    """
    hues = _bit_reversed_hues(n_concepts)
    palette = [colorsys.hsv_to_rgb(h, 0.75, 0.85) for h in hues]

    if frequencies:
        sorted_ids = sorted(range(n_concepts), key=lambda k: -frequencies.get(k, 0.0))
    else:
        sorted_ids = list(range(n_concepts))

    return {cid: palette[rank] for rank, cid in enumerate(sorted_ids)}


# ---------------------------------------------------------------------------
# PIDGIN description loading and parsing
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(
    r"^(Name|Description|Key Actions|Strategic Importance)\s*:\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_description(text: str) -> Dict[str, str]:
    """
    Parse a PIDGIN description string into its four fields.

    Returns a dict with keys: Name, Description, Key Actions,
    Strategic Importance. Missing fields are empty strings.
    """
    result = {
        "Name": "",
        "Description": "",
        "Key Actions": "",
        "Strategic Importance": "",
    }
    # Split on field labels to handle multi-line field values
    # Use a forward-lookahead split so each label anchors its own block
    parts = re.split(
        r"\n?(?=(?:Name|Description|Key Actions|Strategic Importance)\s*:)",
        text,
        flags=re.IGNORECASE,
    )
    for part in parts:
        m = _FIELD_RE.match(part.strip())
        if m:
            key = m.group(1).strip()
            # Normalise capitalisation to match result dict keys
            for k in result:
                if k.lower() == key.lower():
                    key = k
                    break
            # Collect everything after the first line as continuation
            rest_of_block = part.strip()[len(m.group(0)):].strip()
            value = m.group(2).strip()
            if rest_of_block:
                value = value + " " + rest_of_block
            result[key] = value.strip()
    return result


def load_descriptions(
    desc_path: str,
    generic_path: Optional[str] = None,
    n_concepts: int = 64,
) -> Dict[int, Dict[str, str]]:
    """
    Load and parse PIDGIN descriptions, falling back to generic for missing concepts.

    Returns dict mapping concept_id → parsed field dict with an extra
    "_source" key ("textgrad" | "generic" | "none").
    """
    import json

    def _load_raw(path: str) -> Dict[int, str]:
        try:
            with open(path) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    optimized = _load_raw(desc_path)
    generic = _load_raw(generic_path) if generic_path else {}

    result: Dict[int, Dict[str, str]] = {}
    for k in range(n_concepts):
        if k in optimized and optimized[k].strip():
            parsed = parse_description(optimized[k])
            parsed["_source"] = "textgrad"
            parsed["_raw"] = optimized[k]
        elif k in generic and generic[k].strip():
            parsed = parse_description(generic[k])
            parsed["_source"] = "generic"
            parsed["_raw"] = generic[k]
        else:
            parsed = {
                "Name": f"Concept {k}",
                "Description": "",
                "Key Actions": "",
                "Strategic Importance": "",
                "_source": "none",
                "_raw": "",
            }
        result[k] = parsed
    return result


# ---------------------------------------------------------------------------
# Action → Go notation helper
# ---------------------------------------------------------------------------

_COLS = "ABCDEFG"


def action_to_notation(action: int) -> str:
    """Convert action index to Go algebraic notation (e.g., 9 → 'C6')."""
    if action == 49:
        return "Pass"
    row, col = divmod(action, 7)
    return f"{_COLS[col]}{7 - row}"


# ---------------------------------------------------------------------------
# FrameRenderer
# ---------------------------------------------------------------------------

class FrameRenderer:
    """
    Renders FrameData objects into RGB numpy arrays using matplotlib.

    Args:
        descriptions: Parsed PIDGIN descriptions keyed by concept_id.
                      Build with load_descriptions().
        concept_colors: concept_id → (r, g, b) colour tuple.
                        Build with build_concept_colors().
        layout: "video" (1280×720) or "paper" (800×1067).
    """

    # Board visual constants
    BOARD_BG     = "#DCB960"   # Classic Go board tan
    GRID_COLOR   = "#3a2a00"
    STAR_COLOR   = "#2a1800"
    BLACK_STONE  = "#1a1a1a"
    WHITE_STONE  = "#f2f0e8"
    STONE_EDGE_B = "#444444"
    STONE_EDGE_W = "#999999"
    CHOSEN_RING  = "#00ff44"   # Lime green chosen-move highlight
    PANEL_BG     = "#1e1e2e"   # Dark panel background
    TEXT_COLOR   = "#e0e0e0"
    DIM_TEXT     = "#888899"
    ACCENT       = "#7c9cff"

    # Star point positions on 7×7 (row, col) — 0-indexed from top-left
    STAR_POINTS = [(2, 2), (2, 4), (4, 2), (4, 4), (3, 3)]

    def __init__(
        self,
        descriptions: Dict[int, Dict[str, str]],
        concept_colors: Dict[int, Tuple[float, float, float]],
        layout: str = "video",
    ):
        self.descriptions = descriptions
        self.concept_colors = concept_colors
        self.layout = layout
        self._lp = LAYOUTS[layout]

        # Heatmap colormap: yellow → orange → red
        self._heatmap_cmap = plt.cm.YlOrRd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frame(self, fd: FrameData) -> np.ndarray:
        """
        Render one FrameData to an RGB numpy array (H × W × 3, uint8).
        """
        lp = self._lp
        fig = plt.figure(figsize=lp["figsize"], dpi=lp["dpi"])
        fig.patch.set_facecolor(self.PANEL_BG)

        # ── Grid layout ───────────────────────────────────────────────
        if self.layout == "video":
            self._render_video_layout(fig, fd)
        else:
            self._render_paper_layout(fig, fd)

        # ── Extract RGB array ──────────────────────────────────────────
        # buffer_rgba() replaced tostring_rgb() in matplotlib 3.9+
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        rgb = buf[:, :, :3].copy()
        plt.close(fig)
        return rgb

    # ------------------------------------------------------------------
    # Layout orchestrators
    # ------------------------------------------------------------------

    def _render_video_layout(self, fig: plt.Figure, fd: FrameData) -> None:
        """Landscape layout: board left, panels right, strips at bottom."""
        lp = self._lp
        gs = gridspec.GridSpec(
            nrows=4, ncols=2,
            figure=fig,
            height_ratios=[0.9, 8, 1.5, 1.0],   # header, main, timeline, movelog
            width_ratios=[1, 1],
            hspace=0.04,
            wspace=0.04,
            left=0.03, right=0.97,
            top=0.97, bottom=0.03,
        )

        # Header
        ax_hdr = fig.add_subplot(gs[0, :])
        self._draw_header(ax_hdr, fd, lp)

        # Board
        ax_board = fig.add_subplot(gs[1, 0])
        self._draw_board(ax_board, fd, lp)

        # Right column: description + top moves
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs[1, 1],
            height_ratios=[3, 2],
            hspace=0.06,
        )
        ax_desc  = fig.add_subplot(gs_right[0])
        ax_moves = fig.add_subplot(gs_right[1])
        self._draw_concept_panel(ax_desc, fd, lp)
        self._draw_top_moves(ax_moves, fd, lp)

        # Timeline
        ax_time = fig.add_subplot(gs[2, :])
        self._draw_timeline(ax_time, fd, lp)

        # Move log
        ax_log = fig.add_subplot(gs[3, :])
        self._draw_move_log(ax_log, fd, lp)

    def _render_paper_layout(self, fig: plt.Figure, fd: FrameData) -> None:
        """Portrait layout: board top, panels stacked below."""
        lp = self._lp
        gs = gridspec.GridSpec(
            nrows=5, ncols=1,
            figure=fig,
            height_ratios=[0.6, 5, 3, 1.5, 0.9],  # header, board, desc+moves, timeline, log
            hspace=0.05,
            left=0.05, right=0.95,
            top=0.97, bottom=0.03,
        )

        ax_hdr   = fig.add_subplot(gs[0])
        ax_board = fig.add_subplot(gs[1])
        gs_mid   = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[2], width_ratios=[3, 2], wspace=0.05
        )
        ax_desc  = fig.add_subplot(gs_mid[0])
        ax_moves = fig.add_subplot(gs_mid[1])
        ax_time  = fig.add_subplot(gs[3])
        ax_log   = fig.add_subplot(gs[4])

        self._draw_header(ax_hdr, fd, lp)
        self._draw_board(ax_board, fd, lp)
        self._draw_concept_panel(ax_desc, fd, lp)
        self._draw_top_moves(ax_moves, fd, lp)
        self._draw_timeline(ax_time, fd, lp)
        self._draw_move_log(ax_log, fd, lp)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _draw_header(self, ax: plt.Axes, fd: FrameData, lp: dict) -> None:
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")
        desc = self.descriptions.get(fd.concept_id, {})
        concept_name = desc.get("Name", f"Concept {fd.concept_id}")
        ax.text(
            0.01, 0.5,
            f"PRISM  ·  {fd.algo.upper()}  ·  "
            f"Move {fd.agent_move_number}  ·  "
            f"Concept {fd.concept_id}: {concept_name}",
            transform=ax.transAxes,
            color=self.TEXT_COLOR,
            fontsize=lp["title_font"],
            fontweight="bold",
            va="center",
        )

    # ------------------------------------------------------------------
    # Board
    # ------------------------------------------------------------------

    def _draw_board(self, ax: plt.Axes, fd: FrameData, lp: dict,
                    show_heatmap: bool = True, show_chosen: bool = True) -> None:
        """Draw the 7×7 Go board with heatmap and stones."""
        ax.set_facecolor(self.BOARD_BG)
        ax.set_aspect("equal")
        ax.set_xlim(-0.7, 6.7)
        ax.set_ylim(-0.7, 6.7)

        # ── Grid lines ────────────────────────────────────────────────
        for i in range(7):
            ax.axhline(i, color=self.GRID_COLOR, lw=0.8, zorder=1)
            ax.axvline(i, color=self.GRID_COLOR, lw=0.8, zorder=1)

        # ── Star points ───────────────────────────────────────────────
        for r, c in self.STAR_POINTS:
            # Board display: row 0 at top → y = 6 - r
            ax.add_patch(mpatches.Circle(
                (c, 6 - r), radius=0.10,
                facecolor=self.STAR_COLOR, edgecolor="none", zorder=2
            ))

        # ── Heatmap overlay ───────────────────────────────────────────
        if show_heatmap:
            display_hm = np.full((7, 7), np.nan, dtype=np.float32)
            for a in range(49):
                r, c = a // 7, a % 7
                if fd.action_mask[a] == 1:
                    display_hm[r, c] = fd.action_heatmap[r, c]

            hm_max = float(np.nanmax(display_hm)) if not np.all(np.isnan(display_hm)) else 1.0
            hm_max = max(hm_max, 1e-6)

            ax.imshow(
                display_hm,
                cmap=self._heatmap_cmap,
                alpha=0.55,
                vmin=0.0,
                vmax=hm_max,
                origin="upper",
                extent=[-0.5, 6.5, -0.5, 6.5],
                aspect="equal",
                zorder=3,
                interpolation="nearest",
            )

        # ── Stones ────────────────────────────────────────────────────
        # NOTE: PettingZoo go_v5 board_history stores the last-to-step agent's
        # stones in plane 0. After white plays (which always precedes our read),
        # plane 0 = white's stones and plane 1 = black's stones — the opposite
        # of the channel names. We correct for this here by swapping the colors.
        obs = fd.obs  # (7, 7, 3): plane0=white_actual, plane1=black_actual
        for r in range(7):
            for c in range(7):
                display_y = 6 - r
                if obs[r, c, 0] > 0.5:   # plane 0 = white's actual stones
                    ax.add_patch(mpatches.Circle(
                        (c, display_y), radius=0.42,
                        facecolor=self.WHITE_STONE,
                        edgecolor=self.STONE_EDGE_W,
                        linewidth=0.6, zorder=5
                    ))
                elif obs[r, c, 1] > 0.5:  # plane 1 = black's actual stones
                    ax.add_patch(mpatches.Circle(
                        (c, display_y), radius=0.42,
                        facecolor=self.BLACK_STONE,
                        edgecolor=self.STONE_EDGE_B,
                        linewidth=0.6, zorder=5
                    ))

        # ── Chosen move highlight ──────────────────────────────────────
        if not show_chosen:
            pass
        elif (chosen := fd.chosen_action) == 49:
            ax.text(
                3, 3, "PASS",
                color=self.CHOSEN_RING, fontsize=lp["board_font"] + 4,
                fontweight="bold", ha="center", va="center",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", edgecolor=self.CHOSEN_RING, alpha=0.7)
            )
        else:
            cr, cc = chosen // 7, chosen % 7
            ax.add_patch(mpatches.Circle(
                (cc, 6 - cr), radius=0.40,
                fill=False,
                edgecolor=self.CHOSEN_RING,
                linewidth=2.2, zorder=7
            ))

        # ── Axis labels (Go notation) ──────────────────────────────────
        ax.set_xticks(range(7))
        ax.set_xticklabels(list("ABCDEFG"), fontsize=lp["board_font"],
                           color=self.GRID_COLOR, fontweight="bold")
        ax.set_yticks(range(7))
        ax.set_yticklabels([str(i) for i in range(1, 8)],
                           fontsize=lp["board_font"],
                           color=self.GRID_COLOR, fontweight="bold")
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.GRID_COLOR)
            spine.set_linewidth(1.5)

    # ------------------------------------------------------------------
    # Concept description panel
    # ------------------------------------------------------------------

    def _draw_concept_panel(self, ax: plt.Axes, fd: FrameData, lp: dict) -> None:
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")

        desc = self.descriptions.get(fd.concept_id, {})
        source = desc.get("_source", "none")
        concept_color = self.concept_colors.get(fd.concept_id, (0.5, 0.5, 0.8))

        # ── Concept badge (coloured rectangle + ID text) ───────────────
        badge_w, badge_h = 0.38, 0.13
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.83), badge_w, badge_h,
            boxstyle="round,pad=0.01",
            facecolor=concept_color, edgecolor="none",
            transform=ax.transAxes, zorder=2
        ))
        ax.text(
            0.01 + badge_w / 2, 0.83 + badge_h / 2,
            f"Concept {fd.concept_id}",
            transform=ax.transAxes,
            color="white", fontsize=lp["desc_font"] + 1,
            fontweight="bold", ha="center", va="center",
        )

        # ── Source indicator ───────────────────────────────────────────
        if source == "generic":
            ax.text(
                0.42, 0.87, "(description pending)",
                transform=ax.transAxes,
                color=self.DIM_TEXT, fontsize=lp["desc_font"] - 1,
                fontstyle="italic", va="center",
            )

        # ── Name ──────────────────────────────────────────────────────
        name = desc.get("Name", f"Concept {fd.concept_id}")
        ax.text(
            0.01, 0.79,
            name,
            transform=ax.transAxes,
            color=self.ACCENT,
            fontsize=lp["desc_font"] + 2,
            fontweight="bold",
            va="top",
        )

        # ── Description, Key Actions, Strategic Importance ────────────
        # Wrap each section to the panel width; y < 0.12 guards the
        # centrality bar — no artificial line-count truncation.
        line_specs = [
            ("Description",          desc.get("Description", ""),          self.TEXT_COLOR,  68),
            ("Key Actions",          desc.get("Key Actions", ""),          self.TEXT_COLOR,  68),
            ("Strategic Importance", desc.get("Strategic Importance", ""), self.DIM_TEXT,    68),
        ]
        y = 0.71
        line_h = 0.060

        for label, value, color, wrap_width in line_specs:
            if not value:
                continue
            # Stop before reaching the centrality bar
            if y < 0.12:
                break

            wrapped_lines = textwrap.wrap(value, width=wrap_width)
            if not wrapped_lines:
                continue
            display_text = "\n".join(wrapped_lines)
            n_lines = len(display_text.split("\n"))

            # Label
            ax.text(
                0.01, y,
                label.upper() + ":",
                transform=ax.transAxes,
                color=self.DIM_TEXT,
                fontsize=lp["desc_font"] - 2,
                fontweight="bold",
                va="top",
                clip_on=True,
            )
            y -= 0.036

            ax.text(
                0.01, y,
                display_text,
                transform=ax.transAxes,
                color=color,
                fontsize=lp["desc_font"],
                va="top",
                wrap=False,
                clip_on=True,
            )
            y -= line_h * n_lines + 0.018

        # ── Distance-to-centroid bar ───────────────────────────────────
        bar_y = 0.04
        bar_h = 0.025
        ax.add_patch(mpatches.Rectangle(
            (0.01, bar_y), 0.98, bar_h,
            facecolor="#333344", edgecolor=self.DIM_TEXT,
            linewidth=0.5, transform=ax.transAxes
        ))
        # Normalise dist: cap at 15 (typical range is 2–12 for K=64 on 128D)
        norm_dist = min(fd.dist_to_centroid / 15.0, 1.0)
        ax.add_patch(mpatches.Rectangle(
            (0.01, bar_y), 0.98 * (1.0 - norm_dist), bar_h,
            facecolor=tuple(concept_color) + (0.9,),
            transform=ax.transAxes
        ))
        ax.text(
            0.01, bar_y - 0.01,
            f"centrality  (dist={fd.dist_to_centroid:.2f})",
            transform=ax.transAxes,
            color=self.DIM_TEXT, fontsize=lp["desc_font"] - 2, va="top",
        )

    # ------------------------------------------------------------------
    # Top-moves panel
    # ------------------------------------------------------------------

    def _draw_top_moves(self, ax: plt.Axes, fd: FrameData, lp: dict) -> None:
        ax.set_facecolor(self.PANEL_BG)

        # Build ranked list of legal moves
        ranked = sorted(
            [(a, p) for a, p in enumerate(fd.action_probs) if fd.action_mask[a] == 1],
            key=lambda x: -x[1],
        )
        # Show top 8 + pass (action 49) if it has any probability
        top = ranked[:8]
        if (49, fd.action_probs[49]) not in top and fd.action_probs[49] > 0.001:
            top.append((49, fd.action_probs[49]))

        if not top:
            ax.axis("off")
            return

        labels = [action_to_notation(a) for a, _ in top]
        probs  = [p for _, p in top]
        colors = [
            self._heatmap_cmap(min(p / (probs[0] + 1e-8), 1.0))
            for p in probs
        ]

        y_pos = range(len(top) - 1, -1, -1)   # bottom-to-top
        bars = ax.barh(
            list(y_pos), probs, color=colors, edgecolor="none", height=0.7
        )

        # Labels on each bar
        for y, (label, p) in zip(y_pos, zip(labels, probs)):
            ax.text(
                max(p + 0.005, 0.03), y,
                f"{label}  {p:.1%}",
                va="center", color=self.TEXT_COLOR,
                fontsize=lp["desc_font"] - 1,
            )

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, max(probs) * 1.5)
        ax.set_facecolor(self.PANEL_BG)
        ax.spines[:].set_visible(False)

        ax.text(
            0.01, 0.99, "TOP MOVES",
            transform=ax.transAxes,
            color=self.DIM_TEXT, fontsize=lp["desc_font"] - 1,
            fontweight="bold", va="top",
        )

    # ------------------------------------------------------------------
    # Concept history timeline
    # ------------------------------------------------------------------

    def _draw_timeline(self, ax: plt.Axes, fd: FrameData, lp: dict) -> None:
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")

        history = fd.concept_history
        n = len(history)
        max_visible = 48
        start = max(0, n - max_visible)
        visible = history[start:]

        if not visible:
            return

        cell_w = 1.0 / max(len(visible), 1)
        # Cells fill most of the strip height; leave a sliver at top for the
        # label and a sliver at bottom for the move-number ticks.
        cell_bot = 0.18
        cell_top = 0.90
        cell_h   = cell_top - cell_bot

        # Font size for the concept ID inside each block — scale down for
        # many cells so the number always fits.
        n_visible = len(visible)
        id_font = max(lp["log_font"] - (1 if n_visible > 30 else 0), 4)

        for i, cid in enumerate(visible):
            is_current = (start + i == n - 1)
            color = self.concept_colors.get(cid, (0.5, 0.5, 0.8))
            if is_current:
                color = tuple(min(c * 1.35, 1.0) for c in color)
                edge_color = "white"
                edge_lw = 1.5
            else:
                edge_color = "#2a2a3e"
                edge_lw = 0.5

            x = i * cell_w
            ax.add_patch(mpatches.Rectangle(
                (x, cell_bot), cell_w * 0.92, cell_h,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_lw,
                transform=ax.transAxes,
                clip_on=True,
            ))

            # Concept ID number centred inside the block
            ax.text(
                x + cell_w * 0.46,
                cell_bot + cell_h * 0.5,
                str(cid),
                transform=ax.transAxes,
                color="white",
                fontsize=id_font,
                fontweight="bold",
                ha="center", va="center",
                clip_on=True,
            )

            # Move number below block, every 5 moves and at endpoints
            abs_idx = start + i + 1
            if abs_idx == 1 or abs_idx % 5 == 0 or is_current:
                ax.text(
                    x + cell_w * 0.46, cell_bot - 0.02,
                    str(abs_idx),
                    transform=ax.transAxes,
                    color=self.DIM_TEXT,
                    fontsize=max(id_font - 1, 4),
                    ha="center", va="top",
                    clip_on=True,
                )

        ax.text(
            0.0, 0.98,
            "CONCEPT TIMELINE",
            transform=ax.transAxes,
            color=self.DIM_TEXT,
            fontsize=lp["log_font"] - 1,
            fontweight="bold",
            va="top",
        )

    # ------------------------------------------------------------------
    # Move log
    # ------------------------------------------------------------------

    def _draw_move_log(self, ax: plt.Axes, fd: FrameData, lp: dict) -> None:
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")

        # Show the last 24 half-moves from the log
        log = fd.move_log[-24:]
        parts = []
        for entry in log:
            hm = entry["half_move"]
            player = entry["player"]
            action = entry.get("action")
            if action is None:
                notation = "–"
            else:
                notation = action_to_notation(action)
            if player == "black":
                # Black's move: show as "N.D4"
                move_num = (hm + 1) // 2
                parts.append(f"{move_num}.{notation}")
            else:
                # White's move: append after the preceding black entry
                parts.append(notation)

        log_text = "  ".join(parts)
        if len(fd.move_log) > 24:
            log_text = "… " + log_text

        # Label
        ax.text(
            0.01, 0.5,
            "MOVE LOG",
            transform=ax.transAxes,
            color=self.DIM_TEXT,
            fontsize=lp["log_font"] - 1,
            fontweight="bold",
            va="center",
        )

        # Moves text, offset right of the label
        ax.text(
            0.095, 0.5,
            log_text,
            transform=ax.transAxes,
            color=self.TEXT_COLOR,
            fontsize=lp["log_font"],
            va="center",
            fontfamily="monospace",
        )

    # ------------------------------------------------------------------
    # Game summary frame
    # ------------------------------------------------------------------

    def render_summary_frame(self, frames, result) -> np.ndarray:
        """
        Render a standalone game-end summary frame.

        Shows the final board position, outcome banner, per-concept usage
        histogram, full concept timeline, and game statistics.

        Args:
            frames: Full list of FrameData from the game (used for the
                    complete timeline and the fallback final board).
            result: GameResult produced by record_game().

        Returns:
            HxWx3 uint8 RGB array, same dimensions as regular frames.
        """
        lp = self._lp
        fig = plt.figure(figsize=lp["figsize"], dpi=lp["dpi"])
        fig.patch.set_facecolor(self.PANEL_BG)

        if self.layout == "video":
            self._render_summary_video(fig, frames, result, lp)
        else:
            self._render_summary_paper(fig, frames, result, lp)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        rgb = buf[:, :, :3].copy()
        plt.close(fig)
        return rgb

    def _render_summary_video(self, fig, frames, result, lp) -> None:
        """Landscape summary layout matching the video frame grid."""
        gs = gridspec.GridSpec(
            nrows=4, ncols=2,
            figure=fig,
            height_ratios=[0.9, 8, 1.5, 1.0],
            width_ratios=[1, 1],
            hspace=0.04,
            wspace=0.04,
            left=0.03, right=0.97,
            top=0.97, bottom=0.03,
        )

        # Header
        ax_hdr = fig.add_subplot(gs[0, :])
        self._draw_summary_header(ax_hdr, result, lp)

        # Final board (no heatmap, no chosen ring)
        ax_board = fig.add_subplot(gs[1, 0])
        self._draw_final_board(ax_board, result, lp)

        # Right column: outcome + stats (top), concept usage histogram (bottom)
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[1, 1],
            height_ratios=[2, 3], hspace=0.06,
        )
        ax_outcome = fig.add_subplot(gs_right[0])
        ax_usage   = fig.add_subplot(gs_right[1])
        self._draw_outcome_panel(ax_outcome, result, lp)
        self._draw_concept_usage(ax_usage, result, lp)

        # Full concept timeline (all moves, not windowed)
        ax_time = fig.add_subplot(gs[2, :])
        self._draw_full_timeline(ax_time, result, lp)

        # Summary stats strip
        ax_stats = fig.add_subplot(gs[3, :])
        self._draw_summary_stats_strip(ax_stats, result, lp)

    def _render_summary_paper(self, fig, frames, result, lp) -> None:
        """Portrait summary layout."""
        gs = gridspec.GridSpec(
            nrows=5, ncols=1, figure=fig,
            height_ratios=[0.6, 5, 3, 1.5, 0.9],
            hspace=0.05,
            left=0.05, right=0.95, top=0.97, bottom=0.03,
        )
        ax_hdr   = fig.add_subplot(gs[0])
        ax_board = fig.add_subplot(gs[1])
        gs_mid   = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[2], width_ratios=[1, 1], wspace=0.05
        )
        ax_outcome = fig.add_subplot(gs_mid[0])
        ax_usage   = fig.add_subplot(gs_mid[1])
        ax_time    = fig.add_subplot(gs[3])
        ax_stats   = fig.add_subplot(gs[4])

        self._draw_summary_header(ax_hdr, result, lp)
        self._draw_final_board(ax_board, result, lp)
        self._draw_outcome_panel(ax_outcome, result, lp)
        self._draw_concept_usage(ax_usage, result, lp)
        self._draw_full_timeline(ax_time, result, lp)
        self._draw_summary_stats_strip(ax_stats, result, lp)

    # ── Summary sub-panels ─────────────────────────────────────────────

    def _draw_summary_header(self, ax, result, lp) -> None:
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")
        ax.text(
            0.01, 0.5,
            f"PRISM  ·  {result.algo.upper()}  ·  GAME SUMMARY  ·  "
            f"seed={result.game_seed}  ·  vs {result.opponent_name}",
            transform=ax.transAxes,
            color=self.TEXT_COLOR,
            fontsize=lp["title_font"],
            fontweight="bold",
            va="center",
        )

    def _draw_final_board(self, ax, result, lp) -> None:
        """Draw the final board position using a minimal FrameData stub."""
        # Build a minimal FrameData-like object with zero action probs
        # so _draw_board renders cleanly without heatmap/ring.
        from visualizer.game_recorder import FrameData
        stub = FrameData(
            agent_move_number=result.n_agent_moves,
            total_half_moves=result.n_total_half_moves,
            obs=result.final_obs,
            action_mask=np.zeros(50, dtype=np.int8),
            concept_id=0,
            dist_to_centroid=0.0,
            features=np.zeros(128, dtype=np.float32),
            action_probs=np.zeros(50, dtype=np.float32),
            action_heatmap=np.zeros((7, 7), dtype=np.float32),
            chosen_action=49,
            game_seed=result.game_seed,
            algo=result.algo,
        )
        self._draw_board(ax, stub, lp, show_heatmap=False, show_chosen=False)

        # "FINAL POSITION" label in a small badge at the top of the board axes
        ax.text(
            0.5, 1.01, "FINAL POSITION",
            transform=ax.transAxes,
            color=self.DIM_TEXT, fontsize=lp["board_font"] - 1,
            fontweight="bold", ha="center", va="bottom",
        )

    def _draw_outcome_panel(self, ax, result, lp) -> None:
        """Big winner banner with margin and key game facts."""
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")

        winner = result.winner
        if winner == "black":
            banner_color = "#1a1a1a"
            banner_text_color = "#ffffff"
            banner_label = "⬤  BLACK WINS"
        elif winner == "white":
            banner_color = "#e8e6de"
            banner_text_color = "#1a1a1a"
            banner_label = "○  WHITE WINS"
        elif winner == "draw":
            banner_color = "#4a4a5e"
            banner_text_color = "#ffffff"
            banner_label = "DRAW"
        else:
            banner_color = "#3a3a4e"
            banner_text_color = "#aaaacc"
            banner_label = "INCOMPLETE"

        # Banner background
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.60), 0.96, 0.34,
            boxstyle="round,pad=0.01",
            facecolor=banner_color, edgecolor="none",
            transform=ax.transAxes,
        ))
        ax.text(
            0.5, 0.77, banner_label,
            transform=ax.transAxes,
            color=banner_text_color, fontsize=lp["title_font"] + 2,
            fontweight="bold", ha="center", va="center",
        )

        # Score margin sub-label
        if result.score_margin is not None:
            margin_abs = abs(result.score_margin)
            ax.text(
                0.5, 0.63,
                f"by {margin_abs:.1f} pts",
                transform=ax.transAxes,
                color=banner_text_color, fontsize=lp["desc_font"],
                ha="center", va="center", alpha=0.85,
            )

        # Key facts
        facts = [
            ("Black moves",   str(result.n_agent_moves)),
            ("Total plies",   str(result.n_total_half_moves)),
            ("Concepts used", f"{result.n_unique_concepts} / 64"),
            ("Diversity",     f"{result.diversity_score:.2f}"),
        ]
        y = 0.52
        for label, value in facts:
            ax.text(0.04, y, label,
                    transform=ax.transAxes, color=self.DIM_TEXT,
                    fontsize=lp["desc_font"] - 1, va="top")
            ax.text(0.55, y, value,
                    transform=ax.transAxes, color=self.TEXT_COLOR,
                    fontsize=lp["desc_font"] - 1, va="top", fontweight="bold")
            y -= 0.115

    def _draw_concept_usage(self, ax, result, lp) -> None:
        """Horizontal bar chart: concept frequency over the whole game."""
        ax.set_facecolor(self.PANEL_BG)

        if not result.concept_counts:
            ax.axis("off")
            return

        # Sort by count descending, take top 10
        sorted_items = sorted(result.concept_counts.items(),
                              key=lambda x: -x[1])[:10]
        cids   = [cid for cid, _ in sorted_items]
        counts = [cnt for _, cnt in sorted_items]
        total  = sum(result.concept_counts.values())

        # Labels: "C{id}: {name_short}"
        labels = []
        for cid in cids:
            desc = self.descriptions.get(cid, {})
            name = desc.get("Name", f"Concept {cid}")
            short = name[:26] + "…" if len(name) > 26 else name
            labels.append(f"C{cid}: {short}")

        colors = [self.concept_colors.get(cid, (0.5, 0.5, 0.8)) for cid in cids]

        y_pos = range(len(cids) - 1, -1, -1)
        ax.barh(list(y_pos), counts, color=colors, edgecolor="none", height=0.7)

        for y, (label, cnt) in zip(y_pos, zip(labels, counts)):
            pct = cnt / total * 100
            ax.text(
                cnt + 0.2, y,
                f"{label}  {cnt}×  ({pct:.0f}%)",
                va="center", color=self.TEXT_COLOR,
                fontsize=lp["desc_font"] - 2,
            )

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, max(counts) * 2.2)
        ax.set_facecolor(self.PANEL_BG)
        ax.spines[:].set_visible(False)
        ax.text(
            0.01, 0.99, "CONCEPT USAGE (top 10)",
            transform=ax.transAxes,
            color=self.DIM_TEXT, fontsize=lp["desc_font"] - 1,
            fontweight="bold", va="top",
        )

    def _draw_full_timeline(self, ax, result, lp) -> None:
        """Full concept timeline — same style as in-game but showing all moves."""
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")

        history = result.concept_history
        n = len(history)
        if n == 0:
            return

        cell_w  = 1.0 / n
        cell_bot, cell_top = 0.18, 0.90
        cell_h  = cell_top - cell_bot
        id_font = max(lp["log_font"] - (1 if n > 30 else 0), 4)

        for i, cid in enumerate(history):
            color = self.concept_colors.get(cid, (0.5, 0.5, 0.8))
            x = i * cell_w
            ax.add_patch(mpatches.Rectangle(
                (x, cell_bot), cell_w * 0.92, cell_h,
                facecolor=color, edgecolor="#2a2a3e", linewidth=0.5,
                transform=ax.transAxes, clip_on=True,
            ))
            ax.text(
                x + cell_w * 0.46, cell_bot + cell_h * 0.5, str(cid),
                transform=ax.transAxes,
                color="white", fontsize=id_font, fontweight="bold",
                ha="center", va="center", clip_on=True,
            )
            abs_idx = i + 1
            if abs_idx == 1 or abs_idx % 5 == 0 or abs_idx == n:
                ax.text(
                    x + cell_w * 0.46, cell_bot - 0.02, str(abs_idx),
                    transform=ax.transAxes,
                    color=self.DIM_TEXT, fontsize=max(id_font - 1, 4),
                    ha="center", va="top", clip_on=True,
                )

        ax.text(
            0.0, 0.98, "FULL CONCEPT TIMELINE",
            transform=ax.transAxes,
            color=self.DIM_TEXT, fontsize=lp["log_font"] - 1,
            fontweight="bold", va="top",
        )

    def _draw_summary_stats_strip(self, ax, result, lp) -> None:
        """Bottom strip with game identity details."""
        ax.set_facecolor(self.PANEL_BG)
        ax.axis("off")

        # Dominant concept (most-used)
        dominant_cid = max(result.concept_counts, key=result.concept_counts.get) \
            if result.concept_counts else None
        dominant_name = ""
        if dominant_cid is not None:
            desc = self.descriptions.get(dominant_cid, {})
            dominant_name = desc.get("Name", f"Concept {dominant_cid}")

        parts = [
            f"Algo: {result.algo.upper()}",
            f"Seed: {result.game_seed}",
            f"Opponent: {result.opponent_name}",
            f"Moves: {result.n_agent_moves}B / {result.n_total_half_moves} total",
            f"Transitions: {sum(1 for i in range(1, len(result.concept_history)) if result.concept_history[i] != result.concept_history[i-1])}",
        ]
        if dominant_cid is not None:
            cnt = result.concept_counts[dominant_cid]
            pct = cnt / max(len(result.concept_history), 1) * 100
            parts.append(f"Dominant concept: C{dominant_cid} '{dominant_name}' ({pct:.0f}%)")

        ax.text(
            0.01, 0.5, "  |  ".join(parts),
            transform=ax.transAxes,
            color=self.DIM_TEXT, fontsize=lp["log_font"] - 1,
            va="center", fontfamily="monospace",
        )
