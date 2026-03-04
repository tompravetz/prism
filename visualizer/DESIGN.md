# PRISM Visualizer — Design Document

## 0. Executive Summary

The PRISM Visualizer renders a PRISM concept-bottleneck agent playing Go 7×7 into a
multi-panel interpretability display. Each frame of the output shows: the board state
with an action-probability heatmap overlay, the current PIDGIN concept description,
a concept-history timeline, and a ranked list of candidate moves. Output is exported
as an MP4 video, a GIF, and optionally high-DPI PNG frames for paper figures.

The tool is entirely offline (render-then-export, not interactive), built on
**matplotlib + Pillow**, and requires no new mandatory dependencies beyond what PRISM
already uses. FFmpeg integration for MP4 is optional.

---

## 1. Library Choices and Rationale

| Concern | Choice | Reason |
|---|---|---|
| Rendering | **matplotlib** | Already installed; publication-quality output; FuncAnimation; tight control over DPI and figure layout |
| GIF export | **matplotlib PillowWriter** | Pillow is already installed; zero new deps |
| MP4 export | **imageio + imageio-ffmpeg** (optional) | Small install; no system ffmpeg required; degrades gracefully if absent |
| Board stones | **matplotlib.patches.Circle** | Circular stones over a grid look authentic; imshow alone would give square cells |
| Heatmap overlay | **matplotlib.axes.imshow** (alpha ≈ 0.55) | Composited under the stone layer; blends naturally with board color |
| Concept palette | **matplotlib.cm.hsv** sampled at 64 equidistant hues | 64 visually distinct colours, consistent across all frames |
| Text layout | **matplotlib.axes.text** with bbox | Wrappable paragraph text; compatible with tight_layout |

**Why not pygame?**
Pygame is optimised for interactive real-time loops and screen display. Our output is
always written to a file (video/GIF/PNG). Matplotlib's figure → bytes pipeline is
simpler, more portable, and directly exports at arbitrary DPI without an off-screen
rendering workaround.

**Why not OpenCV?**
OpenCV is not installed and its text/font handling is noticeably inferior to
matplotlib for multi-line descriptions. Its main advantage (fast video encoding) is
covered by imageio-ffmpeg.

---

## 2. Architecture Overview

```
GoEnv (7×7)  +  ConceptBottleneckAgent
        │
        ▼
  GameRecorder.record_game()
  Returns: List[FrameData]   ←─── one entry per agent (black) move
        │
        │  +  PIDGIN descriptions (pidgin/results/concept_descriptions.json)
        │  +  evidence cache    (pidgin/results/evidence_cache.pkl)  [optional]
        ▼
  FrameRenderer.render_frame(frame_data, descriptions)
  Returns: np.ndarray  (H × W × 3, uint8)   ─── matplotlib figure → RGB array
        │
        ├──▶  VideoExporter.to_gif()    →  game.gif
        ├──▶  VideoExporter.to_mp4()    →  game.mp4   [optional, needs imageio]
        └──▶  FrameExporter.save_png()  →  frame_NNN.png  (selected frames)
```

No GUI window is ever opened. All rendering happens in Matplotlib's non-interactive
`Agg` backend.

---

## 3. Module Structure

```
prism/visualizer/
├── __init__.py
├── DESIGN.md               ← this file
├── game_recorder.py        ← play a game, collect FrameData per agent move
├── frame_renderer.py       ← render one FrameData into an RGB numpy array
├── exporter.py             ← assemble frames into GIF / MP4 / PNG
└── run_visualizer.py       ← CLI entry point
```

---

## 4. Data Flow: FrameData Schema

`FrameData` is the single unit of information passed from the recorder to the
renderer. It contains everything needed to draw one frame — no further lookups into
the environment are needed after recording.

```python
@dataclass
class FrameData:
    # ── Identity ──────────────────────────────────────────────────────
    agent_move_number: int      # 1-indexed count of the agent's moves this game
    total_half_moves: int       # total half-moves (black + white) so far
    player: str                 # always "black" (agent is black)

    # ── Board state (BEFORE the agent's move is applied) ──────────────
    obs: np.ndarray             # (7, 7, 3) float32  — channels: [black, white, empty]
    action_mask: np.ndarray     # (50,)  int8  — 1 = legal, 0 = illegal

    # ── Concept assignment ────────────────────────────────────────────
    concept_id: int             # 0–63
    dist_to_centroid: float     # L2 distance from features to cluster centre
    features: np.ndarray        # (128,) float32 — encoder output [kept for analysis]

    # ── Action distribution (masked softmax) ─────────────────────────
    action_probs: np.ndarray    # (50,) float32 — softmax probabilities
    action_heatmap: np.ndarray  # (7, 7) float32 — probs re-shaped onto board
    chosen_action: int          # 0–49

    # ── Move log ─────────────────────────────────────────────────────
    # All moves in this game up to and including the current agent move.
    # Each entry: {"move": int, "player": "black"|"white", "action": int}
    move_log: List[Dict]

    # ── Concept history ───────────────────────────────────────────────
    # Concept ID for each of the agent's moves so far (including this one).
    concept_history: List[int]

    # ── Game metadata ─────────────────────────────────────────────────
    game_seed: int
    algo: str                   # "ppo" or "dqn"
```

**Recording discipline:** The agent plays deterministically (`deterministic=True`).
After each black move + white response, we capture the board state the agent *would
see next*. This means `obs` in frame N is the board before black move N, and
`chosen_action` in frame N is what black played on move N.

---

## 5. Game Recording

`GameRecorder` wraps the standard PRISM inference loop and returns
`List[FrameData]`.

```
load_artifacts()
    │
    ├── encoder ← models/baseline/ppo_go_encoder.pt
    ├── concept_manager ← models/bottleneck/concepts_ppo_k64.pkl
    └── policy ← models/bottleneck/ppo_bottleneck_final.pt

record_game(seed, max_moves) → List[FrameData]
    │
    ├── env.reset(seed=seed)
    ├── Loop until done:
    │     1. encoder(obs) → features (128,)
    │     2. concept_manager.assign_concept(features) → concept_id
    │     3. policy.forward(concept_id, mask) → logits, _ → softmax → probs
    │     4. policy.get_action(concept_id, mask, deterministic=True) → action
    │     5. Append FrameData (obs before move, probs, action, etc.)
    │     6. env.step(action) → next obs, reward, done
    └── Return list of FrameData (one per black move)
```

**Action probabilities are extracted before masking is applied to the display layer
(so the heatmap shows the raw policy preference), but a separate `legal_mask` field
marks which cells are actually playable.** This makes the difference between
"structurally preferred but illegal" and "both preferred and legal" visible.

---

## 6. Frame Layout Specification

The rendered frame is a fixed-size figure. Two layout profiles are defined:

### 6a. Video Layout  (1280 × 720 px, 96 dpi → 13.33″ × 7.5″)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PRISM  │  algo: PPO  │  Move 12  │  Concept 23: Corner Invasion        │
├─────────────────────────┬────────────────────────────────────────────────┤
│                         │  ┌─ CONCEPT DESCRIPTION ──────────────────┐   │
│                         │  │ Name: Corner Invasion                  │   │
│     GO BOARD            │  │                                        │   │
│     (7 × 7)             │  │ Description: Activates when the        │   │
│     +                   │  │ corner regions are contested…          │   │
│     HEATMAP OVERLAY     │  │                                        │   │
│     +                   │  │ Key Actions: Prefers A1 (34%), G7…    │   │
│     STONE CIRCLES       │  │                                        │   │
│                         │  │ Strategic Importance: high — top 25%  │   │
│                         │  └────────────────────────────────────────┘   │
│                         │  ┌─ TOP MOVES ─────────────────────────────┐  │
│                         │  │  1. A1    34.1%  ██████████             │  │
│                         │  │  2. G7    18.3%  █████                  │  │
│                         │  │  3. A7    12.2%  ████                   │  │
│                         │  │  4. G1     8.9%  ██                     │  │
│                         │  │  5. D4     5.1%  █                      │  │
│                         │  │  …  (pass: 0.0%)                        │  │
│                         │  └────────────────────────────────────────┘   │
├─────────────────────────┴────────────────────────────────────────────────┤
│ CONCEPT TIMELINE:  [▪][▪][▪][▪][▪][▪][▪][▪][▪][▪][▪][■]               │
│                    1  2  3  4  5  6  7  8  9  10 11 12                   │
├──────────────────────────────────────────────────────────────────────────┤
│ MOVE LOG:  1.D4  2.–  3.E5  4.–  5.C6  6.–  7.B3  8.–  9.A2  10.–  … │
└──────────────────────────────────────────────────────────────────────────┘
```

Row heights (approximate fractions):
- Header row: 4%
- Board + panels: 72%
- Timeline: 12%
- Move log: 12%

Column widths:
- Board panel: 50%
- Description + top-moves panel: 50%

### 6b. Paper Frame Layout  (900 × 1200 px, 150 dpi → 6″ × 8″)

For individual frame export the layout is portrait-oriented so it fits as a figure
panel:

```
┌─────────────────────┐
│   GO BOARD          │  (square, ~50% of height)
│   + heatmap         │
│   + stones          │
├─────────────────────┤
│   CONCEPT           │  (~25% of height)
│   description text  │
├─────────────────────┤
│   TOP MOVES         │  (~13% of height)
│   bar chart         │
├─────────────────────┤
│   TIMELINE          │  (6%)
│   MOVE LOG          │  (6%)
└─────────────────────┘
```

The paper layout is selected automatically when `--paper` is passed to the CLI or
when `FrameExporter.save_png(..., paper=True)` is called.

---

## 7. Board Rendering

### 7a. Coordinate System

Go column labels: A–G (left → right). Row labels: 7–1 (top → bottom in display
convention; row 7 is the top of the board). Action `a` maps to:
- `row = a // 7` (0-indexed from top), `col = a % 7` (0-indexed from left)
- Display coordinates (matplotlib y-axis inverted): plot at `(col, 6 - row)` with
  y=0 at the bottom

```
Display:     A  B  C  D  E  F  G
         7   .  .  .  .  .  .  .    ← obs row 0
         6   .  .  .  .  .  .  .    ← obs row 1
         ...
         1   .  .  .  .  .  .  .    ← obs row 6
```

### 7b. Rendering Layers (bottom-to-top)

1. **Board background**: `ax.set_facecolor('#DCB960')` — classic Go board tan.
   Fixed square axes with `ax.set_aspect('equal')`.

2. **Grid lines**: 7 horizontal + 7 vertical lines at integer coordinates, thin
   black lines (`lw=0.8`). Drawn across the full 0–6 range.

3. **Star points (hoshi)**: On a 7×7 board the standard star points are at
   (2,2), (2,4), (4,2), (4,4), and the centre (3,3). Rendered as small filled
   circles (`r=0.08`) in dark brown.

4. **Heatmap overlay**: `ax.imshow` of the `(7,7)` `action_heatmap` array.
   - `extent=[-0.5, 6.5, -0.5, 6.5]`, `origin='lower'`, `aspect='equal'`
   - Colormap: `'YlOrRd'` (yellow→orange→red) — warm tones contrast with the
     tan board and dark stones without clashing
   - `alpha=0.55` — board grid and stones visible through the overlay
   - `vmin=0`, `vmax=max(action_heatmap)` — normalise to the actual max probability
     so a high-entropy distribution still uses full colour range
   - Illegal positions (`action_mask == 0`) set to `NaN` in the displayed array;
     matplotlib renders NaN as transparent, making them visually absent from the
     heatmap even if the raw softmax assigned them a tiny value

5. **Illegal position markers**: Semi-transparent grey `×` glyphs at each cell
   where `action_mask == 0` and the cell is occupied (a stone is already there).
   Cells that are illegal for other reasons (ko, suicide) but empty can optionally
   be annotated with a faint `×`.

6. **Stone circles**: `matplotlib.patches.Circle` for each occupied cell.
   - Black stones: `facecolor='#1a1a1a'`, `edgecolor='#444444'`, `radius=0.42`
   - White stones: `facecolor='#f5f5f5'`, `edgecolor='#888888'`, `radius=0.42`
   - Drawn after the heatmap so stones are always fully visible on top

7. **Chosen move indicator**: For the cell the agent chose:
   - If placing a stone: add a concentric ring (`Circle`, `fill=False`,
     `edgecolor='lime'`, `lw=2.5`, `radius=0.38`) around the stone after it is
     drawn
   - If passing: a "PASS" text banner across the board

8. **Axis labels**: Column letters (A–G) along the top and bottom; row numbers
   (7–1) along the left and right. `ax.tick_params(...)` replaces the default
   numeric ticks with the Go notation.

### 7c. Stone rendering order

Stones are drawn after the heatmap so they always appear fully opaque. The chosen
action ring is drawn last (top layer).

---

## 8. Concept Panel Rendering

The description panel occupies the upper half of the right column.

### 8a. Description parsing

PIDGIN descriptions follow the 4-field format:

```
Name: Corner Invasion
Description: Activates when the corner regions are contested by an opponent approach...
Key Actions: Prefers A1 (34%), G7 (18%), A7 (12%)...
Strategic Importance: high — top 25% most impactful...
```

`parse_description(text: str) → dict[str, str]` splits on the label prefixes using
a regex, returning `{"Name": ..., "Description": ..., "Key Actions": ...,
"Strategic Importance": ...}`. If a field is absent, the value is an empty string.
Robust to blank lines and minor formatting variations.

### 8b. Fallback when description is unavailable

PIDGIN is still optimizing (currently ~10/64 done). For un-described concepts:
1. Check `descriptions_generic.json` (always present) — use the generic template
   with a "(not yet described)" annotation.
2. Display a subtle badge: `"description pending"` in grey italic.

### 8c. Visual design

- **Concept ID badge**: Coloured rectangle in the top-left corner of the panel,
  using the concept's fixed colour from the 64-colour HSV palette. White text:
  `"Concept 23"`. This badge directly mirrors the colour used in the timeline,
  establishing a visual link.
- **Name**: Large bold font (~14pt). Provides the concept's human-readable handle
  at a glance.
- **Description, Key Actions, Strategic Importance**: Regular weight, wrapped at the
  panel width. Each field preceded by a faint grey label in small caps.
- **Distance-to-centroid bar**: A narrow horizontal bar below the description,
  labelled "centrality". Full-width = maximum observed distance across all concepts;
  current position marked. Indicates whether this state is a "canonical" instance of
  the concept or a boundary case.

All text rendering uses `ax.text(...)` with `transform=ax.transAxes` for
figure-relative positioning. `ax.axis('off')` hides the axes spine.

---

## 9. Action Weights Panel

The lower half of the right column shows the full action distribution as a ranked
list.

### 9a. Content

- Top 8 moves by probability, sorted descending.
- Each row: move notation (e.g., `D4`) + probability percentage + horizontal bar.
- Pass (action 49) always shown as a final row, even if probability ≈ 0%.
- Illegal moves are excluded entirely (they have probability 0 after masking).

### 9b. Rendering

A minimal horizontal bar chart via `ax.barh`. Because only ~8 rows are shown, this
is a regular `ax.barh` call rather than a custom patch loop.

Move notation is computed as:
```python
cols = "ABCDEFG"
row, col = divmod(action, 7)
label = f"{cols[col]}{7 - row}"  # e.g., action 9 → "C6"
```

Bar colours match the board heatmap colormap so there is a visual correspondence
between the bar length and the cell intensity on the board.

---

## 10. Concept History Timeline

The timeline is a horizontal strip that occupies the full width of the figure below
the main panels.

### 10a. Design

- One square cell per agent move, filled with the concept's HSV palette colour.
- Current move: brighter fill + white border to highlight the "now" position.
- Move number labels at every 5th move (to avoid clutter).
- If the game has more than 40 agent moves, the timeline shows only the last 40
  (the window slides right as the game progresses).
- The timeline is rendered as a sequence of `Rectangle` patches in a dedicated axes.

### 10b. Colour palette

```python
CONCEPT_COLORS = [
    matplotlib.colors.hsv_to_rgb((k / 64.0, 0.75, 0.85))
    for k in range(64)
]
```

This gives 64 perceptually spaced hues at fixed saturation and value, so no two
neighbouring concept IDs share the same colour. The same palette is used for the
concept ID badge in the description panel.

---

## 11. Move Log

The move log is a single line of text showing recent moves in algebraic notation.
It occupies the bottom strip of the figure.

Format: `1.D4  2.–  3.E5  4.–  5.C6  6.–  …`

- Odd entries (1, 3, 5, …) = agent (black) moves
- Even entries (2, 4, 6, …) = opponent (white) moves; shown as `–` since we do not
  analyse the opponent
- Pass is shown as `Ps`
- The log shows the last 20 half-moves; earlier moves are replaced with `…`
- The current half-move is bold

Rendered as a single `ax.text(...)` call with wrapped monospace formatting.

---

## 12. Output Formats

### 12a. GIF (primary, no extra deps)

```python
from matplotlib.animation import FuncAnimation, PillowWriter

anim = FuncAnimation(fig, update_fn, frames=len(frame_list), ...)
writer = PillowWriter(fps=fps)
anim.save("game.gif", writer=writer)
```

`fps` defaults to 1.5 (slow enough to read concept descriptions). Configurable via
`--fps`.

Pillow's GIF encoder uses an 8-bit palette (256 colours max). For a complex
visualization with gradients and many colours, this will quantise. To mitigate:
- The board background, stones, and UI chrome are rendered in a narrow palette
- The heatmap uses a sequential colormap that quantises more gracefully than
  diverging colormaps

For a sharp GIF, each frame is first rendered to a `BytesIO` buffer as PNG (full
colour), then the sequence is assembled with `imageio` if available, falling back to
`PillowWriter` otherwise.

### 12b. MP4 (optional, requires imageio-ffmpeg)

```python
import imageio
writer = imageio.get_writer("game.mp4", fps=fps, codec="h264", quality=8)
for frame_rgb in rendered_frames:
    writer.append_data(frame_rgb)
writer.close()
```

If `imageio` is not installed, `run_visualizer.py` prints an install hint and skips
MP4 export without crashing.

MP4 at ~1.5 fps with h264 encoding produces much smaller files than GIF for the same
visual quality (typically 5–20×). Recommended for the project page.

### 12c. Individual frame PNG (paper figures)

```python
fig.savefig(f"frames/frame_{n:03d}.png", dpi=300, bbox_inches='tight')
```

When `--export-frames` is passed, every frame is saved as a separate PNG at the
requested DPI (default 300). A sidecar file `frames/frame_index.json` is written:

```json
[
  {
    "frame": 12,
    "agent_move": 12,
    "concept_id": 23,
    "concept_name": "Corner Invasion",
    "action": 0,
    "action_notation": "A7",
    "action_prob": 0.341
  },
  ...
]
```

This lets the authors quickly find frames showing specific concepts or dramatic
concept transitions for inclusion in the paper.

---

## 13. CLI Interface

```
python -m visualizer.run_visualizer [OPTIONS]

Options:
  --algo {ppo,dqn}        Which trained agent to use  [default: ppo]
  --seed INT              RNG seed for the game  [default: 0]
  --max-moves INT         Maximum agent moves to record  [default: 60]
  --fps FLOAT             Frames per second in video/GIF output  [default: 1.5]
  --out-dir PATH          Output directory  [default: visualizer/outputs/]
  --gif                   Export GIF  [default: on]
  --no-gif                Skip GIF export
  --mp4                   Export MP4 (requires imageio-ffmpeg)
  --export-frames         Save every frame as PNG (for paper selection)
  --frame-dpi INT         DPI for PNG frame export  [default: 300]
  --paper                 Use portrait layout for frame export
  --descriptions PATH     Path to PIDGIN descriptions JSON
                          [default: pidgin/results/concept_descriptions.json]
  --no-descriptions       Skip PIDGIN panel, show concept ID only

Examples:
  # Standard run: GIF output
  python -m visualizer.run_visualizer

  # Export MP4 + all frames for paper selection
  python -m visualizer.run_visualizer --mp4 --export-frames --seed 7

  # Quick test: first 10 agent moves, show result
  python -m visualizer.run_visualizer --max-moves 10 --fps 2
```

---

## 14. Dependencies

### Required (already installed)
| Package | Version in venv | Purpose |
|---|---|---|
| matplotlib | 3.10.8 | All rendering, animation, GIF (via PillowWriter) |
| Pillow | 12.1.0 | PillowWriter for GIF; PNG frame saving |
| numpy | 2.4.2 | Array operations throughout |
| torch | 2.10.0 | Model inference |
| pygame | 2.6.1 | GoEnv dependency (PettingZoo) |

### Optional (install for MP4)
```
venv/Scripts/python.exe -m pip install imageio imageio-ffmpeg
```

### Nothing else is needed.

---

## 15. Description Loading Strategy

```python
def load_descriptions(path: str) -> Dict[int, Dict[str, str]]:
    """
    Load and parse PIDGIN descriptions.

    Returns a dict mapping concept_id (int) → parsed field dict:
        {"Name": ..., "Description": ..., "Key Actions": ...,
         "Strategic Importance": ..., "_raw": ...}

    Falls back to generic descriptions for any concept_id not present in path.
    """
    # 1. Load textgrad-optimized descriptions (may be partial)
    optimized = {}
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            optimized[int(k)] = parse_description(v)

    # 2. Load generic fallback
    generic_path = os.path.join(os.path.dirname(path), "descriptions_generic.json")
    generic = {}
    if os.path.exists(generic_path):
        with open(generic_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            generic[int(k)] = parse_description(v)

    # 3. Merge: prefer optimized, fall back to generic
    result = {}
    for k in range(64):
        if k in optimized:
            result[k] = optimized[k]
            result[k]["_source"] = "textgrad"
        elif k in generic:
            result[k] = generic[k]
            result[k]["_source"] = "generic"
        else:
            result[k] = {"Name": f"Concept {k}", "Description": "",
                         "Key Actions": "", "Strategic Importance": "",
                         "_source": "none"}

    return result
```

This means the visualizer works today (with 10/64 descriptions) and automatically
improves as PIDGIN finishes optimizing.

---

## 16. Resolved Design Questions

### Q3 (Concept colour assignment) — RESOLVED

**Decision:** Assign palette colours by **frequency rank** rather than by concept ID.

The 64-colour palette is generated using a **bit-reversal permutation** of the hue
wheel. Reversing the 6-bit binary representation of each rank index gives a sequence
where consecutive ranks land on maximally separated hues:

```
rank 0 → hue 0/64   (0°)
rank 1 → hue 32/64  (180°)
rank 2 → hue 16/64  (90°)
rank 3 → hue 48/64  (270°)
rank 4 → hue 8/64   (45°)
rank 5 → hue 40/64  (225°)
…
```

The most frequent concept (rank 0) gets hue 0°; the second most frequent (rank 1)
gets hue 180° — the maximum possible distance. This ensures that concepts which
co-appear in the same game are visually distinct even when they have nearby IDs.

Frequency data is loaded from `pidgin/results/evidence_cache.pkl` if it exists.
If the cache is absent, the tool falls back to ID-order assignment without error.

```python
def _build_concept_colors(frequencies: Dict[int, float]) -> Dict[int, Tuple]:
    """concept_id → RGB color, ordered by descending frequency."""
    bit_reversed_hues = [
        int(format(i, '06b')[::-1], 2) / 64.0
        for i in range(64)
    ]
    palette = [colorsys.hsv_to_rgb(h, 0.75, 0.85) for h in bit_reversed_hues]
    sorted_concepts = sorted(frequencies, key=lambda k: -frequencies[k])
    return {cid: palette[rank] for rank, cid in enumerate(sorted_concepts)}
```

### Q4 (Game selection strategy) — RESOLVED

**Decision:** Run `N` candidate games (default 20) and select the one that maximises:

```
diversity_score = n_unique_concepts × (1 + 0.5 × dynamism)
dynamism        = n_concept_transitions / max(n_agent_moves − 1, 1)
```

`n_unique_concepts` is the primary driver (rewards visiting many distinct strategic
situations). `dynamism` is a secondary multiplier capped at 0.5× — it rewards games
where concepts change frequently rather than staying stuck. A game with 20 unique
concepts and 80% transition rate scores 20 × 1.4 = 28; a game with 10 unique
concepts and 90% rate scores only 10 × 1.45 = 14.5.

`record_best_game(n_candidates=20, max_moves=60)` returns the winning frames plus a
stats dict for all N candidates (`seed`, `score`, `n_unique`, `n_transitions`),
which is written to `outputs/game_selection_log.json` for reproducibility.

Default `n_candidates=20` explores enough variation without taking excessive time
(each game is ~5 s on CPU at deterministic inference).

---

## 17. Key Design Decisions and Tradeoffs

### One frame per agent move, not per half-move

**Decision:** Each rendered frame corresponds to one black (agent) move. White
(opponent) moves are silently applied to the board but do not produce separate
frames.

**Rationale:** The interpretability story centres on the agent's decision process:
what concept activated, what the policy considered, what it chose. White's moves
(random opponent) carry no interpretability signal. Including them would double the
frame count and interrupt the interpretability narrative with uninformative frames.

**Implication:** The move log shows both colours to preserve game coherence, but the
concept timeline and heatmap track only black's moves.

### Offline (render-to-file) not interactive

**Decision:** No real-time interactive window; all output is written to files.

**Rationale:** Primary use is paper figure extraction and video embedding. An
interactive display would add pygame/tkinter complexity without benefiting the main
use case. The CLI can be run overnight while PIDGIN continues optimizing.

### Heatmap vmax = observed max, not 1.0

**Decision:** Normalise the heatmap colourmap to `vmax=max(action_heatmap)` rather
than `vmax=1.0`.

**Rationale:** Action distributions are often concentrated (KL from uniform is high).
Normalising to 1.0 would make a 34% top move appear pale. Normalising to the actual
max makes the dominant move clearly saturated while secondary moves show relative
contrast. The absolute percentage is always visible in the top-moves panel.

### Separate description-parsing module

**Decision:** `parse_description()` is a standalone function, not embedded in the
renderer.

**Rationale:** The PIDGIN output format may need adjustment (e.g., if the LLM
occasionally produces slightly different field labels). Centralising the parser means
format variations are handled in one place without touching the rendering logic.

### Paper layout vs video layout as named presets

**Decision:** Two named layout presets rather than fully parametric layout.

**Rationale:** The two use cases (paper figure, video) have different aspect ratios
and font size requirements. Parametric layout would add complexity for marginal
benefit — the two presets cover 100% of the expected use cases.

---

## 18. Open Questions (for implementation phase)

1. **Heatmap transparency on stones**: The heatmap is rendered behind stone circles
   (current plan). Stones are always fully opaque on top. The alternative (tinting
   stone colour by heatmap intensity) is visually confusing — a black stone tinted
   towards orange looks like a captured stone.

2. **Opponent move display**: Show only the board state the agent observes at the
   start of each of its turns (after white has already moved). White's stone
   placements are not individually animated — they appear as a fait accompli on
   the next agent-turn frame. This keeps every frame focused on the agent's
   decision context.
