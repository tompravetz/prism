# PRISM: Policy Reuse via Interpretable Strategy Mapping

**Zero-shot concept transfer between reinforcement learning agents**

[Paper (PDF)](paper/prism.pdf) | March 2026

---

## Overview

PRISM grounds RL agents' decisions in discrete, causally validated concepts and uses those
concepts as a transfer interface between agents trained with different algorithms. Each
agent's encoder features are clustered into *K* concepts via K-means; Hungarian matching
then aligns concept spaces across agents, enabling zero-shot policy transfer with no
gradient updates at transfer time.

The key finding is that transfer quality depends on **source policy strength**, not on
geometric alignment similarity between concept spaces. The framework is scoped to domains
with naturally discrete strategic structure — it fails on Atari Breakout, where continuous
ball dynamics drive action selection, confirming the scope condition empirically.

## Architecture

```
  Observation → Encoder → 128D features → K-means → Concept ID → Bottleneck Policy → Action
       ↑            ↑                          ↑            ↑
   Stage 1:    frozen after        Stage 2:    fitted    Stage 3: trained
   train with  Stage 1             cluster     on        via generational
   PPO/DQN/                        features    collected RL loop
   DAgger                                      features
```

**Transfer:** align source/target concept spaces with Hungarian matching → remap source
policy embedding table into target concept space → zero-shot.

## Key Results

All Go 7×7 results evaluated against GnuGo (10 seeds × 100 games per seed).

### Zero-Shot Transfer (Go 7×7)

| Source | Target | Align Sim | Win Rate | *p* |
|--------|--------|:---------:|:--------:|:---:|
| BC     | DQN    | 0.045     | **76.4%** ± 3.4% | < 0.001 |
| PPO    | DQN    | 0.021     | **69.5%** ± 3.2% | < 0.001 |
| DQN    | PPO    | 0.021     | 49.8% ± 2.6% | 0.82 |
| DAgger | PPO    | 0.045     | 41.5% ± 6.0% | 0.002† |
| DQN    | BC     | 0.045     | 38.7% ± 4.9% | 0.0001† |
| PPO    | BC     | —         | 0.0% (degenerate) | — |

† Significantly *below* 50%, but well above the 3.5% random floor.

Random agent baseline: 3.5% ± 2.0%. No-alignment (identity mapping): 9.2% ± 4.0%.

### Alignment Method Comparison (PPO→DQN)

| Method | Win Rate |
|--------|:--------:|
| Hungarian (PRISM) | **68.8%** ± 5.4% |
| Procrustes + Hungarian | 51.6% ± 8.3% |
| Random permutation | 50.7% ± 39.0% |
| Identity (no alignment) | 9.2% ± 4.0% |

### Causal Validation

- **Intervention:** overriding concept assignments changes the selected action in **69.4%**
  of cases (*p* = 8.6 × 10⁻⁸⁶, 2500 interventions)
- **Ablation:** removing concept C16 drops PPO win rate from 100% to 51.8% — the
  most-*used* concept (C47) is not the most *important* one

### Fine-Tuning Advantage

After zero-shot initialization, REINFORCE fine-tuning on the transferred policy reaches
60% win rate at generation 5 (50K steps). A policy trained from scratch does not reach
60% within 40 generations (400K steps) — an **8× step advantage** (single seed, indicative).

### Scope: Atari Breakout

The identical pipeline on Atari Breakout (ALE/Breakout-v5) produces bottleneck policies
at random-agent performance (0.3–0.5 reward/life vs. 15.1 PPO baseline), and transfer
achieves the same floor. This confirms the framework's scope condition: PRISM requires
domains where strategic state is naturally discrete.

## Installation

```bash
git clone https://github.com/tompravetz/prism.git
cd prism

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

GnuGo is required for Go evaluation. Place the `gnugo-3.8/` binary in the project root
(or set `GNUGO_PATH` environment variable).

## Reproducing Results

### Step 1 — Train baselines (run in parallel)

```bash
python train_baseline.py --algo ppo --seed 42
python train_baseline.py --algo dqn --seed 42
python train_dagger.py --level 5 --bc-samples 285000 --dagger-rounds 2
```

### Step 2 — Train concept bottlenecks (after Step 1)

```bash
python train_bottleneck.py --algo ppo --seed 42
python train_bottleneck.py --algo dqn --seed 42
```

### Step 3 — Run experiments

```bash
# Core transfer result (Table 1)
python experiments/transfer_same_task.py

# Alignment method comparison (Table 2)
python experiments/alignment_comparison.py

# Transfer baselines and fine-tuning (Table 3)
python experiments/transfer_baselines.py
python experiments/finetune_transfer.py

# Causal validation
python experiments/intervention.py
python experiments/ablation.py

# Concept stability
python experiments/stability.py

# Atari boundary condition
python train_atari.py --algo both
python train_atari_bottleneck.py --algo both
python experiments/atari_transfer.py
```

Results are written to `results/` (not tracked by git; regenerate from experiments).

## Project Structure

```
prism/
├── src/
│   ├── environments/
│   │   ├── go_env.py          # Go 7×7 + GnuGo curriculum opponent
│   │   └── simple_env.py      # CartPole / LunarLander wrappers
│   ├── networks.py            # GoCNNEncoder (128D), QNetwork, PolicyNetwork
│   ├── concept_manager.py     # MiniBatchKMeans concept discovery
│   ├── concept_policy.py      # ConceptBottleneckPolicy, ConceptDQNPolicy
│   ├── concept_aligner.py     # Hungarian / Procrustes / greedy alignment
│   └── utils.py               # CurriculumPhase, shared utilities
├── experiments/
│   ├── transfer_same_task.py  # Main agent-to-agent transfer
│   ├── alignment_comparison.py
│   ├── transfer_baselines.py
│   ├── finetune_transfer.py
│   ├── intervention.py        # Causal intervention study
│   ├── ablation.py            # Concept ablation
│   ├── stability.py           # Concept stability (ARI / NMI)
│   ├── eval_strong.py         # Evaluation vs GnuGo
│   ├── eval_capability.py
│   ├── random_baseline.py
│   ├── k_ablation.py          # K-value sensitivity sweep
│   ├── atari_transfer.py      # Atari Breakout transfer
│   ├── atari_k_sweep.py
│   └── archive/               # Archived experiments (not in paper)
├── analysis/
│   ├── figures.py             # Figure generation from results/
│   └── visualize.py
├── paper/
│   ├── prism.tex              # LaTeX source
│   ├── prism.bib              # Bibliography
│   └── prism.pdf              # Compiled paper
├── pidgin/                    # PIDGIN extension (LLM concept naming)
├── visualizer/                # Go game visualization
├── train_baseline.py          # PPO / DQN curriculum training
├── train_bottleneck.py        # Generational concept bottleneck training
├── train_dagger.py            # DAgger behavioral cloning
├── train_atari.py             # Atari Breakout baseline training
├── train_atari_bottleneck.py  # Atari concept bottleneck
└── requirements.txt
```

## Paper

Full paper: [`paper/prism.pdf`](paper/prism.pdf)

The paper covers: three-stage pipeline, Hungarian concept alignment, causal intervention
protocol, ablation study, alignment method comparison, fine-tuning advantage, and Atari
Breakout boundary condition.

## Citation

```bibtex
@techreport{pravetz2026prism,
  title   = {{PRISM}: Policy Reuse via Interpretable Strategy Mapping},
  author  = {Pravetz, Thomas},
  year    = {2026},
  note    = {Preprint}
}
```

## License

MIT License
