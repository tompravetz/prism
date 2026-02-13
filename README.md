# PRISM: Policy Reuse via Interpretable Strategy Mapping

**Concept-mediated strategy transfer for reinforcement learning agents**

---

## Overview

PRISM discovers discrete concepts from trained RL encoders using K-means clustering, creating a universal interface layer between observation encoding and policy execution. By aligning concept spaces across different agents via Hungarian matching, PRISM enables zero-shot policy transfer between agents trained with different algorithms (PPO, DQN, DAgger), across different domains (Go, CartPole, LunarLander, Acrobot), and across task scales (Go 5x5 to 7x7 curriculum transfer).

## Architecture

```
                        PRISM Concept Bottleneck Pipeline

  +-------------+     +---------+     +----------+     +------------+     +------------------+     +--------+
  | Observation | --> | Encoder | --> | 128D     | --> | K-means    | --> | Bottleneck       | --> | Action |
  |             |     |         |     | Features |     | Concept ID |     | Policy           |     |        |
  +-------------+     +---------+     +----------+     +------------+     +------------------+     +--------+

  Stage 1: Train encoder + policy end-to-end with standard RL (PPO / DQN / DAgger)
  Stage 2: Freeze encoder, cluster 128D features into 64 discrete concepts via K-means
  Stage 3: Train bottleneck policy mapping concept IDs directly to actions

  Transfer: Align source/target concept spaces with Hungarian matching, then reuse policy
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tompravetz/prism.git
cd prism

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Train a baseline RL agent
python train_baseline.py --algo ppo

# 2. Discover concepts and train the bottleneck policy
python train_bottleneck.py --algo ppo --generations 100

# 3. Run same-task agent-to-agent transfer
python experiments/transfer_same_task.py

# 4. Run cross-domain transfer
python experiments/transfer_cross_domain.py
```

## Project Structure

```
prism/
├── src/
│   ├── environments/          # Environment wrappers (Go, CartPole, LunarLander, Acrobot)
│   ├── networks.py            # CNN/MLP encoders and policy/Q-value heads
│   ├── concept_manager.py     # K-means concept discovery
│   ├── concept_policy.py      # Bottleneck policy (concept ID -> action)
│   ├── concept_aligner.py     # Hungarian matching for concept space alignment
│   ├── strategy_library.py    # Cross-domain strategy registry and retrieval
│   └── utils.py               # Shared utility functions
├── experiments/
│   ├── transfer_same_task.py      # Agent-to-agent transfer within Go 7x7
│   ├── transfer_cross_domain.py   # Transfer across different domains
│   ├── transfer_baselines.py      # Baseline comparisons for transfer
│   ├── strategy_composition.py    # Composing specialist strategies
│   ├── curriculum.py              # 5x5 -> 7x7 curriculum transfer
│   ├── strategy_library_demo.py   # Strategy library demonstration
│   ├── transitive_transfer.py    # Transitive alignment composition
│   ├── alignment_comparison.py  # Alignment method comparison (5 methods)
│   ├── transfer_prediction.py   # Transfer quality prediction analysis
│   └── cross_domain.py           # LunarLander PRISM pipeline
├── analysis/
│   ├── create_hero_figure.py          # Main paper figure generation
│   ├── cross_domain_analysis.py       # Cross-domain transfer analysis
│   ├── concept_interpretation.py      # Concept visualization and interpretation
│   ├── saliency_comparison.py         # Saliency map comparisons
│   ├── failure_analysis.py            # Failure mode categorization
│   └── concept_quality_metrics.py     # Quantitative concept quality metrics
├── paper/
│   └── prism.tex              # LaTeX source
├── train_baseline.py          # Baseline RL training (SB3 MaskablePPO / DQN)
├── train_bottleneck.py        # Generational concept bottleneck training
├── train_cloned.py            # Behavioral cloning training
├── run_dagger_pipeline.py     # DAgger training loop + evaluation
├── eval_gnugo.py              # GnuGo evaluation harness
├── requirements.txt
└── README.md
```

## Key Results

All results reported as mean over 5 independent seeds.

### Same-Task Transfer (Go 7x7, Win Rate vs Random Opponent)

| Source | Target | Alignment | Win Rate (5 seeds) |
|:-------|:-------|:---------:|:------------------:|
| PPO    | DQN    | 0.262     | 92.0% ± 2.6       |
| PPO    | DAgger | 0.310     | 90.6% ± 2.2       |
| DQN    | PPO    | 0.262     | 64.4% ± 6.1       |
| DQN    | DAgger | 0.240     | 64.4% ± 4.0       |
| DAgger | PPO    | 0.310     | 55.2% ± 8.2       |
| DAgger | DQN    | 0.240     | 53.6% ± 3.0       |

RL-trained agents (PPO, DQN) transfer well to each other; DAgger (behavioral cloning) transfers less effectively, consistent with richer concept utilization by RL encoders.

### Cross-Domain Transfer (Fine-Tuned Improvement over From-Scratch)

| Source      | Target      | Alignment | Improvement |
|:------------|:------------|:---------:|:-----------:|
| CartPole    | LunarLander | 0.428     | +21.1%      |
| LunarLander | CartPole    | 0.428     | +8.6%       |
| Acrobot     | LunarLander | 0.413     | +4.2%       |
| Acrobot     | CartPole    | 0.407     | +4.1%       |
| CartPole    | Acrobot     | 0.407     | -1.1%       |
| LunarLander | Acrobot     | 0.413     | -8.0%       |

4 of 6 cross-domain pairs show positive transfer (4--21% improvement).

### Curriculum Transfer (Go 5x5 → 7x7, 5 seeds)

| Method                  | Final Win Rate | Generations to 95% |
|:------------------------|:--------------:|:-------------------:|
| Transferred + fine-tune | 96.4% ± 3.7   | 26.0 ± 4.9         |
| From scratch            | 94.0% ± 5.1   | 35.0 ± 6.3         |
| **Speedup**             |                | **1.35x** (p=0.027) |

## Paper

The full paper is available at [`paper/prism.tex`](paper/prism.tex) (23 pages, 6 figures). It includes an information-theoretic transfer bound, transitive alignment composition analysis, and five alignment method comparisons.

## Citation

```bibtex
@article{pravetz2026prism,
  title={{PRISM}: Policy Reuse via Interpretable Strategy Mapping},
  author={Pravetz, Thomas},
  year={2026}
}
```

## License

This project is released under the MIT License.
