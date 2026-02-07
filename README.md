# ASTRIA: Unsupervised Concept Bottleneck Models for Interpretable RL

**A**utonomous **St**rategy **R**easoning via **I**nterpretable **A**gents

ASTRIA forces reinforcement learning agents to reason through discrete concepts discovered via unsupervised clustering -- requiring **zero human supervision** for concept definition. Unlike prior work (e.g., SCoBots, Delfosse et al. NeurIPS 2024) that requires hand-labeled concepts, ASTRIA discovers concepts automatically from trained encoder features.

## Architecture

```
Board (7x7x3) --> CNN Encoder --> 128D Features --> K-means --> Concept ID (0-63) --> Bottleneck Policy --> Action
     |                |                                |                                    |
  [Stage 1]      [Frozen]                        [Stage 2]                             [Stage 3]
  Train RL      Reuse encoder                  Unsupervised                       Policy sees ONLY
  baseline      features                       discovery                          a single integer
```

**Key insight**: A single integer (6 bits) is sufficient for 92-100% win rates on Go 7x7, despite the observation containing 4,704 bits.

## Results Summary

| Metric | PPO Bottleneck | DQN Bottleneck | VQ End-to-End |
|--------|---------------|---------------|---------------|
| Win Rate (Best) | **100%** | 96% | 92% |
| Causal Intervention | **81.2%** (p < 1e-200) | -- | -- |
| Active Concepts | 64/64 | 64/64 | -- |
| Concept Pairwise KL | 1.72 | -- | -- |

**Cross-domain**: Also validated on CartPole (KL divergence = 0.676 under intervention).

## Project Structure

```
astria/
  train_baseline.py          # Stage 1: Train PPO/DQN encoders
  train_bottleneck.py        # Stage 3: Train concept bottleneck policies
  train_vq.py                # Alternative: VQ-VAE end-to-end training
  train_simple.py            # CartPole cross-domain experiments
  evaluate.py                # Baseline vs bottleneck comparison

  src/
    networks.py              # CNN (Go) and MLP (CartPole) encoders
    concept_manager.py       # K-means concept discovery (Stage 2)
    concept_policy.py        # Bottleneck policy architecture
    strategy_memory.py       # Concept-action-outcome tracking
    vq_layer.py              # Vector Quantization with straight-through estimator
    dynamics_model.py        # P(concept_t+1 | concept_t, action_t) predictor
    opponent_pool.py         # Self-play opponent management
    environments/
      go_env.py              # Go 7x7 wrapper (PettingZoo -> Gymnasium)
      simple_env.py          # CartPole/LunarLander wrappers

  experiments/
    intervention.py          # Causal concept override (key experiment)
    ablation.py              # Concept importance via ablation
    stability.py             # Rotation invariance testing
    dynamics.py              # Concept dynamics accuracy
    simple_intervention.py   # CartPole intervention

  analysis/
    figures.py               # Generate all paper figures
    visualize.py             # Individual plot functions
    concept_viz.py           # t-SNE, board examples, transition graphs

  paper/
    astria.tex               # Full research report (LaTeX)

  results/                   # Experiment outputs (JSON + figures)
  models/                    # Trained models (not in repo -- see below)
```

## Quick Start

### Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch stable-baselines3 sb3-contrib pettingzoo gymnasium scikit-learn matplotlib seaborn
```

### Reproduce All Results
```bash
# Stage 1: Train baselines (~30 min)
python train_baseline.py --env go --algo ppo --steps 200000
python train_baseline.py --env go --algo dqn --steps 200000

# Stages 2-3: Discover concepts & train bottleneck (~60 min)
python train_bottleneck.py --algo both --generations 100 --steps-per-gen 20000

# Run all experiments (~20 min)
python experiments/intervention.py
python experiments/stability.py
python experiments/ablation.py
python experiments/dynamics.py

# Cross-domain (CartPole)
python train_simple.py
python experiments/simple_intervention.py

# VQ-VAE variant
python train_vq.py --generations 100 --steps-per-gen 20000

# Generate figures
python analysis/figures.py
python analysis/concept_viz.py --algo ppo
```

## Key Experiments

### 1. Causal Intervention (81.2% action change)
Override the concept assignment and measure whether the agent's action changes. Result: 81.2% change rate (p < 1e-200), proving concepts causally determine behavior.

### 2. Concept Ablation (1 critical concept found)
Disable individual concepts during gameplay. Concept C17 drops win rate by 5.4% -- removing it makes the agent lose games it would otherwise win.

### 3. Concept Dynamics (52% top-1, 76% top-5)
A simple model predicting next concept from (current concept, action) achieves 52% accuracy (vs 1.6% random baseline), showing temporal structure in concept space.

### 4. Cross-Domain (CartPole KL = 0.676)
Same pipeline applied to CartPole with MLP encoder. Concepts meaningfully shift action distributions (KL divergence = 0.676), confirming domain-agnosticism.

## Comparison with Prior Work

| | SCoBots (Delfosse et al., NeurIPS 2024) | ASTRIA (Ours) |
|---|---|---|
| Concept source | **Supervised** (object labels required) | **Unsupervised** (K-means + VQ-VAE) |
| Human labels needed | Yes | **No** |
| Environments | Atari | Go 7x7 + CartPole |
| Algorithms compared | Single | PPO vs DQN |
| Differentiable variant | No | VQ-VAE end-to-end |
| Dynamics modeling | No | Yes (52% accuracy) |

## Citation

```bibtex
@misc{kumar2026astria,
  title={ASTRIA: Unsupervised Concept Bottleneck Models for Interpretable Reinforcement Learning},
  author={Pravetz, Thomas},
  year={2026},
  note={Available at: github.com/tompravetz/astria}
}
```

## License

MIT
