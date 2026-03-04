# PIDGIN: Policy Interpretability via Domain-Grounded Integrated Naming

## Design Document v2.0

**Author:** Thomas Pravetz
**Date:** February 2026
**Status:** Active design

---

## 0. Executive Summary

PIDGIN is a framework for developing human-readable strategy representations in reinforcement learning agents. It extends PRISM's concept bottleneck architecture by co-evolving strategy representations with natural language descriptions throughout training — not applying language post-hoc to finished concepts.

The central claim: **concepts should be intrinsically describable, not retrospectively labeled.** If a description is applied to a finished concept, the concept was never shaped to be expressible — language is being forced onto alien structure. PIDGIN makes human-expressibility a training objective, so that the resulting concepts are constitutively described by language rather than annotated after the fact.

PIDGIN is **task-agnostic**. It makes no assumptions about the domain. It is validated in domains where the LLM has no prior knowledge, demonstrating that vocabulary emerges from observed agent behavior rather than being imported from existing human expertise.

**Research lineage:**
- PRISM: machine → machine (concept transfer between RL agents via centroid alignment)
- PIDGIN: machine → human (strategy concepts expressible in natural language, emergent not imposed)
- (Future): human → machine (natural language specifies strategies for agents to execute)

---

## 1. Motivation and Problem Statement

### 1.1 The Post-Hoc Interpretability Problem

The dominant paradigm in interpretable RL is post-hoc: train an agent, then apply an interpretability method to explain what the trained agent has learned. This includes saliency maps, concept probes, prototype networks, TCAV, and the original PIDGIN v1 design (TextGrad description optimization after PRISM training completes).

Post-hoc methods share a structural flaw: **the agent was not trained to be interpretable.** Its representations optimize for performance, not human-expressibility. The interpretability layer is forced to map alien representational structure onto human vocabulary. Failure modes:

- **Vocabulary mismatch**: Human terms may not carve the agent's concept space at its joints. A concept the agent treats as atomic may require three human words to approximate; a human concept may split across two agent concepts.
- **Confabulation**: An LLM asked to describe a cluster it has no prior knowledge of will generate plausible-sounding text that may not accurately reflect the cluster's defining characteristics.
- **Novel domain failure**: If the domain is genuinely new — a synthetic game designed for this research, with no prior human theory — the LLM has no vocabulary to draw from. Post-hoc description fails completely.

### 1.2 The Novel Domain Argument

Consider training an RL agent on a game designed specifically for this research — one that no human has extensively analyzed and no LLM has encountered in training data. Post-hoc interpretability:

1. The LLM does not know the game's rules, strategic depth, or relevant patterns.
2. It cannot recognize meaningful positions from any rendering of the observation space.
3. Its descriptions will analogize to known games or produce generic text.
4. The resulting descriptions are neither accurate nor novel — they import irrelevant human concepts.

PIDGIN's response: **if the LLM participates in shaping what a concept is during training, concepts crystallize around describable structure.** The LLM generates observational descriptions grounded in what it can see (spatial patterns, outcome correlations, action frequencies). Those descriptions become the concept's vocabulary. Later concepts build on established vocabulary. The agent and the LLM develop a shared language for a domain neither fully understood at the outset.

This is the killer motivating example. Strategy development is well-studied in existing games (chess, Go, Atari); there is little room for PIDGIN to coin genuinely new strategies there. PIDGIN's proper domain is **novel environments** where the strategy space is unexplored and no prior human vocabulary exists.

### 1.3 What PIDGIN Provides

PIDGIN produces a **concept library**: a mapping from concept IDs to persistent natural language descriptions, where:

1. Descriptions co-evolved with the concepts they describe (not applied afterward).
2. Vocabulary is persistent — terms coined early are used consistently throughout training.
3. The policy's behavior is mechanistically downstream of the descriptions — descriptions are functional inputs to the policy, not annotations.
4. The system is domain-agnostic, operating via a plugin interface for observation rendering.

---

## 2. Framework Overview

### 2.1 The Three Components

PIDGIN extends PRISM's three-stage pipeline with three interacting components:

```
[PRISM Stage 1]  Baseline RL training → frozen domain encoder
      │
      ▼
[PIDGIN Component 2]  Describability-Regularized Concept Formation
      │  (Modified Stage 2: clustering shaped by LLM describability)
      │  ←→ [PIDGIN Component 1: Vocabulary Manager]
      ▼
[PIDGIN Component 3]  Description-Conditioned Policy
      │  (Modified Stage 3: policy conditions on text embeddings)
      │  ←→ [PIDGIN Component 1: Vocabulary Manager]
      ▼
  Human-readable concept library + explainable policy
```

**Component 1 — Vocabulary Manager**
A persistent, append-only dictionary mapping concept IDs to descriptions and their text embeddings. All LLM calls are made with full vocabulary context, enforcing terminological consistency.

**Component 2 — Describability-Regularized Concept Formation**
Modified clustering that optimizes jointly for intra-cluster compactness and LLM describability. Low-describability clusters are split; clusters with near-identical descriptions are merge candidates. TextGrad runs within the loop to refine descriptions.

**Component 3 — Description-Conditioned Policy**
The bottleneck policy conditions on the text embedding of the concept's description rather than an anonymous learned embedding. The policy's behavior is mechanistically linked to language.

### 2.2 Relationship to PRISM

PIDGIN is a drop-in extension to PRISM. Stage 1 is unchanged. Stages 2 and 3 are augmented. PRISM's artifacts (encoder, ablation results, intervention data) feed into PIDGIN's components.

| Stage | PRISM | PIDGIN |
|---|---|---|
| 1. Baseline training | PPO/DQN → encoder | Unchanged |
| 2. Concept formation | K-means (compactness only) | K-means + describability objective |
| 3. Policy training | Embedding(K, d) → MLP → action | sentence_embed(description) → MLP → action |
| Evaluation | Win rate, ARI, transfer win rate | + BDM, DBM, vocab consistency, policy explainability |

---

## 3. Component 1: Vocabulary Manager

### 3.1 Structure

The Vocabulary Manager maintains a persistent dictionary:

```python
@dataclass
class VocabEntry:
    concept_id: int
    name: str                    # 2-5 word coined name
    description: str             # Full structured description
    text_embedding: np.ndarray   # sentence-transformers embedding (384D)
    version: int                 # Incremented on significant update
    coined_at_step: int          # Training step when first coined
    update_history: list[str]    # Previous descriptions for drift tracking

vocab: dict[int, VocabEntry]
```

### 3.2 Vocabulary Persistence

When the LLM generates or updates any description, the full current vocabulary is injected into context:

```
VOCABULARY IN USE:
  Concept 0: "Boundary Anchor" — agent clusters actions near the state-space
             boundary, forming connected groups
  Concept 3: "Interior Pressure" — agent occupies central positions adjacent
             to opponent groups
  ...

INSTRUCTIONS:
  - Use the above terms where applicable.
  - If this concept relates to "Boundary Anchor" behavior, reference that term.
  - If this concept is genuinely novel and not covered by existing terms,
    coin a new term. New terms must be descriptive and grounded in observable
    behavior — not analogies to other known domains.
  - Do NOT import terminology from other known domains (chess, Go, Atari, etc.)
    unless the current domain demonstrably shares that structure.
  - Describe observationally. Ground descriptions in the state examples and
    behavioral data provided.
```

This enforces consistency through prompt engineering, with no additional architectural mechanism required.

### 3.3 Update Policy

Descriptions are updated only when a concept drifts significantly (cosine similarity between old and new centroid < 0.85). Minor drift triggers a description review; major drift triggers re-optimization via TextGrad. The append-only policy means no terms are deleted. If two concepts merge, the merged entry references both prior names.

### 3.4 Domain-Agnosticism

For a novel domain, the Vocabulary Manager starts empty. The LLM's first descriptions are purely observational:

> "Name: Edge Cluster\nDescription: The agent places units adjacent to the boundary of the state space, forming connected groups. Actions concentrate on boundary-adjacent positions."

These terms become the vocabulary. Subsequent concepts are described relative to them:

> "Name: Edge Counter\nDescription: A response pattern to Edge Cluster formations. The agent occupies interior positions adjacent to established edge groups."

No prior domain knowledge is required. The vocabulary emerges from observable agent behavior.

---

## 4. Component 2: Describability-Regularized Concept Formation

### 4.1 Modified Clustering Objective

Standard K-means minimizes intra-cluster variance:

```
L_kmeans = Σ_k Σ_{i∈C_k} ||f(s_i) - μ_k||²
```

PIDGIN adds a describability regularization term:

```
L_PIDGIN = L_kmeans + λ · L_desc
```

Where `L_desc` penalizes clusters that are incoherent (LLM cannot generate a specific description) or indistinguishable (two clusters receive near-identical descriptions, cosine similarity > θ between their text embeddings).

### 4.2 Describability Score

After each clustering iteration, each cluster is scored:

```
describability(k) = α · specificity(desc_k)
                  + β · (1 - max_{k'≠k} sim(embed(desc_k), embed(desc_{k'})))
                  + γ · action_consistency(desc_k, P(a|c=k))
```

Where:
- `specificity`: does the description contain specific, falsifiable claims about observable patterns? Scored via structured LLM evaluation (binary pass/fail per criterion, not a scalar rating).
- `distinctiveness`: how different is this description from all others in embedding space?
- `action_consistency`: do the described actions match the actual action distribution?

Low-describability clusters (score below threshold) trigger:
1. **Split candidate**: if the cluster contains subclusters with different action distributions, propose a split and re-optimize descriptions for each half.
2. **Merge candidate**: if two clusters have near-identical descriptions, propose a merge and re-optimize the merged description.

### 4.3 TextGrad Integration

TextGrad (Yuksekgonul et al., 2024) is used for description optimization within the loop:

```
For each cluster k:
    context_var = build_context(k)          # State examples, action dist, neighbors
    description_var = tg.Variable(
        value=vocab[k].description or TEMPLATE,
        requires_grad=True
    )
    loss = TextLoss(context_var + description_var, eval_criteria)
    loss.backward()
    optimizer.step()
    vocab.update(k, description_var.value)
```

For online training (clustering updates every N steps), a single TextGrad step may suffice. For offline post-formation refinement, the full T=10 loop runs.

### 4.4 Online vs. Offline Clustering

**Offline mode**: Collect encoder features from evaluation episodes, run describability-regularized clustering, then train the description-conditioned policy. Compatible with PRISM's existing Stage 2.

**Online mode**: Clustering runs incrementally during RL training. Every M training steps, cluster assignments are updated and descriptions are re-evaluated. More expensive but allows concepts to co-evolve with the encoder during early training.

The comparison study tests both modes.

---

## 5. Component 3: Description-Conditioned Policy

### 5.1 Architecture

Standard PRISM bottleneck policy:
```
concept_id (int) → Embedding(K, d) → MLP → action logits
```

PIDGIN replaces this with:
```
description (str) → sentence_embed (frozen, 384D) → MLP → action logits
```

The sentence embedding is produced by a frozen `sentence-transformers/all-MiniLM-L6-v2`. The MLP maps from 384D to action logits.

### 5.2 Why This Matters

This is the central architectural link between PIDGIN's language and behavior:

1. **Descriptions are functional**: Changing the description changes the policy's behavior. The description is not a label — it is an input.
2. **Novel description generalization**: The policy can potentially generalize to descriptions geometrically close to training descriptions in embedding space — including descriptions of situations never encountered during training.
3. **Vocabulary changes propagate automatically**: When the Vocabulary Manager updates a description due to concept drift, the policy's effective input changes without retraining.
4. **Human language queries**: A human (or any system) can query the policy with a natural language description of a situation and receive a meaningful action distribution. This is the machine → human interface.

### 5.3 Training

Training proceeds as in PRISM's Stage 3 (encoder and clustering frozen), except:
- Input: `sentence_embed(vocab[c(s)].description)` instead of integer concept ID.
- The sentence encoder is frozen throughout.
- A contrastive loss prevents collapse when descriptions are semantically similar:

```
L_policy = L_RL + δ · L_contrastive
L_contrastive = Σ_{k≠k'} max(0, margin - ||embed(desc_k) - embed(desc_{k'})||)
```

---

## 6. Comparison Study: Method Variants

PIDGIN is a framework, not a single method. The paper compares five variants to isolate each component's contribution:

| Method | LLM integration | Policy input | Label |
|---|---|---|---|
| A | Post-hoc, single LLM pass | concept ID | Baseline |
| B | Post-hoc TextGrad (T=10) | concept ID | v1 DESIGN.md approach |
| C | Describability-regularized clustering only | concept ID | Component 2 only |
| D | Description-conditioned policy only (post-hoc descriptions) | text embedding | Component 3 only |
| E | Full PIDGIN (Components 1+2+3) | text embedding | Primary |

Methods A and B show the state of the art in post-hoc description. C and D isolate individual components. E is the full system.

The comparison is run in a domain chosen to minimize prior LLM knowledge, to stress-test the novel domain claim. Results in well-known domains are reported separately to distinguish the two regimes.

---

## 7. Evaluation

### 7.1 Concept Discriminativeness Score (CDS)

Generalized from v1 to be domain-agnostic. All "board state" references become "observation":

- **BDM-10** (Observation-to-Description Matching, 10-way): Given an observation where concept k fired, select the correct description from 10 options (9 hard distractors = nearest-neighbor concepts by centroid similarity). Chance = 10%. Target > 50%.
- **DBM-K** (Description-to-Observation Matching, K-way): Given an observation, identify the correct description from all K. Chance = 1/K. Target > 25% on high-importance concepts.
- **BPA** (Behavioral Prediction Accuracy): Given only the description, predict action region and ablation importance class.
- **SNA** (Semantic Neighbor Agreement): Spearman correlation between description embedding similarity and centroid cosine similarity.

```
CDS = 0.4 · BDM-10_norm + 0.3 · DBM-K_norm + 0.15 · BPA_norm + 0.15 · SNA_norm
```

### 7.2 New Metrics

**Vocabulary Consistency (VC)**
For each coined term, compute the fraction of descriptions that use it where contextually applicable (judged by a fresh LLM instance with the full vocabulary as ground truth). Target: VC > 0.85.

**Policy Explainability (PE)**
Given only a description, can a fresh LLM instance predict the agent's top-3 most likely actions? Correct if ground truth is in the predicted set. Target: PE > 40% (chance depends on action space size).

**Vocabulary Novelty (VN)**
For novel-domain experiments: what fraction of coined terms are genuinely new (not present in LLM's prior knowledge)? Judged by prompting a separate LLM to classify each term as "known domain terminology" vs. "novel coinage." This validates the domain-agnosticism claim. Target: VN > 0.3 in novel-domain experiments.

**Description Stability (DS)**
Rolling cosine similarity between descriptions at step T and T+N across training. High DS = vocabulary has settled. Target: DS > 0.90 after the first 20% of training.

### 7.3 Comparison Study Metrics

| Comparison | What it tests |
|---|---|
| CDS(A) vs CDS(B) | Does TextGrad iteration help post-hoc? |
| CDS(B) vs CDS(C) | Does integrating LLM into clustering beat post-hoc? |
| PE(D) vs PE(A/B/C) | Does description-conditioned policy improve explainability? |
| VC(E) vs VC(A/B) | Does the Vocabulary Manager improve consistency? |
| VN (novel domain only) | Can PIDGIN generate genuinely novel vocabulary? |
| Transfer win rate (all) | Does interpretability hurt performance? |

### 7.4 Null Hypotheses

**H0a (Integration null)**: Full PIDGIN (Method E) does not improve CDS over post-hoc TextGrad (Method B).

**H0b (Policy null)**: Description-conditioned policy (D) does not improve PE over post-hoc methods (A, B).

**H0c (Novelty null)**: In novel-domain experiments, PIDGIN descriptions are not meaningfully different from post-hoc descriptions (VN ≈ 0).

**H0d (Stability null)**: Vocabulary Manager does not produce more stable vocabulary than post-hoc methods (VC < 0.70).

**H0e (Performance null)**: PIDGIN's additional objectives degrade transfer win rate by more than 5 percentage points compared to PRISM.

### 7.5 Success Criteria

**Minimum publishable result**: CDS(E) > CDS(B) with p < 0.05 + VC > 0.85 + H0e rejected. This establishes that co-training improves discriminativeness and stability without degrading performance.

**Strong result**: All null hypotheses rejected + VN > 0.3 in novel-domain experiment. This establishes PIDGIN as a genuine domain-agnostic framework for emergent strategy vocabulary.

---

## 8. Architecture Summary

### 8.1 File Structure

```
prism/pidgin/
├── __init__.py
├── DESIGN.md                          ← This document
├── vocabulary_manager.py              ← Component 1
│                                         Persistent vocab dict, update policy,
│                                         domain-agnostic prompt construction
├── data_collector.py                  ← Domain-agnostic observation collector
│                                         Collects (obs, concept_id, action) tuples
│                                         Rendering via domain plugin
├── concept_prompter.py                ← Formats evidence into LLM context blocks
│                                         Injects vocabulary, structures context
├── describability_clusterer.py        ← Component 2
│                                         Modified K-means + describability objective
│                                         TextGrad integration for description optimization
├── description_policy.py              ← Component 3
│                                         sentence_embed → MLP → actions
│                                         Contrastive loss, training loop
├── evaluator.py                       ← All evaluation metrics
│                                         BDM, DBM, BPA, SNA, VC, PE, VN, DS
├── run_pidgin.py                      ← Main entry point
│                                         Orchestrates all components
└── results/
    ├── concept_descriptions.json      ← Final concept library
    ├── concept_library.json           ← Full metadata + convergence stats
    ├── vocabulary_log.json            ← Term-by-term vocabulary evolution over training
    └── optimization_logs/             ← Per-concept convergence curves
        ├── concept_000_log.json
        └── ...
```

### 8.2 Domain Plugin Interface

PIDGIN is task-agnostic via a plugin interface for domain-specific rendering:

```python
class DomainPlugin(Protocol):
    def render_observation(self, obs: np.ndarray) -> str:
        """Render an observation as text for the LLM context."""
        ...

    def action_name(self, action: int) -> str:
        """Human-readable action name, or generic 'Action {i}' if unknown."""
        ...

    def observation_metadata(self, obs: np.ndarray) -> dict:
        """Domain-specific metadata: step number, player, score, etc."""
        ...
```

For known domains, plugins provide rich rendering. For novel domains, the generic plugin describes observations spatially and numerically. All other PIDGIN components are fully domain-agnostic.

### 8.3 Key Design Parameters

| Parameter | Value | Rationale |
|---|---|---|
| K (concepts) | Inherited from PRISM | Configurable per domain |
| T (TextGrad max iterations) | 10 | Convergence detection stops early |
| Describability threshold | 0.5 | Below: flag for split/merge |
| Contrastive margin | 0.3 | Prevents description embedding collapse |
| Vocab drift threshold | cosine sim < 0.85 | Prevents unnecessary vocab churn |
| LLM backend | claude-sonnet-4-20250514 | Strong instruction following |
| Sentence encoder | all-MiniLM-L6-v2 | Fast, CPU-friendly, 384D |
| λ (describability regularization weight) | Sweep: {0.01, 0.1, 1.0} | TBD empirically |

---

## 9. Open Questions

1. **Online vs. offline clustering**: Online clustering co-evolves concepts with the encoder but requires LLM calls during training. What is the tradeoff? Is online necessary for the vocabulary novelty claim?

2. **Sentence embedding collapse**: If the domain produces semantically similar descriptions, the sentence encoder may not separate them well. Does the contrastive loss suffice, or do we need a description encoder fine-tuned on the domain?

3. **LLM observational grounding for non-visual domains**: If the observation space is high-dimensional or non-visual (physics simulation, raw sensor data), can the LLM generate meaningful descriptions from numerical features? What representations are minimally required?

4. **Vocabulary bootstrapping**: At step 0, the vocabulary is empty. Early descriptions will be generic. How many concepts need to be established before the vocabulary becomes self-referential? Is there an explicit bootstrapping phase where generic descriptions are acceptable?

5. **Optimal λ**: Too high forces artificial splits for describability at the cost of compactness. Too low reverts to standard K-means. Requires sweep.

6. **Novel domain selection**: The experiment requires a domain with (a) no prior LLM knowledge, (b) non-trivial strategy space, (c) fast training on CPU. Options: procedurally generated grid world with novel mechanics, custom multi-agent environment, synthetic combinatorial game.

---

## 10. Summary

PIDGIN is a framework for emergent, vocabulary-grounded concept formation in RL. Unlike post-hoc interpretability methods that impose human vocabulary on machine-discovered concepts, PIDGIN co-evolves strategy representations and natural language descriptions throughout training. The result is a system where the agent's concepts are intrinsically human-expressible, with a persistent vocabulary that can capture genuinely novel strategic patterns in domains beyond existing human expertise.

Three components work together:

1. **Vocabulary Manager**: Persistent, append-only dictionary. All LLM calls receive full vocabulary context. Coined terms persist; updates require evidence of concept drift.

2. **Describability-Regularized Concept Formation**: Clustering objective that jointly optimizes compactness and LLM describability. Uses TextGrad in the loop. Splits incoherent clusters; merges redundant ones.

3. **Description-Conditioned Policy**: Policy conditions on text embeddings of concept descriptions. Behavior is mechanistically downstream of language — descriptions are functional inputs, not annotations.

The framework is task-agnostic via a domain plugin interface. Its primary validation target is a novel domain where no prior LLM knowledge exists — the regime where post-hoc interpretation fails and PIDGIN's emergent vocabulary is the only viable approach.

**Research lineage:**
- PRISM: machine → machine
- PIDGIN: machine → human
- (Future): human → machine
