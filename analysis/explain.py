"""
LLM Explanation Pipeline.

Uses Claude API to generate natural language explanations for discovered
Go strategies. For each top strategy (concept-action pair), we:

    1. Collect example board states that trigger this concept
    2. Render them as ASCII text
    3. Ask Claude to:
        - Name the strategy
        - Explain what the board states have in common
        - Compare to known Go theory/principles
        - Assess whether the strategy is novel or standard

This produces human-readable interpretations of what the bottleneck agent
has learned, bridging the gap between the agent's discrete concepts and
human understanding of Go.

Expected cost: ~$2-3 total for 15-20 strategy explanations.

Usage:
    python -m analysis.explain --algo ppo --top-k 15
    python -m analysis.explain --algo both
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.strategy_memory import StrategyMemory
from src.utils import (get_device, ensure_dir, set_seed,
                       render_go_board_ascii, action_to_coord)


def collect_strategy_examples(encoder, concept_manager, env,
                               target_concepts, n_examples=5,
                               n_episodes=500, device=None):
    """
    Collect example board states for specific concepts.

    Plays random games and records board states that map to each target concept.
    For each concept, we collect n_examples diverse examples.

    Args:
        encoder: Frozen encoder.
        concept_manager: Fitted ConceptManager.
        env: Go environment.
        target_concepts: List of concept IDs to find examples for.
        n_examples: Number of example boards per concept.
        n_episodes: Max episodes to search.
        device: Torch device.

    Returns:
        Dictionary: {concept_id: [(obs, action_mask), ...]}
    """
    device = device or get_device()
    examples = defaultdict(list)
    target_set = set(target_concepts)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            concept = concept_manager.assign_concept_from_obs(encoder, obs, device)

            if concept in target_set and len(examples[concept]) < n_examples:
                mask = info.get("action_mask", np.ones(50, dtype=np.int8))
                examples[concept].append((obs.copy(), mask.copy()))

            # Check if we have enough examples for all target concepts
            if all(len(examples[c]) >= n_examples for c in target_concepts):
                return dict(examples)

            # Random action to advance game
            mask = info.get("action_mask", np.ones(50, dtype=np.int8))
            legal = np.where(mask == 1)[0]
            action = np.random.choice(legal) if len(legal) > 0 else 49
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return dict(examples)


def create_strategy_prompt(strategy, examples, board_size=7):
    """
    Create a prompt for Claude to explain a strategy.

    The prompt includes:
        - The strategy metadata (concept ID, action, win rate)
        - 3-5 example board states rendered as ASCII
        - Instructions for naming and explaining the strategy

    Args:
        strategy: Dict with concept_id, action, win_rate, count.
        examples: List of (obs, action_mask) tuples for this concept.
        board_size: Board size.

    Returns:
        String prompt for Claude API.
    """
    concept_id = strategy["concept_id"]
    action = strategy["action"]
    win_rate = strategy["win_rate"]
    count = strategy["count"]
    coord = action_to_coord(action, board_size)

    # Render example boards as ASCII
    board_texts = []
    for i, (obs, mask) in enumerate(examples[:5]):
        board_ascii = render_go_board_ascii(obs, board_size)
        board_texts.append(f"Example {i+1}:\n{board_ascii}")

    boards_str = "\n\n".join(board_texts)

    action_desc = f"position {coord}" if coord != "pass" else "pass"

    prompt = f"""I'm analyzing an AI agent that learned to play Go (7x7 board) using a
concept bottleneck architecture. The agent groups board states into discrete
"concepts" and learns strategies based on these concepts.

Here is one discovered strategy:
- Concept ID: {concept_id}
- Preferred Action: {action_desc}
- Win Rate: {win_rate:.1%}
- Times Used: {count}

Below are {min(len(examples), 5)} example board states that trigger this concept.
X = Black stones (the AI), O = White stones, . = empty.

{boards_str}

Please analyze this strategy:
1. **Name**: Give this strategy a short, descriptive name (2-5 words).
2. **Pattern**: What do these board states have in common? Describe the
   structural features that define this concept.
3. **Action Rationale**: Why is the preferred action ({action_desc}) a good
   move in these situations?
4. **Go Theory**: Does this strategy correspond to any known Go concept,
   principle, or pattern? (e.g., territory control, influence, connecting
   groups, protecting weaknesses, attacking, joseki patterns)
5. **Assessment**: Is this a standard/expected strategy, a creative discovery,
   or a suboptimal pattern?

Be concise — aim for 3-5 sentences per point."""

    return prompt


def explain_strategies_with_llm(strategies, examples_by_concept,
                                 api_key=None, model="claude-sonnet-4-5-20250929"):
    """
    Send strategy descriptions to Claude API for explanation.

    Args:
        strategies: List of strategy dicts.
        examples_by_concept: Dict of {concept_id: [(obs, mask), ...]}.
        api_key: Anthropic API key (from env var if not provided).
        model: Claude model to use.

    Returns:
        List of dicts with strategy info + LLM explanation.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Install with: pip install anthropic")
        return None

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set. Set it with: "
                  "export ANTHROPIC_API_KEY='your-key'")
            return None

    client = anthropic.Anthropic(api_key=api_key)

    explanations = []

    for i, strategy in enumerate(strategies):
        concept_id = strategy["concept_id"]
        examples = examples_by_concept.get(concept_id, [])

        if not examples:
            print(f"  Skipping concept {concept_id}: no examples found")
            continue

        prompt = create_strategy_prompt(strategy, examples)

        print(f"  Explaining strategy {i+1}/{len(strategies)} "
              f"(concept {concept_id})...", end=" ")

        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            explanation = response.content[0].text
            print("done")
        except Exception as e:
            explanation = f"ERROR: {e}"
            print(f"error: {e}")

        explanations.append({
            **strategy,
            "explanation": explanation,
            "n_examples": len(examples),
        })

    return explanations


def run_explanation_pipeline(algo="ppo", top_k=15, n_concepts=64, seed=42,
                              model_dir="models/bottleneck",
                              baseline_dir="models/baseline"):
    """
    Complete explanation pipeline:
        1. Load strategy memory
        2. Collect example board states for top strategies
        3. Send to Claude for explanation
        4. Save results as JSON and formatted text
    """
    set_seed(seed)
    device = get_device()

    # ---- Load models ----
    env = GoEnv(board_size=7)

    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder_path = os.path.join(baseline_dir, f"{algo}_go_encoder.pt")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
    encoder.to(device)
    encoder.eval()

    cm = ConceptManager(n_concepts=n_concepts)
    concept_path = os.path.join(model_dir, f"concepts_{algo}_k{n_concepts}.pkl")
    if os.path.exists(concept_path):
        cm.load(concept_path)
    else:
        print(f"ERROR: Concepts not found at {concept_path}")
        return

    # Load strategy memory
    sm = StrategyMemory(n_concepts=n_concepts, n_actions=50)
    sm_path = os.path.join(model_dir, f"strategy_memory_{algo}.pkl")
    if os.path.exists(sm_path):
        sm.load(sm_path)
    else:
        print(f"ERROR: Strategy memory not found at {sm_path}")
        return

    # Get top strategies
    strategies = sm.get_top_strategies(min_count=20, min_win_rate=0.5, top_k=top_k)
    if not strategies:
        strategies = sm.get_top_strategies(min_count=5, min_win_rate=0.3, top_k=top_k)

    if not strategies:
        print("No strategies found. Run training first.")
        return

    print(f"Found {len(strategies)} strategies to explain.")

    # ---- Collect example board states ----
    target_concepts = list(set(s["concept_id"] for s in strategies))
    print(f"Collecting example boards for {len(target_concepts)} concepts...")
    examples = collect_strategy_examples(
        encoder, cm, env, target_concepts,
        n_examples=5, n_episodes=500, device=device,
    )
    print(f"Found examples for {len(examples)} concepts.")

    # ---- Get LLM explanations ----
    print("\nSending strategies to Claude for explanation...")
    explanations = explain_strategies_with_llm(strategies, examples)

    if explanations:
        # Save as JSON
        ensure_dir("results")
        output_path = f"results/explanations_{algo}.json"
        with open(output_path, "w") as f:
            json.dump(explanations, f, indent=2, default=str)
        print(f"\nExplanations saved to {output_path}")

        # Save as formatted text report
        report_path = f"results/strategy_report_{algo}.txt"
        with open(report_path, "w") as f:
            f.write(f"STRATEGY EXPLANATIONS — {algo.upper()}\n")
            f.write("=" * 60 + "\n\n")
            for i, exp in enumerate(explanations):
                f.write(f"Strategy {i+1}: Concept {exp['concept_id']}, "
                        f"Action {exp['action']}\n")
                f.write(f"Win Rate: {exp['win_rate']:.1%}, "
                        f"Count: {exp['count']}\n")
                f.write("-" * 40 + "\n")
                f.write(exp.get("explanation", "No explanation available") + "\n")
                f.write("\n\n")
        print(f"Report saved to {report_path}")
    else:
        print("No explanations generated. Check API key.")

    env.close()
    return explanations


def main():
    parser = argparse.ArgumentParser(description="LLM explanation pipeline")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "both"])
    parser.add_argument("--top-k", type=int, default=15,
                        help="Number of strategies to explain")
    parser.add_argument("--n-concepts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/bottleneck")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "both" else ["ppo", "dqn"]
    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Strategy Explanation — {algo.upper()}")
        print(f"{'='*60}")
        run_explanation_pipeline(
            algo=algo, top_k=args.top_k,
            n_concepts=args.n_concepts, seed=args.seed,
            model_dir=args.model_dir, baseline_dir=args.baseline_dir,
        )


if __name__ == "__main__":
    main()
