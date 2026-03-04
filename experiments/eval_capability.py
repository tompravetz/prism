"""
PRISM Capability Sweep — find the GnuGo level boundary.

Runs a concept bottleneck agent (and optionally the pre-bottleneck baseline)
against increasing GnuGo levels, stopping when the win rate is no longer
significantly above 50% (one-sided binomial test, default α=0.05).

Running both lets you measure the actual cost of interpretability: the gap
between the bottleneck and the unconstrained baseline at each GnuGo level.

Each level is evaluated with n_seeds × n_games games. Fixed sequential
seeds (0, 1, …, n_seeds−1) are used for reproducibility — game diversity
comes from GnuGo's own internal randomness, not the Python seed.

Statistical protocol matches the paper's agent-transfer experiments:
5 seeds × 100 games per condition, significance vs. 50% baseline.

Usage:
    python experiments/eval_capability.py --algo ppo
    python experiments/eval_capability.py --algo ppo --include-baseline
    python experiments/eval_capability.py --algo dqn --include-baseline
    python experiments/eval_capability.py --algo ppo --start-level 2 --n-seeds 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.networks import GoCNNEncoder
from src.concept_manager import ConceptManager
from src.concept_policy import ConceptBottleneckPolicy, ConceptDQNPolicy
from src.utils import set_seed, get_device, ensure_dir
from experiments.eval_strong import eval_agent_vs_gnugo


def load_baseline_agent(algo: str, device):
    """
    Load the pre-bottleneck baseline agent (full policy, no concept layer).

    PPO baseline: SB3 MaskablePPO zip — predict(obs, action_masks=mask).
    DQN baseline: custom .pt file — requires knowing the saved class.
    """
    if algo == "ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            raise ImportError("sb3-contrib required: pip install sb3-contrib")
        model = MaskablePPO.load(
            "models/baseline/ppo_go_baseline.zip",
            device=device,
        )

        def agent_fn(obs, mask):
            action, _ = model.predict(
                obs, deterministic=True, action_masks=mask
            )
            return int(action)

        return agent_fn

    elif algo == "dqn":
        # Checkpoint saved by train_baseline.py as a top-level dict:
        # {"encoder": state_dict, "q_head": state_dict, "target_encoder": ...,
        #  "target_q_head": ..., "optimizer": ..., "steps_done": int}
        state = torch.load(
            "models/baseline/dqn_go_baseline.pt",
            map_location=device, weights_only=True,
        )
        from src.networks import GoCNNEncoder, QNetwork
        env = GoEnv(board_size=7)
        encoder = GoCNNEncoder(env.observation_space, features_dim=128)
        env.close()

        encoder.load_state_dict(state["encoder"])
        encoder.to(device).eval()

        n_actions = state["q_head"]["net.4.weight"].shape[0]
        q_head = QNetwork(features_dim=128, n_actions=n_actions).to(device)
        q_head.load_state_dict(state["q_head"])
        q_head.eval()

        def agent_fn(obs, mask):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = (torch.FloatTensor(mask).to(device)
                      if mask is not None else None)
            with torch.no_grad():
                feats = encoder(obs_t)
                q_vals = q_head(feats, mask_t)[0]
            return int(q_vals.argmax().item())

        return agent_fn

    else:
        raise ValueError(f"Unknown algo: {algo}")


def load_agent(algo: str, device):
    """Load encoder, concept manager, and policy for the given algo."""
    env = GoEnv(board_size=7)

    encoder = GoCNNEncoder(env.observation_space, features_dim=128)
    encoder.load_state_dict(
        torch.load(f"models/baseline/{algo}_go_encoder.pt",
                   map_location=device, weights_only=True)
    )
    encoder.to(device).eval()

    cm = ConceptManager(n_concepts=64)
    cm.load(f"models/bottleneck/concepts_{algo}_k64.pkl")

    if algo == "dqn":
        policy = ConceptDQNPolicy(
            n_concepts=64, embed_dim=64, hidden_dim=128, n_actions=50,
        )
        policy.load_state_dict(
            torch.load(f"models/bottleneck/{algo}_bottleneck_final.pt",
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
            torch.load(f"models/bottleneck/{algo}_bottleneck_final.pt",
                       map_location=device, weights_only=True)
        )
        policy.to(device).eval()

        def agent_fn(obs, mask):
            c = cm.assign_concept_from_obs(encoder, obs, device)
            return policy.get_action(c, mask, deterministic=True)

    env.close()
    return agent_fn, encoder, cm


def eval_level(agent_fn, encoder, cm, algo, level, n_seeds, n_games, alpha, device):
    """
    Evaluate one GnuGo level across n_seeds seeds.

    Returns a dict with aggregate stats and significance test result.
    """
    print(f"\n{'='*60}")
    print(f"  {algo.upper()} vs GnuGo Level {level}"
          f"  ({n_seeds} seeds × {n_games} games = {n_seeds * n_games} total)")
    print(f"{'='*60}")

    all_wins = 0
    all_games = 0
    seed_win_rates = []
    t0 = time.time()

    for seed in range(n_seeds):
        set_seed(seed)
        result = eval_agent_vs_gnugo(
            agent_fn, encoder, cm,
            gnugo_level=level, n_games=n_games, device=device,
        )
        wr = result["win_rate"]
        seed_win_rates.append(wr)
        all_wins += result["wins"]
        all_games += result["total_games"]
        print(f"    seed {seed}: {result['wins']:3d}W / {result['losses']:3d}L"
              f"  ({wr:.1%})")

    win_rate = all_wins / all_games
    mean_wr  = float(np.mean(seed_win_rates))
    std_wr   = float(np.std(seed_win_rates, ddof=1)) if n_seeds > 1 else 0.0

    # One-sided binomial test: H0: p ≤ 0.50, H1: p > 0.50
    binom = stats.binomtest(all_wins, all_games, p=0.5, alternative="greater")
    p_value = float(binom.pvalue)
    significant = p_value < alpha

    elapsed = time.time() - t0
    ci_lo = float(binom.proportion_ci(confidence_level=0.95).low)
    ci_hi = float(binom.proportion_ci(confidence_level=0.95).high)

    print(f"\n  Level {level} result:")
    print(f"    Win rate : {win_rate:.1%}  ({all_wins}/{all_games})")
    print(f"    95% CI   : [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"    Mean ± SD: {mean_wr:.1%} ± {std_wr:.1%}  (across seeds)")
    print(f"    p-value  : {p_value:.4f}  "
          f"({'✓ significant' if significant else '✗ not significant'}, α={alpha})")
    print(f"    Time     : {elapsed:.0f}s")

    return {
        "level":           level,
        "wins":            all_wins,
        "games":           all_games,
        "win_rate":        round(win_rate, 4),
        "mean_wr":         round(mean_wr, 4),
        "std_wr":          round(std_wr, 4),
        "ci_95":           [round(ci_lo, 4), round(ci_hi, 4)],
        "p_value":         round(p_value, 6),
        "significant":     significant,
        "seed_win_rates":  [round(r, 4) for r in seed_win_rates],
        "elapsed_s":       round(elapsed, 1),
    }


def run_sweep(algo, start_level, max_level, n_seeds, n_games, alpha, device,
              include_baseline=False):
    print(f"\n{'='*60}")
    print(f"PRISM CAPABILITY SWEEP  |  algo={algo.upper()}")
    print(f"Levels {start_level}–{max_level}  |  {n_seeds} seeds × {n_games} games  |  α={alpha}")
    if include_baseline:
        print(f"Comparing bottleneck vs pre-bottleneck baseline")
    print(f"Sweeping all levels — significance column marks p < {alpha}")
    print(f"{'='*60}")

    agent_fn, encoder, cm = load_agent(algo, device)

    baseline_fn = None
    if include_baseline:
        print(f"\nLoading {algo.upper()} baseline (no bottleneck)...")
        baseline_fn = load_baseline_agent(algo, device)
        print("Baseline loaded.")

    results = []

    for level in range(start_level, max_level + 1):
        r = eval_level(agent_fn, encoder, cm, algo,
                       level, n_seeds, n_games, alpha, device)

        if baseline_fn is not None:
            print(f"\n  --- Baseline ({algo.upper()}, no bottleneck) vs Level {level} ---")
            b = eval_level(baseline_fn, None, None, f"{algo}_baseline",
                           level, n_seeds, n_games, alpha, device)
            r["baseline"] = b
            gap = r["win_rate"] - b["win_rate"]
            print(f"\n  Bottleneck vs Baseline gap at Level {level}: "
                  f"{gap:+.1%}  "
                  f"({'bottleneck ahead' if gap > 0 else 'baseline ahead'})")

        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PRISM capability sweep — find GnuGo level boundary"
    )
    parser.add_argument(
        "--algo", choices=["ppo", "dqn"], default="ppo",
        help="Agent to evaluate (default: ppo)"
    )
    parser.add_argument(
        "--start-level", type=int, default=1, metavar="N",
        help="First GnuGo level to test (default: 1)"
    )
    parser.add_argument(
        "--max-level", type=int, default=10, metavar="N",
        help="Maximum GnuGo level to test (default: 10)"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5, metavar="N",
        help="Number of seeds per level (default: 5)"
    )
    parser.add_argument(
        "--n-games", type=int, default=100, metavar="N",
        help="Games per seed (default: 100)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for stopping (default: 0.05)"
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Also sweep the pre-bottleneck baseline for gap comparison"
    )
    parser.add_argument(
        "--out-dir", default="results",
        help="Directory to save JSON results (default: results/)"
    )
    args = parser.parse_args()

    device = get_device()
    t_start = time.time()

    results = run_sweep(
        algo=args.algo,
        start_level=args.start_level,
        max_level=args.max_level,
        n_seeds=args.n_seeds,
        n_games=args.n_games,
        alpha=args.alpha,
        device=device,
        include_baseline=args.include_baseline,
    )

    # Save results
    ensure_dir(args.out_dir)
    out_path = os.path.join(args.out_dir, f"capability_sweep_{args.algo}.json")
    with open(out_path, "w") as f:
        json.dump({
            "algo":      args.algo,
            "n_seeds":   args.n_seeds,
            "n_games":   args.n_games,
            "alpha":     args.alpha,
            "levels":    results,
        }, f, indent=2)

    # Final summary table
    total_time = time.time() - t_start
    has_baseline = any("baseline" in r for r in results)
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE — {args.algo.upper()}")
    print(f"{'='*60}")
    if has_baseline:
        print(f"  {'Level':>5}  {'Bottleneck':>10}  {'Baseline':>9}  {'Gap':>6}  {'p-value':>9}  {'Sig?':>5}")
        print(f"  {'-'*57}")
        for r in results:
            sig = "✓" if r["significant"] else "✗"
            if "baseline" in r:
                gap = r["win_rate"] - r["baseline"]["win_rate"]
                print(f"  L{r['level']:>4}   {r['win_rate']:>9.1%}   "
                      f"{r['baseline']['win_rate']:>8.1%}   {gap:>+6.1%}  "
                      f"{r['p_value']:>9.4f}  {sig:>5}")
            else:
                print(f"  L{r['level']:>4}   {r['win_rate']:>9.1%}   "
                      f"{'—':>8}   {'—':>6}  {r['p_value']:>9.4f}  {sig:>5}")
    else:
        print(f"  {'Level':>5}  {'Win Rate':>9}  {'±SD':>7}  {'95% CI':>15}  {'p-value':>9}  {'Sig?':>5}")
        print(f"  {'-'*57}")
        for r in results:
            ci = f"[{r['ci_95'][0]:.1%}, {r['ci_95'][1]:.1%}]"
            sig = "✓" if r["significant"] else "✗"
            print(f"  L{r['level']:>4}   {r['win_rate']:>8.1%}   {r['std_wr']:>6.1%}  "
                  f"{ci:>15}  {r['p_value']:>9.4f}  {sig:>5}")
    print(f"\nResults saved: {out_path}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
