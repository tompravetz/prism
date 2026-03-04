"""
Random Agent Baseline vs GnuGo.

Measures the true random-play win rate vs GnuGo at a given level.
Used to establish the correct null hypothesis for transfer experiment
p-values (replacing the incorrect H0=0.5 assumption).

Usage:
    python experiments/random_baseline.py --level 1 --n-seeds 10 --n-eval 100
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.go_env import GoEnv
from src.utils import set_seed, ensure_dir


def run_random_baseline(gnugo_level=1, n_seeds=10, n_eval=100):
    """
    Evaluate a purely random agent vs GnuGo at the given level.

    The random agent selects uniformly from all legal moves (including pass),
    matching GoEnv's _random_opponent exactly. This establishes the true floor
    for GnuGo-evaluated transfer experiments.
    """
    from visualizer.opponents import GnuGoOpponent

    print(f"Random Agent Baseline vs GnuGo Level {gnugo_level}")
    print(f"  {n_seeds} seeds x {n_eval} games = {n_seeds * n_eval} total games")

    seed_wrs = []

    for seed in range(n_seeds):
        set_seed(seed * 1000)
        opponent = GnuGoOpponent(level=gnugo_level)
        env = GoEnv(board_size=7, opponent_fn=opponent)

        wins = 0
        for _ in range(n_eval):
            obs, info = env.reset()
            done = False
            while not done:
                mask = info.get("action_mask", np.ones(env.action_count, dtype="int8"))
                legal = np.where(mask == 1)[0]
                action = int(np.random.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if reward > 0:
                wins += 1

        wr = wins / n_eval
        seed_wrs.append(wr)
        print(f"  Seed {seed}: {wins}/{n_eval} = {wr:.1%}")
        env.close()
        opponent.close()

    mean_wr = float(np.mean(seed_wrs))
    std_wr = float(np.std(seed_wrs, ddof=1))
    ci = stats.t.interval(0.95, df=n_seeds - 1,
                           loc=mean_wr, scale=std_wr / np.sqrt(n_seeds))

    print(f"\nRandom baseline vs GnuGo L{gnugo_level}:")
    print(f"  Mean WR : {mean_wr:.2%} +/- {std_wr:.2%}")
    print(f"  95% CI  : [{ci[0]:.2%}, {ci[1]:.2%}]")

    # Re-test transfer results against this baseline
    result_path = f"results/transfer_same_task_10seed_L{gnugo_level}.json"
    if os.path.exists(result_path):
        print(f"\nRe-testing transfer results against random baseline "
              f"({mean_wr:.2%})...")
        with open(result_path) as f:
            transfer_data = json.load(f)

        print(f"\n{'Source':<8} {'Target':<8} {'Mean WR':>9} {'vs Random':>11} "
              f"{'p (vs random)':>14} {'significant':>12}")
        print("-" * 65)
        for pair in transfer_data["pairs"]:
            wrs = pair["win_rates"]
            t_stat, p_val = stats.ttest_1samp(wrs, mean_wr)
            sig = p_val < 0.05 and np.mean(wrs) > mean_wr
            delta = np.mean(wrs) - mean_wr
            print(f"{pair['source']:<8} {pair['target']:<8} "
                  f"{pair['mean_wr']:>8.2%} "
                  f"{delta:>+10.2%} "
                  f"{p_val:>14.4f} "
                  f"{'YES' if sig else 'no':>12}")

    # Save
    ensure_dir("results")
    out = {
        "gnugo_level": gnugo_level,
        "n_seeds": n_seeds,
        "n_eval": n_eval,
        "seed_win_rates": [float(x) for x in seed_wrs],
        "mean_wr": mean_wr,
        "std_wr": std_wr,
        "ci_95_lower": float(ci[0]),
        "ci_95_upper": float(ci[1]),
    }
    out_path = f"results/random_baseline_L{gnugo_level}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random agent baseline vs GnuGo")
    parser.add_argument("--level", type=int, default=1,
                        help="GnuGo level (default: 1)")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Evaluation seeds (default: 10)")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Games per seed (default: 100)")
    args = parser.parse_args()
    run_random_baseline(gnugo_level=args.level,
                        n_seeds=args.n_seeds, n_eval=args.n_eval)
