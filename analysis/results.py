"""
Compile experimental results into structured tables and summaries.

Reads all result JSON files from the results/ directory and produces:
    - Comparison tables (Markdown and LaTeX format)
    - Summary statistics
    - Success criteria checklist

This is used to generate the data for the research report's tables.

Usage:
    python -m analysis.results
"""

import os
import sys
import json
import numpy as np

# Add project root to path so 'src' package is importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import ensure_dir


def load_result(path, default=None):
    """Load a JSON result file, return default if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def compile_comparison_table(results_dir="results"):
    """
    Build the main comparison matrix:

    |                  | PPO Full | PPO Bottleneck | DQN Full | DQN Bottleneck |
    |------------------|----------|----------------|----------|----------------|
    | Win Rate         |          |                |          |                |
    | Concept Causality|   N/A    |                |   N/A    |                |
    | Strategy Quality |   N/A    |                |   N/A    |                |

    Returns: List of row dicts and formatted string.
    """
    # Load evaluation results
    eval_data = load_result(os.path.join(results_dir, "evaluation.json"), [])

    # Build lookup by variant name
    eval_by_variant = {}
    for r in eval_data:
        eval_by_variant[r.get("variant", "")] = r

    # Load intervention results
    interv_ppo = load_result(os.path.join(results_dir, "intervention_ppo.json"))
    interv_dqn = load_result(os.path.join(results_dir, "intervention_dqn.json"))

    # Load ablation results
    ablation_ppo = load_result(os.path.join(results_dir, "ablation_ppo.json"))
    ablation_dqn = load_result(os.path.join(results_dir, "ablation_dqn.json"))

    # Build table rows
    rows = []

    # Row 1: Win Rate
    rows.append({
        "Metric": "Win Rate vs Random",
        "PPO Full": _fmt_wr(eval_by_variant.get("PPO Full Baseline (Go)", {})),
        "PPO Bottleneck": _fmt_wr(eval_by_variant.get("PPO Bottleneck (Go)", {})),
        "DQN Full": _fmt_wr(eval_by_variant.get("DQN Full Baseline (Go)", {})),
        "DQN Bottleneck": _fmt_wr(eval_by_variant.get("DQN Bottleneck (Go)", {})),
    })

    # Row 2: Concept Causality (intervention change rate)
    rows.append({
        "Metric": "Concept Causality",
        "PPO Full": "N/A",
        "PPO Bottleneck": f"{interv_ppo['overall_change_rate']:.1%}" if interv_ppo else "—",
        "DQN Full": "N/A",
        "DQN Bottleneck": f"{interv_dqn['overall_change_rate']:.1%}" if interv_dqn else "—",
    })

    # Row 3: Strategy Count (significant ablation impact)
    rows.append({
        "Metric": "Significant Strategies",
        "PPO Full": "N/A",
        "PPO Bottleneck": str(ablation_ppo["n_significant_strategies"]) if ablation_ppo else "—",
        "DQN Full": "N/A",
        "DQN Bottleneck": str(ablation_dqn["n_significant_strategies"]) if ablation_dqn else "—",
    })

    # Row 4: Active Concepts
    rows.append({
        "Metric": "Active Concepts",
        "PPO Full": "N/A",
        "PPO Bottleneck": f"{interv_ppo['n_active_concepts']}/64" if interv_ppo else "—",
        "DQN Full": "N/A",
        "DQN Bottleneck": f"{interv_dqn['n_active_concepts']}/64" if interv_dqn else "—",
    })

    return rows


def _fmt_wr(result_dict):
    """Format win rate from result dict."""
    wr = result_dict.get("win_rate")
    if wr is not None:
        return f"{wr:.1%}"
    return "—"


def format_markdown_table(rows):
    """Format rows as a Markdown table."""
    if not rows:
        return "No data available."

    headers = list(rows[0].keys())
    lines = []

    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for row in rows:
        values = [str(row.get(h, "—")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def format_latex_table(rows):
    """Format rows as a LaTeX table."""
    if not rows:
        return "% No data available"

    headers = list(rows[0].keys())
    n_cols = len(headers)

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{'l' + 'c' * (n_cols - 1)}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    lines.append("\\midrule")

    for row in rows:
        values = [str(row.get(h, "—")) for h in headers]
        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparison of concept bottleneck agents across algorithms.}")
    lines.append("\\label{tab:comparison}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def check_success_criteria(results_dir="results"):
    """
    Check the Phase 1 success criteria.

    Returns a list of (criterion, met: bool, details: str).
    """
    eval_data = load_result(os.path.join(results_dir, "evaluation.json"), [])
    interv_ppo = load_result(os.path.join(results_dir, "intervention_ppo.json"))
    interv_dqn = load_result(os.path.join(results_dir, "intervention_dqn.json"))
    ablation_ppo = load_result(os.path.join(results_dir, "ablation_ppo.json"))
    ablation_dqn = load_result(os.path.join(results_dir, "ablation_dqn.json"))

    checks = []

    # 1. Both baselines beat random 60%+
    for variant_name in ["PPO Full Baseline (Go)", "DQN Full Baseline (Go)"]:
        for r in eval_data:
            if r.get("variant") == variant_name:
                met = r.get("win_rate", 0) >= 0.6
                checks.append((
                    f"{variant_name} beats random 60%+",
                    met,
                    f"Win rate: {r.get('win_rate', 0):.1%}",
                ))

    # 2. Both bottleneck agents learn
    for variant_name in ["PPO Bottleneck (Go)", "DQN Bottleneck (Go)"]:
        for r in eval_data:
            if r.get("variant") == variant_name:
                wr = r.get("win_rate", 0)
                met = wr > 0.1  # Any improvement over random
                checks.append((
                    f"{variant_name} learns (win rate improves)",
                    met,
                    f"Win rate: {wr:.1%}",
                ))

    # 3. Concept intervention changes actions >50%
    for name, data in [("PPO", interv_ppo), ("DQN", interv_dqn)]:
        if data:
            cr = data.get("overall_change_rate", 0)
            met = cr > 0.5
            checks.append((
                f"{name} concept intervention >50% change rate",
                met,
                f"Change rate: {cr:.1%}",
            ))

    # 4. 5+ strategies with measurable ablation impact
    for name, data in [("PPO", ablation_ppo), ("DQN", ablation_dqn)]:
        if data:
            n_sig = data.get("n_significant_strategies", 0)
            met = n_sig >= 5
            checks.append((
                f"{name} has 5+ significant strategies",
                met,
                f"Found: {n_sig}",
            ))

    return checks


def compile_all(results_dir="results"):
    """
    Compile all results and save formatted outputs.
    """
    ensure_dir(results_dir)

    # Comparison table
    rows = compile_comparison_table(results_dir)
    md_table = format_markdown_table(rows)
    latex_table = format_latex_table(rows)

    # Success criteria
    criteria = check_success_criteria(results_dir)

    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\n## Comparison Table\n")
    print(md_table)

    print("\n\n## Success Criteria Checklist\n")
    for criterion, met, details in criteria:
        status = "[PASS]" if met else "[FAIL]"
        print(f"  {status} {criterion}: {details}")

    n_passed = sum(1 for _, met, _ in criteria if met)
    n_total = len(criteria)
    print(f"\n  Score: {n_passed}/{n_total} criteria met")

    # Save outputs
    with open(os.path.join(results_dir, "comparison_table.md"), "w") as f:
        f.write(md_table)

    with open(os.path.join(results_dir, "comparison_table.tex"), "w") as f:
        f.write(latex_table)

    summary = {
        "comparison_table": rows,
        "success_criteria": [
            {"criterion": c, "met": m, "details": d}
            for c, m, d in criteria
        ],
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults compiled to {results_dir}/")
    return summary


if __name__ == "__main__":
    compile_all()
