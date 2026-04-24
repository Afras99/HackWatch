"""
Format training log metrics into a clean summary for Claude.

Usage:
  python research/format_metrics.py --log /tmp/train_monitor_v6.log
  python research/format_metrics.py --log /tmp/train_monitor_v6.log --tail 30
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_log(log_path: str) -> list[dict]:
    metrics = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\{['\"]loss['\"].*?\}", line)
            if m:
                try:
                    metrics.append(ast.literal_eval(m.group()))
                except Exception:
                    pass
    return metrics


def format_summary(log_path: str, tail_lines: int = 20) -> str:
    metrics = parse_log(log_path)

    if not metrics:
        return f"No metric dicts found in {log_path}.\nMake sure training has started and `logging_steps=1` is set."

    def _vals(key: str) -> list[float]:
        return [m[key] for m in metrics if key in m]

    rewards      = _vals("reward")
    stds         = _vals("reward_std")
    frac_zero    = _vals("frac_reward_zero_std")
    kls          = _vals("kl")
    clipped      = _vals("completions/clipped_ratio")
    gnorms       = [g for g in _vals("grad_norm") if g == g]  # strip NaN
    losses       = _vals("loss")
    rescued      = _vals("dynamic_sampling/frac_rescued")

    def _last(lst, n=20):
        slc = lst[-n:] if len(lst) >= n else lst
        return round(mean(slc), 4) if slc else None

    def _trend(lst):
        if len(lst) < 5:
            return "insufficient_data"
        d = mean(lst[-10:]) - mean(lst[:10])
        return "rising" if d > 0.05 else "falling" if d < -0.05 else "flat"

    lines = [
        f"# Training Metrics Summary",
        f"Log: {log_path}",
        f"Steps completed: {len(metrics)}",
        "",
        "## Key Metrics (last 20 steps)",
        f"  reward            : {_last(rewards)} (trend: {_trend(rewards)})",
        f"  reward_std        : {_last(stds)}",
        f"  frac_zero_std     : {_last(frac_zero)}   ← 1.0 = all groups degenerate",
        f"  kl                : {_last(kls)}",
        f"  clipped_ratio     : {_last(clipped)}",
        f"  grad_norm (last5) : {round(max(gnorms[-5:]), 4) if gnorms else 'N/A'}",
        f"  loss              : {_last(losses)}",
    ]

    if rescued:
        lines.append(f"  frac_rescued (DS) : {_last(rescued)}   ← DynamicSampling rescue rate")

    # Per-step table (last N)
    lines += ["", f"## Last {tail_lines} Steps"]
    header = f"{'step':>5}  {'reward':>7}  {'frac_zero':>9}  {'kl':>7}  {'grad_norm':>9}  {'loss':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, m in enumerate(metrics[-tail_lines:]):
        step = len(metrics) - tail_lines + i + 1
        lines.append(
            f"{step:>5}  "
            f"{m.get('reward', '?'):>7}  "
            f"{m.get('frac_reward_zero_std', '?'):>9}  "
            f"{m.get('kl', '?'):>7}  "
            f"{m.get('grad_norm', '?'):>9}  "
            f"{m.get('loss', '?'):>8}"
        )

    # Append RESEARCH_AGENT.md as context
    agent_md = Path(__file__).parent / "RESEARCH_AGENT.md"
    if agent_md.exists():
        lines += [
            "",
            "---",
            "# Instructions for Claude (from RESEARCH_AGENT.md)",
            "",
            agent_md.read_text(),
        ]

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Format training log for Claude analysis")
    p.add_argument("--log", required=True, help="Path to training log")
    p.add_argument("--tail", type=int, default=20, help="Number of recent steps to show")
    p.add_argument("--out", default=None, help="Write output to file instead of stdout")
    args = p.parse_args()

    summary = format_summary(args.log, tail_lines=args.tail)

    if args.out:
        Path(args.out).write_text(summary)
        print(f"Written to {args.out}")
        print("Paste the contents into Claude for analysis.")
    else:
        print(summary)


if __name__ == "__main__":
    main()
