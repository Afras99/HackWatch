"""
Live training monitor — tails a log file and applies patches when kill criteria trigger.

Usage:
  python research/live_monitor.py --log /tmp/train_monitor_v2b.log [--dry-run]

Kill criteria (from GUARDRAILS.md + literature):
  - clipped_ratio > 0.3 for 5+ steps → raise max_completion_length
  - frac_reward_zero_std = 1.0 for 20+ steps AND reward > 0.9 → ceiling hit, log it
  - grad_norm > 50 → log explosion warning
  - kl > 0.5 → log divergence warning
  - reward_std = 0 for 20+ steps AND reward < 0.4 → collapsed, log it

Note: This monitor LOGS and REPORTS. It does not auto-edit training configs mid-run
(would require process signaling that is fragile). Instead it writes a health report
to research/live_health.json every 30 seconds, which auto_research.py can act on
between training runs.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import deque
from pathlib import Path


HEALTH_OUTPUT = Path("research/live_health.json")
CHECK_INTERVAL = 30  # seconds


def parse_last_metrics(log_path: str, window: int = 30) -> list[dict]:
    metrics = []
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\{['\"]loss['\"].*?\}", line)
                if m:
                    try:
                        metrics.append(ast.literal_eval(m.group()))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return metrics[-window:]


def check_kill_criteria(metrics: list[dict]) -> list[str]:
    if not metrics:
        return []

    alerts = []
    recent = metrics[-5:]
    long_window = metrics[-20:] if len(metrics) >= 20 else metrics

    # Clipping
    if all(m.get("completions/clipped_ratio", 0) > 0.3 for m in recent):
        alerts.append("TRUNCATION: clipped_ratio > 0.3 for last 5 steps → raise max_completion_length +128")

    # Gradient explosion
    for m in recent:
        gn = m.get("grad_norm")
        if gn and gn == gn and gn > 50:  # not NaN
            alerts.append(f"EXPLOSION: grad_norm={gn:.1f} > 50 → halve learning_rate")

    # KL divergence
    last_kl = metrics[-1].get("kl", 0)
    if last_kl > 0.5:
        alerts.append(f"DIVERGENCE: kl={last_kl:.4f} > 0.5 → raise beta toward 0.08")

    # Ceiling hit
    long_frac_zero = [m.get("frac_reward_zero_std", 0) for m in long_window]
    long_reward = [m.get("reward", 0) for m in long_window]
    if len(long_frac_zero) >= 20:
        if sum(f >= 0.9 for f in long_frac_zero) >= 18 and sum(r >= 0.9 for r in long_reward) >= 18:
            alerts.append(
                "CEILING: frac_zero_std≈1.0 and reward≈1.0 for 20+ steps. "
                "Model has mastered heuristic reward. Next run needs finer reward or curriculum."
            )

    # Collapse
    if len(long_reward) >= 20:
        if sum(f >= 0.9 for f in long_frac_zero) >= 18 and sum(r < 0.4 for r in long_reward) >= 18:
            alerts.append(
                "COLLAPSE: frac_zero_std≈1.0 and reward < 0.4 for 20+ steps. "
                "Lower beta to 0.01 or run SFT warmstart first."
            )

    return alerts


def compute_summary(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    rewards = [m.get("reward", 0) for m in metrics]
    kls = [m.get("kl", 0) for m in metrics]
    return {
        "steps": len(metrics),
        "last_step_reward": round(rewards[-1], 4),
        "mean_reward_last20": round(sum(rewards[-20:]) / max(1, len(rewards[-20:])), 4),
        "last_kl": round(kls[-1], 6),
        "last_clipped_ratio": round(metrics[-1].get("completions/clipped_ratio", 0), 4),
        "last_frac_zero_std": round(metrics[-1].get("frac_reward_zero_std", 0), 4),
    }


def monitor_loop(log_path: str, dry_run: bool = False):
    print(f"[live_monitor] watching {log_path} (interval={CHECK_INTERVAL}s)")
    seen_alerts: set[str] = set()

    while True:
        metrics = parse_last_metrics(log_path)
        alerts = check_kill_criteria(metrics)
        summary = compute_summary(metrics)

        health = {"summary": summary, "alerts": alerts, "timestamp": time.time()}
        HEALTH_OUTPUT.write_text(json.dumps(health, indent=2))

        new_alerts = [a for a in alerts if a not in seen_alerts]
        for alert in new_alerts:
            print(f"\n🚨 {alert}")
            seen_alerts.add(alert)

        if summary:
            step = summary.get("steps", 0)
            reward = summary.get("last_step_reward", 0)
            kl = summary.get("last_kl", 0)
            sys.stdout.write(f"\r[step {step:>3}] reward={reward:.3f}  kl={kl:.5f}  alerts={len(alerts)}")
            sys.stdout.flush()

        time.sleep(CHECK_INTERVAL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--dry-run", action="store_true", help="Run once and exit")
    args = p.parse_args()

    if args.dry_run:
        metrics = parse_last_metrics(args.log)
        alerts = check_kill_criteria(metrics)
        summary = compute_summary(metrics)
        print(json.dumps({"summary": summary, "alerts": alerts}, indent=2))
    else:
        monitor_loop(args.log, dry_run=False)


if __name__ == "__main__":
    main()
