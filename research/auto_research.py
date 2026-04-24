"""
Auto-research: formats training metrics and saves them for review.

After each training run, this is called by auto_loop.py to snapshot
the metrics. Claude (Claude Code) then reads these when you ask for
analysis, following research/RESEARCH_AGENT.md.

Usage:
  python research/auto_research.py --log /tmp/train_monitor_v6.log
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.format_metrics import parse_log, format_summary

RESEARCH_LOG = Path("research/log.jsonl")
SNAPSHOTS_DIR = Path("research/snapshots")


def run_once(log_path: str) -> dict:
    metrics = parse_log(log_path)
    n = len(metrics)

    if not metrics:
        print(f"[auto_research] No metrics found in {log_path}")
        entry = {"timestamp": time.time(), "log_path": log_path, "steps": 0}
    else:
        rewards = [m.get("reward", 0) for m in metrics]
        frac_zero = [m.get("frac_reward_zero_std", 1.0) for m in metrics]
        last20_reward = sum(rewards[-20:]) / max(1, len(rewards[-20:]))
        last20_fz = sum(frac_zero[-20:]) / max(1, len(frac_zero[-20:]))

        entry = {
            "timestamp": time.time(),
            "log_path": log_path,
            "steps": n,
            "mean_reward_last20": round(last20_reward, 4),
            "frac_zero_std_last20": round(last20_fz, 4),
        }

        print(f"[auto_research] {log_path}: {n} steps, "
              f"reward={last20_reward:.4f}, frac_zero_std={last20_fz:.4f}")

    # Save formatted snapshot for Claude to read
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    run_name = Path(log_path).stem
    snap_path = SNAPSHOTS_DIR / f"{run_name}.md"
    snap_path.write_text(format_summary(log_path, tail_lines=30))
    print(f"[auto_research] Snapshot saved → {snap_path}")
    print(f"[auto_research] To analyze: show Claude the snapshot or run:")
    print(f"  python research/format_metrics.py --log {log_path}")

    # Append to log
    RESEARCH_LOG.parent.mkdir(exist_ok=True)
    with open(RESEARCH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    args = p.parse_args()
    run_once(args.log)


if __name__ == "__main__":
    main()
