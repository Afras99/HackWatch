"""
Experiment analyzer — parses training logs and W&B to produce a health report.

Usage:
  python research/experiment_analyzer.py --log /tmp/train_monitor_v2b.log
  python research/experiment_analyzer.py --log /tmp/train_monitor_v2b.log --wandb-run soxdq4iy
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def parse_log(log_path: str) -> list[dict]:
    """Extract metric dicts from a training log file."""
    metrics = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\{['\"]loss['\"].*?\}", line)
            if m:
                try:
                    d = ast.literal_eval(m.group())
                    metrics.append(d)
                except Exception:
                    pass
    return metrics


def classify_trend(values: list[float]) -> str:
    if len(values) < 5:
        return "insufficient_data"
    recent = values[-10:]
    early = values[:10]
    delta = mean(recent) - mean(early)
    if delta > 0.05:
        return "rising"
    if delta < -0.05:
        return "falling"
    return "flat"


def diagnose(metrics: list[dict]) -> dict[str, Any]:
    if not metrics:
        return {"diagnosis": "no_data", "steps_completed": 0}

    rewards = [m.get("reward", 0) for m in metrics]
    stds = [m.get("reward_std", 0) for m in metrics]
    frac_zero = [m.get("frac_reward_zero_std", 1.0) for m in metrics]
    clipped = [m.get("completions/clipped_ratio", 1.0) for m in metrics]
    kls = [m.get("kl", 0) for m in metrics]
    gnorms = [m.get("grad_norm", 0) for m in metrics if m.get("grad_norm") is not None]
    gnorms = [g for g in gnorms if g == g]  # filter NaN

    last20_reward = mean(rewards[-20:]) if len(rewards) >= 20 else mean(rewards)
    last20_frac_zero = mean(frac_zero[-20:]) if len(frac_zero) >= 20 else mean(frac_zero)
    last_kl = kls[-1] if kls else 0
    last_clipped = clipped[-1] if clipped else 0

    # Diagnose
    diagnosis = "healthy"
    recommended_action = "continue"

    if last_clipped > 0.3:
        diagnosis = "truncation"
        recommended_action = "raise_max_completion_length"
    elif last_kl > 0.5:
        diagnosis = "diverging"
        recommended_action = "raise_beta"
    elif gnorms and max(gnorms[-5:]) > 50:
        diagnosis = "exploding_gradients"
        recommended_action = "halve_lr"
    elif last20_frac_zero > 0.8 and last20_reward > 0.9:
        diagnosis = "ceiling_hit"
        recommended_action = "finer_reward_or_harder_tasks"
    elif last20_frac_zero > 0.8 and last20_reward < 0.5:
        diagnosis = "collapsed"
        recommended_action = "lower_beta_or_sft_warmstart"
    elif last20_frac_zero > 0.5:
        diagnosis = "low_diversity"
        recommended_action = "raise_num_generations_or_curriculum"
    elif classify_trend(rewards) == "flat" and last20_reward < 0.7:
        diagnosis = "learning_slowly"
        recommended_action = "dr_grpo_loss_or_curriculum"

    return {
        "steps_completed": len(metrics),
        "reward_trend": classify_trend(rewards),
        "mean_reward_last20": round(last20_reward, 4),
        "mean_reward_std_last20": round(mean(stds[-20:]) if len(stds) >= 20 else mean(stds) if stds else 0, 4),
        "frac_zero_std_last20": round(last20_frac_zero, 4),
        "clipped_ratio_last": round(last_clipped, 4),
        "kl_last": round(last_kl, 6),
        "max_grad_norm_last5": round(max(gnorms[-5:]), 4) if gnorms else None,
        "diagnosis": diagnosis,
        "recommended_action": recommended_action,
    }


def fetch_wandb(run_id: str, project: str = "afrasvellora777-student/hackwatch") -> list[dict]:
    """Fetch run history from W&B API."""
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")
        hist = run.history(samples=500)
        return hist.to_dict(orient="records")
    except Exception as e:
        print(f"[wandb] fetch failed: {e}", file=sys.stderr)
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Path to training log file")
    p.add_argument("--wandb-run", default=None, help="W&B run ID to merge")
    p.add_argument("--out", default=None, help="Output JSON path (default: print)")
    args = p.parse_args()

    metrics = parse_log(args.log)
    if args.wandb_run:
        wandb_metrics = fetch_wandb(args.wandb_run)
        if wandb_metrics:
            metrics = wandb_metrics  # W&B is authoritative if available

    report = diagnose(metrics)
    report["log_path"] = args.log
    report["wandb_run"] = args.wandb_run

    out = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(out)
        print(f"Report written to {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
