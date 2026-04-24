"""
Self-improving training loop — the full Karpathy auto-research cycle.

Each iteration:
  1. Build dataset with current UCB curriculum stats
  2. Train for --steps steps (or until done)
  3. Run auto_research: diagnose → literature → proposals → apply GUARDRAILS-safe patches
  4. If patches were applied and budget remains, loop (v+1 run with updated config)
  5. Stop when: (a) no patches applied, (b) max iterations reached, or (c) reward
     plateau (healthy diagnosis and reward > 0.97 for 3 consecutive runs)

Usage:
  # Full autonomous loop (recommended):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \\
  python -m training.auto_loop \\
      --env-url http://localhost:<port> \\
      --base-dir ./runs \\
      --max-iterations 5 \\
      --steps-per-run 400

  # Dry run (no GPU needed — verifies pipeline only):
  python -m training.auto_loop --dry-run --env-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


RESEARCH_LOG = Path("research/log.jsonl")
LOG_DIR = Path("/tmp")


def _next_version(base_dir: str) -> int:
    """Find next unused monitor version number under base_dir."""
    base = Path(base_dir)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("monitor_v")] if base.exists() else []
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_v")[-1]))
        except ValueError:
            pass
    return max(nums, default=2) + 1


def _run_training(
    version: int,
    env_url: str,
    base_dir: str,
    steps: int,
    no_wandb: bool,
    dry_run: bool,
) -> tuple[str, str]:
    """Launch training subprocess. Returns (output_dir, log_path)."""
    out_dir = str(Path(base_dir) / f"monitor_v{version}")
    log_path = str(LOG_DIR / f"train_monitor_v{version}.log")

    cmd = [
        sys.executable, "-m", "training.train_monitor",
        "--env-url", env_url,
        "--output-dir", out_dir,
        "--max-steps", str(steps),
    ]
    if no_wandb:
        cmd.append("--no-wandb")
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'='*60}")
    print(f"[auto_loop] Starting monitor_v{version}")
    print(f"  out_dir:  {out_dir}")
    print(f"  log:      {log_path}")
    print(f"  steps:    {steps}")
    print(f"{'='*60}\n")

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )

    if proc.returncode != 0:
        print(f"[auto_loop] WARNING: training exited with code {proc.returncode}")

    return out_dir, log_path


def _run_auto_research(log_path: str) -> dict:
    """Snapshot metrics for Claude review. Returns last log entry."""
    cmd = [sys.executable, "-m", "research.auto_research", "--log", log_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[auto_loop] auto_research error: {result.stderr[:500]}")

    try:
        lines = RESEARCH_LOG.read_text().strip().splitlines()
        return json.loads(lines[-1]) if lines else {}
    except Exception:
        return {}


def _get_mean_reward(log_path: str) -> float:
    """Parse last 20 steps of reward from training log."""
    import ast, re
    rewards = []
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\{['\"]loss['\"].*?\}", line)
                if m:
                    try:
                        d = ast.literal_eval(m.group())
                        if "reward" in d:
                            rewards.append(d["reward"])
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    if not rewards:
        return 0.0
    return sum(rewards[-20:]) / max(1, len(rewards[-20:]))


def main():
    p = argparse.ArgumentParser(description="Self-improving GRPO training loop")
    p.add_argument("--env-url",       default="http://localhost:8000")
    p.add_argument("--base-dir",      default="./runs")
    p.add_argument("--max-iterations",type=int, default=5)
    p.add_argument("--steps-per-run", type=int, default=400)
    p.add_argument("--no-wandb",      action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--start-version", type=int, default=None,
                   help="Start from this version number (default: auto-detect)")
    args = p.parse_args()

    version = args.start_version or _next_version(args.base_dir)
    plateau_count = 0  # consecutive "healthy" / no-patch runs at high reward
    last_reward = 0.0

    print(f"[auto_loop] Starting self-improving loop")
    print(f"  max_iterations: {args.max_iterations}")
    print(f"  steps_per_run:  {args.steps_per_run}")
    print(f"  start_version:  v{version}")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration}/{args.max_iterations} — monitor_v{version}")
        print(f"{'#'*60}")

        # --- Train ---
        out_dir, log_path = _run_training(
            version=version,
            env_url=args.env_url,
            base_dir=args.base_dir,
            steps=args.steps_per_run,
            no_wandb=args.no_wandb,
            dry_run=args.dry_run,
        )

        mean_reward = _get_mean_reward(log_path)
        print(f"\n[auto_loop] Training done. mean_reward_last20 = {mean_reward:.4f}")

        # --- Snapshot metrics for Claude review ---
        entry = _run_auto_research(log_path=log_path)

        diagnosis = entry.get("diagnosis", "unknown")
        applied = []  # patches are now applied manually via research/apply_patch.py
        papers = []

        print(f"\n[auto_loop] Iteration {iteration} summary:")
        print(f"  diagnosis:     {diagnosis}")
        print(f"  mean_reward:   {mean_reward:.4f}")
        print(f"  snapshot:      research/snapshots/train_monitor_v{version}.md")
        print(f"  → Ask Claude to review the snapshot and apply patches before next run.")

        # --- Stopping criteria ---
        # Plateau: high reward for 2 consecutive runs
        if mean_reward >= 0.97:
            plateau_count += 1
            print(f"[auto_loop] Plateau count: {plateau_count}/2")
            if plateau_count >= 2:
                print(f"\n[auto_loop] High reward plateau — review snapshot then decide next step.")
                print(f"  Best checkpoint: {out_dir}/final")
                break
        else:
            plateau_count = 0

        # Low reward — stop and flag for manual review
        if mean_reward < 0.5 and not args.dry_run:
            print(f"[auto_loop] Low reward ({mean_reward:.4f}) — review snapshot for diagnosis.")
            break

        # Last iteration
        if iteration == args.max_iterations:
            print(f"\n[auto_loop] Max iterations reached.")
            break

        version += 1
        last_reward = mean_reward
        print(f"[auto_loop] Continuing to v{version}")

    print(f"\n[auto_loop] Done. Final version: monitor_v{version-1}")
    print(f"  Run eval: python eval/evaluate_monitor.py --trajectories data/trajectories.jsonl")


if __name__ == "__main__":
    main()
