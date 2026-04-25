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
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


RESEARCH_LOG = Path("research/log.jsonl")
LOG_DIR = Path("runs/logs")


class AutoLoop:
    """Self-improving GRPO training loop.

    Each iteration trains the monitor, runs auto_research diagnostics, and
    optionally applies GUARDRAILS-safe patches before the next version.
    Stops when the reward plateaus, drops too low, or ``max_iterations`` is
    reached.

    All config is stored in ``__init__``; no config is loaded inside methods.

    Args:
        env_url: URL of the running HackWatch env server.
        base_dir: Root directory for monitor run outputs.
        max_iterations: Maximum number of training iterations.
        steps_per_run: Training steps per iteration.
        no_wandb: Disable W&B logging when ``True``.
        dry_run: Validate pipeline without GPU training when ``True``.
        start_version: Override the starting version number (auto-detected when ``None``).
    """

    def __init__(
        self,
        env_url: str = "http://localhost:8000",
        base_dir: str = "./runs",
        max_iterations: int = 5,
        steps_per_run: int = 400,
        no_wandb: bool = False,
        dry_run: bool = False,
        start_version: int | None = None,
    ) -> None:
        self.env_url = env_url
        self.base_dir = base_dir
        self.max_iterations = max_iterations
        self.steps_per_run = steps_per_run
        self.no_wandb = no_wandb
        self.dry_run = dry_run
        self.start_version = start_version

    # ------------------------------------------------------------------
    # Step 1 — determine next unused version number
    # ------------------------------------------------------------------

    def _next_version(self) -> int:
        """Find the next unused ``monitor_vN`` version number under ``base_dir``.

        Returns:
            Integer version number one higher than the current maximum.
        """
        base = Path(self.base_dir)
        existing = (
            [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("monitor_v")]
            if base.exists()
            else []
        )
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_v")[-1]))
            except ValueError:
                pass
        return max(nums, default=0) + 1

    # ------------------------------------------------------------------
    # Step 2 — launch training subprocess for one version
    # ------------------------------------------------------------------

    def _run_training(self, version: int) -> tuple[str, str]:
        """Launch the training subprocess for ``monitor_v{version}``.

        Args:
            version: Version integer; output goes to ``{base_dir}/monitor_v{version}``.

        Returns:
            ``(output_dir, log_path)`` tuple.
        """
        out_dir = str(Path(self.base_dir) / f"monitor_v{version}")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = str(LOG_DIR / f"train_monitor_v{version}.log")

        cmd = [
            sys.executable, "-m", "training.train_monitor",
            "--env-url", self.env_url,
            "--output-dir", out_dir,
            "--max-steps", str(self.steps_per_run),
        ]
        if self.no_wandb:
            cmd.append("--no-wandb")
        if self.dry_run:
            cmd.append("--dry-run")

        print(f"\n{'='*60}")
        print(f"[auto_loop] Starting monitor_v{version}")
        print(f"  out_dir:  {out_dir}")
        print(f"  log:      {log_path}")
        print(f"  steps:    {self.steps_per_run}")
        print(f"{'='*60}\n")

        _timeout_s = self.steps_per_run * 12 + 600
        with open(log_path, "w") as log_f:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ},
                    timeout=_timeout_s,
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[auto_loop] WARNING: training timed out after {_timeout_s}s — killing"
                )
                return out_dir, log_path

        if proc.returncode != 0:
            print(f"[auto_loop] WARNING: training exited with code {proc.returncode}")

        return out_dir, log_path

    # ------------------------------------------------------------------
    # Step 3 — run auto_research diagnostics
    # ------------------------------------------------------------------

    def _run_auto_research(self, log_path: str) -> dict:
        """Snapshot metrics for Claude review.

        Args:
            log_path: Path to the training log file.

        Returns:
            Last entry from the research log, or empty dict on failure.
        """
        cmd = [sys.executable, "-m", "research.auto_research", "--log", log_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            err_msg = result.stderr[:2000]
            print(f"[auto_loop] auto_research error: {err_msg}")
            err_path = Path(log_path).with_suffix(".research_err.txt")
            err_path.write_text(err_msg)

        try:
            lines = RESEARCH_LOG.read_text().strip().splitlines()
            return json.loads(lines[-1]) if lines else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Step 4 — parse mean reward from training log
    # ------------------------------------------------------------------

    def _get_mean_reward(self, log_path: str) -> float | None:
        """Parse mean of last 20 reward values from the training log.

        Returns ``None`` when the log is missing or contains no reward lines,
        so callers can distinguish a parse failure from a genuinely bad run.

        Args:
            log_path: Path to the training log file.

        Returns:
            Mean reward float or ``None``.
        """
        rewards = []
        try:
            with open(log_path) as f:
                for line in f:
                    m = re.search(r"\{.*\}", line)
                    if m:
                        try:
                            d = ast.literal_eval(m.group())
                            if isinstance(d, dict) and "reward" in d:
                                rewards.append(float(d["reward"]))
                        except Exception:
                            pass
        except FileNotFoundError:
            return None
        if not rewards:
            return None
        return sum(rewards[-20:]) / len(rewards[-20:])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full self-improving loop up to ``max_iterations`` times."""
        version = self.start_version or self._next_version()
        plateau_count = 0

        print("[auto_loop] Starting self-improving loop")
        print(f"  max_iterations: {self.max_iterations}")
        print(f"  steps_per_run:  {self.steps_per_run}")
        print(f"  start_version:  v{version}")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'#'*60}")
            print(f"# Iteration {iteration}/{self.max_iterations} — monitor_v{version}")
            print(f"{'#'*60}")

            out_dir, log_path = self._run_training(version=version)

            mean_reward = self._get_mean_reward(log_path)
            if mean_reward is None:
                print(
                    "[auto_loop] WARNING: could not parse reward from log "
                    "— skipping stopping criteria this iteration"
                )
                mean_reward_display = "N/A (parse failure)"
            else:
                mean_reward_display = f"{mean_reward:.4f}"
            print(f"\n[auto_loop] Training done. mean_reward_last20 = {mean_reward_display}")

            entry = self._run_auto_research(log_path=log_path)
            diagnosis = entry.get("diagnosis", "unknown")

            print(f"\n[auto_loop] Iteration {iteration} summary:")
            print(f"  diagnosis:     {diagnosis}")
            print(f"  mean_reward:   {mean_reward_display}")
            print(
                f"  snapshot:      research/snapshots/train_monitor_v{version}.md"
            )
            print(
                "  → Ask Claude to review the snapshot and apply patches before next run."
            )

            if mean_reward is None:
                version += 1
                print(f"[auto_loop] Reward parse failed — continuing to v{version}")
                continue

            if mean_reward >= 0.97:
                plateau_count += 1
                print(f"[auto_loop] Plateau count: {plateau_count}/2")
                if plateau_count >= 2:
                    print(
                        "\n[auto_loop] High reward plateau — review snapshot then decide next step."
                    )
                    print(f"  Best checkpoint: {out_dir}/final")
                    break
            else:
                plateau_count = 0

            if mean_reward < 0.5 and not self.dry_run:
                print(
                    f"[auto_loop] Low reward ({mean_reward:.4f}) — review snapshot for diagnosis."
                )
                break

            if iteration == self.max_iterations:
                print("\n[auto_loop] Max iterations reached.")
                break

            version += 1
            print(f"[auto_loop] Continuing to v{version}")

        print(f"\n[auto_loop] Done. Final version: monitor_v{version - 1}")
        print(
            "  Run eval: python eval/evaluate_monitor.py "
            "--trajectories data/trajectories.jsonl"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Self-improving GRPO training loop")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--base-dir", default="./runs")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--steps-per-run", type=int, default=400)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--start-version", type=int, default=None,
        help="Start from this version number (default: auto-detect)",
    )
    args = p.parse_args()

    AutoLoop(
        env_url=args.env_url,
        base_dir=args.base_dir,
        max_iterations=args.max_iterations,
        steps_per_run=args.steps_per_run,
        no_wandb=args.no_wandb,
        dry_run=args.dry_run,
        start_version=args.start_version,
    ).run()

# Run on terminal:
# python -m training.auto_loop --dry-run --env-url http://localhost:8000

