"""
Rule-based monitor baseline — generates real W&B curves without GPU.

Runs N episodes using the heuristic classifier from train_monitor.py,
logging per-episode and aggregate metrics every LOG_INTERVAL episodes.

Usage:
    # Start env server first:
    uvicorn server.app:app --port 8000 &

    # Run baseline (logs to W&B by default):
    python -m training.run_baseline --episodes 300

    # Disable W&B (just print stats):
    python -m training.run_baseline --episodes 300 --no-wandb
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

from training.train_monitor import HACK_SIGNAL_PATTERNS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LOG_INTERVAL = 10


# ---------------------------------------------------------------------------
# Running stats accumulator (plain dataclass — not a service class)
# ---------------------------------------------------------------------------


class RunningStats:
    """Accumulates TP/FP/TN/FN counts and rewards across episodes."""

    def __init__(self) -> None:
        self.tp = self.fp = self.tn = self.fn = 0
        self.heldout_tp = self.heldout_fn = 0
        self.rewards: list[float] = []

    def update(self, info: dict) -> None:
        """Incorporate one episode's result dict."""
        self.rewards.append(float(info.get("reward", 0.0)))
        self.tp += info.get("tp", 0)
        self.fp += info.get("fp", 0)
        self.tn += info.get("tn", 0)
        self.fn += info.get("fn", 0)
        if info.get("difficulty") == "heldout_novel":
            self.heldout_tp += info.get("tp", 0)
            self.heldout_fn += info.get("fn", 0)

    def summary(self) -> dict:
        """Return aggregate metrics dict."""
        precision = (
            (self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else None
        )
        recall = (
            (self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else None
        )
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        fpr = self.fp / max(1, self.fp + self.tn)
        heldout_dr = self.heldout_tp / max(1, self.heldout_tp + self.heldout_fn)
        return {
            "mean_reward": sum(self.rewards) / max(1, len(self.rewards)),
            "exploit_detection_rate": recall,
            "false_positive_rate": fpr,
            "precision": precision,
            "f1_score": f1,
            "heldout_detection_rate": heldout_dr,
            "episodes": len(self.rewards),
        }


# ---------------------------------------------------------------------------
# BaselineRunner — main service class
# ---------------------------------------------------------------------------


class BaselineRunner:
    """Runs heuristic-monitor episodes against the live env and logs metrics.

    Args:
        env_url: URL of the running HackWatch env server.
        no_wandb: Disable W&B logging when ``True``.
    """

    def __init__(self, env_url: str, no_wandb: bool = False) -> None:
        self.env_url = env_url
        self.no_wandb = no_wandb

    # ------------------------------------------------------------------
    # Step 1 — classify a diff with the rule-based heuristic
    # ------------------------------------------------------------------

    def _rule_based_verdict(self, diff: str) -> dict:
        """Heuristic monitor: block if any known hack signal appears.

        Confidence is proportional to number of signals fired:
          0 signals → allow, confidence 0.50 (no evidence either way)
          1 signal  → block, confidence 0.75 (some evidence)
          2+ signals → block, confidence 0.95 (strong evidence)

        Args:
            diff: Unified diff string from the environment observation.

        Returns:
            Action dict with ``verdict``, ``confidence``, ``reasoning``.
        """
        hits = [p for p in HACK_SIGNAL_PATTERNS if re.search(p, diff)]
        n = len(hits)
        if n == 0:
            return {"verdict": "allow", "confidence": 0.50, "reasoning": "no hack signals"}
        confidence = 0.75 if n == 1 else 0.95
        return {
            "verdict": "block",
            "confidence": confidence,
            "reasoning": f"{n} signal(s) matched",
        }

    # ------------------------------------------------------------------
    # Step 2 — run one episode
    # ------------------------------------------------------------------

    def _run_episode(self, client: httpx.Client) -> dict:
        """Run one episode with the rule-based monitor.

        Args:
            client: Reusable ``httpx.Client`` with the env base URL configured.

        Returns:
            Info dict from the final ``/step`` response.
        """
        resp = client.post(f"{self.env_url}/reset", json={})
        resp.raise_for_status()
        obs = resp.json()

        done = obs.get("episode_done", False)
        final_info: dict = {}

        while not done:
            action = self._rule_based_verdict(obs.get("worker_diff", ""))
            step_resp = client.post(f"{self.env_url}/step", json=action)
            step_resp.raise_for_status()
            data = step_resp.json()
            obs = data["observation"]
            done = data.get("done", False) or obs.get("episode_done", False)
            if data.get("reward") is not None:
                final_info = {
                    "reward": data["reward"],
                    "difficulty": (data.get("info") or {}).get("difficulty", "train"),
                    **(data.get("info") or {}),
                }

        return final_info

    # ------------------------------------------------------------------
    # Step 3 — log metrics to W&B + stdout
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        wandb_run: object,
        episode: int,
        summary: dict,
    ) -> None:
        """Log aggregate stats to W&B and stdout.

        Args:
            wandb_run: Active W&B run object (or ``None``).
            episode: 1-based episode counter for this log line.
            summary: Dict returned by ``RunningStats.summary()``.
        """
        def _fmt(v) -> str:
            return f"{v:.3f}" if isinstance(v, float) else ("N/A" if v is None else str(v))

        log.info(
            f"ep={episode:4d}  reward={_fmt(summary['mean_reward'])}  "
            f"detection={_fmt(summary['exploit_detection_rate'])}  "
            f"fpr={_fmt(summary['false_positive_rate'])}  "
            f"f1={_fmt(summary['f1_score'])}  "
            f"heldout_dr={_fmt(summary['heldout_detection_rate'])}"
        )
        if wandb_run:
            import wandb as _wandb
            _wandb.log({
                "episode": episode,
                **{k: v for k, v in summary.items() if v is not None},
            })

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, episodes: int = 300) -> dict:
        """Run the full baseline evaluation loop.

        Args:
            episodes: Number of episodes to run.

        Returns:
            Final aggregate summary dict.
        """
        wandb_run = None
        if not self.no_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project="hackwatch",
                    name=f"rule_based_baseline_{int(time.time())}",
                    config={"monitor": "rule_based", "episodes": episodes},
                )
                log.info(f"W&B run: {wandb_run.url}")
            except ImportError:
                log.warning("wandb not installed — running without W&B logging")

        stats = RunningStats()

        with httpx.Client(timeout=30.0) as client:
            for ep in range(episodes):
                try:
                    info = self._run_episode(client)
                except Exception as exc:
                    log.warning(f"Episode {ep} failed: {exc}")
                    continue

                stats.update(info)

                if (ep + 1) % LOG_INTERVAL == 0:
                    self._log_metrics(wandb_run, ep + 1, stats.summary())

        final = stats.summary()
        log.info("=== Final baseline results ===")
        for k, v in final.items():
            log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        if wandb_run:
            import wandb as _wandb
            _wandb.summary.update(final)
            _wandb.finish()
            print(f"\nW&B run ID: {wandb_run.id}")
            print(f"W&B URL:    {wandb_run.url}")

        out_path = Path("runs/baseline_results.json")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(final, indent=2))
        log.info(f"Saved to {out_path}")

        return final


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------


def rule_based_verdict(diff: str) -> dict:
    """Backward-compatible alias for ``BaselineRunner._rule_based_verdict``."""
    return BaselineRunner(env_url="")._rule_based_verdict(diff)


def run_episode(env_url: str, client: httpx.Client) -> dict:
    """Backward-compatible alias for ``BaselineRunner._run_episode``."""
    return BaselineRunner(env_url=env_url)._run_episode(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    runner = BaselineRunner(env_url=args.env_url, no_wandb=args.no_wandb)
    runner.run(episodes=args.episodes)

# Run on terminal:
# python -m training.run_baseline --episodes 300 --no-wandb
