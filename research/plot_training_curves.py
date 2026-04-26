"""
Generate publication-quality training curves for HackWatch demo/judging.

Usage:
    python research/plot_training_curves.py \
        --runs runs/monitor_v9 runs/monitor_v16 \
        --labels "3B baseline" "1.5B+temp1.3" \
        --out research/training_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> list[dict]:
    """Load all log_history entries from the latest checkpoint in a run dir."""
    run_path = Path(run_dir)
    states = sorted(run_path.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        # Try root-level trainer_state.json
        root = run_path / "trainer_state.json"
        if root.exists():
            states = [root]
        else:
            raise FileNotFoundError(f"No trainer_state.json found under {run_dir}")
    data = json.loads(states[-1].read_text())
    return data["log_history"]


def smooth(values: list[float], window: int = 20) -> list[float]:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[:len(values)])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

PANEL_METRICS = [
    ("reward",          "Mean Reward",           "tab:blue",   True),
    ("reward_std",      "Within-group Reward Std","tab:orange", False),
    ("kl",              "KL Divergence",          "tab:red",    False),
    ("entropy",         "Entropy",                "tab:green",  False),
    ("completions/mean_length", "Mean Completion Length (tokens)", "tab:purple", False),
    ("frac_reward_zero_std", "Frac Zero-Std Groups", "tab:brown", False),
]


def plot_curves(
    runs: list[str],
    labels: list[str],
    out: str,
    smooth_window: int = 20,
) -> None:
    n_panels = len(PANEL_METRICS)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    all_histories = []
    for run_dir in runs:
        try:
            all_histories.append(load_run(run_dir))
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            all_histories.append([])

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for panel_idx, (metric, title, _, do_smooth) in enumerate(PANEL_METRICS):
        ax = axes[panel_idx]
        for run_idx, (history, label) in enumerate(zip(all_histories, labels)):
            steps = [e["step"] for e in history if metric in e]
            vals  = [e[metric] for e in history if metric in e]
            if not steps:
                continue
            color = colors[run_idx % len(colors)]
            vals_smoothed = smooth(vals, smooth_window) if do_smooth else vals
            ax.plot(steps, vals, alpha=0.2, color=color, linewidth=0.8)
            ax.plot(steps, vals_smoothed, label=label, color=color, linewidth=2.0)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Step", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.grid(True, alpha=0.3, linestyle="--")
        if panel_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused panels
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("HackWatch Monitor — GRPO Training Curves", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",   nargs="+", default=["runs/monitor_v9"])
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out",    default="research/training_curves.png")
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) < len(args.runs):
        labels += [Path(r).name for r in args.runs[len(labels):]]

    plot_curves(args.runs, labels, args.out, args.smooth)


if __name__ == "__main__":
    main()
