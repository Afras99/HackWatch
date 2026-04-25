"""
Parse cotrain_v2 W&B output.log and generate training curve PNGs for submission.
Outputs: docs/plots/reward_curve.png, docs/plots/loss_curve.png
"""
import ast
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG = Path(__file__).parent.parent / "wandb/run-20260425_031047-w9osws3o/files/output.log"
OUT = Path(__file__).parent.parent / "docs/plots"
OUT.mkdir(parents=True, exist_ok=True)

AMBER = "#e8a000"
AMBER_DARK = "#b07000"
DIM = "#555555"

steps, rewards, losses, kls = [], [], [], []

for line in LOG.read_text().splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        d = ast.literal_eval(line)
    except Exception:
        continue
    if "reward" not in d or "loss" not in d:
        continue
    steps.append(len(steps) + 1)
    rewards.append(float(d["reward"]))
    losses.append(float(d["loss"]))
    kls.append(float(d.get("kl", 0)))

steps = np.array(steps)
rewards = np.array(rewards)
losses = np.array(losses)

# EMA smoothing
def ema(values, alpha=0.12):
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

reward_ema = ema(rewards)

# --- Reward curve ---
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")

ax.plot(steps, rewards, color=DIM, linewidth=0.7, alpha=0.6, label="Raw reward")
ax.plot(steps, reward_ema, color=AMBER, linewidth=2.2, label="EMA reward (α=0.12)")

# Shade under EMA
ax.fill_between(steps, 0, reward_ema, color=AMBER, alpha=0.08)

# Warmup annotation
ax.axvline(x=30, color="#555577", linewidth=1, linestyle="--", alpha=0.7)
ax.text(32, 0.05, "warmup end", color="#7777aa", fontsize=8)

# Final value annotation
final_ema = reward_ema[-1]
ax.annotate(
    f"  final EMA: {final_ema:.3f}",
    xy=(steps[-1], final_ema),
    color=AMBER,
    fontsize=9,
    va="center",
)

ax.set_xlim(1, steps[-1])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Training Step", color="#aaaaaa", fontsize=11)
ax.set_ylabel("GRPO Reward", color="#aaaaaa", fontsize=11)
ax.set_title("HackWatch — Monitor Reward During GRPO Training (cotrain_v2, 300 steps)",
             color="#dddddd", fontsize=12, pad=12)
ax.tick_params(colors="#777777")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

legend = ax.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#444444",
                   labelcolor="#cccccc", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "reward_curve.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved {OUT / 'reward_curve.png'}")

# --- Loss curve ---
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")

loss_ema = ema(losses, alpha=0.15)
ax.plot(steps, losses, color=DIM, linewidth=0.7, alpha=0.6, label="Raw loss")
ax.plot(steps, loss_ema, color="#4488ff", linewidth=2.0, label="EMA loss")
ax.fill_between(steps, 0, loss_ema, color="#4488ff", alpha=0.06)

ax.set_xlim(1, steps[-1])
ax.set_ylim(0, max(losses) * 1.15)
ax.set_xlabel("Training Step", color="#aaaaaa", fontsize=11)
ax.set_ylabel("DR-GRPO Loss", color="#aaaaaa", fontsize=11)
ax.set_title("HackWatch — Training Loss (cotrain_v2, 300 steps)",
             color="#dddddd", fontsize=12, pad=12)
ax.tick_params(colors="#777777")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

ax.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#444444",
          labelcolor="#cccccc", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "loss_curve.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved {OUT / 'loss_curve.png'}")
print(f"\nParsed {len(steps)} training steps.")
print(f"Reward: min={rewards.min():.3f}  max={rewards.max():.3f}  final_ema={reward_ema[-1]:.3f}")
