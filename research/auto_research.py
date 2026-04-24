"""
Auto-research orchestrator — runs the full Karpathy-style loop:
  1. Analyze current training run metrics
  2. Diagnose the issue
  3. Generate paper-backed hypotheses
  4. Print/apply safe proposals

Usage:
  # Analyze + propose (dry run, no changes):
  python research/auto_research.py --log /tmp/train_monitor_v2b.log

  # Analyze + apply GUARDRAILS-safe config patches to train_monitor.py:
  python research/auto_research.py --log /tmp/train_monitor_v2b.log --apply

  # Full loop with W&B:
  python research/auto_research.py --log /tmp/train_monitor_v2b.log --wandb-run soxdq4iy --apply
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.experiment_analyzer import parse_log, diagnose, fetch_wandb
from research.hypothesis_generator import generate, PROPOSALS


TRAIN_SCRIPT = Path("training/train_monitor.py")
RESEARCH_LOG = Path("research/log.jsonl")


# ---------------------------------------------------------------------------
# Config patching — edits GRPOConfig in train_monitor.py
# ---------------------------------------------------------------------------

def apply_config_patch(patch: dict) -> bool:
    """Edit GRPOConfig kwargs in training/train_monitor.py. Returns True if changed."""
    if not patch:
        return False

    src = TRAIN_SCRIPT.read_text()
    changed = False

    for key, value in patch.items():
        # Match: key=<old_value>,  (handles int, float, str, bool)
        pattern = rf'({re.escape(key)}\s*=\s*)([^\s,\)]+)'
        new_val = repr(value) if isinstance(value, str) else str(value)
        new_src, n = re.subn(pattern, rf'\g<1>{new_val}', src)
        if n > 0:
            src = new_src
            changed = True
            print(f"  ✏️  {key} → {new_val}")
        else:
            print(f"  ⚠️  Could not find '{key}' in {TRAIN_SCRIPT} — skipping")

    if changed:
        TRAIN_SCRIPT.write_text(src)
    return changed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_once(log_path: str, wandb_run: str | None, apply: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"[auto_research] analyzing {log_path}")
    print(f"{'='*60}")

    # Step 1 — load metrics
    metrics = parse_log(log_path)
    if wandb_run:
        wb = fetch_wandb(wandb_run)
        if wb:
            metrics = wb

    # Step 2 — diagnose
    report = diagnose(metrics)
    print(f"\n📊 Health report:")
    for k, v in report.items():
        print(f"   {k}: {v}")

    # Step 3 — generate proposals
    diagnosis = report["diagnosis"]
    proposals = generate(diagnosis)

    print(f"\n💡 Proposals for '{diagnosis}':")
    for p in proposals:
        safe = "✅" if p["guardrails_safe"] else "⚠️ CAUTION"
        print(f"   [{safe}] {p['id']} ({p['confidence']:.0%}): {p['title']}")
        if p.get("config_patch"):
            print(f"       Config: {p['config_patch']}")
        if p.get("code_change"):
            print(f"       Code: {p['code_change'][:120]}...")

    # Step 4 — apply safe proposals
    applied = []
    if apply:
        print(f"\n🔧 Applying GUARDRAILS-safe proposals...")
        for p in proposals:
            if p["guardrails_safe"] and p["confidence"] >= 0.75 and p.get("config_patch"):
                print(f"\n  Applying: {p['title']}")
                if apply_config_patch(p["config_patch"]):
                    applied.append(p["id"])

        if applied:
            print(f"\n✅ Applied: {applied}")
            print(f"   Run tests: python -m pytest tests/ -v")
        else:
            print("  (no safe config patches to apply)")

    # Step 5 — log
    entry = {
        "timestamp": time.time(),
        "log_path": log_path,
        "wandb_run": wandb_run,
        "report": report,
        "proposals": [p["id"] for p in proposals],
        "applied": applied,
    }
    with open(RESEARCH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--apply", action="store_true",
                   help="Apply GUARDRAILS-safe config patches to train_monitor.py")
    p.add_argument("--loop", action="store_true",
                   help="Keep running every 5 minutes (for continuous monitoring)")
    args = p.parse_args()

    if args.loop:
        while True:
            run_once(args.log, args.wandb_run, args.apply)
            print(f"\n⏳ Next check in 5 minutes...")
            time.sleep(300)
    else:
        run_once(args.log, args.wandb_run, args.apply)


if __name__ == "__main__":
    main()
