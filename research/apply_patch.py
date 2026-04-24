"""
Apply a GRPOConfig patch to training/train_monitor.py.

Usage:
  python research/apply_patch.py '{"num_generations": 12, "per_device_train_batch_size": 12}'
  python research/apply_patch.py '{"beta": 0.02, "temperature": 1.3}'
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TRAIN_SCRIPT = Path("training/train_monitor.py")

GUARDRAILS = {
    "beta":           (0.01, 0.1),
    "learning_rate":  (1e-6, 1e-5),
    "lora_rank":      (8, 64),
    "num_generations":(4, 32),
}


def apply_patch(patch: dict) -> list[str]:
    """Edit GRPOConfig kwargs in train_monitor.py. Returns list of applied keys."""
    src = TRAIN_SCRIPT.read_text()
    applied = []

    for key, value in patch.items():
        # Guardrails bounds check
        if key in GUARDRAILS:
            lo, hi = GUARDRAILS[key]
            if not (lo <= value <= hi):
                print(f"  BLOCKED {key}={value} — out of GUARDRAILS bounds [{lo}, {hi}]")
                continue

        pattern = rf'({re.escape(key)}\s*=\s*)([^\s,\)]+)'
        new_val = repr(value) if isinstance(value, str) else str(value)
        new_src, n = re.subn(pattern, rf'\g<1>{new_val}', src)
        if n > 0:
            src = new_src
            applied.append(key)
            print(f"  ✅ {key} → {new_val}")
        else:
            print(f"  ⚠️  '{key}' not found in {TRAIN_SCRIPT} — skipping")

    if applied:
        TRAIN_SCRIPT.write_text(src)
        print(f"\nApplied {len(applied)} change(s) to {TRAIN_SCRIPT}")
    else:
        print("No changes applied.")

    return applied


def main():
    if len(sys.argv) != 2:
        print("Usage: python research/apply_patch.py '<json_dict>'")
        sys.exit(1)

    try:
        patch = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)

    apply_patch(patch)


if __name__ == "__main__":
    main()
