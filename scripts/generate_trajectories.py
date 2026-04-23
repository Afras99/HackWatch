"""
Generate 135 cached trajectories: 9 workers × 15 tasks.

Usage:
  python scripts/generate_trajectories.py [--seed 42] [--out data/trajectories.jsonl]

Output: data/trajectories.jsonl — one JSON object per line:
  {task_id, worker_id, steps, final_state, ground_truth_hack: bool, hack_primitive: str|null}

This file is training data, eval data, and demo evidence — keep it deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.tasks import ALL_TASKS
from server.workers import WORKER_BANK
from server.exploits import ALL_PRIMITIVES


def generate_all(seed: int = 42) -> list[dict]:
    """Run every worker against every task. Returns list of trajectory dicts."""
    records = []

    for task in ALL_TASKS:
        for worker_id, worker_fn in WORKER_BANK.items():
            try:
                trajectory = worker_fn(task)
            except Exception as e:
                print(f"  WARN: worker={worker_id} task={task['task_id']} failed: {e}", file=sys.stderr)
                continue

            planted_label = trajectory[0].get("planted_label", "legit") if trajectory else "legit"
            hack_primitive = planted_label if planted_label != "legit" else None
            is_hack = planted_label != "legit"

            steps = []
            for step in trajectory:
                safe_step = {k: v for k, v in step.items() if k != "planted_label"}
                steps.append(safe_step)

            records.append({
                "task_id": task["task_id"],
                "worker_id": worker_id,
                "steps": steps,
                "final_state": {
                    "task_id": task["task_id"],
                    "worker_id": worker_id,
                    "step_count": len(steps),
                },
                "ground_truth_hack": is_hack,
                "hack_primitive": hack_primitive,
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/trajectories.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating trajectories (seed={args.seed})...")
    records = generate_all(seed=args.seed)

    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Summary
    total = len(records)
    hacks = sum(1 for r in records if r["ground_truth_hack"])
    legit = total - hacks
    primitives: dict[str, int] = {}
    for r in records:
        p = r["hack_primitive"] or "legit"
        primitives[p] = primitives.get(p, 0) + 1

    print(f"Written {total} trajectories to {out_path}")
    print(f"  hack: {hacks} ({100*hacks//total}%)  legit: {legit} ({100*legit//total}%)")
    print("  primitive breakdown:")
    for p, count in sorted(primitives.items()):
        print(f"    {p}: {count}")


if __name__ == "__main__":
    main()
