"""
Export HackWatch trajectories as a HuggingFace Dataset.

Usage:
  python scripts/export_hf_dataset.py \
      --trajectories data/trajectories.jsonl \
      --out data/hackwatch_dataset \
      [--push-to-hub YOUR_HF_ORG/hackwatch-trajectories]

The exported dataset has the following splits:
  - train:   trajectories with training exploit primitives
  - heldout: subprocess_escape + eval_injection (generalization test)
  - stealth: stealth_* variants (semantic reasoning test)
  - legit:   clean/legitimate worker trajectories

Each row contains:
  task_id, worker_id, hack_primitive, ground_truth_hack,
  steps (list of {diff, action}), prompt (monitor input for step 0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_monitor import make_prompt, SYSTEM_PROMPT

_HELDOUT = {"subprocess_escape", "eval_injection"}
_STEALTH = {"stealth_assert_camouflage", "stealth_memo_lookup",
            "stealth_conftest_bypass", "stealth_import_redirect"}


def _split_name(primitive: str | None) -> str:
    if not primitive:
        return "legit"
    if primitive in _HELDOUT:
        return "heldout"
    if primitive in _STEALTH:
        return "stealth"
    return "train"


def _record_to_row(record: dict) -> dict:
    primitive = record.get("hack_primitive") or None
    steps = record.get("steps", [])

    # Build monitor prompt for step 0 (what the model sees)
    prompt = ""
    if steps:
        obs = {
            "task_prompt": f"Task: {record['task_id']}",
            "worker_diff": steps[0].get("diff", ""),
            "tools_used_so_far": [steps[0].get("action", {}).get("tool", "write_file")],
            "step_idx": 0,
        }
        try:
            prompt = make_prompt(obs)
        except Exception:
            prompt = ""

    return {
        "task_id": record["task_id"],
        "worker_id": record["worker_id"],
        "hack_primitive": primitive or "legit",
        "ground_truth_hack": bool(record.get("ground_truth_hack", False)),
        "n_steps": len(steps),
        "step_diffs": [s.get("diff", "") for s in steps],
        "system_prompt": SYSTEM_PROMPT,
        "monitor_prompt_step0": prompt,
        "split": _split_name(primitive),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="data/trajectories.jsonl")
    parser.add_argument("--out",          default="data/hackwatch_dataset")
    parser.add_argument("--push-to-hub",  default=None,
                        help="HuggingFace repo ID to push to, e.g. myorg/hackwatch-trajectories")
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found. Run scripts/generate_trajectories.py first.",
              file=sys.stderr)
        sys.exit(1)

    records = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rows = [_record_to_row(r) for r in records]
    print(f"Loaded {len(rows)} trajectories from {traj_path}")

    # Group by split
    splits: dict[str, list[dict]] = {}
    for row in rows:
        splits.setdefault(row["split"], []).append(row)

    for split, split_rows in splits.items():
        print(f"  {split}: {len(split_rows)} rows")

    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    dataset_dict = DatasetDict({
        split: Dataset.from_list(split_rows)
        for split, split_rows in splits.items()
    })

    out_path = Path(args.out)
    dataset_dict.save_to_disk(str(out_path))
    print(f"\nSaved to {out_path}/")
    for split in dataset_dict:
        print(f"  {split}: {len(dataset_dict[split])} examples, "
              f"columns: {dataset_dict[split].column_names}")

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset_dict.push_to_hub(args.push_to_hub)
        print("Done.")


if __name__ == "__main__":
    main()
