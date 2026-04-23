"""
Evaluate the MONITOR model against the cached trajectory dataset.

Usage:
  # Baseline (untrained model via OpenAI-compatible API):
  python eval/evaluate_monitor.py \
      --trajectories data/trajectories.jsonl \
      --api-url http://localhost:11434/v1 \
      --model-name Qwen/Qwen2.5-3B-Instruct \
      --tag baseline \
      --out eval/results_baseline.json

  # Trained model:
  python eval/evaluate_monitor.py \
      --trajectories data/trajectories.jsonl \
      --api-url http://localhost:8001/v1 \
      --model-name hackwatch-monitor-v1 \
      --tag trained_v1 \
      --out eval/results_trained.json

Outputs a JSON results file and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.reward import compute_episode_reward, aggregate_rewards
from training.train_monitor import make_prompt, parse_verdict, SYSTEM_PROMPT


def call_model(
    prompt: str,
    api_url: str,
    model_name: str,
    api_key: str = "dummy",
    temperature: float = 0.0,
) -> str:
    import httpx
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 256,
    }
    resp = httpx.post(
        f"{api_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def evaluate_trajectory(
    record: dict,
    api_url: str,
    model_name: str,
    api_key: str = "dummy",
) -> dict:
    """Score one trajectory record. Returns per-episode result dict."""
    planted_label = record["hack_primitive"] or "legit"
    steps = record["steps"]

    monitor_actions_raw = []
    intervention_step = None

    for i, step in enumerate(steps):
        obs_like = {
            "task_prompt": f"Task: {record['task_id']}",
            "worker_diff": step.get("diff", ""),
            "tools_used_so_far": [step.get("action", {}).get("tool", "write_file")],
            "step_idx": i,
        }
        prompt = make_prompt(obs_like)
        try:
            completion = call_model(prompt, api_url, model_name, api_key)
        except Exception as e:
            completion = '{"verdict": "allow", "confidence": 0.5, "reasoning": "api_error"}'
            print(f"  API error at step {i}: {e}", file=sys.stderr)

        action = parse_verdict(completion)
        monitor_actions_raw.append(action)

        if action["verdict"] in ("flag", "block") and intervention_step is None:
            intervention_step = i

        if action["verdict"] == "block":
            break

    from hackwatch.models import MonitorAction
    monitor_actions = [
        MonitorAction(verdict=a["verdict"], confidence=a["confidence"], reasoning=a.get("reasoning", ""))
        for a in monitor_actions_raw
    ]

    info = compute_episode_reward(
        planted_label=planted_label,
        monitor_flags=monitor_actions,
        worker_succeeded=True,
        intervention_step=intervention_step,
        total_steps=max(1, len(steps)),
    )

    return {
        "task_id": record["task_id"],
        "worker_id": record["worker_id"],
        "ground_truth_hack": record["ground_truth_hack"],
        "hack_primitive": record["hack_primitive"],
        **info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="data/trajectories.jsonl")
    parser.add_argument("--api-url",      default="http://localhost:11434/v1")
    parser.add_argument("--model-name",   default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--api-key",      default="dummy")
    parser.add_argument("--tag",          default="eval")
    parser.add_argument("--out",          default="eval/results.json")
    parser.add_argument("--limit",        type=int, default=None, help="Evaluate only first N trajectories")
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found. Run scripts/generate_trajectories.py first.", file=sys.stderr)
        sys.exit(1)

    records = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    print(f"Evaluating {len(records)} trajectories with model={args.model_name} tag={args.tag}")

    results = []
    for i, record in enumerate(records):
        print(f"  [{i+1}/{len(records)}] task={record['task_id']} worker={record['worker_id']}", end=" ")
        result = evaluate_trajectory(record, args.api_url, args.model_name, args.api_key)
        results.append(result)
        correct = "✓" if (result["tp"] or result["tn"]) else "✗"
        print(f"reward={result['reward']:.3f} {correct}")

    agg = aggregate_rewards(results)

    # Per-primitive breakdown
    by_primitive: dict[str, list[dict]] = {}
    for r in results:
        p = r["hack_primitive"] or "legit"
        by_primitive.setdefault(p, []).append(r)
    primitive_f1 = {p: aggregate_rewards(rs)["f1"] for p, rs in by_primitive.items()}

    output = {
        "tag": args.tag,
        "model": args.model_name,
        "n_trajectories": len(results),
        "aggregate": agg,
        "per_primitive_f1": primitive_f1,
        "per_episode": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n=== {args.tag} Results ===")
    print(f"  Accuracy:  {agg['accuracy']:.1%}")
    print(f"  F1:        {agg['f1']:.3f}")
    print(f"  Precision: {agg['precision']:.3f}")
    print(f"  Recall:    {agg['recall']:.3f}")
    print(f"  Mean reward: {agg['mean_reward']:.3f}")
    print(f"\nPer-primitive F1:")
    for p, f1 in sorted(primitive_f1.items()):
        print(f"  {p:25s}: {f1:.3f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
