"""
Optuna hyperparameter optimisation for the HackWatch MONITOR GRPO training.

Two phases:

  CPU phase  — no GPU, no model load. Scores param combos using the heuristic
               reward function directly on a sample of training prompts.
               Runs in <60 seconds for 20 trials. Use to eliminate clearly bad
               combos before committing GPU time.

  GPU phase  — real 50-step GRPO training per trial on DGX. Returns mean reward
               over steps 30-50 (avoids noisy warmup). 2-3 trials sufficient.

Usage:

  # 1. CPU sweep (fast, no GPU needed):
  python -m training.hpo --phase cpu --n-trials 20

  # 2. GPU sweep (3 trials, ~30-45 min total on A100):
  python -m training.hpo --phase gpu --n-trials 3 --env-url http://localhost:9876

  # 3. Print best params from a completed study:
  python -m training.hpo --phase report --storage sqlite:///hpo.db

After GPU phase, plug best params into:
  training/train_monitor.py MonitorTrainer.__init__ defaults
and run a full 400-step training.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Search space — tuned based on log analysis of 14 failed runs.
# See training notes: ECG caused by temperature=1.1 (near-greedy) + num_iterations=2.
SEARCH_SPACE: dict[str, Any] = {
    "learning_rate":     ("log_float",   1e-6,  5e-5),
    "beta":              ("log_float",   0.01,  0.10),   # floor=0.01 per CLAUDE.md
    "temperature":       ("float",       1.0,   1.6),    # primary fix: must be >1.2
    "num_iterations":    ("categorical", [1, 2]),
    "num_train_epochs":  ("categorical", [2, 3, 4]),     # current 8 causes memorisation
}


def _suggest(trial, name: str, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown spec kind: {kind}")


# ---------------------------------------------------------------------------
# CPU phase — heuristic reward only, no GPU
# ---------------------------------------------------------------------------

def cpu_objective(trial) -> float:
    """
    Score a hyperparameter combo using the heuristic reward function only.
    No model is loaded — this runs entirely on CPU in a few seconds.

    Limitation: can't test actual training dynamics (clip_ratio, KL). Only
    filters combos where the reward distribution is obviously degenerate.

    We generate synthetic completions by randomly sampling verdicts/confidences
    at the given temperature (higher temp = more uniform distribution over
    verdicts) and score them with the heuristic reward function.
    """
    params = {name: _suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()}
    temperature = params["temperature"]

    from training.train_monitor import build_prompt_dataset, score_action_heuristically

    dataset = build_prompt_dataset(env_url=None)
    sample_size = min(50, len(dataset))
    indices = random.sample(range(len(dataset)), sample_size)

    rewards: list[float] = []
    for i in indices:
        row = dataset[i]
        prompt_msgs = row["prompt"]
        prompt_text = ""
        for msg in reversed(prompt_msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break

        # Simulate what a model would output at this temperature.
        # Higher temperature → more uniform verdict distribution → more variance.
        # This is a proxy: real model behaviour depends on weights, not just temp.
        t = max(0.1, temperature)
        # Softmax-like weights: at t=1.0 model strongly prefers block (trained base);
        # at t=1.6 distribution is more uniform.
        logits = {"block": 2.0 / t, "flag": 1.0 / t, "allow": 0.5 / t}
        total = sum(2.718 ** v for v in logits.values())
        probs = {k: 2.718 ** v / total for k, v in logits.items()}

        verdict = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        confidence = round(random.uniform(0.6, 0.99), 2) if verdict != "allow" else round(random.uniform(0.1, 0.5), 2)
        reasoning = "suspicious pattern detected" if verdict != "allow" else "code looks clean"

        action = {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}
        scored = score_action_heuristically(action, prompt_text)
        rewards.append(scored["total"])

    mean_r = sum(rewards) / len(rewards)
    std_r  = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

    # Objective: maximise mean reward AND reward std (std = GRPO can learn).
    # Low std means all completions score the same → zero gradient.
    objective = mean_r + 0.5 * std_r
    log.info(
        f"Trial {trial.number} CPU | params={params} "
        f"mean={mean_r:.4f} std={std_r:.4f} obj={objective:.4f}"
    )
    return objective


# ---------------------------------------------------------------------------
# GPU phase — real 50-step GRPO training
# ---------------------------------------------------------------------------

def gpu_objective(trial, env_url: str, output_base: str) -> float:
    """
    Run real GRPO training for 50 steps. Each trial takes ~10-15 min on A100.
    Returns mean reward over steps 30-50 (avoids warmup noise).

    The metric to watch in W&B: clip_ratio > 0 means advantages are non-zero
    and actual learning is happening.
    """
    params = {name: _suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()}

    from training.train_monitor import MonitorTrainer

    output_dir = str(Path(output_base) / f"trial_{trial.number}")
    log.info(f"Trial {trial.number} GPU | params={params} | output={output_dir}")

    MonitorTrainer(
        env_url=env_url,
        output_dir=output_dir,
        max_steps=50,
        no_wandb=True,   # suppress W&B noise during HPO; enable for main run
        learning_rate=params["learning_rate"],
        beta=params["beta"],
        temperature=params["temperature"],
        num_iterations=params["num_iterations"],
        num_train_epochs=params["num_train_epochs"],
    ).run()

    # Read trainer_state.json from the checkpoint-50 dir
    state_path = Path(output_dir) / "checkpoint-50" / "trainer_state.json"
    if not state_path.exists():
        # Fallback: find any checkpoint
        candidates = sorted(Path(output_dir).glob("checkpoint-*/trainer_state.json"))
        if not candidates:
            log.warning(f"Trial {trial.number}: no trainer_state.json found, returning 0")
            return 0.0
        state_path = candidates[-1]

    state = json.loads(state_path.read_text())
    all_rewards = [e["reward"] for e in state.get("log_history", []) if "reward" in e]

    if not all_rewards:
        log.warning(f"Trial {trial.number}: no reward in log_history, returning 0")
        return 0.0

    # Mean over last 20 steps (or all steps if fewer than 20)
    tail = all_rewards[-20:]
    mean_r = sum(tail) / len(tail)
    log.info(f"Trial {trial.number} GPU | mean_reward(last20)={mean_r:.4f}")
    return mean_r


# ---------------------------------------------------------------------------
# Report — print best params from completed study
# ---------------------------------------------------------------------------

def print_report(study) -> None:
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial: #{best.number}  value={best.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:<20} = {v}")
    print()
    print("Top 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)
    for t in trials[:5]:
        print(f"  #{t.number:>2}  value={t.value:.4f}  params={t.params}")
    print("=" * 60)

    print("\nTo use best params, update MonitorTrainer defaults in train_monitor.py:")
    print("  MonitorTrainer(")
    for k, v in best.params.items():
        print(f"      {k}={repr(v)},")
    print("  )")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for HackWatch GRPO")
    parser.add_argument("--phase",      choices=["cpu", "gpu", "report"], default="cpu")
    parser.add_argument("--n-trials",   type=int, default=None,
                        help="Number of Optuna trials (default: 20 for cpu, 3 for gpu)")
    parser.add_argument("--env-url",    default="http://localhost:8000")
    parser.add_argument("--study-name", default="hackwatch_hpo")
    parser.add_argument("--storage",    default=None,
                        help="Optuna storage URL e.g. sqlite:///hpo.db (persists across runs)")
    parser.add_argument("--output-dir", default="./runs/hpo",
                        help="Base dir for GPU trial checkpoints")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        raise SystemExit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    if args.phase == "report":
        if not study.trials:
            print("No completed trials found in study.")
            return
        print_report(study)
        return

    if args.phase == "cpu":
        n = args.n_trials or 20
        log.info(f"CPU phase: {n} trials (no GPU required)")
        study.optimize(cpu_objective, n_trials=n, show_progress_bar=True)
        print_report(study)

    elif args.phase == "gpu":
        n = args.n_trials or 3

        # Seed GPU trials with best params from CPU phase if available
        cpu_trials = [t for t in study.trials if t.value is not None]
        if cpu_trials:
            top = sorted(cpu_trials, key=lambda t: t.value, reverse=True)[:n]
            log.info(f"GPU phase: seeding {n} trials from top CPU results")
            for t in top:
                study.enqueue_trial(t.params)

        log.info(f"GPU phase: {n} trials × 50 steps each on DGX")
        study.optimize(
            lambda trial: gpu_objective(trial, args.env_url, args.output_dir),
            n_trials=n,
            show_progress_bar=True,
        )
        print_report(study)


if __name__ == "__main__":
    main()
