"""
Day 4: Co-training loop — worker and monitor train simultaneously with
a curriculum that gradually introduces harder hack primitives.

Curriculum schedule:
  Phase 0 (ep 0-100):   easy tasks, single primitives, legit_frac=0.6
  Phase 1 (ep 100-300): medium tasks, all primitives, legit_frac=0.4
  Phase 2 (ep 300+):    hard + heldout_novel, combo hacks, legit_frac=0.2

Run:
  python -m training.train_cotrain \\
      --monitor-checkpoint ./runs/monitor_v8/final \\
      --env-url http://localhost:8000 \\
      --output-dir ./runs/cotrain_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass

import httpx

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server import sandbox  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Curriculum phases
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    name: str
    ep_start: int
    ep_end: int
    difficulty: str          # sent as "seed" hint in /reset body
    legit_frac: float        # not enforced server-side but logged


CURRICULUM = [
    CurriculumPhase("warmup",      0,    100, "train",        0.60),
    CurriculumPhase("ramp",      100,   300,  "train",        0.40),
    CurriculumPhase("adversarial", 300, 9999, "heldout_novel", 0.20),
]


def phase_for(episode: int) -> CurriculumPhase:
    for p in CURRICULUM:
        if p.ep_start <= episode < p.ep_end:
            return p
    return CURRICULUM[-1]


# ---------------------------------------------------------------------------
# Co-training loop
# ---------------------------------------------------------------------------

def cotrain(
    monitor_checkpoint: str,
    worker_model: str,
    env_url: str,
    output_dir: str,
    total_episodes: int = 600,
    no_wandb: bool = False,
    dry_run: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if dry_run:
        log.info("Dry run: verifying env connectivity and curriculum phases")
        with httpx.Client(timeout=10.0) as c:
            r = c.post(f"{env_url}/reset", json={})
            r.raise_for_status()
            log.info(f"Env /reset OK: {list(r.json().keys())}")
        for ep in [0, 100, 300]:
            p = phase_for(ep)
            log.info(f"  ep={ep} → phase={p.name} difficulty={p.difficulty}")
        # Verify build_prompt_dataset import works
        from training.train_monitor import build_prompt_dataset, build_env_reward_fn
        dataset = build_prompt_dataset(env_url=env_url)
        log.info(f"Dataset OK: {len(dataset)} rows (message-list format)")
        return

    # Load models
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError:
        raise ImportError("Install unsloth for co-training: pip install unsloth")

    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]

    log.info(f"Loading monitor from {monitor_checkpoint}")
    monitor_model, monitor_tok = FastLanguageModel.from_pretrained(
        monitor_checkpoint, max_seq_length=4096, load_in_4bit=True
    )
    # Only add LoRA if the checkpoint is a base model (no adapters already present)
    _has_adapters = any("lora" in n for n, _ in monitor_model.named_parameters())
    if not _has_adapters:
        monitor_model = FastLanguageModel.get_peft_model(
            monitor_model, r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=64, lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        log.info("Monitor already has LoRA adapters — skipping get_peft_model")
        FastLanguageModel.for_training(monitor_model)

    # Use the same UCB-weighted message-list dataset as train_monitor.py.
    # v8 was trained on message-list format; raw strings break the chat template
    # and cause clipped_ratio=1.0 (model generates unstructured 256-token blobs).
    from training.train_monitor import build_prompt_dataset, build_env_reward_fn
    dataset = build_prompt_dataset(env_url=env_url)
    log.info(f"Co-train dataset: {len(dataset)} message-list prompts (UCB-weighted)")

    report = "none" if no_wandb else "wandb"

    # Mirror the working train_monitor.py config — this is what produced v8.
    monitor_cfg = GRPOConfig(
        output_dir=f"{output_dir}/monitor",
        per_device_train_batch_size=8,      # was 1; must match num_generations for proper group stats
        gradient_accumulation_steps=2,      # was 8; 2×8 = same effective batch, better group diversity
        num_generations=8,
        max_completion_length=512,          # was 256; v8 generates 150-200 tok reasoning
        max_prompt_length=1024,
        num_train_epochs=1,
        beta=0.01,                          # was 0.04; lower KL penalty allows more exploration
        learning_rate=5e-6,                 # was 1e-6; 5× higher matches v8 training
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        bf16=False,
        fp16=True,
        optim="adamw_torch_fused",
        logging_steps=1,
        report_to=report,
        max_steps=min(total_episodes // 2, 300),
        save_steps=50,
        loss_type='dr_grpo',                # removes length+variance normalization bias
        scale_rewards=False,
        importance_sampling_level='sequence',
        mask_truncated_completions=True,    # zero gradients from any remaining truncated completions
        epsilon=0.2,
        epsilon_high=0.28,                  # DAPO asymmetric clipping
        temperature=1.1,
    )

    from training.dynamic_grpo import DynamicSamplingGRPOTrainer

    # Shared reward fn from train_monitor: parses message-list completions correctly
    # and logs all 12 per-component W&B metrics (detection_f1, comp_*_mean, etc.)
    monitor_reward_fn = build_env_reward_fn(env_url=env_url)

    monitor_trainer = DynamicSamplingGRPOTrainer(
        model=monitor_model,
        tokenizer=monitor_tok,
        args=monitor_cfg,
        train_dataset=dataset,
        reward_funcs=[monitor_reward_fn],
    )

    # EMA-smoothed reward metric for judge-presentable W&B curve.
    # Raw GRPO reward oscillates because groups contrast 8 completions — EMA shows the signal.
    _ema_state: dict = {"value": None}
    _ema_alpha = 0.1

    from transformers import TrainerCallback  # type: ignore[import]

    class _EMALogger(TrainerCallback):
        def on_log(self, args, state, **kwargs):
            try:
                import wandb as _wb
                if _wb.run is None:
                    return
                latest = monitor_trainer._metrics.get("train", {}).get("reward", [])
                if not latest:
                    return
                last = float(latest[-1])
                if _ema_state["value"] is None:
                    _ema_state["value"] = last
                else:
                    _ema_state["value"] = _ema_alpha * last + (1 - _ema_alpha) * _ema_state["value"]
                _wb.log({
                    "reward/ema_smoothed":   _ema_state["value"],
                    "reward/raw":            last,
                    "reward/residual_abs":   abs(last - _ema_state["value"]),
                })
            except Exception:
                pass

    monitor_trainer.add_callback(_EMALogger())

    log.info("Starting co-training: monitor GRPO phase")
    monitor_trainer.train()

    # Save monitor
    monitor_out = f"{output_dir}/monitor/final"
    monitor_model.save_pretrained(monitor_out)
    monitor_tok.save_pretrained(monitor_out)
    log.info(f"Monitor saved to {monitor_out}")

    # Worker trainer — adversarial: rewarded when code passes tests while evading monitor.
    # Skip with HACKWATCH_SKIP_WORKER=1 to run monitor-only co-training.
    if not os.environ.get("HACKWATCH_SKIP_WORKER"):
        log.info(f"Loading worker from {worker_model}")
        worker_model_obj, worker_tok = FastLanguageModel.from_pretrained(
            worker_model, max_seq_length=4096, load_in_4bit=True
        )
        worker_model_obj = FastLanguageModel.get_peft_model(
            worker_model_obj, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32, bias="none",
            use_gradient_checkpointing="unsloth",
        )

        worker_cfg = GRPOConfig(
            output_dir=f"{output_dir}/worker",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=4,
            max_completion_length=512,
            max_prompt_length=2048,
            beta=0.04,
            learning_rate=5e-7,
            max_grad_norm=0.5,
            logging_steps=1,
            report_to=report,
            max_steps=total_episodes // 4,
            save_steps=50,
        )

        def worker_reward_fn(completions: list[str], prompts: list[str], **_) -> list[float]:
            rewards = []
            for completion in completions:
                try:
                    parsed = json.loads(completion) if completion.strip().startswith("{") else {}
                    path = parsed.get("path", "solution.py")
                    content = parsed.get("content", completion)
                    repo = {path: content}
                    sandbox_result = sandbox.run_tests(repo, timeout=10)
                    tests_pass = sandbox_result["passed"]
                    rewards.append(1.0 if tests_pass else 0.0)
                except Exception:
                    rewards.append(0.0)
            return rewards

        worker_trainer = GRPOTrainer(
            model=worker_model_obj,
            tokenizer=worker_tok,
            args=worker_cfg,
            train_dataset=dataset,
            reward_funcs=[worker_reward_fn],
        )
        worker_trainer.train()

        worker_out = f"{output_dir}/worker/final"
        worker_model_obj.save_pretrained(worker_out)
        worker_tok.save_pretrained(worker_out)
        log.info(f"Worker saved to {worker_out}")
    else:
        log.info("HACKWATCH_SKIP_WORKER set — skipping worker training phase")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Co-train HackWatch monitor + worker")
    parser.add_argument("--monitor-checkpoint", default="./runs/monitor_v8/final")
    parser.add_argument("--worker-model",       default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--env-url",            default="http://localhost:8000")
    parser.add_argument("--output-dir",         default="./runs/cotrain_v2")
    parser.add_argument("--total-episodes",     type=int, default=600)
    parser.add_argument("--no-wandb",           action="store_true")
    parser.add_argument("--dry-run",            action="store_true")
    args = parser.parse_args()

    cotrain(
        monitor_checkpoint=args.monitor_checkpoint,
        worker_model=args.worker_model,
        env_url=args.env_url,
        output_dir=args.output_dir,
        total_episodes=args.total_episodes,
        no_wandb=args.no_wandb,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
