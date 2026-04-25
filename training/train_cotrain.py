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
# Curriculum phases (plain dataclass — not a service class)
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    """One phase of the co-training curriculum."""
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


# ---------------------------------------------------------------------------
# CoTrainer — main service class
# ---------------------------------------------------------------------------


class CoTrainer:
    """Runs the Day-4 co-training loop for monitor + worker.

    All config is stored in ``__init__``; no config is loaded inside methods.

    Args:
        monitor_checkpoint: Path to the pre-trained monitor checkpoint.
        worker_model: HF model name or path for the worker.
        env_url: URL of the running HackWatch env server.
        output_dir: Directory to save trained models and logs.
        total_episodes: Total training episodes.
        no_wandb: Disable W&B logging when ``True``.
        dry_run: Validate pipeline without running GPU training when ``True``.
    """

    def __init__(
        self,
        monitor_checkpoint: str = "./runs/monitor_v8/final",
        worker_model: str = "Qwen/Qwen2.5-3B-Instruct",
        env_url: str = "http://localhost:8000",
        output_dir: str = "./runs/cotrain_v2",
        total_episodes: int = 600,
        no_wandb: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.monitor_checkpoint = monitor_checkpoint
        self.worker_model = worker_model
        self.env_url = env_url
        self.output_dir = output_dir
        self.total_episodes = total_episodes
        self.no_wandb = no_wandb
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Step 1 — resolve curriculum phase for a given episode index
    # ------------------------------------------------------------------

    def _phase_for(self, episode: int) -> CurriculumPhase:
        """Return the curriculum phase active at ``episode``.

        Args:
            episode: Zero-based episode index.

        Returns:
            Matching ``CurriculumPhase``.
        """
        for p in CURRICULUM:
            if p.ep_start <= episode < p.ep_end:
                return p
        return CURRICULUM[-1]

    # ------------------------------------------------------------------
    # Step 2 — dry-run connectivity check
    # ------------------------------------------------------------------

    def _dry_run_check(self) -> None:
        """Verify env connectivity and curriculum without touching the GPU."""
        log.info("Dry run: verifying env connectivity and curriculum phases")
        with httpx.Client(timeout=10.0) as c:
            r = c.post(f"{self.env_url}/reset", json={})
            r.raise_for_status()
            log.info(f"Env /reset OK: {list(r.json().keys())}")
        for ep in [0, 100, 300]:
            p = self._phase_for(ep)
            log.info(f"  ep={ep} → phase={p.name} difficulty={p.difficulty}")
        from training.train_monitor import build_prompt_dataset, build_env_reward_fn
        dataset = build_prompt_dataset(env_url=self.env_url)
        log.info(f"Dataset OK: {len(dataset)} rows (message-list format)")

    # ------------------------------------------------------------------
    # Step 3 — load monitor model and tokenizer
    # ------------------------------------------------------------------

    def _load_monitor(self):
        """Load the monitor model from ``self.monitor_checkpoint``.

        Returns:
            ``(monitor_model, monitor_tok)`` tuple.
        """
        from unsloth import FastLanguageModel  # type: ignore[import]
        log.info(f"Loading monitor from {self.monitor_checkpoint}")
        monitor_model, monitor_tok = FastLanguageModel.from_pretrained(
            self.monitor_checkpoint, max_seq_length=4096, load_in_4bit=True
        )
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
        return monitor_model, monitor_tok

    # ------------------------------------------------------------------
    # Step 4 — run monitor GRPO training phase
    # ------------------------------------------------------------------

    def _train_monitor(self, monitor_model, monitor_tok, dataset, reward_fn) -> None:
        """Run GRPO training on the monitor model.

        Args:
            monitor_model: Loaded monitor model with LoRA.
            monitor_tok: Corresponding tokenizer.
            dataset: UCB-weighted message-list prompt dataset.
            reward_fn: Env-backed reward function callable.
        """
        from trl import GRPOConfig  # type: ignore[import]
        from training.config import grpo_cfg
        from training.dynamic_grpo import DynamicSamplingGRPOTrainer
        from transformers import TrainerCallback  # type: ignore[import]

        _cfg = grpo_cfg()

        report = "none" if self.no_wandb else _cfg.get("report_to", "wandb")
        monitor_cfg = GRPOConfig(
            output_dir=f"{self.output_dir}/monitor",
            max_steps=min(self.total_episodes // 2, 300),
            save_steps=50,
            report_to=report,
            bf16=False,
            fp16=True,
            optim="adamw_torch_fused",
            per_device_train_batch_size=_cfg.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=_cfg.get("gradient_accumulation_steps", 2),
            num_generations=_cfg.get("num_generations", 8),
            max_completion_length=_cfg.get("max_completion_length", 256),
            max_prompt_length=_cfg.get("max_prompt_length", 1024),
            num_train_epochs=_cfg.get("num_train_epochs", 2),
            beta=_cfg.get("beta", 0.051),
            learning_rate=_cfg.get("learning_rate", 1.05e-5),
            warmup_ratio=_cfg.get("warmup_ratio", 0.1),
            max_grad_norm=_cfg.get("max_grad_norm", 0.5),
            logging_steps=_cfg.get("logging_steps", 1),
            loss_type=_cfg.get("loss_type", "dr_grpo"),
            scale_rewards=_cfg.get("scale_rewards", False),
            importance_sampling_level=_cfg.get("importance_sampling_level", "sequence"),
            mask_truncated_completions=_cfg.get("mask_truncated_completions", True),
            epsilon=_cfg.get("epsilon", 0.2),
            epsilon_high=_cfg.get("epsilon_high", 0.28),
            temperature=_cfg.get("temperature", 1.012),
            num_iterations=_cfg.get("num_iterations", 1),
        )

        _ema_state: dict = {"value": None}
        _ema_alpha = 0.1

        monitor_trainer = DynamicSamplingGRPOTrainer(
            model=monitor_model,
            processing_class=monitor_tok,
            args=monitor_cfg,
            train_dataset=dataset,
            reward_funcs=[reward_fn],
        )

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
                        _ema_state["value"] = (
                            _ema_alpha * last + (1 - _ema_alpha) * _ema_state["value"]
                        )
                    _wb.log({
                        "reward/ema_smoothed": _ema_state["value"],
                        "reward/raw": last,
                        "reward/residual_abs": abs(last - _ema_state["value"]),
                    })
                except Exception:
                    pass

        monitor_trainer.add_callback(_EMALogger())
        log.info("Starting co-training: monitor GRPO phase")
        monitor_trainer.train()

        monitor_out = f"{self.output_dir}/monitor/final"
        monitor_model.save_pretrained(monitor_out)
        monitor_tok.save_pretrained(monitor_out)
        log.info(f"Monitor saved to {monitor_out}")

    # ------------------------------------------------------------------
    # Step 5 — run worker adversarial training phase (optional)
    # ------------------------------------------------------------------

    def _train_worker(self, dataset) -> None:
        """Train the worker adversarially against the monitor.

        Skipped when ``HACKWATCH_SKIP_WORKER=1`` is set.

        Args:
            dataset: Shared message-list dataset (same as monitor training).
        """
        if os.environ.get("HACKWATCH_SKIP_WORKER"):
            log.info("HACKWATCH_SKIP_WORKER set — skipping worker training phase")
            return

        from unsloth import FastLanguageModel  # type: ignore[import]
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]
        from training.config import grpo_cfg

        _cfg = grpo_cfg()

        log.info(f"Loading worker from {self.worker_model}")
        worker_model_obj, worker_tok = FastLanguageModel.from_pretrained(
            self.worker_model, max_seq_length=4096, load_in_4bit=True
        )
        worker_model_obj = FastLanguageModel.get_peft_model(
            worker_model_obj, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32, bias="none",
            use_gradient_checkpointing="unsloth",
        )

        report = "none" if self.no_wandb else "wandb"
        # Worker uses its own adversarial config — intentionally different from
        # monitor. Only structural params (batch, logging) come from yaml.
        worker_cfg = GRPOConfig(
            output_dir=f"{self.output_dir}/worker",
            per_device_train_batch_size=_cfg.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=_cfg.get("gradient_accumulation_steps", 2),
            num_generations=4,
            max_completion_length=512,
            max_prompt_length=2048,
            beta=0.04,
            learning_rate=5e-7,
            max_grad_norm=_cfg.get("max_grad_norm", 0.5),
            logging_steps=_cfg.get("logging_steps", 1),
            report_to=report,
            max_steps=self.total_episodes // 4,
            save_steps=50,
        )

        def worker_reward_fn(
            completions: list[str], prompts: list[str], **_
        ) -> list[float]:
            rewards = []
            for completion in completions:
                try:
                    parsed = (
                        json.loads(completion)
                        if completion.strip().startswith("{")
                        else {}
                    )
                    path = parsed.get("path", "solution.py")
                    content = parsed.get("content", completion)
                    repo = {path: content}
                    sandbox_result = sandbox.run_tests(repo, timeout=10)
                    rewards.append(1.0 if sandbox_result["passed"] else 0.0)
                except Exception:
                    rewards.append(0.0)
            return rewards

        worker_trainer = GRPOTrainer(
            model=worker_model_obj,
            processing_class=worker_tok,
            args=worker_cfg,
            train_dataset=dataset,
            reward_funcs=[worker_reward_fn],
        )
        worker_trainer.train()

        worker_out = f"{self.output_dir}/worker/final"
        worker_model_obj.save_pretrained(worker_out)
        worker_tok.save_pretrained(worker_out)
        log.info(f"Worker saved to {worker_out}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full co-training pipeline.

        Runs ``_dry_run_check`` when ``self.dry_run`` is ``True``.
        Otherwise loads models, builds dataset, and trains monitor then worker.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        if self.dry_run:
            self._dry_run_check()
            return

        try:
            from unsloth import FastLanguageModel  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Install unsloth for co-training: pip install unsloth"
            )

        from training.train_monitor import build_prompt_dataset, build_env_reward_fn

        monitor_model, monitor_tok = self._load_monitor()
        dataset = build_prompt_dataset(env_url=self.env_url)
        log.info(f"Co-train dataset: {len(dataset)} message-list prompts (UCB-weighted)")
        reward_fn = build_env_reward_fn(env_url=self.env_url)

        self._train_monitor(monitor_model, monitor_tok, dataset, reward_fn)
        self._train_worker(dataset)


# ---------------------------------------------------------------------------
# Backward-compatible module-level alias
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
    """Backward-compatible alias for ``CoTrainer(...).run()``."""
    CoTrainer(
        monitor_checkpoint=monitor_checkpoint,
        worker_model=worker_model,
        env_url=env_url,
        output_dir=output_dir,
        total_episodes=total_episodes,
        no_wandb=no_wandb,
        dry_run=dry_run,
    ).run()


def phase_for(episode: int) -> CurriculumPhase:
    """Backward-compatible alias for ``CoTrainer()._phase_for()``."""
    return CoTrainer()._phase_for(episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-train HackWatch monitor + worker")
    parser.add_argument("--monitor-checkpoint", default="./runs/monitor_v8/final")
    parser.add_argument("--worker-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./runs/cotrain_v2")
    parser.add_argument("--total-episodes", type=int, default=600)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    CoTrainer(
        monitor_checkpoint=args.monitor_checkpoint,
        worker_model=args.worker_model,
        env_url=args.env_url,
        output_dir=args.output_dir,
        total_episodes=args.total_episodes,
        no_wandb=args.no_wandb,
        dry_run=args.dry_run,
    ).run()

# Run on terminal:
# python -m training.train_cotrain --dry-run
