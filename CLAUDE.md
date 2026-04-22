# HackWatch — Claude Code Context

## Project

OpenEnv-compatible RL environment for the Meta PyTorch OpenEnv Hackathon (finale Apr 25-26, 2026).
Trains a MONITOR agent to detect reward-hacking by a WORKER agent on coding tasks.

## Conda environment

Always use: `/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python`

Run tests: `/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m pytest tests/ -v`

Install deps: `/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/pip install -e ".[dev]"`

Start server: `/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m uvicorn server.app:app --reload`

## Critical invariants (never break these)

1. `planted_label` must NEVER appear in `MonitorObservation` — monitor cannot see ground truth
2. No exploit primitive may write its own name into generated file content or paths (label leakage)
3. `HackWatchState.to_dict()` excludes `planted_label`; only `to_dict_full()` includes it (internal logging only)
4. Sandbox (`server/sandbox.py`) runs pytest in a temp dir — never shares fs with agent I/O

## Stack

- Python 3.11, FastAPI, Pydantic v2
- Training: TRL 0.24 + Unsloth + Qwen2.5-3B-Instruct (LoRA r=32)
- Tests: pytest (92 tests, all green)
- Demo: `/demo/build/index.html` served at `/demo` by FastAPI

## Current status (as of Apr 22, 2026)

**Done:**
- Full environment: reset/step, 8 exploit primitives, 15 tasks, 92 tests green
- reward.py fully implemented (F1 + calibration_penalty + intervention_cost + latency_bonus)
- Demo frontend: real API wired, 6 features (multi-step loop, confidence slider, verdict buttons, episode history, reasoning textarea, exploit banner), pre-transpiled React.createElement (no Babel)
- training/train_monitor.py: complete GRPO pipeline with dataset builder, reward fn, model loading

**Stubbed / next up:**
- training/train_cotrain.py: co-training loop skeleton — needs the adversarial curriculum activated (Day 4)
- No real W&B training curves yet — demo shows illustrative data; replace with real after Day 3 run
- HF Space not pushed yet

**Blocked:**
- GPU access needed to run actual GRPO training (Unsloth requires CUDA)

## Design decisions (do not re-litigate)

- **Single difficulty bucket**: easy/medium/hard tiers were dropped. All tasks draw from one pool. The `heldout_novel` split (6 train primitives, 2 holdout) is the only stratification that matters.
- **heldout_novel split**: `subprocess_escape` and `eval_injection` are held out. Never add them to the training set — they are the generalization test.
- **GRPO beta=0.04**: Do not lower below 0.01 — collapses KL penalty and destroys diversity.

## DO NOT (training code guardrails)

- Do NOT remove or weaken the `heldout_novel` split in `server/exploits.py`
- Do NOT set `beta < 0.01` in any GRPOConfig
- Do NOT remove `calibration_penalty` from the reward combinator — it prevents confidence gaming
- Do NOT add `planted_label` to `MonitorObservation` (invariant #1)
- Do NOT add difficulty tier logic (easy/medium/hard) — decision was made to drop it

## Day milestones

- Day 1 ✓: repo scaffold, models, 8 exploits, 15 tasks, env reset/step, 92 tests green
- Day 2 ✓: reward.py combinator, leakage audit, demo frontend live
- Day 3: run GRPO training on GPU, capture real W&B curves, swap illustrative chart in demo
- Day 4: co-training worker + monitor with adversarial curriculum
- Day 5: HF Space push, validator run, final demo polish

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
