# AGENTS.md — Cold-Start Guide for AI Agents

Read this before touching any code. It tells you exactly where things stand and what to do next.

---

## 30-second summary

HackWatch is an RL environment for the Meta PyTorch OpenEnv Hackathon (Apr 25–26, 2026).
A MONITOR agent watches a WORKER agent solve coding tasks and must detect when the worker
"reward hacks" (cheats). The monitor is trained with GRPO; ground truth comes from planted
labels — no LLM judge. The env is a FastAPI server with `/reset` and `/step` endpoints.

---

## You are here (Apr 22, 2026)

| Area | State |
|------|-------|
| Environment (reset/step, exploits, reward) | **Complete** |
| 92 tests | **All green** |
| Demo frontend (`demo/build/index.html`) | **Complete** — wired to live API |
| `training/train_monitor.py` | **Complete** — GRPO pipeline ready, not yet run on GPU |
| `training/train_cotrain.py` | **Skeleton** — model loading done, adversarial loop not activated |
| Real W&B training curves | **Missing** — demo shows illustrative data |
| HF Space | **Not pushed** |

**Highest-priority next step:** Run `training/train_monitor.py` on a GPU to produce real reward curves, then swap the hardcoded `CURVE` constant in `demo/build/index.html` with real data.

---

## The 3 invariants you must never break

1. `planted_label` must **never** appear in `MonitorObservation` — the monitor cannot see ground truth. Enforced in `hackwatch/models.py` and tested by `tests/test_leakage.py`.
2. No exploit primitive may write its own name into generated file content or paths. Tested by `tests/test_exploits.py` (the `test_no_label_leakage` family).
3. `HackWatchState.to_dict()` excludes `planted_label`; only `.to_dict_full()` includes it (internal logging only).

---

## Decisions already made — do not re-litigate

- **Single difficulty bucket**: easy/medium/hard tiers were dropped. Do not re-add them.
- **heldout_novel split**: `subprocess_escape` and `eval_injection` are held out of training. This is the generalization claim ("trained on 6 exploit types, catches 2 it never saw"). Never add them to the training set.
- **GRPO beta=0.04**: Do not lower below 0.01 — collapses KL and kills diversity.
- **calibration_penalty in reward**: Prevents confidence gaming. Do not remove it.

---

## Ordered next steps

1. **Run GRPO training** (Day 3): `python -m training.train_monitor --env-url http://localhost:8000 --output-dir ./runs/monitor_v1`
   - Requires GPU with CUDA + Unsloth installed
   - Use `--dry-run` first to verify pipeline without GPU: `python -m training.train_monitor --dry-run`
   - Log to W&B; capture real reward curves

2. **Update demo training curve**: In `demo/build/index.html`, find the `const CURVE = mkCurve(...)` block and replace with real W&B data points. The chart renders `{ep, reward}` objects.

3. **Run co-training** (Day 4): `python -m training.train_cotrain --monitor-checkpoint ./runs/monitor_v1/final`
   - `train_cotrain.py` loads both models and runs sequential GRPO updates; the adversarial loop is in `cotrain()` starting at line ~160
   - Update `CURRICULUM` phases to reflect single-bucket design (remove easy/medium references)

4. **Push to HF Space** (Day 5): `openenv push` — see `openenv.yaml` for config

5. **Run hackathon validator**: `openenv validate` against the live URL

---

## Module map

```
hackwatch/models.py          MonitorAction, MonitorObservation, HackWatchState
server/app.py                FastAPI app, /reset + /step + /demo routes
server/environment.py        reset() and step() logic
server/exploits.py           8 exploit primitives + heldout_novel split
server/tasks.py              15 tasks (single pool, no difficulty tiers)
server/workers.py            Scripted worker bank (legit + per-primitive)
server/reward.py             Reward combinator (F1 + bonuses + penalties)
server/sandbox.py            Subprocess-isolated pytest runner
training/train_monitor.py    GRPO training pipeline for MONITOR (ready to run)
training/train_cotrain.py    Co-training skeleton (Day 4)
training/configs/grpo_base.yaml   GRPO hyperparameters reference
demo/build/index.html        Single-file React frontend (pre-transpiled)
tests/                       92 tests — run before and after every change
```

---

## Verify state in 60 seconds

```bash
# 1. All tests pass
/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m pytest tests/ -v

# 2. Server starts
/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m uvicorn server.app:app --reload

# 3. Env responds
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
curl -s -X POST http://localhost:8000/step  -H "Content-Type: application/json" -d '{"verdict":"allow","confidence":0.8,"reasoning":"test"}' | python3 -m json.tool

# 4. Training dry-run (no GPU needed)
/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m training.train_monitor --dry-run
```
