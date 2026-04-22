# HackWatch — Implementation Plan

Meta PyTorch OpenEnv Hackathon (finale April 25–26, 2026)

**Goal:** OpenEnv-compatible RL environment that trains a MONITOR agent to detect
reward-hacking by a WORKER agent on coding tasks. Ground truth from planted hack labels
— reward function is 100% deterministic (no LLM judge).

**Stack:** Python 3.11, FastAPI, Pydantic v2, openenv-core, TRL 0.24 + Unsloth +
Qwen2.5-3B-Instruct, Docker, HF Spaces, React + recharts demo.

---

## Decisions made (do not re-open)

- **Single difficulty bucket**: easy/medium/hard tiers dropped. All 15 tasks draw from one pool.
- **heldout_novel split**: `subprocess_escape` and `eval_injection` held out. This is the generalization claim. Never add them to training set.
- **GRPO beta=0.04**: Do not lower below 0.01.
- **No LLM judge**: reward is fully deterministic from `planted_label` vs. monitor verdict.

---

## Day 1 (Apr 20) ✓ COMPLETE

- [x] T1: git init, pyproject.toml, scaffold repo layout
- [x] T2: models.py, stub server/environment.py, curl /reset + /step roundtrip
- [x] T3: server/exploits.py — 8 hack primitives, all unit-tested
- [x] T4: server/tasks.py — 15 base tasks (single pool, no tiers)
- [x] T5: server/workers.py — scripted worker bank (legit + per-primitive)
- [x] T6: Wire reset() + step() in environment.py
- [x] T7: inference.py baseline runner with [START]/[STEP]/[END] tags
- [x] T8: Dockerfile — FROM python:3.11-slim, uvicorn entrypoint

## Day 2 (Apr 21–22) ✓ COMPLETE

- [x] Implement server/reward.py (F1 + calibration_penalty + intervention_cost + latency_bonus)
- [x] heldout_novel split enforced in exploits.py
- [x] Leakage audit — 92 tests green including label-leakage assertions
- [x] Demo frontend: wired to live API, 6 features, pre-transpiled React

---

## Day 3 Tasks

- [ ] Run GRPO training on GPU: `python -m training.train_monitor --env-url http://localhost:8000 --output-dir ./runs/monitor_v1`
  - Use `--dry-run` first to verify pipeline
  - Log to W&B; verify reward signal non-zero on first batch
- [ ] Save checkpoint after 100 steps; verify LoRA adapter serializes cleanly
- [ ] Replace hardcoded `CURVE` in `demo/build/index.html` with real W&B data points

---

## Day 4 Tasks

- [ ] Activate co-training loop: `python -m training.train_cotrain --monitor-checkpoint ./runs/monitor_v1/final`
  - Update CURRICULUM phases in train_cotrain.py — remove easy/medium references, keep heldout_novel phase
  - Monitor trains against adversarial worker; both improve together
- [ ] Held-out evaluation on heldout_novel primitives (subprocess_escape, eval_injection)
- [ ] Record eval metrics: precision, recall, F1, false-positive rate

---

## Day 5 (Demo Day — Apr 25–26)

- [ ] HF Space: push, README with obs/action/state spec, demo GIF
- [ ] Run hackathon validator (`openenv validate`), fix deviations, tag v1.0
- [ ] GitHub: submit to hackathon

---

## Repo Layout

```
hackwatch/
├── __init__.py
├── models.py              ✅ MonitorAction, MonitorObservation, HackWatchState
├── client.py              ✅ HackWatchEnvClient(EnvClient)
├── openenv.yaml           ✅
├── pyproject.toml         ✅
├── README.md              ✅ (needs obs/action/state spec expansion)
├── Dockerfile             ✅
├── inference.py           ✅ [START]/[STEP]/[END] tags
├── server/
│   ├── app.py             ✅ create_fastapi_app + /demo mount
│   ├── environment.py     ✅ reset/step wired
│   ├── tasks.py           ✅ 15 tasks
│   ├── exploits.py        ✅ 8 primitives + combinator stub
│   ├── workers.py         ✅ scripted worker bank
│   ├── reward.py          ⚠️  skeleton only — needs full implementation (Day 2)
│   └── sandbox.py         ✅ subprocess-isolated pytest runner
├── training/
│   ├── train_monitor.py   ✅ complete GRPO pipeline — ready to run on GPU
│   ├── train_cotrain.py   ⚠️  Day 4 — update CURRICULUM, activate adversarial loop
│   └── configs/grpo_base.yaml ✅
├── demo/
│   └── build/index.html   ✅ live API, 6 features, pre-transpiled React (illustrative training curve)
└── tests/
    └── (92 tests green — test_env_smoke, test_exploits, test_reward, test_leakage, ...)
```

---

## Key Design Decisions

1. **planted_label never in MonitorObservation** — enforced at dataclass level; reward.py
   reads it from server-side state only.
2. **Deterministic reward** — no LLM judge; F1 on single episode aggregated across batch.
3. **Scripted workers Day 3, adversarial co-train Day 4** — avoids cold-start problem.
4. **Beta=0.04 in GRPO** — Lewis Tunstall's collapse warning; do not set to 0.001.
5. **Label leakage mitigations** — all injected code uses generic names (_PRECOMP, _rc, _HW_*);
   test_exploits.py asserts no label string appears verbatim in any diff.
