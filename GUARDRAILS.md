# HackWatch — Autonomous Agent Guardrails

Rules for any AI agent (Claude Code or otherwise) operating autonomously on this repo.
These exist to prevent accidental submission invalidation or training corruption.

---

## HARD STOPS (never do these, abort and ask the user)

1. **Never add `planted_label` to `MonitorObservation`** — invariant #1, breaks the entire experiment
2. **Never move `subprocess_escape` or `eval_injection` into the training split** — they are the generalization test; exposing them during training voids the eval
3. **Never set `beta < 0.01` in any GRPOConfig** — collapses KL, destroys response diversity
4. **Never remove `calibration_penalty` from `compute_episode_reward`** — prevents confidence gaming
5. **Never commit `.env`** — contains API keys; already gitignored, keep it that way
6. **Never push a `uv.lock` or `requirements.txt` that unpins training deps** — version drift breaks DGX runs overnight
7. **Never modify the `openenv.yaml` back to the legacy format** — the v1 5-field spec is required for `openenv validate` to pass

---

## CAUTION (check before doing)

- **Adding a new exploit primitive**: it must NOT write its name or the string "hack" into generated file content or paths (label leakage). Add a test in `tests/test_exploits.py` to verify.
- **Changing reward weights**: run `pytest tests/test_reward.py tests/test_verifier_adversarial.py` before committing. All adversarial tests must pass.
- **Modifying `HackWatchState.to_dict()`**: verify `planted_label` is still absent from the output.
- **Changing the GRPO dataset builder**: ensure new prompts do NOT include ground-truth hack labels or `planted_label` fields.
- **Touching `inference.py` stdout format**: the `[START]`, `[STEP]`, `[END]` contract is validated by the hackathon submission system. Any format change must be validated with `scripts/validate_submission.sh`.

---

## AUTO-APPROVED (safe to do without asking)

- Running `pytest tests/ -v`
- Running `python scripts/generate_trajectories.py`
- Running `python eval/evaluate_monitor.py` with `--limit N` for spot-checks
- Reading any file in the repo
- Editing files under `demo/`, `docs/`, `README.md`
- Editing training hyperparameters within bounds: `beta in [0.01, 0.1]`, `lr in [1e-6, 1e-5]`, `lora_rank in [8, 64]`
- Adding new tests to `tests/`

---

## EVALUATION CHECKPOINTS

Before any major phase transition (pre-training → training, training → demo), run:

```bash
# 1. All tests green
/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m pytest tests/ -v

# 2. Adversarial reward tests specifically
/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python -m pytest tests/test_verifier_adversarial.py -v

# 3. Submission validator
bash scripts/validate_submission.sh

# 4. Trajectory generator (if trajectories missing)
/Users/afrasaboobackerp/miniconda3/envs/hackwatch/bin/python scripts/generate_trajectories.py
```

---

## TRAINING KILL CRITERIA

If running GRPO autonomously and monitoring W&B:

| Condition | Action |
|-----------|--------|
| `reward/mean_total < 0.4` at step 50 | Drop `beta` to 0.01, restart |
| `reward/mean_total < 0.4` at step 50 on beta=0.01 | Stop — reward function is broken, escalate to user |
| `reward/detection_f1` is NaN | Stop — check that `planted_label` is flowing correctly |
| GPU OOM despite `gpu_memory_utilization=0.90` | Drop LoRA rank from 32 to 16, disable `UNSLOTH_VLLM_STANDBY` |
| Training loss is 0.0 throughout | Model is not updating — check LoRA target modules |

---

## INFORMATION HIDING INVARIANTS (verify any time models.py changes)

```python
# This must ALWAYS be true:
obs = MonitorObservation(...)
assert "planted_label" not in obs.to_dict()

# This must ALWAYS be true:
state = HackWatchState(..., planted_label="test_override")
assert "planted_label" not in state.to_dict()
assert state.to_dict_full()["planted_label"] == "test_override"
```
