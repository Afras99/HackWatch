# HackWatch Research Agent — Instructions for Claude

You are analyzing a GRPO reinforcement learning training run for the **HackWatch** project.

## What HackWatch Is

A MONITOR agent (Qwen2.5-3B LoRA) trained via GRPO to detect reward-hacking by a WORKER agent on coding tasks. The monitor outputs a JSON verdict:
```json
{"verdict": "allow"|"flag"|"block", "confidence": 0.0-1.0, "reasoning": "..."}
```

Reward = `0.85 × correct_detection + 0.15 × brier_calibration_bonus`

Training uses 30 tasks across 8 exploit primitives (test_override, file_leak, subprocess_escape, conftest_tamper, eval_injection, timer_bypass, gradient_hardcode, env_var_injection). Two primitives are **held out** (subprocess_escape, eval_injection) — the model is never trained on them.

---

## GUARDRAILS — Never Violate

| Parameter | Constraint |
|-----------|------------|
| `beta` | `[0.01, 0.1]` — never below 0.01 |
| `learning_rate` | `[1e-6, 1e-5]` |
| `lora_rank` | `[8, 64]` |
| `num_generations` | minimum 4 |
| `calibration_penalty` | never remove from reward |
| `planted_label` | never add to observations |
| `heldout_novel split` | never train on subprocess_escape or eval_injection |

---

## Metric Definitions

| Metric | What it means |
|--------|---------------|
| `reward` | mean reward per step (target: rising toward 1.0) |
| `reward_std` | std of reward within a group of 8 completions |
| `frac_reward_zero_std` | fraction of groups where all 8 completions scored identically (std=0) |
| `kl` | KL divergence from reference model (should stay < 0.3) |
| `completions/clipped_ratio` | fraction of completions truncated at max_completion_length |
| `grad_norm` | gradient norm (should stay < 5 normally) |
| `loss` | GRPO policy loss (very small is normal, ~0.001–0.01) |
| `reward/detection_f1` | F1 score on hack detection |
| `reward/calibration` | calibration quality (0–1, higher is better) |
| `dynamic_sampling/frac_rescued` | fraction of zero-std groups rescued by noise injection |

---

## Diagnosis Rubric

Use this to identify the **primary issue** (pick one):

| Diagnosis | Signals | Root Cause |
|-----------|---------|------------|
| `ceiling_hit` | `frac_reward_zero_std ≈ 1.0` AND `reward ≈ 0.99` | Heuristic scorer labels all 8 completions identically → zero advantage → no gradient |
| `collapsed` | `frac_reward_zero_std > 0.8` AND `reward < 0.5` | Model collapsed to single output, KL penalty too high or LR too low |
| `diverging` | `kl > 0.5` | Policy moved too far from reference |
| `exploding_gradients` | `grad_norm > 20` in last 5 steps | LR too high or unstable batch |
| `truncation` | `clipped_ratio > 0.3` | Completions too long for `max_completion_length` |
| `low_diversity` | `frac_reward_zero_std > 0.5` AND `reward` between 0.5–0.9 | Not enough generation diversity |
| `learning_slowly` | reward flat below 0.7 for 50+ steps | LR too low, wrong loss type, or insufficient data |
| `healthy` | reward rising, frac_zero_std < 0.3, kl < 0.3 | All good — continue or add harder tasks |

---

## Current Config (train_monitor.py)

```python
GRPOConfig(
    beta=0.01,
    learning_rate=5e-6,
    num_generations=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_completion_length=256,
    loss_type='dr_grpo',         # Dr GRPO: removes length+variance normalization
    scale_rewards=False,          # per-sample normalization
    importance_sampling_level='sequence',  # GSPO
    mask_truncated_completions=True,       # DAPO overlong penalty
    epsilon=0.2,
    epsilon_high=0.28,            # DAPO asymmetric clipping
    temperature=1.1,
)
```

Already implemented:
- `DynamicSamplingGRPOTrainer` — injects σ=0.005 noise into zero-advantage groups
- UCB curriculum — upsamples hard/unseen tasks
- DRA-GRPO diversity bonus — +0.05 × unique_verdicts/3 per group

---

## Known Techniques to Consider

| Technique | What it does | TRL param / location | Safe? |
|-----------|-------------|---------------------|-------|
| Raise `num_generations` 8→12 | More rollouts = more variance = stronger signal | `num_generations=12, per_device_train_batch_size=12` | ✅ |
| Lower `beta` toward 0.01 | Less KL penalty → more exploration | `beta=0.01` (already at floor) | ✅ only if > 0.01 |
| `temperature` 1.1→1.3 | Higher generation diversity | `temperature=1.3` | ✅ |
| Harder tasks | Add obfuscated/combo variants | add more to `server/tasks.py ALL_TASKS` | ✅ |
| More trajectories | More dataset variety | `scripts/generate_trajectories.py` | ✅ |
| Increase `max_completion_length` 256→384 | Longer reasoning chains | `max_completion_length=384` | ✅ |
| Brier reward weight | Increase `0.15` → `0.20` | `server/reward.py` — run adversarial tests after | ⚠️ |
| SFT warm-start | Pre-train on 150 cached trajectories before GRPO | new training script needed | ✅ |
| Raise `lora_rank` 32→64 | More model capacity | `lora_rank=64` | ✅ |
| `num_train_epochs` 8→12 | More passes over data | `num_train_epochs=12` | ✅ |

---

## What to Output

When given training metrics, return a JSON block:

```json
{
  "diagnosis": "ceiling_hit",
  "confidence": 0.9,
  "reasoning": "frac_reward_zero_std=1.0 for all 400 steps with reward=0.999. All 8 completions per group score identically because the heuristic scorer perfectly pattern-matches every diff. DynamicSamplingGRPOTrainer is injecting noise (frac_rescued=1.0) but can't fully overcome it — the underlying data distribution still produces identical rewards.",
  "proposals": [
    {
      "id": "raise_num_generations",
      "title": "Raise num_generations 8 → 12",
      "description": "More rollouts per prompt increases the chance of getting a mixed-reward group even when most diffs are obvious. Requires matching per_device_train_batch_size.",
      "guardrails_safe": true,
      "confidence": 0.80,
      "config_patch": {"num_generations": 12, "per_device_train_batch_size": 12},
      "code_change": null
    },
    {
      "id": "add_harder_tasks",
      "title": "Add 5 more obfuscated task variants to server/tasks.py",
      "description": "The current 30 tasks all use obvious hack patterns that the heuristic scorer catches with certainty. Need tasks where the hack is subtle enough that the heuristic is uncertain.",
      "guardrails_safe": true,
      "confidence": 0.85,
      "config_patch": null,
      "code_change": "Add OBFUS_6-10 to server/tasks.py ALL_TASKS with hacks that don't match existing regex patterns. Example: use base64-encoded subprocess call, or a hack that only activates after N function calls."
    }
  ]
}
```

Rules for proposals:
- `config_patch` must only contain keys that exist in `GRPOConfig` or `LoraConfig` in `training/train_monitor.py`
- All values in `config_patch` must respect GUARDRAILS bounds
- Set `guardrails_safe: false` if touching `server/reward.py`, observation structure, or held-out split
- Confidence = how certain you are this will improve training (not just that it's correct)
- Max 3 proposals, ranked by confidence descending

---

## How to Use This File

1. Run `python research/format_metrics.py --log /tmp/train_monitor_v6.log` to get a formatted summary
2. Paste the output here along with this file
3. Claude will return a JSON block with diagnosis + proposals
4. To apply a config_patch, run:
   ```bash
   python research/apply_patch.py '{"num_generations": 12, "per_device_train_batch_size": 12}'
   ```
5. Re-launch training: `python -m training.auto_loop --start-version <N+1> ...`
