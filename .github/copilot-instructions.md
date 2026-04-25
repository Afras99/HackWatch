# HackWatch — Copilot Instructions

## What this project is

RL environment for the Meta PyTorch OpenEnv Hackathon. A MONITOR agent (Qwen2.5-3B LoRA)
watches a WORKER agent solve coding tasks and detects reward hacking. Trained with GRPO via TRL.

## Import style

Always absolute imports — never relative.

```python
# CORRECT
from server.exploits import ALL_PRIMITIVES
from training.dynamic_grpo import DynamicSamplingGRPOTrainer

# WRONG
from .exploits import ALL_PRIMITIVES
```

Lazy imports inside functions are intentional for heavy libraries (e.g. `datasets`, `torch`,
`peft`). Do not hoist them to the top level — they are only needed on specific code paths.

```python
# CORRECT — only pay import cost when actually training
def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    ...

# WRONG — pays cost on every import of the module
import torch
from transformers import AutoModelForCausalLM
```

Remove unused imports immediately — dead imports are a code smell.

## Code standards

- No bare `except:` — always catch specific exceptions (`except Exception:` at minimum)
- No hardcoded secrets — credentials via environment variables only
- No mutation of shared state in reward functions (they run concurrently per batch)
- No unused variables or imports
- Functions over 50 lines should be split
- Files over 400 lines should be considered for splitting

## Critical invariants — never break these

1. `planted_label` must NEVER appear in `MonitorObservation`
2. No exploit primitive may write its own name into generated file content or paths
3. `HackWatchState.to_dict()` excludes `planted_label`; only `.to_dict_full()` includes it
4. `heldout_novel` split: `subprocess_escape` and `eval_injection` are held out of training — never add them to the training set

## Training stack

- **Model loading**: `transformers.AutoModelForCausalLM` + PEFT LoRA (r=32)
- **Unsloth**: commented out in `load_model()` — bug in 2026.4.8 causes KL explosion (grad_norm=NaN). Re-enable when fixed.
- **TRL**: `GRPOTrainer` via `DynamicSamplingGRPOTrainer` subclass
- **TRL API**: use `processing_class=tok` not `tokenizer=tok` (renamed in TRL 0.22)
- **GRPO beta**: never set below 0.01

## Key files

| File | Purpose |
|------|---------|
| `server/app.py` | FastAPI app, `/reset` + `/step` + `/demo` |
| `server/environment.py` | `reset()` and `step()` logic |
| `server/exploits.py` | 8 exploit primitives + heldout split |
| `server/reward.py` | F1 + calibration + intervention + latency reward |
| `training/train_monitor.py` | GRPO training pipeline |
| `training/dynamic_grpo.py` | DAPO dynamic sampling subclass |
| `hackwatch/models.py` | `MonitorAction`, `MonitorObservation`, `HackWatchState` |

## GRPOConfig guardrails

- `beta >= 0.01` always
- `loss_type='dr_grpo'` (removes length/variance normalisation bias)
- `processing_class=tok` not `tokenizer=tok`
- `fp16=True`, `bf16=False` on V100 (CUDA 7.0 has no bf16)
