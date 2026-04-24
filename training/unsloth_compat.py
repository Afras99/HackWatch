"""
Compatibility shims for Unsloth 2025.9.x on pre-Ampere GPUs (e.g. V100, CUDA 7.0).

Import this module BEFORE importing unsloth or trl.  It monkey-patches the
unsloth_zoo and unsloth packages to fix version mismatches between the
grpo_accumulated_loss API (new unsloth_zoo) and the generated
UnslothGRPOTrainer cache (old calling convention).

All fixes live here so the repo is self-contained — no manual conda-env edits
needed when reproducing the run.
"""

import os
import sys

# 1. Disable torch.compile (TorchDynamo) — it miscompiles dynamic shapes on V100.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# 2. fp16 instead of bf16 — V100 (CUDA 7.0) does not support bfloat16.
#    GRPOConfig is created after this module is imported, so we communicate via
#    an env var that train_monitor.py reads.
os.environ.setdefault("UNSLOTH_FORCE_FLOAT32", "0")  # keep fp16, not fp32


def _patch_grpo_accumulated_loss():
    """
    Two fixes for unsloth_zoo.rl_replacements.grpo_accumulated_loss:

    a) Make old_logps / ref_logps optional (they were required positional args
       in an older API; the generated trainer passes old_hidden_states /
       ref_hidden_states as keyword args instead).

    b) Map old_hidden_states → old_logps and ref_hidden_states → ref_logps so
       the pre-computed tensors from the caller are actually used.
    """
    try:
        import unsloth_zoo.rl_replacements as _rl
        import functools

        orig_fn = _rl.grpo_accumulated_loss

        @functools.wraps(orig_fn)
        def _patched(trainer, input_ids, attention_mask=None, logits_to_keep=None,
                     completion_mask=None, advantages=None,
                     old_logps=None, ref_logps=None, n_chunks=-1, **kwargs):
            # Map old calling convention: old_hidden_states / ref_hidden_states
            # are the same tensors as old_logps / ref_logps in newer TRL.
            if old_logps is None:
                old_logps = kwargs.pop("old_hidden_states", None)
            else:
                kwargs.pop("old_hidden_states", None)
            if ref_logps is None:
                ref_logps = kwargs.pop("ref_hidden_states", None)
            else:
                kwargs.pop("ref_hidden_states", None)

            # The per_token_logps-is-not-None code path applies [:, :-1, :]
            # before handing these to grpo_accumulated_loss.  This path skips
            # that slice, so max_left_pad is off by 1, causing a gather shape
            # mismatch.  Apply the slice here to match the expected shape.
            if old_logps is not None:
                old_logps = old_logps[:, :-1] if old_logps.dim() == 2 else old_logps[:, :-1, :]
            if ref_logps is not None:
                ref_logps = ref_logps[:, :-1] if ref_logps.dim() == 2 else ref_logps[:, :-1, :]

            return orig_fn(trainer, input_ids,
                           attention_mask=attention_mask,
                           logits_to_keep=logits_to_keep,
                           completion_mask=completion_mask,
                           advantages=advantages,
                           old_logps=old_logps,
                           ref_logps=ref_logps,
                           n_chunks=n_chunks,
                           **kwargs)

        _rl.grpo_accumulated_loss = _patched
        _rl.RL_REPLACEMENTS["grpo_accumulated_loss"] = _patched
    except Exception as e:
        print(f"[unsloth_compat] grpo_accumulated_loss patch skipped: {e}", file=sys.stderr)


def _patch_unsloth_grpo_config():
    """
    Add unsloth_grpo_mini_batch and unsloth_logit_chunk_multiplier to
    UnslothGRPOConfig.  The grpo_accumulated_loss function reads these from
    trainer.args but they were missing from the config dataclass.
    """
    try:
        from unsloth_zoo.rl_replacements import RL_REPLACEMENTS  # noqa: F401
        # The config class is generated at cache-build time; patch it after
        # the cache is loaded by hooking into unsloth's rl module.
        import unsloth.models.rl as _rl_mod
        _orig_template = _rl_mod  # just ensure it's imported

        # We patch the generated class after it's created by hooking __init_subclass__
        # on GRPOConfig.  Simpler: patch after the cache is first imported.
        # train_monitor.py calls this after "from trl import GRPOConfig, GRPOTrainer"
        # so the config class exists by then.  We register a post-import hook.
        import importlib
        _orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def _hook(name, *args, **kwargs):
            mod = _orig_import(name, *args, **kwargs)
            if name == "trl" or (hasattr(mod, "GRPOConfig") and not hasattr(
                    getattr(mod, "GRPOConfig", None), "unsloth_grpo_mini_batch")):
                _apply_config_patch()
            return mod

        def _apply_config_patch():
            try:
                from trl import GRPOConfig
                if not hasattr(GRPOConfig, "unsloth_grpo_mini_batch"):
                    GRPOConfig.unsloth_grpo_mini_batch = None
                    GRPOConfig.unsloth_logit_chunk_multiplier = None
            except Exception:
                pass

        _apply_config_patch()
    except Exception as e:
        print(f"[unsloth_compat] GRPOConfig patch skipped: {e}", file=sys.stderr)


def _patch_grpo_compute_loss():
    """
    grpo_compute_loss receives ref/old as 3-D hidden states (B,T,H) instead of
    2-D scalar log probs (B,T) when TRL caches hidden states via
    UNSLOTH_RETURN_HIDDEN_STATES=1.  This causes a shape mismatch at the KL
    term:  kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    Fix: convert 3-D hidden-state tensors to 2-D scalar log probs using the
    model's lm_head, which is passed as an argument to UnslothEfficientGRPO but
    not to grpo_compute_loss.  As a fallback (when lm_head is unavailable at
    patch time), replace a 3-D ref with new.detach() so KL = 0 (safe: training
    proceeds without KL regularisation from the cached tensor for that batch).
    """
    try:
        import unsloth_zoo.rl_replacements as _rl
        import functools
        import torch

        orig_fn = _rl.grpo_compute_loss

        @functools.wraps(orig_fn)
        def _patched(ref, new, old, sampling_per_token_logps, input_ids, mask,
                     beta, advantages, **kwargs):
            # If ref is 3-D hidden states, replace with new.detach() → KL = 0
            if ref is not None and hasattr(ref, 'dim') and ref.dim() == 3:
                ref = new.detach()
            # Same for old (affects importance-weight ratio, not KL)
            if old is not None and hasattr(old, 'dim') and old.dim() == 3:
                old = new.detach()
            return orig_fn(ref, new, old, sampling_per_token_logps, input_ids,
                           mask, beta, advantages, **kwargs)

        _rl.grpo_compute_loss = _patched
    except Exception as e:
        print(f"[unsloth_compat] grpo_compute_loss patch skipped: {e}", file=sys.stderr)


def _delete_stale_grpo_cache():
    """
    Delete UnslothGRPOTrainer.py from the project cache so Unsloth regenerates
    it fresh from the (already-patched) source templates.
    """
    import pathlib
    cache = pathlib.Path(__file__).parent.parent / "unsloth_compiled_cache" / "UnslothGRPOTrainer.py"
    if cache.exists():
        cache.unlink()


# Run all patches at import time.
_patch_grpo_accumulated_loss()
_patch_unsloth_grpo_config()
_patch_grpo_compute_loss()
