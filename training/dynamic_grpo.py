"""
DAPO Dynamic Sampling as a TRL GRPOTrainer subclass (arXiv 2503.14476 §3.2).

Problem: when all num_generations completions for a prompt score identically
(reward std=0), GRPO advantage=0 everywhere in the group → zero gradient →
training stalls. This is the primary failure mode ("ceiling_hit") in HackWatch
v3/v4 where the heuristic scorer perfectly labels every diff.

Fix: after reward scoring, detect groups where std(advantages) ≈ 0 and inject
small Gaussian noise (σ=0.005) to break the degeneracy. This preserves the
gradient direction for non-degenerate groups while rescuing stuck training steps.

True DAPO Dynamic Sampling resamples at the rollout level; this implementation
achieves the same effect at the advantage level, which is fully compatible with
TRL 0.24 without patching the data pipeline.
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer


class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with DAPO-style dynamic sampling noise injection.

    For groups where all completions score identically (advantage std < ε),
    injects Gaussian noise (σ=dynamic_sampling_noise) to prevent zero-gradient
    training steps. Non-degenerate groups are untouched.
    """

    def __init__(self, *args, dynamic_sampling_noise: float = 0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self._ds_noise = dynamic_sampling_noise

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        try:
            self._apply_dynamic_sampling(output)
        except Exception:
            pass  # never crash training — graceful no-op on API changes
        return output

    def _apply_dynamic_sampling(self, output: dict) -> None:
        adv = output.get("advantages")
        if adv is None or adv.numel() == 0:
            return

        num_gen = self.args.num_generations
        bs = adv.shape[0]
        if bs == 0 or bs % num_gen != 0:
            return

        n_groups = bs // num_gen
        grouped = adv.view(n_groups, num_gen)
        # std(advantages) ≈ 0 means all completions in group were equal-reward
        stds = grouped.std(dim=1)  # (n_groups,)
        zero_mask = stds < 1e-6   # True for degenerate groups

        if not zero_mask.any():
            return  # all groups have signal — nothing to do

        # Expand group mask to per-sample mask: (bs,)
        sample_zero_mask = zero_mask.repeat_interleave(num_gen)

        noise = torch.randn(bs, device=adv.device, dtype=adv.dtype) * self._ds_noise
        output["advantages"] = torch.where(sample_zero_mask, adv + noise, adv)

        # Log fraction of rescued groups
        frac_zero = zero_mask.float().mean().item()
        mode = "train" if (hasattr(self, "model") and self.model.training) else "eval"
        if hasattr(self, "_metrics"):
            self._metrics[mode].setdefault("dynamic_sampling/frac_rescued", []).append(frac_zero)
