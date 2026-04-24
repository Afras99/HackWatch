"""
Maps experiment analyzer diagnosis → paper-backed improvement proposals.

All proposals are pre-checked against GUARDRAILS.md bounds:
  - beta ∈ [0.01, 0.1]
  - lr ∈ [1e-6, 1e-5]
  - lora_rank ∈ [8, 64]

Usage:
  python research/hypothesis_generator.py --diagnosis ceiling_hit
  python research/hypothesis_generator.py --report research/health_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Proposal:
    id: str
    title: str
    description: str
    source: str          # paper / doc reference
    guardrails_safe: bool
    confidence: float    # 0-1: how confident we are this will help
    config_patch: dict[str, Any] | None  # GRPOConfig kwarg changes
    code_change: str | None              # free-text description of code change


PROPOSALS: dict[str, list[Proposal]] = {
    "ceiling_hit": [
        Proposal(
            id="dr_grpo_loss",
            title="Switch to Dr GRPO loss (remove length bias)",
            description=(
                "HackWatch monitor outputs are short JSON (45-130 tok). "
                "Standard GRPO divides advantage by response length — incentivising padding. "
                "Dr GRPO removes length normalization, keeping gradient signal honest. "
                "Also removes group-std normalization which causes calibration overconfidence "
                "(arXiv 2509.23870)."
            ),
            source="arXiv 2503.20783 (Dr GRPO); arXiv 2509.23870 (calibration)",
            guardrails_safe=True,
            confidence=0.85,
            config_patch={"loss_type": "dr_grpo", "scale_rewards": "batch"},
            code_change=None,
        ),
        Proposal(
            id="finer_reward_confidence",
            title="Add continuous confidence reward component",
            description=(
                "Current heuristic reward is nearly discrete (0, 0.5, 1.0). "
                "Adding a continuous component based on confidence calibration "
                "breaks reward ties within a group, restoring GRPO gradient signal. "
                "Use Brier score: r += 0.2 * (1 - (conf - correct)^2). "
                "CAUTION: requires modifying server/reward.py — run adversarial tests after."
            ),
            source="Brier (1950); arXiv 2408.10215 RLCR §calibration",
            guardrails_safe=False,  # CAUTION — reward change
            confidence=0.80,
            config_patch=None,
            code_change=(
                "In server/reward.py compute_episode_reward(): "
                "add brier_term = 1.0 - mean((conf - int(correct))**2 for each step). "
                "Weight: 0.2 * brier_term. Run pytest tests/test_reward.py tests/test_verifier_adversarial.py."
            ),
        ),
        Proposal(
            id="curriculum_ucb",
            title="Curriculum sampling via UCB over exploit primitives",
            description=(
                "When reward=1.0 across batches, all tasks are equally easy. "
                "UCB curriculum (DUMP, 2025) upweights primitives with lowest average reward "
                "so training focuses on the hardest cases. "
                "Implement as a weighted sampler over dataset rows keyed by hack_primitive."
            ),
            source="arXiv 2504.13592 RCS+GRPO; DUMP UCB curriculum 2025",
            guardrails_safe=True,
            confidence=0.75,
            config_patch=None,
            code_change=(
                "In training/train_monitor.py build_prompt_dataset(): "
                "add curriculum_weights dict initialized uniform. "
                "After each checkpoint eval, reweight via UCB: w_i = r_i_mean + sqrt(2*log(N)/n_i). "
                "Pass weights to Dataset.from_list() sampling."
            ),
        ),
    ],
    "low_diversity": [
        Proposal(
            id="raise_num_generations",
            title="Raise num_generations 8 → 12",
            description="More rollouts per prompt = more within-group reward variance = stronger GRPO signal.",
            source="GRPO paper (Shao et al. 2024)",
            guardrails_safe=True,
            confidence=0.70,
            config_patch={"num_generations": 12, "per_device_train_batch_size": 12},
            code_change=None,
        ),
        Proposal(
            id="dra_grpo_diversity_reward",
            title="Diversity-Aware Reward Adjustment (DRA-GRPO)",
            description=(
                "Add a semantic diversity bonus: within each group of 12 completions, "
                "compute pairwise similarity (by verdict+confidence), penalize near-duplicates. "
                "r_adjusted = r + 0.1 * diversity_score. Encourages exploration."
            ),
            source="arXiv 2505.09655 DRA-GRPO",
            guardrails_safe=True,
            confidence=0.65,
            config_patch=None,
            code_change=(
                "In training/train_monitor.py env_reward_fn(): "
                "after computing rewards list, compute group diversity: "
                "verdicts = [parse_verdict(t)['verdict'] for t in texts]; "
                "diversity = len(set(verdicts)) / len(verdicts); "
                "rewards = [r + 0.1 * diversity for r in rewards]."
            ),
        ),
    ],
    "learning_slowly": [
        Proposal(
            id="dr_grpo_loss",
            title="Switch to Dr GRPO loss",
            description="Removes length and variance normalization biases slowing learning.",
            source="arXiv 2503.20783",
            guardrails_safe=True,
            confidence=0.80,
            config_patch={"loss_type": "dr_grpo", "scale_rewards": "batch"},
            code_change=None,
        ),
        Proposal(
            id="raise_lr",
            title="Raise learning rate 2e-6 → 5e-6",
            description="Learning rate may be too conservative for 200 steps on a classification task.",
            source="GUARDRAILS bounds: lr ∈ [1e-6, 1e-5]",
            guardrails_safe=True,
            confidence=0.65,
            config_patch={"learning_rate": 5e-6},
            code_change=None,
        ),
    ],
    "diverging": [
        Proposal(
            id="raise_beta",
            title="Raise beta to 0.08 (increase KL penalty)",
            description="KL > 0.5 means the policy is drifting too far from reference. Raise beta.",
            source="GUARDRAILS bounds: beta ∈ [0.01, 0.1]",
            guardrails_safe=True,
            confidence=0.90,
            config_patch={"beta": 0.08},
            code_change=None,
        ),
    ],
    "truncation": [
        Proposal(
            id="raise_completion_length",
            title="Raise max_completion_length 256 → 512",
            description="clipped_ratio > 0.3 means completions are hitting the ceiling. Increase budget.",
            source="TRL GRPOConfig docs",
            guardrails_safe=True,
            confidence=0.95,
            config_patch={"max_completion_length": 512},
            code_change=None,
        ),
    ],
    "collapsed": [
        Proposal(
            id="lower_beta",
            title="Lower beta to 0.01 (minimum allowed)",
            description="reward near 0 with zero std suggests KL penalty dominating. Relax it.",
            source="GUARDRAILS bounds: beta ≥ 0.01",
            guardrails_safe=True,
            confidence=0.80,
            config_patch={"beta": 0.01},
            code_change=None,
        ),
    ],
    "healthy": [
        Proposal(
            id="continue",
            title="Continue training — no changes needed",
            description="Metrics look healthy. Let the run finish.",
            source="observation",
            guardrails_safe=True,
            confidence=1.0,
            config_patch=None,
            code_change=None,
        ),
    ],
}


def generate(diagnosis: str) -> list[dict]:
    proposals = PROPOSALS.get(diagnosis, PROPOSALS.get("healthy"))
    return [asdict(p) for p in proposals]


def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--diagnosis", choices=list(PROPOSALS.keys()))
    grp.add_argument("--report", help="Path to health report JSON from experiment_analyzer.py")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.report:
        report = json.loads(open(args.report).read())
        diagnosis = report.get("diagnosis", "healthy")
    else:
        diagnosis = args.diagnosis

    proposals = generate(diagnosis)

    print(f"\nDiagnosis: {diagnosis}")
    print(f"Proposals ({len(proposals)}):")
    for pr in proposals:
        safe = "✅" if pr["guardrails_safe"] else "⚠️ CAUTION"
        print(f"  [{safe}] {pr['id']} (confidence={pr['confidence']:.0%}): {pr['title']}")
    print()

    out = json.dumps({"diagnosis": diagnosis, "proposals": proposals}, indent=2)
    if args.out:
        open(args.out, "w").write(out)
        print(f"Proposals written to {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
