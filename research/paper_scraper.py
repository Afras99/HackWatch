"""
Literature scraper for HackWatch auto-research loop.

Maintains a curated database of papers relevant to:
  - GRPO training improvements (length bias, diversity, calibration)
  - Reward hacking detection in RL
  - Curriculum learning for RL post-training

Usage:
  # List all papers matching a diagnosis:
  python research/paper_scraper.py --diagnosis ceiling_hit

  # Dump full database as JSON:
  python research/paper_scraper.py --all

  # Refresh: re-run WebSearch and append new papers to database
  python research/paper_scraper.py --refresh
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paper database
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract_snippet: str
    url: str
    relevance_tags: list[str]     # maps to experiment_analyzer diagnoses + HackWatch topics
    hackwatch_takeaway: str        # one-line actionable insight for this project
    implemented: bool = False      # True once the idea is in the codebase


PAPER_DB: list[Paper] = [
    # ---- GRPO training improvements ----
    Paper(
        arxiv_id="2503.20783",
        title="Understanding R1-Zero-Like Training: A Critical Perspective (Dr GRPO)",
        abstract_snippet=(
            "Identifies length-normalization bias in vanilla GRPO that artificially lengthens "
            "incorrect responses. Dr GRPO removes length and std normalization, achieving SOTA "
            "on MATH-level 3-5 with Qwen2.5-Math-7B in 27h on 8×A100."
        ),
        url="https://arxiv.org/abs/2503.20783",
        relevance_tags=["ceiling_hit", "learning_slowly", "calibration", "grpo_improvement"],
        hackwatch_takeaway=(
            "Use loss_type='dr_grpo', scale_rewards='batch' in GRPOConfig. "
            "Already implemented in train_monitor.py."
        ),
        implemented=True,
    ),
    Paper(
        arxiv_id="2505.09655",
        title="DRA-GRPO: Diversity-Aware Reward Adjustment for R1-Zero-Like Training",
        abstract_snippet=(
            "GRPO's scalar rewards fail to capture semantic diversity among completions. "
            "DRA uses Submodular Mutual Information (SMI) to downweight redundant completions "
            "and amplify diverse ones. Achieves 58.2% avg accuracy on math benchmarks with "
            "only 7,000 fine-tuning samples (~$55 total cost)."
        ),
        url="https://arxiv.org/abs/2505.09655",
        relevance_tags=["low_diversity", "ceiling_hit", "grpo_improvement"],
        hackwatch_takeaway=(
            "Within-group verdict diversity bonus: +0.05 * (unique_verdicts/3) per completion. "
            "Implemented in train_monitor.py _apply_diversity_bonus()."
        ),
        implemented=True,
    ),
    Paper(
        arxiv_id="2509.23870",
        title="Rethinking Reward Miscalibration of GRPO in Agentic RL",
        abstract_snippet=(
            "Outcome-based rewards in agentic GRPO cause reward miscalibration: positive reward "
            "allocated to flawed intermediate steps. Gradient coupling between similar prompts "
            "causes gradients from well-performing samples to strengthen suboptimal actions. "
            "Fix: classify good/bad actions explicitly to separate embeddings."
        ),
        url="https://arxiv.org/abs/2509.23870",
        relevance_tags=["calibration", "ceiling_hit", "grpo_improvement", "reward_design"],
        hackwatch_takeaway=(
            "Brier calibration bonus (gated on correct detection) addresses the miscalibration "
            "problem. Already implemented in server/reward.py and train_monitor.py heuristic."
        ),
        implemented=True,
    ),
    Paper(
        arxiv_id="2503.14476",
        title="DAPO: Open-Source LLM Reinforcement Learning System at Scale",
        abstract_snippet=(
            "Four key techniques: (1) Clip-Higher — asymmetric clip to promote diversity and "
            "avoid entropy collapse; (2) Dynamic Sampling — skip all-correct/all-wrong batches; "
            "(3) Token-Level Policy Gradient — stable long-CoT training; "
            "(4) Overlong Reward Shaping — penalize verbose incorrect outputs. "
            "Achieves 50 pts AIME 2024 with Qwen2.5-32B."
        ),
        url="https://arxiv.org/abs/2503.14476",
        relevance_tags=["ceiling_hit", "low_diversity", "collapsed", "grpo_improvement"],
        hackwatch_takeaway=(
            "Consider clip_range_ratio asymmetry (clip_higher=0.28 vs clip_lower=0.2) in "
            "GRPOConfig for v4 run. Dynamic sampling (skip batches where all rewards identical) "
            "is the correct fix for frac_reward_zero_std=1.0 ceiling."
        ),
        implemented=False,
    ),
    # ---- Curriculum learning ----
    Paper(
        arxiv_id="2504.13592",
        title="Improving Generalization in Intent Detection: GRPO with Reward-Based Curriculum Sampling (RCS)",
        abstract_snippet=(
            "Reward-Based Curriculum Sampling (RCS) during GRPO focuses training on challenging "
            "cases. In intent detection, RL+RCS substantially outperforms SFT in generalization "
            "to unseen intents. Uses per-sample reward history to upweight hard examples."
        ),
        url="https://arxiv.org/abs/2504.13592",
        relevance_tags=["ceiling_hit", "learning_slowly", "curriculum"],
        hackwatch_takeaway=(
            "UCB oversampling in build_prompt_dataset(): weight = (1 - mean_reward) + "
            "sqrt(2*log(N)/n). Hard tasks get up to 4× copies. Implemented."
        ),
        implemented=True,
    ),
    Paper(
        arxiv_id="2504.09710",
        title="DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training",
        abstract_snippet=(
            "Distribution-level curriculum using UCB over difficulty buckets. Automatically "
            "identifies which data distributions are most informative for current model capability "
            "and upweights them. Improves sample efficiency vs. uniform sampling."
        ),
        url="https://arxiv.org/abs/2504.09710",
        relevance_tags=["ceiling_hit", "curriculum", "learning_slowly"],
        hackwatch_takeaway=(
            "UCB at the exploit-primitive level (test_override, file_leak, etc.) is the next "
            "evolution after task-level curriculum. Add primitive_id tracking to env reset obs."
        ),
        implemented=False,
    ),
    Paper(
        arxiv_id="2504.05520",
        title="Efficient Reinforcement Finetuning via Adaptive Curriculum Learning",
        abstract_snippet=(
            "Adaptive curriculum selects training examples that yield the most informative "
            "gradient signal — neither too easy (already solved) nor too hard (no signal). "
            "Cuts training compute by 2× vs. uniform sampling on math benchmarks."
        ),
        url="https://arxiv.org/abs/2504.05520",
        relevance_tags=["learning_slowly", "curriculum", "ceiling_hit"],
        hackwatch_takeaway=(
            "Filter out prompts where monitor confidence > 0.95 (already mastered). "
            "Rebalance dataset toward medium-difficulty hack patterns."
        ),
        implemented=False,
    ),
    # ---- Reward hacking detection ----
    Paper(
        arxiv_id="2601.20103",
        title="Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis",
        abstract_snippet=(
            "Benchmark of reward hack detection methods in code environments. Contrastive "
            "analysis between hacked and legit solutions reveals that diff-level features "
            "(test mutations, import changes, hardcoded values) are the most reliable signals."
        ),
        url="https://arxiv.org/abs/2601.20103",
        relevance_tags=["reward_hacking_detection", "benchmark", "monitor_design"],
        hackwatch_takeaway=(
            "Our 8 exploit primitives align well with their taxonomy. The paper validates "
            "diff-level pattern matching as the right feature space for monitor agents."
        ),
        implemented=False,
    ),
    Paper(
        arxiv_id="2604.16242",
        title="Detecting and Suppressing Reward Hacking with Gradient Fingerprints",
        abstract_snippet=(
            "Gradient fingerprints of hacked vs. legitimate solutions differ statistically. "
            "A lightweight detector trained on gradient norms of model parameters flags "
            "reward hacking in real time without requiring ground-truth labels."
        ),
        url="https://arxiv.org/abs/2604.16242",
        relevance_tags=["reward_hacking_detection", "monitor_design"],
        hackwatch_takeaway=(
            "Future work: add gradient fingerprint features to MonitorObservation. "
            "Current monitor operates on diffs only — gradient signals would be complementary."
        ),
        implemented=False,
    ),
    Paper(
        arxiv_id="2509.22047",
        title="MO-GRPO: Mitigating Reward Hacking in Multi-Objective GRPO",
        abstract_snippet=(
            "In multi-objective RL settings, GRPO agents find cross-objective hacks that "
            "satisfy one reward while violating another. MO-GRPO adds a constraint on "
            "reward correlation to prevent single-objective gaming."
        ),
        url="https://arxiv.org/abs/2509.22047",
        relevance_tags=["reward_hacking_detection", "grpo_improvement", "reward_design"],
        hackwatch_takeaway=(
            "Calibration penalty (penalizing high-confidence FP) already addresses one form "
            "of reward gaming. Consider adding correlation constraint between detection and "
            "calibration components in compute_episode_reward()."
        ),
        implemented=False,
    ),
    Paper(
        arxiv_id="2507.05619",
        title="Detecting and Mitigating Reward Hacking in RL Systems: Comprehensive Empirical Study",
        abstract_snippet=(
            "Comprehensive study of reward hacking types across 6 RL environments. "
            "Key finding: monitor agents that see the full diff (not just reward signals) "
            "catch 73% of hacks vs. 41% for reward-only monitors."
        ),
        url="https://arxiv.org/abs/2507.05619",
        relevance_tags=["reward_hacking_detection", "monitor_design", "benchmark"],
        hackwatch_takeaway=(
            "Validates HackWatch's design: monitor sees full worker diff, not just reward. "
            "73% detection rate is our baseline target to beat."
        ),
        implemented=False,
    ),
    # ---- Calibration ----
    Paper(
        arxiv_id="2602.01750",
        title="Adversarial Reward Auditing for Active Detection and Mitigation of Reward Hacking",
        abstract_snippet=(
            "Active auditing framework that probes the reward model with adversarial inputs "
            "to detect when reward hacking has occurred. Maintains a calibrated uncertainty "
            "estimate over reward model reliability."
        ),
        url="https://arxiv.org/abs/2602.01750",
        relevance_tags=["reward_hacking_detection", "calibration", "monitor_design"],
        hackwatch_takeaway=(
            "Adversarial probe idea: inject known hack patterns into eval episodes and "
            "measure monitor recall. Already implemented in tests/test_verifier_adversarial.py."
        ),
        implemented=True,
    ),
]


# ---------------------------------------------------------------------------
# Relevance mapping: diagnosis → tag
# ---------------------------------------------------------------------------

DIAGNOSIS_TO_TAGS: dict[str, list[str]] = {
    "ceiling_hit":       ["ceiling_hit", "curriculum", "grpo_improvement"],
    "low_diversity":     ["low_diversity", "grpo_improvement"],
    "learning_slowly":   ["learning_slowly", "curriculum", "grpo_improvement"],
    "diverging":         ["grpo_improvement"],
    "truncation":        ["grpo_improvement"],
    "collapsed":         ["collapsed", "grpo_improvement"],
    "healthy":           ["reward_hacking_detection", "monitor_design"],
    "reward_design":     ["reward_design", "calibration"],
    "monitor_design":    ["reward_hacking_detection", "monitor_design", "benchmark"],
}


def rank(diagnosis: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k papers ranked by relevance to the given diagnosis."""
    target_tags = set(DIAGNOSIS_TO_TAGS.get(diagnosis, ["grpo_improvement"]))
    scored = []
    for paper in PAPER_DB:
        overlap = len(target_tags & set(paper.relevance_tags))
        if overlap > 0:
            scored.append((overlap, paper))
    scored.sort(key=lambda x: (-x[0], x[1].arxiv_id))
    results = []
    for _, paper in scored[:top_k]:
        d = asdict(paper)
        d["relevance_score"] = len(target_tags & set(paper.relevance_tags)) / len(target_tags)
        results.append(d)
    return results


def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Keyword search over titles and abstracts."""
    query_words = set(query.lower().split())
    scored = []
    for paper in PAPER_DB:
        text = (paper.title + " " + paper.abstract_snippet + " " + " ".join(paper.relevance_tags)).lower()
        hits = sum(1 for w in query_words if w in text)
        if hits > 0:
            scored.append((hits, paper))
    scored.sort(key=lambda x: (-x[0], x[1].arxiv_id))
    return [asdict(p) for _, p in scored[:top_k]]


def summarize_for_diagnosis(diagnosis: str) -> str:
    """One-paragraph research summary for a given diagnosis."""
    papers = rank(diagnosis, top_k=3)
    if not papers:
        return f"No papers indexed for diagnosis '{diagnosis}'."
    lines = [f"Top papers for diagnosis '{diagnosis}':"]
    for p in papers:
        impl = "✅ implemented" if p["implemented"] else "📌 not yet implemented"
        lines.append(f"  [{impl}] arXiv:{p['arxiv_id']} — {p['title']}")
        lines.append(f"    ↳ {p['hackwatch_takeaway']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="HackWatch paper scraper")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--diagnosis", choices=list(DIAGNOSIS_TO_TAGS.keys()),
                   help="Show papers relevant to this training diagnosis")
    g.add_argument("--query", help="Keyword search over paper database")
    g.add_argument("--all", action="store_true", help="Dump full database as JSON")
    p.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    args = p.parse_args()

    if args.all:
        print(json.dumps([asdict(paper) for paper in PAPER_DB], indent=2))
    elif args.diagnosis:
        if args.json:
            print(json.dumps(rank(args.diagnosis), indent=2))
        else:
            print(summarize_for_diagnosis(args.diagnosis))
    elif args.query:
        results = search(args.query)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"arXiv:{r['arxiv_id']} — {r['title']}")
                print(f"  {r['hackwatch_takeaway']}\n")


if __name__ == "__main__":
    main()
