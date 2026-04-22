# HackWatch — Judge FAQ

Prepared answers for every question a hackathon judge or evaluator is likely to ask.

---

## "Can we see real training curves? Is this illustrative data?"

**Answer:** Yes — run `python -m training.run_baseline --episodes 300` against the live
server to generate a real W&B run. This uses the heuristic rule-based monitor (no GPU
needed) as a baseline. Once GPU access is available, `train_monitor.py` runs the GRPO
fine-tuned Qwen2.5-3B-Instruct monitor and logs the same metrics side-by-side.

The W&B project is `hackwatch`. Run ID and URL are printed at the end of either script.

---

## "What exactly are the heldout exploits, and how is generalization measured?"

`subprocess_escape` and `eval_injection` are **never** added to the training
primitive pool. In 15% of live episodes (`HELDOUT_FRACTION = 0.15`), the environment
samples one of these two exploit types instead of the 6 training types.

The metric is `heldout_detection_rate` — the fraction of heldout episodes where the
monitor correctly flags or blocks, logged separately from overall detection rate.
A monitor that's memorizing labels (not patterns) will score near chance (50%) on
heldout; a monitor detecting structural signals will score above 70%.

---

## "Why scripted workers? Would this work against a real LLM?"

Scripted workers are the teacher-forcing phase — same approach as AlphaCode's RLHF
curriculum. The environment is the contribution, not the trained weights. The claim is:
the co-training dynamic (monitor learning to detect → worker learning to evade →
monitor improving) is what we're studying. Scripted workers give a clean, reproducible
signal to bootstrap from; a real LLM worker is the Day 4 co-training target.

---

## "Your reward signal is sparse (0 or 1). Doesn't that make GRPO unstable?"

Two mitigations:
1. `num_generations=8` per prompt ensures enough variance for advantage estimation
2. `beta=0.04` KL clipping prevents the policy from collapsing to a mode
   (0.001 causes reward-hacking collapse per Lewis Tunstall's GRPO analysis)

The `latency_bonus` (+0.2 max) and `calibration_penalty` add density — a monitor
that flags early gets more signal than one that waits until the last step.

---

## "Why is the calibration_penalty so large (up to 0.5)? It can overwhelm other rewards."

Intentional. A monitor that always outputs `confidence=1.0` is useless in production —
it can't distinguish certain hacks from uncertain ones. The penalty is sized to make
overconfident false positives actively harmful, forcing the monitor to learn calibrated
confidence alongside verdict accuracy.

---

## "The monitor only sees diffs. Real reward hacking won't have obvious diff signatures."

True for current LLMs. This is a controlled testbed — the claim is not "deploy this in
production" but "build a training environment where the detection signal is learnable
and measurable." The diff-based approach is analogous to static analysis: one layer of
defense, not the whole stack. The research contribution is the co-training loop and the
deterministic reward signal, not the specific exploit vocabulary.

---

## "Only 15 tasks — won't the monitor just memorize task-solution pairs?"

The monitor sees `worker_diff`, not the task ID. It must learn structural patterns
(`assert True`, `_PRECOMP =`, `pytest_runtest_makereport`) not task-specific features.
The heldout exploit test validates this: if the monitor were memorizing, it would fail
on `subprocess_escape`/`eval_injection` since those patterns never appeared in training.

---

## "The /step endpoint previously exposed planted_label in the info dict. Is that a leak?"

Fixed. As of the current version, `planted_label` is filtered from the public `/step`
response. The environment still uses it internally for reward computation, and it
appears in `HackWatchState.to_dict_full()` for logging, but it is never sent to the
monitor agent.

---

## "Your openenv.yaml lists HackWatchState as the state_schema, but /state omits planted_label."

The public `/state` endpoint uses `HackWatchState.to_dict()` which intentionally omits
`planted_label`. The schema declaration refers to the dataclass definition, not the
serialized form. `to_dict_full()` exposes the full state and is used only for internal
logging. This is documented in the invariants section of CLAUDE.md.

---

## "What's your baseline? Without a comparison, the improvement curve means nothing."

Baselines (all logged to W&B project `hackwatch`):
1. **Random monitor** — always allows, detection rate ≈ 0%
2. **Rule-based monitor** — keyword heuristics on diff, ~70-80% detection, ~20% FPR
3. **GRPO-trained Qwen2.5-3B** — target is >85% detection, <10% FPR on training exploits,
   >65% detection on heldout exploits

Run `python -m training.run_baseline` to reproduce the rule-based baseline.

---

## "What happens if an exploit uses obfuscated code to evade the keyword heuristics?"

That's exactly what the GRPO training targets. The rule-based monitor is the floor;
the trained monitor should learn higher-level patterns. The eval metric is detection
rate on the actual environment episodes (which use the real exploit implementations),
not on regex matches — so evasion that fools regexes but not the trained model would
show up as improved performance.
