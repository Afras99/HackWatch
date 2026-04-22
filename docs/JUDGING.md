# Hackathon Judging Criteria & Themes

Meta PyTorch OpenEnv Hackathon — Finale April 25–26, 2026

---

## Minimum Requirements (must hit all 4)

1. **OpenEnv** — use latest release (`openenv-core`)
2. **Training script** — minimal GRPO/TRL script runnable in Colab (Unsloth or HF TRL)
3. **Mini-blog or video** — HuggingFace post or YouTube video, <2 minutes
4. **HF Spaces deployment** — OpenEnv-compliant environment hosted publicly

---

## Judging Criteria (First Round — 3 min pitch + 2 min Q&A)

| Criterion | Weight | What judges look for |
|-----------|--------|----------------------|
| Environment Innovation | **40%** | Novel, creative, or challenging? Does it meaningfully test agent behavior? |
| Storytelling | **30%** | Clear problem + environment + agent behavior explanation? Engaging demo? |
| Showing Improvement in Rewards | **20%** | Observable training progress: reward curves, metrics, before/after behavior |
| Reward & Training Pipeline | **10%** | Coherent reward logic, meaningful improvement in agent inference |

---

## Our Theme Alignment

**Primary: Theme #1 — Multi-Agent Interactions**  
Monitor agent watches worker agent. Partially observable (monitor never sees planted_label).
Theory-of-mind: monitor must model worker's incentives to detect cheating.

**Sub-theme Bonus: Fleet AI — Scalable Oversight**  
"Environments that train oversight agents to monitor, analyze, and explain the behavior
of other AI agents." This is exactly HackWatch. Nic Ouporov's framing: residency training
for agents before they touch real infrastructure.

**Secondary: Theme #4 — Self-Improvement**  
Hack curriculum auto-escalates once monitor F1 > 0.7. Monitor and worker improve
through adversarial co-training. Adaptive curriculum = recursive skill amplification.

**Secondary bonus: Patronus AI** — trajectory debugging (our episode logs + reward
breakdown per step match their Percival framing).

---

## Themes Reference (full)

### Theme 1 — Multi-Agent Interactions
Cooperation, competition, negotiation, coalition formation. Theory-of-mind reasoning,
emergent strategic behavior.  
*Examples: Market simulations, compute-allocation negotiations, collaborative puzzle worlds.*

Sub-theme bonuses:
- **Fleet AI** — Scalable Oversight: train oversight agents to monitor other AI agents
- **Halluminate** — Multi-Actor Environments: agent manages multiple actors

### Theme 2 — Long-Horizon Planning & Instruction Following
Deep multi-step reasoning with sparse/delayed rewards. Decompose goals, track state
over extended trajectories, recover from mistakes.  
*Examples: Research-planning simulators, large-scale codebase refactoring.*

Sub-theme bonuses:
- **Scale AI** — long horizon workflows for non-code business use cases (Sales/PM/HR)
- **Mercor** — capped/uncapped rewards that scale with token output

### Theme 3 — World Modeling
3.1 Professional: real tool/API interaction, no shortcuts  
3.2 Personalized: personal assistant tasks, conflict management

Sub-theme bonuses:
- **Scaler AI Labs** — Multi-App RL for Enterprise Workflows
- **Patronus AI** — Consumer Workflows with Schema Drift

### Theme 4 — Self-Improvement
Agents generate new challenges, escalate difficulty, improve through self-play.  
*Examples: Self-play negotiation arenas, evolving coding competitions.*

Sub-theme bonus:
- **Snorkel AI** — Simulated Experts-in-the-Loop with changing requirements

### Theme 5 — Wild Card
Out-of-box ideas that add value to LLM training. Rewarded for creativity.

---

## What "Scaler Validator" Was (clarification)

The original plan mentioned a "Scaler validator script" — **this does not exist**.
It was a confusion with Scaler AI Labs (Theme 3 sub-theme bonus prize), which is a
*different* company sponsoring a bonus prize, not the hackathon organizer.

There is no validator script to download. The minimum requirements are the 4 items
listed above. The closest thing to "validation" is:
- `pytest tests/` — all 81 tests must pass
- The environment must respond correctly to `/reset` and `/step` over HTTP
- The OpenEnv manifest (`openenv.yaml`) must be valid

---

## Pre-Onsite Checklist (before April 25)

- [ ] Environment runs on HF Spaces — `/reset` returns 200 over public URL
- [ ] Training script runnable in Colab (test `train_monitor.py --dry-run`)
- [ ] Mini-blog on HuggingFace (model card or Space README) or YouTube video <2 min
- [ ] `git tag v0.1` and push to GitHub
- [ ] Reward curves captured to W&B or screenshots for the pitch deck
- [ ] Demo UI live at `/demo` (drop Claude Design build into `demo/build/`)
- [ ] 90-second pitch rehearsed 15 times

## Onsite (April 25–26)

- Scale monitor to Qwen2.5-7B (LoRA) to push F1 > 0.85 on novel exploits
- Ablation run: monitor-only without calibration term → shows "always-flag" collapse
- Record backup demo video in case Wi-Fi fails
