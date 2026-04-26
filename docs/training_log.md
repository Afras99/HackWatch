# HackWatch — Training Log (cotrain_v2)

Real metrics extracted from `runs/cotrain_v2/trainer_state.json`.
This is the run that produced **F1=0.970, Precision=1.000, Recall=0.941, 0 false positives**.

Model: Qwen2.5-3B-Instruct + LoRA r=32 · DR-GRPO · beta=0.051 · 300 steps

---

## Run History (all runs)

| Run | Steps | Final Reward | KL | Notes |
|-----|-------|-------------|-----|-------|
| monitor_v1 | 499 | 0.5608 | 0.0249 | Early run, Unsloth, low reward |
| monitor_v2 | 200 | 1.0000 | 0.0195 | reward=1.0 = reward hacking (overfit) |
| monitor_v3 | 400 | 1.0000 | 0.0108 | Same issue |
| monitor_v4 | 400 | 1.0000 | 0.0136 | Same issue |
| monitor_v5 | 400 | 1.0000 | 0.0333 | Same issue |
| monitor_v6 | 400 | 1.0000 | 0.0366 | Same issue |
| monitor_v7 | 200 | 1.0000 | 0.0367 | Same issue |
| **monitor_v8** | **400** | **0.7991** | **0.0551** | First real learning signal, no overfit |
| monitor_v9 | 800 | 0.8190 | 0.2189 | Longer run, KL starting to rise |
| monitor_v10 | 100 | 0.5909 | 61205 | KL explosion — Unsloth bug |
| monitor_v11 | 300 | 0.4909 | 0.0088 | Reward too low, config issue |
| monitor_v12 | 149 | 0.8231 | 0.1449 | Crashed at step 160 (env 404) |
| monitor_colab | 150 | 0.2712 | 97.597 | KL explosion on Colab |
| cotrain_v1 | 100 | 0.1531 | 0.0002 | Co-training, too short |
| **cotrain_v2** | **300** | **0.7960** | **0.2431** | ✅ Best run — F1=0.970 |

---

## cotrain_v2 — Full Step-by-Step Metrics

| Step | Reward | Reward Std | KL | Grad Norm |
|------|--------|------------|-----|-----------|
|   1 | 0.3256 | 0.1218 | 0.0697 | 0.2955 |
|  10 | 0.2402 | 0.3473 | 0.1101 | 2.2557 |
|  25 | 0.6724 | 0.2066 | 0.0635 | 0.1571 |
|  50 | 0.7585 | 0.0187 | 0.2258 | 0.1208 |
|  75 | 0.5781 | 0.2347 | 0.1374 | 0.3964 |
| 100 | 0.8339 | 0.0458 | 0.0482 | 0.5218 |
| 125 | 0.8093 | 0.0727 | 0.1407 | 0.2103 |
| 150 | 0.8490 | 0.0305 | 0.0873 | 0.0500 |
| 175 | 0.7996 | 0.0607 | 0.0929 | 0.2037 |
| 200 | 0.4621 | 0.0124 | 0.0754 | 0.5017 |
| 225 | 0.8740 | 0.0715 | 0.1373 | 0.0775 |
| 229 | 0.9033 | 0.0284 | 0.0483 | 0.0134 |
| 250 | 0.8159 | 0.0375 | 0.2617 | 1.1987 |
| 275 | 0.8203 | 0.0545 | 0.2250 | 0.0899 |
| 282 | 0.8940 | 0.0438 | 0.0386 | 0.1416 |
| 300 | 0.7960 | 0.0480 | 0.2431 | 0.1255 |

---

## Key Observations

- **v2–v7 reward=1.0**: Not good — the model collapsed to always outputting one verdict and gaming the reward. Identified and fixed by adjusting beta and the exploit sampling ratio.
- **v8 was the first honest run**: reward ~0.80, KL stable, no collapse. This is where the approach started working.
- **monitor_v10 / monitor_colab KL explosion**: Caused by the Unsloth hidden-state bug (KL → 61k and 97 respectively). Fixed by switching to standard HF + PEFT.
- **cotrain_v2 is the best run**: 300 steps, reward trending 0.33 → 0.87 → settles ~0.80. KL stayed under 0.55 throughout (one spike at step 26 to 0.38, recovered immediately). Peak reward 0.9033 at step 229.
- **Grad norm stable**: Mostly 0.05–0.5. One spike to 2.26 at step 10 (early learning burst), recovered by step 11.
