#!/usr/bin/env bash
set -euo pipefail
HW_PY=/home/afrasaboobackerp/.conda/envs/hackwatch/bin/python
VLLM_PY=/home/afrasaboobackerp/.conda/envs/hackwatch/bin/python
cd /home/afrasaboobackerp/HackWatch

VLLM_PORT=8001
VLLM_PID=""

cleanup() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null && echo "vLLM stopped"
}
trap cleanup EXIT

start_vllm() {
    local ckpt="$1"
    echo "[$(date +%H:%M)] Starting inference server for: $ckpt"
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null; sleep 2
    CUDA_VISIBLE_DEVICES=1 $VLLM_PY scripts/serve_checkpoint.py \
        --checkpoint "$ckpt" \
        --port $VLLM_PORT \
        --device cuda:0 \
        > logs/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "Inference server PID: $VLLM_PID — waiting 45s for model load..."
    sleep 45
    curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1 && \
        echo "[$(date +%H:%M)] Inference server ready" || \
        echo "[$(date +%H:%M)] WARNING: server may not be ready yet, check logs/vllm.log"
}

run_eval() {
    local tag="$1"
    local use_heuristic="$2"
    local out="eval/results_${tag}.json"
    
    if [ "$use_heuristic" = "true" ]; then
        echo "[$(date +%H:%M)] Running HEURISTIC eval (no checkpoint yet)..."
        $HW_PY eval/evaluate_monitor.py \
            --trajectories data/trajectories.jsonl \
            --heuristic --tag "$tag" --out "$out" 2>/dev/null
    else
        echo "[$(date +%H:%M)] Running MODEL eval against vLLM:$VLLM_PORT..."
        $HW_PY eval/evaluate_monitor.py \
            --trajectories data/trajectories.jsonl \
            --api-url "http://localhost:$VLLM_PORT/v1" \
            --model-name hackwatch-monitor \
            --tag "$tag" --out "$out" 2>/dev/null
    fi

    $HW_PY -c "
import json, sys
try:
    r = json.load(open('$out'))
    a = r.get('aggregate', {})
    ppf = r.get('per_primitive_f1', {})
    print(f\"[$(date +%H:%M)] TAG=$tag\")
    print(f\"  F1={a.get('f1',0):.3f}  P={a.get('precision',0):.3f}  R={a.get('recall',0):.3f}  acc={a.get('accuracy',0):.1%}\")
    print(f\"  Heldout(subprocess/eval_inj): {r.get('heldout_detection_rate', 0):.3f}\")
    heldout = {k:v for k,v in ppf.items() if 'subprocess' in k or 'eval_inj' in k}
    if heldout:
        for k,v in heldout.items(): print(f'    {k}: {v:.3f}')
    print(f\"  Baseline to beat: F1=0.966  heldout=0.667\")
except Exception as e:
    print(f'  [eval parse error: {e}]')
" 2>/dev/null || echo "  [eval output not available]"
}

LAST_CKPT=""
EVAL_COUNT=0

echo "[$(date +%H:%M)] Eval loop started. Checks every 5 min."
echo "  - Uses heuristic until first checkpoint appears"
echo "  - Spins up vLLM on GPU 1 for model eval once checkpoint ready"
echo "  - Baseline: F1=0.966, heldout=0.667"
echo ""

while true; do
    # Find latest checkpoint
    CKPT=$(ls -td runs/monitor_final/checkpoint-* 2>/dev/null | head -1)
    
    if [ -z "$CKPT" ]; then
        # No checkpoint yet — heuristic eval
        run_eval "heuristic_${EVAL_COUNT}" "true"
    elif [ "$CKPT" != "$LAST_CKPT" ]; then
        # New checkpoint — restart vLLM and eval with model
        echo ""
        echo "[$(date +%H:%M)] NEW CHECKPOINT: $CKPT"
        LAST_CKPT="$CKPT"
        start_vllm "$CKPT"
        STEP=$(echo "$CKPT" | grep -oP '\d+$')
        run_eval "model_step${STEP}" "false"
    else
        # Same checkpoint — just re-run model eval
        STEP=$(echo "$CKPT" | grep -oP '\d+$')
        run_eval "model_step${STEP}_e${EVAL_COUNT}" "false"
    fi
    
    EVAL_COUNT=$((EVAL_COUNT + 1))
    echo "  Sleeping 5 min..."
    sleep 300
done
