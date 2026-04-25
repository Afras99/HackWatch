#!/usr/bin/env bash
# Validate the HackWatch submission before pushing.
# Run this after every commit on Friday.
#
# Usage: bash scripts/validate_submission.sh

set -e
PYTHON=${PYTHON:-python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "=== HackWatch Submission Validator ==="
echo ""

# 1. openenv.yaml format check
echo "[1/6] Checking openenv.yaml spec_version..."
if grep -q "spec_version: 1" openenv.yaml; then
    echo "  ✓ spec_version: 1 present"
else
    echo "  ✗ FAIL: openenv.yaml missing spec_version: 1"
    exit 1
fi
for field in "name:" "type:" "runtime:" "app:" "port:"; do
    if grep -q "$field" openenv.yaml; then
        echo "  ✓ $field present"
    else
        echo "  ✗ FAIL: openenv.yaml missing $field"
        exit 1
    fi
done

# 2. planted_label leakage check
echo ""
echo "[2/6] Checking planted_label leakage..."
if grep -rn "planted_label" hackwatch/models.py | grep -q "MonitorObservation"; then
    # Check that it's only in the class definition comment, not in to_dict
    if python -c "
import sys; sys.path.insert(0, '.')
from hackwatch.models import MonitorObservation
from dataclasses import fields
names = [f.name for f in fields(MonitorObservation)]
assert 'planted_label' not in names, f'planted_label found in MonitorObservation fields: {names}'
obs = MonitorObservation.__new__(MonitorObservation)
print('Fields OK:', names)
" 2>&1; then
        echo "  ✓ planted_label not in MonitorObservation fields"
    else
        echo "  ✗ FAIL: planted_label leakage in MonitorObservation"
        exit 1
    fi
else
    echo "  ✓ planted_label not referenced in MonitorObservation"
fi

# 3. All tests green
echo ""
echo "[3/6] Running test suite..."
$PYTHON -m pytest tests/ -q --tb=short 2>&1 | tail -5
if $PYTHON -m pytest tests/ -q --tb=short > /dev/null 2>&1; then
    echo "  ✓ All tests passed"
else
    echo "  ✗ FAIL: test suite has failures"
    $PYTHON -m pytest tests/ -q --tb=short
    exit 1
fi

# 4. Adversarial reward tests
echo ""
echo "[4/6] Running adversarial reward tests..."
if $PYTHON -m pytest tests/test_verifier_adversarial.py -q --tb=short > /dev/null 2>&1; then
    echo "  ✓ All adversarial tests passed"
else
    echo "  ✗ FAIL: adversarial reward tests failed — reward function is broken"
    $PYTHON -m pytest tests/test_verifier_adversarial.py -v --tb=short
    exit 1
fi

# 5. Server starts and health endpoint responds
echo ""
echo "[5/6] Checking server health endpoint..."
$PYTHON -m uvicorn server.app:app --port 18765 --log-level error &
SERVER_PID=$!
sleep 2
if curl -sf http://localhost:18765/health > /dev/null; then
    echo "  ✓ Health endpoint responds"
else
    echo "  ✗ FAIL: server did not start or health endpoint not responding"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

# 6. inference.py stdout contract
echo ""
echo "[6/6] Checking inference.py stdout contract..."
# Start server briefly for inference
$PYTHON -m uvicorn server.app:app --port 18766 --log-level error &
SERVER_PID=$!
sleep 2
FIRST_LINE=$(HACKWATCH_ENV_URL=http://localhost:18766 $PYTHON inference.py --episodes 1 2>/dev/null | head -1 || true)
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

if echo "$FIRST_LINE" | grep -qE '^\[START\] task='; then
    echo "  ✓ stdout contract OK: $FIRST_LINE"
else
    echo "  ✗ FAIL: inference.py first line does not match [START] contract"
    echo "    Got: $FIRST_LINE"
    exit 1
fi

echo ""
echo "=== ALL CHECKS PASSED — safe to push ==="
