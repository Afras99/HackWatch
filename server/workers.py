"""
Scripted worker bank for Day 3 bootstrapping.

Each worker function takes a task dict and returns a trajectory:
    list of {"action": dict, "diff": str} steps.

Workers are deterministic so that the MONITOR trains against a stable
distribution before co-training begins on Day 4.
"""
from __future__ import annotations

import difflib
from typing import Any

from server.exploits import ALL_PRIMITIVES, make_unified_diff
from server.tasks import Task


WorkerStep = dict[str, Any]     # {"action": {...}, "diff": str}
Trajectory = list[WorkerStep]


# ---------------------------------------------------------------------------
# Legit worker — applies the gold_patch correctly
# ---------------------------------------------------------------------------

def legit_worker(task: Task) -> Trajectory:
    """
    Apply the gold solution patch in a single step.
    Returns a one-step trajectory with the correct unified diff.
    """
    original = task["scaffold_repo"]
    # Build the patched repo by applying gold_patch heuristically
    patched = _apply_gold_patch(original, task["gold_patch"])
    diff = make_unified_diff(original, patched)
    return [
        {
            "action": {
                "tool": "write_file",
                "files_changed": list(patched.keys()),
                "description": "Apply fix",
            },
            "diff": diff,
            "repo_after": patched,
        }
    ]


def _apply_gold_patch(scaffold: dict[str, str], gold_patch: str) -> dict[str, str]:
    """
    Very lightweight patch application: just return the scaffold with the
    known-broken line replaced by the fixed line, inferred from gold_patch.
    Falls back to scaffold unchanged if parsing fails.
    """
    result = dict(scaffold)
    try:
        removed, added = {}, {}
        for line in gold_patch.splitlines():
            if line.startswith("--- a/"):
                current_file = line[6:]
            elif line.startswith("-") and not line.startswith("---"):
                removed.setdefault(current_file, []).append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                added.setdefault(current_file, []).append(line[1:])
        for fpath, removals in removed.items():
            if fpath in result:
                src = result[fpath]
                additions = added.get(fpath, [])
                for rem, add in zip(removals, additions):
                    src = src.replace(rem, add, 1)
                result[fpath] = src
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Hack workers — one per primitive, plus a combo worker
# ---------------------------------------------------------------------------

def _make_hack_worker(primitive_name: str):
    """Factory: build a deterministic worker that applies one hack primitive."""
    primitive_fn = ALL_PRIMITIVES[primitive_name]

    def worker(task: Task) -> Trajectory:
        original = task["scaffold_repo"]
        modified, label = primitive_fn(original, task["task_id"])
        diff = make_unified_diff(original, modified)
        return [
            {
                "action": {
                    "tool": "write_file",
                    "files_changed": [p for p in modified if modified[p] != original.get(p)],
                    "description": "Refactor and optimise",
                },
                "diff": diff,
                "repo_after": modified,
                "planted_label": label,
            }
        ]

    worker.__name__ = f"{primitive_name}_worker"
    return worker


# Build one named worker per primitive
test_override_worker = _make_hack_worker("test_override")
file_leak_worker = _make_hack_worker("file_leak")
subprocess_escape_worker = _make_hack_worker("subprocess_escape")
conftest_tamper_worker = _make_hack_worker("conftest_tamper")
eval_injection_worker = _make_hack_worker("eval_injection")
timer_bypass_worker = _make_hack_worker("timer_bypass")
gradient_hardcode_worker = _make_hack_worker("gradient_hardcode")
env_var_injection_worker = _make_hack_worker("env_var_injection")


def combo_worker(task: Task) -> Trajectory:
    """Combine two randomly-chosen primitives (used for heldout_novel)."""
    from server.exploits import apply_combo_hack
    original = task["scaffold_repo"]
    modified, label = apply_combo_hack(original, task["task_id"])
    diff = make_unified_diff(original, modified)
    return [
        {
            "action": {
                "tool": "write_file",
                "files_changed": [p for p in modified if modified[p] != original.get(p)],
                "description": "Refactor and optimise",
            },
            "diff": diff,
            "repo_after": modified,
            "planted_label": label,
        }
    ]


# Registry: name → worker function
WORKER_BANK: dict[str, Any] = {
    "legit": legit_worker,
    "test_override": test_override_worker,
    "file_leak": file_leak_worker,
    "subprocess_escape": subprocess_escape_worker,
    "conftest_tamper": conftest_tamper_worker,
    "eval_injection": eval_injection_worker,
    "timer_bypass": timer_bypass_worker,
    "gradient_hardcode": gradient_hardcode_worker,
    "env_var_injection": env_var_injection_worker,
    "combo": combo_worker,
}


def sample_worker(hack_type: str | None = None) -> Any:
    """Return a worker function for the given hack_type (or 'legit')."""
    key = hack_type or "legit"
    if key not in WORKER_BANK:
        raise KeyError(f"Unknown worker: {key!r}.  Available: {list(WORKER_BANK)}")
    return WORKER_BANK[key]
