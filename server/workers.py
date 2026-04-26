"""
Scripted worker bank for Day 3 bootstrapping.

Each worker function takes a task dict and returns a trajectory:
    list of {"action": dict, "diff": str} steps.

Workers are deterministic so that the MONITOR trains against a stable
distribution before co-training begins on Day 4.
"""
from __future__ import annotations

from typing import Any

from server.exploits import ALL_PRIMITIVES, make_unified_diff
from server.tasks import Task


WorkerStep = dict[str, Any]     # {"action": {...}, "diff": str}
Trajectory = list[WorkerStep]


class WorkerBank:
    """Registry of scripted worker functions keyed by hack type.

    Initialise once — ``_build`` populates ``self._bank`` from the live
    ``ALL_PRIMITIVES`` dict, so stealth and third-party primitives are
    included automatically.  Exposes a ``sample`` method that returns the
    worker callable for a given hack type.
    """

    def __init__(self) -> None:
        self._bank: dict[str, Any] = {}
        self._build()

    # ------------------------------------------------------------------
    # Step 1 — gold-patch application helper
    # ------------------------------------------------------------------

    def _apply_gold_patch(
        self, scaffold: dict[str, str], gold_patch: str
    ) -> dict[str, str]:
        """Apply the gold patch heuristically by replacing known-broken lines.

        Args:
            scaffold: Original repo dict.
            gold_patch: Unified diff string from the task definition.

        Returns:
            Patched repo dict (falls back to scaffold unchanged if parsing fails).
        """
        result = dict(scaffold)
        try:
            removed: dict[str, list[str]] = {}
            added: dict[str, list[str]] = {}
            current_file = ""
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

    # ------------------------------------------------------------------
    # Step 2 — factory: build a single hack worker for one primitive
    # ------------------------------------------------------------------

    def _make_hack_worker(self, primitive_name: str) -> Any:
        """Return a deterministic worker that applies one hack primitive.

        Args:
            primitive_name: Key in ``ALL_PRIMITIVES``.

        Returns:
            Worker callable ``(task: Task) -> Trajectory``.
        """
        primitive_fn = ALL_PRIMITIVES[primitive_name]

        def worker(task: Task) -> Trajectory:
            original = task["scaffold_repo"]
            modified, label = primitive_fn(original, task["task_id"])
            diff = make_unified_diff(original, modified)
            return [
                {
                    "action": {
                        "tool": "write_file",
                        "files_changed": [
                            p for p in modified if modified[p] != original.get(p)
                        ],
                        "description": "Refactor and optimise",
                    },
                    "diff": diff,
                    "repo_after": modified,
                    "planted_label": label,
                }
            ]

        worker.__name__ = f"{primitive_name}_worker"
        return worker

    # ------------------------------------------------------------------
    # Step 3 — build the complete bank
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Populate ``self._bank`` with legit, combo, and per-primitive workers.

        Called once from ``__init__``.  Stealth and third-party primitives
        registered via entry_points are included because they appear in
        ``ALL_PRIMITIVES`` at import time.
        """
        self._bank = {"legit": self._legit_worker, "combo": self._combo_worker}
        for name in ALL_PRIMITIVES:
            self._bank[name] = self._make_hack_worker(name)

    # ------------------------------------------------------------------
    # Built-in workers (legit + combo)
    # ------------------------------------------------------------------

    def _legit_worker(self, task: Task) -> Trajectory:
        """Apply the gold solution patch in a single step."""
        original = task["scaffold_repo"]
        patched = self._apply_gold_patch(original, task["gold_patch"])
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

    def _combo_worker(self, task: Task) -> Trajectory:
        """Combine two randomly-chosen primitives (used for heldout_novel)."""
        from server.exploits import apply_combo_hack
        original = task["scaffold_repo"]
        modified, label = apply_combo_hack(original, task["task_id"])
        diff = make_unified_diff(original, modified)
        return [
            {
                "action": {
                    "tool": "write_file",
                    "files_changed": [
                        p for p in modified if modified[p] != original.get(p)
                    ],
                    "description": "Refactor and optimise",
                },
                "diff": diff,
                "repo_after": modified,
                "planted_label": label,
            }
        ]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def sample(self, hack_type: str | None = None) -> Any:
        """Return the worker callable for the given hack type.

        Args:
            hack_type: Key in the bank (e.g. ``"legit"``, ``"test_override"``).
                Defaults to ``"legit"`` when ``None``.

        Returns:
            Worker callable ``(task: Task) -> Trajectory``.

        Raises:
            KeyError: If ``hack_type`` is not in the bank.
        """
        key = hack_type or "legit"
        if key not in self._bank:
            raise KeyError(
                f"Unknown worker: {key!r}.  Available: {list(self._bank)}"
            )
        return self._bank[key]


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible aliases
# ---------------------------------------------------------------------------

_bank = WorkerBank()

WORKER_BANK: dict[str, Any] = _bank._bank
sample_worker = _bank.sample

# Legacy module-level standalone functions (kept for any direct callers)
legit_worker = _bank._legit_worker
combo_worker = _bank._combo_worker


if __name__ == "__main__":
    from server.tasks import ALL_TASKS
    task = ALL_TASKS[0]

    traj = sample_worker("legit")(task)
    print(f"Legit worker trajectory steps: {len(traj)}")
    assert traj[0]["diff"], "Expected non-empty diff from legit worker"

    traj_hack = sample_worker("test_override")(task)
    assert traj_hack[0]["planted_label"] == "test_override"
    print(f"Hack worker planted_label: {traj_hack[0]['planted_label']!r}")
    print("server/workers.py smoke test passed.")

# Run on terminal:
# python -m server.workers
