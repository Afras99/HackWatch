"""
Subprocess-isolated pytest runner.

The sandbox NEVER shares a filesystem with the live agent I/O directory.
Each call spins up a fresh temporary directory, writes the repo dict, runs
pytest, and returns structured results.  The calling process is not affected
by any file mutations the worker makes.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


class SandboxRunner:
    """Runs pytest in a subprocess-isolated temporary directory.

    Initialise once and call ``run()`` for each repo snapshot.  The timeout
    can be overridden per-instance; no configuration is loaded inside methods.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Step 1 — write repo files into temp dir
    # ------------------------------------------------------------------

    def _write_repo(self, base: str, repo_dict: dict[str, str]) -> None:
        """Write all repo files into ``base`` directory, creating sub-dirs."""
        for rel_path, content in repo_dict.items():
            full = Path(base) / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 2 — parse pytest output counts
    # ------------------------------------------------------------------

    def _parse_counts(self, stdout: str) -> tuple[int, int]:
        """Parse ``X passed, Y failed`` from pytest ``-q`` output.

        Uses regex so ANSI color codes and varied whitespace don't break parsing.
        """
        # Strip ANSI escape codes first
        clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
        passed = failed = 0
        for m in re.finditer(r"(\d+)\s+(passed|failed)", clean, re.IGNORECASE):
            n, label = int(m.group(1)), m.group(2).lower()
            if label == "passed":
                passed = n
            elif label == "failed":
                failed = n
        return passed, failed

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, repo_dict: dict[str, str], timeout: int | None = None) -> dict:
        """Write ``repo_dict`` into a temp dir and run pytest.

        Args:
            repo_dict: Mapping of relative path → file content.
            timeout: Override the instance-level timeout for this call only.

        Returns:
            Dict with keys ``passed`` (bool), ``returncode`` (int),
            ``stdout`` (str), ``stderr`` (str), ``num_passed`` (int),
            ``num_failed`` (int).
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        with tempfile.TemporaryDirectory(prefix="hw_sandbox_") as tmpdir:
            self._write_repo(tmpdir, repo_dict)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header", "--color=no"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env={**os.environ, "PYTHONPATH": tmpdir},
            )
            num_passed, num_failed = self._parse_counts(result.stdout)
            return {
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "num_passed": num_passed,
                "num_failed": num_failed,
            }


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible alias
# ---------------------------------------------------------------------------

_runner = SandboxRunner()

run_tests = _runner.run


if __name__ == "__main__":
    result = run_tests({
        "solution.py": "def total(numbers):\n    return sum(numbers)\n",
        "test_solution.py": (
            "from solution import total\n"
            "def test_basic():\n"
            "    assert total([1, 2, 3]) == 6\n"
        ),
    })
    print("Sandbox result:", result)
    assert result["passed"], f"Expected tests to pass: {result}"
    print("server/sandbox.py smoke test passed.")

# Run on terminal:
# python -m server.sandbox
