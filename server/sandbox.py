"""
Subprocess-isolated pytest runner.

The sandbox NEVER shares a filesystem with the live agent I/O directory.
Each call spins up a fresh temporary directory, writes the repo dict, runs
pytest, and returns structured results.  The calling process is not affected
by any file mutations the worker makes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_tests(repo_dict: dict[str, str], timeout: int = 30) -> dict:
    """
    Write repo_dict files into a temp dir and run pytest.

    Args:
        repo_dict:  mapping of relative path → file content
        timeout:    seconds before the subprocess is killed

    Returns:
        {
            "passed": bool,
            "returncode": int,
            "stdout": str,
            "stderr": str,
            "num_passed": int,   # parsed from pytest output
            "num_failed": int,
        }
    """
    with tempfile.TemporaryDirectory(prefix="hw_sandbox_") as tmpdir:
        _write_repo(tmpdir, repo_dict)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": tmpdir},
        )
        num_passed, num_failed = _parse_counts(result.stdout)
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "num_passed": num_passed,
            "num_failed": num_failed,
        }


def _write_repo(base: str, repo_dict: dict[str, str]) -> None:
    for rel_path, content in repo_dict.items():
        full = Path(base) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")


def _parse_counts(stdout: str) -> tuple[int, int]:
    """Parse 'X passed, Y failed' from pytest -q output."""
    passed = failed = 0
    for line in stdout.splitlines():
        lower = line.lower()
        for token in lower.split(","):
            token = token.strip()
            parts = token.split()
            if len(parts) >= 2 and parts[0].isdigit():
                n = int(parts[0])
                rest = " ".join(parts[1:])
                if "passed" in rest:
                    passed = n
                elif "failed" in rest:
                    failed = n
    return passed, failed
