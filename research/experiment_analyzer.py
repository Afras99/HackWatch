"""
Parses training log files into metric dicts.
Diagnosis is now done by Claude — see research/RESEARCH_AGENT.md.
"""
from __future__ import annotations

import ast
import re


def parse_log(log_path: str) -> list[dict]:
    """Extract metric dicts from a TRL training log file."""
    metrics = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\{['\"]loss['\"].*?\}", line)
            if m:
                try:
                    metrics.append(ast.literal_eval(m.group()))
                except Exception:
                    pass
    return metrics
