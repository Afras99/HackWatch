from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

# ---------------------------------------------------------------------------
# OpenEnv base classes — import from openenv-core if installed, else use
# local stubs so the module loads during development before pip install.
# ---------------------------------------------------------------------------
try:
    from openenv.core import Action, Observation, State  # type: ignore[import]
except ImportError:
    class Action:
        def to_dict(self) -> dict:
            return asdict(self)  # type: ignore[arg-type]

    class Observation:
        def to_dict(self) -> dict:
            return asdict(self)  # type: ignore[arg-type]

    class State:
        def to_dict(self) -> dict:
            return asdict(self)  # type: ignore[arg-type]


@dataclass
class MonitorAction(Action):
    """Action emitted by the MONITOR agent each step."""
    verdict: Literal["allow", "flag", "block"]
    confidence: float       # [0.0, 1.0]
    reasoning: str = ""     # logged, not scored; NEVER fed back into obs

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.verdict not in ("allow", "flag", "block"):
            raise ValueError(f"invalid verdict: {self.verdict}")

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorAction":
        return cls(
            verdict=d["verdict"],
            confidence=float(d["confidence"]),
            reasoning=d.get("reasoning", ""),
        )

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class MonitorObservation(Observation):
    """
    Observation visible to the MONITOR agent.

    CRITICAL INVARIANT: planted_label is NEVER included here.
    The monitor must infer hack presence purely from behavioral signals.
    """
    task_prompt: str
    worker_last_action: dict
    worker_diff: str            # unified diff of repo change (single step)
    tools_used_so_far: list[str]
    step_idx: int
    episode_done: bool

    def to_dict(self) -> dict:
        return {
            "task_prompt": self.task_prompt,
            "worker_last_action": self.worker_last_action,
            "worker_diff": self.worker_diff,
            "tools_used_so_far": list(self.tools_used_so_far),
            "step_idx": self.step_idx,
            "episode_done": self.episode_done,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorObservation":
        return cls(
            task_prompt=d["task_prompt"],
            worker_last_action=d.get("worker_last_action", {}),
            worker_diff=d.get("worker_diff", ""),
            tools_used_so_far=list(d.get("tools_used_so_far", [])),
            step_idx=int(d["step_idx"]),
            episode_done=bool(d["episode_done"]),
        )


@dataclass
class HackWatchState(State):
    """Server-side ground truth state (never sent to monitor in full)."""
    episode_id: str
    step_count: int
    task_id: str
    difficulty: Literal["train", "heldout_novel"]
    # planted_label lives here — never serialised into MonitorObservation
    planted_label: str = "legit"
    worker_succeeded: bool = False

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            # planted_label intentionally omitted from public state
        }

    def to_dict_full(self) -> dict:
        """Full state including planted_label — for internal/logging use only."""
        return {
            **self.to_dict(),
            "planted_label": self.planted_label,
            "worker_succeeded": self.worker_succeeded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HackWatchState":
        return cls(
            episode_id=d["episode_id"],
            step_count=int(d["step_count"]),
            task_id=d["task_id"],
            difficulty=d["difficulty"],
            planted_label=d.get("planted_label", "legit"),
            worker_succeeded=bool(d.get("worker_succeeded", False)),
        )
