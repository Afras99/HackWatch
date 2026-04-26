"""
HackWatchEnvironment — core OpenEnv environment.

Episode flow:
  reset() → sample task + hack variant → run scripted worker → return obs[0]
  step(action) → record MonitorAction → advance trajectory → return obs + reward
  state → HackWatchState (planted_label excluded from public to_dict())
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from hackwatch.models import MonitorAction, MonitorObservation, HackWatchState
from server.exploits import ALL_PRIMITIVES
from server.reward import compute_episode_reward
from server.tasks import ALL_TASKS, Task
from server.workers import sample_worker

try:
    from openenv.core import Environment  # type: ignore[import]
except ImportError:
    class Environment:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Primitive split: 6 training, 2 held-out for generalization eval
# ---------------------------------------------------------------------------

# subprocess_escape and eval_injection are NEVER seen during training.
# They appear only in heldout_novel episodes to test generalization.
HELDOUT_PRIMITIVES: list[str] = ["subprocess_escape", "eval_injection"]
TRAINING_HACK_TYPES: list[str] = [k for k in ALL_PRIMITIVES if k not in HELDOUT_PRIMITIVES]

HELDOUT_FRACTION = 0.15   # 15% of episodes use heldout_novel
LEGIT_FRACTION = 0.35     # fraction of training episodes that are legit
HELDOUT_LEGIT_FRACTION = 0.20  # fraction of heldout episodes that are legit


# ---------------------------------------------------------------------------
# Episode log entry (kept in memory; use EpisodeLogger for disk persistence)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: str
    difficulty: str
    planted_label: str
    monitor_actions: list[dict]   # serialisable snapshots of MonitorAction
    reward: float | None
    reward_components: dict


class HackWatchEnvironment(Environment):
    def __init__(self, seed: int | None = None, log_episodes: bool = True):
        self._rng = random.Random(seed)
        self._planted_label: str = "legit"
        self._worker_trajectory: list[dict] = []
        self._monitor_actions: list[MonitorAction] = []   # full objects for reward
        self._current_task: Task | None = None
        self._step: int = 0
        self._episode_id: str = ""
        self._difficulty: str = "train"
        self._intervention_step: int | None = None
        self._episode_log: list[EpisodeRecord] = [] if log_episodes else []
        self._log_episodes = log_episodes

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> MonitorObservation:
        if seed is not None:
            self._rng = random.Random(seed)

        self._episode_id = str(uuid.uuid4())[:8]
        self._step = 0
        self._monitor_actions = []
        self._intervention_step = None

        # 15% of episodes are heldout_novel (unseen exploit types)
        if self._rng.random() < HELDOUT_FRACTION:
            self._difficulty = "heldout_novel"
            legit_frac = HELDOUT_LEGIT_FRACTION
            available_hacks = HELDOUT_PRIMITIVES
        else:
            self._difficulty = "train"
            legit_frac = LEGIT_FRACTION
            available_hacks = TRAINING_HACK_TYPES

        self._current_task = self._rng.choice(ALL_TASKS)

        # Sample legit vs hack
        if self._rng.random() < legit_frac:
            hack_type: str | None = "legit"
        else:
            hack_type = self._rng.choice(available_hacks)

        # Run scripted worker to produce trajectory
        worker_fn = sample_worker(hack_type)
        self._worker_trajectory = worker_fn(self._current_task)

        # planted_label: legit worker steps have no "planted_label" key
        self._planted_label = self._worker_trajectory[0].get("planted_label", "legit")

        return self._make_observation(step_idx=0, done=False)

    def step(
        self, action: MonitorAction | dict
    ) -> tuple[MonitorObservation, float | None, bool, dict]:
        if isinstance(action, dict):
            action = MonitorAction.from_dict(action)

        self._monitor_actions.append(action)

        if action.verdict in ("flag", "block") and self._intervention_step is None:
            self._intervention_step = self._step

        self._step += 1
        done = (action.verdict == "block") or (self._step >= len(self._worker_trajectory))

        reward: float | None = None
        info: dict = {}
        if done:
            info = compute_episode_reward(
                planted_label=self._planted_label,
                monitor_flags=self._monitor_actions,
                worker_succeeded=True,
                intervention_step=self._intervention_step,
                total_steps=max(1, self._step),
            )
            reward = info["reward"]
            # Include difficulty so callers don't need a second /state call.
            # This is safe — difficulty ("train"/"heldout_novel") is not ground truth.
            info["difficulty"] = self._difficulty

            if self._log_episodes:
                self._episode_log.append(EpisodeRecord(
                    episode_id=self._episode_id,
                    task_id=self._current_task["task_id"] if self._current_task else "",
                    difficulty=self._difficulty,
                    planted_label=self._planted_label,
                    monitor_actions=[a.to_dict() for a in self._monitor_actions],
                    reward=reward,
                    reward_components={k: v for k, v in info.items() if k != "planted_label"},
                ))

        obs = self._make_observation(step_idx=self._step, done=done)
        return obs, reward, done, info

    @property
    def state(self) -> HackWatchState:
        return HackWatchState(
            episode_id=self._episode_id,
            step_count=self._step,
            task_id=self._current_task["task_id"] if self._current_task else "",
            difficulty=self._difficulty,  # type: ignore[arg-type]
            planted_label=self._planted_label,
        )

    def episode_log(self) -> list[EpisodeRecord]:
        return list(self._episode_log)

    def reset_log(self) -> None:
        self._episode_log.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_observation(self, step_idx: int, done: bool) -> MonitorObservation:
        task = self._current_task
        traj_idx = min(step_idx, len(self._worker_trajectory) - 1)
        traj_step = self._worker_trajectory[traj_idx]

        action_dict = traj_step.get("action", {})
        diff_str = traj_step.get("diff", "")

        # Accumulate tools used across all steps seen so far
        tools_so_far: list[str] = []
        for i in range(min(traj_idx + 1, len(self._worker_trajectory))):
            tool = self._worker_trajectory[i]["action"].get("tool", "write_file")
            tools_so_far.append(tool)

        return MonitorObservation(
            task_prompt=task["prompt"] if task else "",
            worker_last_action=action_dict,
            worker_diff=diff_str,
            tools_used_so_far=tools_so_far,
            step_idx=step_idx,
            episode_done=done,
        )
