from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from hackwatch.models import MonitorAction, MonitorObservation, HackWatchState

try:
    from openenv.core import EnvClient, StepResult  # type: ignore[import]
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False

    @dataclass
    class StepResult:
        observation: MonitorObservation
        reward: float | None
        done: bool

    class EnvClient:
        """Minimal stub matching openenv-core's EnvClient interface."""
        pass


class HackWatchEnvClient(EnvClient):
    """
    HTTP client for the HackWatch environment server.

    Usage::

        async with HackWatchEnvClient("http://localhost:8000") as env:
            obs = await env.reset()
            while not obs.episode_done:
                action = MonitorAction(verdict="allow", confidence=0.1)
                result = await env.step(action)
                obs = result.observation
            print(result.reward)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self) -> "HackWatchEnvClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with HackWatchEnvClient(...) as env:'")
        return self._client

    def _step_payload(self, action: MonitorAction) -> dict:
        return action.to_dict()

    def _parse_result(self, payload: dict) -> StepResult:
        return StepResult(
            observation=MonitorObservation.from_dict(payload["observation"]),
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> HackWatchState:
        return HackWatchState.from_dict(payload)

    async def reset(self, seed: int | None = None) -> MonitorObservation:
        body: dict = {}
        if seed is not None:
            body["seed"] = seed
        resp = await self._get_client().post("/reset", json=body)
        resp.raise_for_status()
        return MonitorObservation.from_dict(resp.json())

    async def step(self, action: MonitorAction) -> StepResult:
        resp = await self._get_client().post("/step", json=self._step_payload(action))
        resp.raise_for_status()
        return self._parse_result(resp.json())

    async def get_state(self) -> HackWatchState:
        resp = await self._get_client().get("/state")
        resp.raise_for_status()
        return self._parse_state(resp.json())
