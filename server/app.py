"""
FastAPI application for HackWatch.

Mounts the REST API under / and the React demo under /demo.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from hackwatch.models import MonitorAction
from server.environment import HackWatchEnvironment

# One environment instance per server process (stateful per episode).
# For multi-worker deployment, use Redis-backed state instead.
_env = HackWatchEnvironment()


def create_app() -> FastAPI:
    app = FastAPI(
        title="HackWatch",
        description="OpenEnv RL environment for reward-hacking detection",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        # Serve index.html directly — HF Space iframes block cross-origin redirects
        demo_index = Path(__file__).parent.parent / "demo" / "build" / "index.html"
        if demo_index.exists():
            return FileResponse(str(demo_index), media_type="text/html")
        return RedirectResponse(url="/demo")

    @app.post("/reset")
    async def reset(body: dict = {}):  # noqa: B006
        seed = body.get("seed") if body else None
        obs = _env.reset(seed=seed)
        return obs.to_dict()

    @app.post("/step")
    async def step(body: dict):
        try:
            action = MonitorAction.from_dict(body)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        obs, reward, done, info = _env.step(action)
        # planted_label is ground truth — omit from public API response
        public_info = {k: v for k, v in info.items() if k != "planted_label"} if info else info
        return {
            "observation": obs.to_dict(),
            "reward": reward,
            "done": done,
            "info": public_info,
        }

    @app.get("/state")
    async def state():
        return _env.state.to_dict()

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    # Mount demo static files if build dir exists
    demo_build = Path(__file__).parent.parent / "demo" / "build"
    if demo_build.exists():
        app.mount("/demo", StaticFiles(directory=str(demo_build), html=True), name="demo")

    return app


app = create_app()
