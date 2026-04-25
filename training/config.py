"""
Central config loader for HackWatch training.

All training files import from here instead of each loading yaml themselves.
The config file is training/configs/grpo_base.yaml — edit that to change defaults.
"""
from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "configs" / "grpo_base.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache grpo_base.yaml. Returns the full config dict."""
    return yaml.safe_load(_CONFIG_PATH.read_text())


def grpo_cfg() -> dict:
    """Return the [grpo] section of grpo_base.yaml."""
    return load_config().get("grpo", {})


def lora_cfg() -> dict:
    """Return the [lora] section of grpo_base.yaml."""
    return load_config().get("lora", {})


def model_cfg() -> dict:
    """Return the [model] section of grpo_base.yaml."""
    return load_config().get("model", {})
