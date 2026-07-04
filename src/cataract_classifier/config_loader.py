from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
ENV_DEFAULTS = {
    "DATA_DIR": "./data/raw",
    "MODEL_DIR": "./models",
    "LOG_FILE": "logs/cataract_classifier.log",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    return resolve_env_vars(cfg)


def resolve_env_vars(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        return {key: resolve_env_vars(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [resolve_env_vars(value) for value in cfg]
    if isinstance(cfg, str):
        return ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), ENV_DEFAULTS.get(match.group(1), match.group(0))),
            cfg,
        )
    return cfg
