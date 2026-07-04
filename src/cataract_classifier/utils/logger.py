from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from cataract_classifier.config_loader import load_config


def get_logger(name: str) -> logging.Logger:
    config_path = Path("configs/logging_config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        for handler in cfg.get("handlers", {}).values():
            filename = handler.get("filename")
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
