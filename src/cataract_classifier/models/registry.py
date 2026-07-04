from __future__ import annotations

from pathlib import Path

from tensorflow.keras.models import load_model

from cataract_classifier.config_loader import load_config
from cataract_classifier.data.transforms import get_preprocess_function


def load_registry(path: str = "configs/model_registry.yaml") -> dict:
    return load_config(path)


def load_trained_model(name: str, registry: dict):
    model_info = _get_model_info(name, registry)
    return load_model(model_info["path"], compile=False)


def get_preprocess_fn(name: str, registry: dict):
    model_info = _get_model_info(name, registry)
    return get_preprocess_function(model_info["preprocess"])


def get_model_path(name: str, registry: dict) -> Path:
    return Path(_get_model_info(name, registry)["path"])


def _get_model_info(name: str, registry: dict) -> dict:
    models = registry.get("models", {})
    key = "resnet50" if name == "resnet" else name
    try:
        return models[key]
    except KeyError as exc:
        raise ValueError(f"Unknown registered model: {name}") from exc
