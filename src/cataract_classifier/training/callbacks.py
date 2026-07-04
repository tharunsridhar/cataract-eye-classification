from __future__ import annotations

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def default_callbacks() -> list:
    return [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]


def checkpoint_callback(path: str) -> ModelCheckpoint:
    return ModelCheckpoint(path, monitor="val_loss", save_best_only=True, verbose=1)
