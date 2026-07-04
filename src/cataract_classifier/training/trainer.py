from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def train_head(model, train_data, valid_data, epochs, lr, class_weights, callbacks):
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss="categorical_crossentropy", metrics=["accuracy"])
    return model.fit(
        train_data,
        validation_data=valid_data,
        epochs=int(epochs),
        class_weight=class_weights,
        callbacks=callbacks,
    )


def unfreeze_layers(base_model, strategy: str, n_or_block) -> None:
    if strategy == "last_n_layers":
        for layer in base_model.layers[-int(n_or_block) :]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        return

    if strategy == "by_block_name":
        block_name = str(n_or_block)
        for layer in base_model.layers:
            layer.trainable = layer.name.startswith(block_name)
        return

    raise ValueError(f"Unknown unfreeze strategy: {strategy}")


def fine_tune(model, train_data, valid_data, epochs, lr, class_weights, callbacks):
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss="categorical_crossentropy", metrics=["accuracy"])
    return model.fit(
        train_data,
        validation_data=valid_data,
        epochs=int(epochs),
        class_weight=class_weights,
        callbacks=callbacks,
    )


def combine_histories(h1, h2) -> dict:
    history = {}
    for key in h1.history.keys():
        history[key] = h1.history[key] + h2.history[key]
    return history
