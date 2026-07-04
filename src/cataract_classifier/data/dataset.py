from __future__ import annotations

import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_generators(data_cfg: dict, preprocess_fn):
    image_size = tuple(data_cfg["image_size"])
    batch_size = data_cfg["batch_size"]
    augmentation = data_cfg.get("augmentation", {})

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn, **augmentation)
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_data = train_datagen.flow_from_directory(
        data_cfg["data_dir"]["train"],
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    valid_data = val_test_datagen.flow_from_directory(
        data_cfg["data_dir"]["valid"],
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_data = val_test_datagen.flow_from_directory(
        data_cfg["data_dir"]["test"],
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, valid_data, test_data


def build_test_generator(data_cfg: dict, preprocess_fn):
    image_size = tuple(data_cfg["image_size"])
    batch_size = data_cfg["batch_size"]
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    return test_datagen.flow_from_directory(
        data_cfg["data_dir"]["test"],
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def compute_class_weights(train_generator) -> dict[int, float]:
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    return dict(enumerate(weights))
