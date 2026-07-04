from __future__ import annotations

from tensorflow.keras.applications import DenseNet121, EfficientNetB3, ResNet50, VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_model(arch_name: str, num_classes: int, img_size: tuple[int, int]) -> Model:
    builders = {
        "vgg19": _build_vgg19,
        "resnet50": _build_resnet50,
        "resnet": _build_resnet50,
        "densenet121": _build_densenet121,
        "efficientnetb3": _build_efficientnetb3,
    }
    try:
        return builders[arch_name.lower()](num_classes, img_size)
    except KeyError as exc:
        raise ValueError(f"Unknown architecture: {arch_name}") from exc


def _build_vgg19(num_classes: int, img_size: tuple[int, int]) -> Model:
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.base_model = base_model
    return model


def _build_resnet50(num_classes: int, img_size: tuple[int, int]) -> Model:
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.base_model = base_model
    return model


def _build_densenet121(num_classes: int, img_size: tuple[int, int]) -> Model:
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.base_model = base_model
    return model


def _build_efficientnetb3(num_classes: int, img_size: tuple[int, int]) -> Model:
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.base_model = base_model
    return model
