from __future__ import annotations

from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as inception_resnet_v2_preprocess,
)
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess


PREPROCESS_FUNCTIONS = {
    "vgg19": vgg19_preprocess,
    "resnet50": resnet50_preprocess,
    "densenet": densenet_preprocess,
    "efficientnet": efficientnet_preprocess,
    "mobilenetv3": mobilenetv3_preprocess,
    "inception_resnet_v2": inception_resnet_v2_preprocess,
}


def get_preprocess_function(name: str):
    try:
        return PREPROCESS_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preprocess function: {name}") from exc
