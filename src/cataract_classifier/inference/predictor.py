from __future__ import annotations

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image


def preprocess_image(img, target_size, preprocess_fn) -> np.ndarray:
    if not isinstance(img, Image.Image):
        img = image.load_img(img, target_size=tuple(target_size))
    else:
        img = img.resize(tuple(target_size))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_fn(arr)


def predict_single(model, img_array, class_names) -> tuple[str, float, np.ndarray]:
    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]), preds
