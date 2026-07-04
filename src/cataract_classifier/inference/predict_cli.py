from __future__ import annotations

import argparse

from cataract_classifier.config_loader import load_config
from cataract_classifier.inference.ensemble import (
    accuracy_weighted_ensemble,
    average_probability_ensemble,
    simple_argmax_ensemble,
)
from cataract_classifier.inference.predictor import preprocess_image
from cataract_classifier.models.registry import get_preprocess_fn, load_registry, load_trained_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--strategy", choices=["simple", "average", "weighted"], default="weighted")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    registry = load_registry(args.registry)
    class_names = data_cfg["class_names"]
    predictions = {}
    accuracy_weights = {}

    for model_name, model_info in registry["models"].items():
        model = load_trained_model(model_name, registry)
        preprocess_fn = get_preprocess_fn(model_name, registry)
        img_array = preprocess_image(args.image, data_cfg["image_size"], preprocess_fn)
        predictions[model_name] = model.predict(img_array, verbose=0)[0]
        accuracy_weights[model_name] = float(model_info["reported_test_accuracy"])

    if args.strategy == "simple":
        for model_name, result in simple_argmax_ensemble(predictions, class_names).items():
            print(f"{model_name}: {result['predicted_class']} ({result['confidence']:.2%})")
        return

    if args.strategy == "average":
        label, confidence, _ = average_probability_ensemble(predictions, class_names)
    else:
        label, confidence, _ = accuracy_weighted_ensemble(predictions, accuracy_weights, class_names)

    print(f"Predicted Class: {label}")
    print(f"Ensemble Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
