from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from cataract_classifier.config_loader import load_config
from cataract_classifier.data.dataset import build_test_generator
from cataract_classifier.evaluation.metrics import (
    get_predictions,
    plot_confusion_matrix,
    print_classification_report,
)
from cataract_classifier.models.registry import get_preprocess_fn, load_registry, load_trained_model
from cataract_classifier.utils.io_utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--output-dir", default="docs/images")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    registry = load_registry(args.registry)
    model = load_trained_model(args.model, registry)
    preprocess_fn = get_preprocess_fn(args.model, registry)
    test_data = build_test_generator(data_cfg, preprocess_fn)
    label_names = list(test_data.class_indices.keys())

    y_true, y_pred = get_predictions(model, test_data)
    print_classification_report(y_true, y_pred, label_names)
    plot_confusion_matrix(y_true, y_pred, label_names, f"{args.model} Confusion Matrix")
    ensure_dir(args.output_dir)
    plt.savefig(Path(args.output_dir) / f"{args.model}_confusion_matrix.png")


if __name__ == "__main__":
    main()
