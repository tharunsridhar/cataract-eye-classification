from __future__ import annotations

import argparse
from pathlib import Path

from cataract_classifier.config_loader import load_config
from cataract_classifier.data.dataset import build_generators, compute_class_weights
from cataract_classifier.data.transforms import get_preprocess_function
from cataract_classifier.models.architectures import build_model
from cataract_classifier.training.callbacks import checkpoint_callback, default_callbacks
from cataract_classifier.training.trainer import combine_histories, fine_tune, train_head, unfreeze_layers
from cataract_classifier.utils.io_utils import ensure_dir, make_versioned_filename
from cataract_classifier.utils.plotting import plot_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["densenet121", "efficientnetb3", "resnet50", "vgg19"])
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)[args.model]
    preprocess_fn = get_preprocess_function(_preprocess_name(args.model))
    train_data, valid_data, test_data = build_generators(data_cfg, preprocess_fn)
    class_weights = compute_class_weights(train_data)

    model = build_model(args.model, len(data_cfg["class_names"]), tuple(data_cfg["image_size"]))
    callbacks = default_callbacks()
    if args.checkpoint:
        callbacks.append(checkpoint_callback(str(Path(args.model_dir) / f"{args.model}_best_model.h5")))

    history_head = train_head(
        model,
        train_data,
        valid_data,
        train_cfg["epochs_head"],
        train_cfg["lr_head"],
        class_weights,
        callbacks,
    )

    n_or_block = train_cfg.get("unfreeze_n", train_cfg.get("unfreeze_block"))
    unfreeze_layers(model.base_model, train_cfg["unfreeze_strategy"], n_or_block)
    history_fine = fine_tune(
        model,
        train_data,
        valid_data,
        train_cfg["epochs_fine"],
        train_cfg["lr_fine"],
        class_weights,
        callbacks,
    )

    full_history = combine_histories(history_head, history_fine)
    plot_training(full_history, args.model)

    loss, acc = model.evaluate(test_data)
    ensure_dir(args.model_dir)
    model.save(Path(args.model_dir) / make_versioned_filename(args.model, acc * 100, "h5"))
    model.save(Path(args.model_dir) / make_versioned_filename(args.model, acc * 100, "keras"))
    print(f"Test accuracy: {acc * 100:.2f}%")


def _preprocess_name(model_name: str) -> str:
    return {
        "densenet121": "densenet",
        "efficientnetb3": "efficientnet",
        "resnet50": "resnet50",
        "vgg19": "vgg19",
    }[model_name]


if __name__ == "__main__":
    main()
