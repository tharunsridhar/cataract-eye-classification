from __future__ import annotations

import numpy as np


def simple_argmax_ensemble(predictions_per_model: dict, class_names) -> dict:
    results = {}
    for model_name, probs in predictions_per_model.items():
        top_idx = int(np.argmax(probs))
        results[model_name] = {
            "predicted_class": class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "class_confidences": {
                class_names[index]: float(confidence) for index, confidence in enumerate(probs)
            },
        }
    return results


def average_probability_ensemble(predictions_per_model: dict, class_names) -> tuple[str, float, np.ndarray]:
    all_probs = np.array(list(predictions_per_model.values()))
    avg_probs = all_probs.mean(axis=0)
    best_idx = int(np.argmax(avg_probs))
    return class_names[best_idx], float(avg_probs[best_idx]), avg_probs


def accuracy_weighted_ensemble(
    predictions_per_model: dict, accuracy_weights: dict, class_names
) -> tuple[str, float, dict]:
    total_accuracy = sum(accuracy_weights.values())
    weighted_confidences = {}
    for class_name in class_names:
        class_idx = class_names.index(class_name)
        weighted_sum = 0.0
        for model_name, probs in predictions_per_model.items():
            model_weight = accuracy_weights[model_name] / total_accuracy
            weighted_sum += float(probs[class_idx]) * model_weight
        weighted_confidences[class_name] = weighted_sum

    final_class = max(weighted_confidences, key=weighted_confidences.get)
    final_confidence = weighted_confidences[final_class]
    individual_preds = {
        name: class_names[int(np.argmax(probs))] for name, probs in predictions_per_model.items()
    }
    if len(set(individual_preds.values())) > 1:
        print("MODEL DISAGREEMENT DETECTED:")
        for model_name, pred in individual_preds.items():
            print(f"  {model_name:20s}: {pred}")
        print("Resolved using weighted confidence matrix approach")
    else:
        print(f"All models agree on: {final_class}")
    return final_class, float(final_confidence), weighted_confidences
