import numpy as np

from cataract_classifier.inference.ensemble import (
    accuracy_weighted_ensemble,
    average_probability_ensemble,
    simple_argmax_ensemble,
)


CLASS_NAMES = ["Immature", "Mature", "Normal"]


def test_simple_argmax_ensemble():
    predictions = {"a": np.array([0.1, 0.8, 0.1]), "b": np.array([0.2, 0.3, 0.5])}
    result = simple_argmax_ensemble(predictions, CLASS_NAMES)
    assert result["a"]["predicted_class"] == "Mature"
    assert result["b"]["predicted_class"] == "Normal"


def test_average_probability_ensemble():
    predictions = {"a": np.array([0.1, 0.8, 0.1]), "b": np.array([0.7, 0.2, 0.1])}
    label, confidence, avg_probs = average_probability_ensemble(predictions, CLASS_NAMES)
    assert label == "Immature"
    assert confidence == avg_probs[0]


def test_accuracy_weighted_ensemble():
    predictions = {"a": np.array([0.1, 0.8, 0.1]), "b": np.array([0.7, 0.2, 0.1])}
    weights = {"a": 99.0, "b": 1.0}
    label, confidence, weighted = accuracy_weighted_ensemble(predictions, weights, CLASS_NAMES)
    assert label == "Mature"
    assert confidence == weighted["Mature"]
