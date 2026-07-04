import matplotlib
import numpy as np

matplotlib.use("Agg")

from cataract_classifier.evaluation.metrics import plot_confusion_matrix, print_classification_report


def test_metrics_helpers_do_not_error():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    labels = ["Immature", "Mature", "Normal"]
    report = print_classification_report(y_true, y_pred, labels)
    assert "precision" in report
    plot_confusion_matrix(y_true, y_pred, labels, "Test")
