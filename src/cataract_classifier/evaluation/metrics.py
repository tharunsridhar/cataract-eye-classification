from __future__ import annotations

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def get_predictions(model, test_data):
    y_true = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)
    return y_true, y_pred


def print_classification_report(y_true, y_pred, label_names) -> str:
    report = classification_report(y_true, y_pred, target_names=label_names)
    print(report)
    return report


def plot_confusion_matrix(y_true, y_pred, label_names, title) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
