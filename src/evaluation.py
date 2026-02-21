"""
evaluation.py

Centralized evaluation utilities for classification pipelines.

Features:
- Accuracy and Macro-F1 computation
- Compact confusion matrix (default)
- Optional full confusion matrix
- Organized output folders by pipeline type
- Reusable across BoW and embedding pipelines
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Base output directory
BASE_OUTPUT_DIR = "outputs"

def get_output_dir(subfolder: str) -> str:
    """
    Create and return output directory for a given subfolder.
    """

    path = os.path.join(BASE_OUTPUT_DIR, subfolder)

    os.makedirs(path, exist_ok=True)

    return path


def evaluate_classifier(
    model,
    X_test,
    y_test,
    label_names,
    model_name: str,
    subfolder: str,
    prefix: str = "",
    save_full_matrix: bool = False,
    save_compact_matrix: bool = True
):
    """
    Evaluate classifier performance and generate evaluation outputs.

    Returns:
        accuracy (float)
        macro_f1 (float)
        y_pred (np.ndarray)
    """

    print(f"\n=== Evaluating: {model_name} ===")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_names
        )
    )

    if save_full_matrix:

        plot_confusion_matrix(
            y_test,
            y_pred,
            label_names,
            model_name,
            subfolder,
            prefix
        )

    if save_compact_matrix:

        plot_compact_confusion_matrix(
            y_test,
            y_pred,
            label_names,
            model_name,
            subfolder,
            prefix
        )

    return accuracy, macro_f1, y_pred


def plot_confusion_matrix(
    y_true,
    y_pred,
    label_names,
    model_name: str,
    subfolder: str,
    prefix: str = ""
):
    """
    Save full confusion matrix heatmap.
    """

    output_dir = get_output_dir(subfolder)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names
    )

    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = os.path.join(
        output_dir,
        f"{prefix}confusion_matrix_full_{model_name.replace(' ', '_')}.png"
    )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Saved full confusion matrix to {filename}")


def plot_compact_confusion_matrix(
    y_true,
    y_pred,
    label_names,
    model_name: str,
    subfolder: str,
    prefix: str = "",
    top_n: int = 10
):
    """
    Save compact confusion matrix showing most confused classes.
    """

    output_dir = get_output_dir(subfolder)

    cm = confusion_matrix(y_true, y_pred)

    # Compute total misclassifications per class
    confusion_totals = cm.sum(axis=1) - np.diag(cm)

    # Get indices of most confused classes
    top_indices = np.argsort(confusion_totals)[-top_n:]

    cm_small = cm[np.ix_(top_indices, top_indices)]

    labels_small = [label_names[i] for i in top_indices]

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm_small,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_small,
        yticklabels=labels_small
    )

    plt.title(f"Top Confusions: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    filename = os.path.join(
        output_dir,
        f"{prefix}confusion_matrix_compact_{model_name.replace(' ', '_')}.png"
    )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Saved compact confusion matrix to {filename}")