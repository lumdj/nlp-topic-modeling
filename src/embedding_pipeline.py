"""
embedding_pipeline.py

Implements document classification using SentenceTransformer embeddings
and classical ML classifiers.

Models included:
- Multinomial Naive Bayes (included for comparison; performs poorly)
- Logistic Regression
- Linear SVM
- Random Forest

Outputs:
- Accuracy
- Macro-F1
- Confusion matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import load_dataset
from src.embeddings import load_embedding_model, generate_embeddings
from src.evaluation import evaluate_classifier


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Classifiers
# -------------------------

def build_classifiers():
    """
    Return classifiers dictionary.
    """

    classifiers = {
        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ),

        "Linear SVM": LinearSVC(),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42
        )
    }

    return classifiers

# -------------------------
# Main experiment
# -------------------------

def run_experiment():

    print("\nLoading dataset...")
    X_train, X_test, y_train, y_test, label_names = load_dataset()

    embedding_model = load_embedding_model()

    # Generate embeddings
    X_train_emb = generate_embeddings(embedding_model, X_train)
    X_test_emb = generate_embeddings(embedding_model, X_test)

    X_train_emb = normalize(X_train_emb)
    X_test_emb = normalize(X_test_emb)

    classifiers = build_classifiers()

    results = {}

    for name, classifier in classifiers.items():

        print(f"\nTraining {name}...")

        # MultinomialNB requires non-negative input
        if name == "Naive Bayes":
            X_train_use = np.abs(X_train_emb)
            X_test_use = np.abs(X_test_emb)
        else:
            X_train_use = X_train_emb
            X_test_use = X_test_emb

        classifier.fit(X_train_use, y_train)

        accuracy, macro_f1, y_pred = evaluate_classifier(
            classifier,
            X_test_emb,
            y_test,
            label_names,
            model_name=name,
            subfolder="embeddings"
        )

        results[name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

    print("\nFinal Results Summary (Embeddings):")

    for name, metrics in results.items():
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )

    return results


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    run_experiment()
