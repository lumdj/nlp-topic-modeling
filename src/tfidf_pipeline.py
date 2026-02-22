"""
bow_pipeline.py

Implements Bag-of-Words and TF-IDF classification pipelines.

Models included:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest

Outputs:
- Accuracy
- Macro-F1
- Confusion matrix
"""

import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import load_dataset
from src.evaluation import evaluate_classifier

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Pipeline builder
# -------------------------

def build_pipelines():
    """
    Build pipelines for all models.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2
    )

    pipelines = {
        "Naive Bayes": Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", LogisticRegression(
                max_iter=1000,
                n_jobs=-1
            ))
        ]),

        "Linear SVM": Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", LinearSVC())
        ]),

        "Random Forest": Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=42
            ))
        ])
    }

    return pipelines

# -------------------------
# Main runner
# -------------------------

def run_experiment():
    """
    Run full experiment.
    """

    print(f"\nLoading dataset...")
    X_train, X_test, y_train, y_test, label_names = load_dataset()

    pipelines = build_pipelines()

    results = {}

    for name, pipeline in pipelines.items():

        print(f"\nTraining {name}...")

        pipeline.fit(X_train, y_train)

        accuracy, macro_f1, y_pred = evaluate_classifier(
            pipeline,
            X_test,
            y_test,
            label_names,
            model_name=name,
            subfolder="tfidf",
            prefix="tfidf_"
        )

        results[name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

    print("\nFinal Results Summary:")
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

    print("\nRunning TF-IDF experiment...")
    run_experiment()

