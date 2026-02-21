"""
compare_models.py

Runs TF-IDF and SentenceTransformer pipelines and compares results.
Saves comparison table to outputs/comparisons/.
"""

import os
import pandas as pd

from src.tfidf_pipeline import run_experiment as run_tfidf_experiment
from src.embedding_pipeline import run_experiment as run_embedding_experiment
from src.evaluation import get_output_dir


def run_comparison():

    print("Running TF-IDF experiment...")
    tfidf_results = run_tfidf_experiment()

    print("\nRunning embedding experiment...")
    embedding_results = run_embedding_experiment()

    tfidf_df = pd.DataFrame(tfidf_results).T
    emb_df = pd.DataFrame(embedding_results).T

    comparison = pd.DataFrame({
        "TF-IDF Accuracy": tfidf_df["accuracy"],
        "Embedding Accuracy": emb_df["accuracy"],
        "TF-IDF Macro-F1": tfidf_df["macro_f1"],
        "Embedding Macro-F1": emb_df["macro_f1"],
    })

    print("\n=== Model Comparison ===")
    print(comparison)

    # Save results
    output_dir = get_output_dir("comparisons")

    output_path = os.path.join(
        output_dir,
        "model_comparison.csv"
    )

    comparison.to_csv(output_path)

    print(f"\nSaved comparison to {output_path}")

    return comparison


if __name__ == "__main__":

    run_comparison()