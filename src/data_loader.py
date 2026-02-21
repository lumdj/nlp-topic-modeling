"""
data_loader.py

Responsible for loading and preparing the 20 Newsgroups dataset.
Provides clean train/test splits and optional subsampling.

Used by:
- BoW / TF-IDF pipeline
- SentenceTransformer pipeline
- Clustering pipeline
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np


def load_raw_dataset(remove_headers=True):
    """
    Load the full 20 Newsgroups dataset.

    Returns:
        texts (list[str])
        labels (np.ndarray)
        label_names (list[str])
    """

    remove = ("headers", "footers", "quotes") if remove_headers else ()

    dataset = fetch_20newsgroups(
        subset="all",
        remove=remove
    )

    texts = dataset.data
    labels = dataset.target
    label_names = dataset.target_names

    return texts, labels, label_names


def subsample_dataset(texts, labels, max_samples=10000, random_state=42):
    """
    Subsample dataset to a fixed size while preserving class balance.
    """

    if len(texts) <= max_samples:
        return texts, labels

    rng = np.random.default_rng(random_state)

    indices = rng.choice(
        len(texts),
        size=max_samples,
        replace=False
    )

    texts_sub = [texts[i] for i in indices]
    labels_sub = labels[indices]

    return texts_sub, labels_sub


def create_train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
):
    """
    Create stratified train/test split.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def load_dataset(
    max_samples=10000,
    test_size=0.2,
    random_state=42
):
    """
    Main entrypoint.

    Loads, subsamples, and splits dataset.

    Returns:
        X_train
        X_test
        y_train
        y_test
        label_names
    """

    texts, labels, label_names = load_raw_dataset()

    texts, labels = subsample_dataset(
        texts,
        labels,
        max_samples=max_samples,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = create_train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, label_names


if __name__ == "__main__":
    # Quick test

    X_train, X_test, y_train, y_test, label_names = load_dataset()

    print("Dataset loaded successfully")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Number of classes: {len(label_names)}")
