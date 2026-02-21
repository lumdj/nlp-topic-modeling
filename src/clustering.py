"""
clustering.py

Builds topic clusters and generates a 2-level topic tree using:

- SentenceTransformer embeddings
- KMeans clustering
- OpenAI LLM topic labeling

Outputs (saved to outputs/clustering/):
- elbow_plot.png
- topic_tree.txt
- topic_tree.json
"""

import os
import json
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

from src.data_loader import load_raw_dataset, subsample_dataset
from src.embeddings import load_embedding_model, generate_embeddings
from src.api.openai_client import generate_topic_label
from src.evaluation import get_output_dir


# Ensure clustering output folder exists
OUTPUT_DIR = get_output_dir("clustering")


# -------------------------
# Clustering helpers
# -------------------------

def find_optimal_k(embeddings, max_k=9, plot=True):
    """
    Automatically determine optimal K using elbow method.
    Saves elbow plot to outputs/clustering/.
    """

    inertias = []
    k_values = list(range(2, max_k + 1))

    print("Computing inertia for elbow method...")

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    # Detect elbow automatically
    knee = KneeLocator(
        k_values,
        inertias,
        curve="convex",
        direction="decreasing"
    )

    optimal_k = knee.knee

    # Safe fallback if KneeLocator fails
    if optimal_k is None:
        optimal_k = 8
        print("KneeLocator failed. Defaulting to K=8")

    print(f"Optimal K selected: {optimal_k}")

    # Save elbow plot
    if plot:

        plt.figure(figsize=(6, 4))

        plt.plot(k_values, inertias, marker="o")

        plt.axvline(
            optimal_k,
            color="red",
            linestyle="--",
            label=f"Optimal K={optimal_k}"
        )

        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.legend()

        path = os.path.join(OUTPUT_DIR, "elbow_plot.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        print(f"Saved elbow plot to {path}")

    return optimal_k


def run_kmeans(embeddings, k):

    print(f"Running KMeans with k={k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(embeddings)

    return kmeans, labels


def get_representative_docs(
    texts,
    embeddings,
    kmeans,
    labels,
    cluster_id,
    n_docs=5
):

    cluster_indices = np.where(labels == cluster_id)[0]

    cluster_embeddings = embeddings[cluster_indices]

    centroid = kmeans.cluster_centers_[cluster_id]

    distances = np.linalg.norm(
        cluster_embeddings - centroid,
        axis=1
    )

    sorted_indices = np.argsort(distances)

    representative_docs = []

    for i in sorted_indices:

        doc = texts[cluster_indices[i]].strip()

        # Only skip completely empty docs
        if len(doc) > 0:
            representative_docs.append(doc)

        if len(representative_docs) >= n_docs:
            break

    return representative_docs


# -------------------------
# Topic labeling
# -------------------------

def label_clusters(texts, embeddings, kmeans, labels, k):

    cluster_labels = {}

    for cluster_id in range(k):

        print(f"\nLabeling cluster {cluster_id}")

        docs = get_representative_docs(
            texts,
            embeddings,
            kmeans,
            labels,
            cluster_id
        )

        print(f"Representative docs count: {len(docs)}")
        print("Sample doc preview:", docs[0][:100] if docs else "NONE")

        label = generate_topic_label(docs)

        cluster_labels[cluster_id] = label

        print(f"Cluster {cluster_id}: {label}")

    return cluster_labels


# -------------------------
# Topic tree construction
# -------------------------

def find_largest_clusters(labels, n=2):

    counts = Counter(labels)

    largest = sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [cluster_id for cluster_id, _ in largest[:n]]


def build_topic_tree(texts, embeddings, top_k):

    print("\n=== TOP LEVEL CLUSTERING ===")

    top_kmeans, top_labels = run_kmeans(
        embeddings,
        top_k
    )

    top_cluster_names = label_clusters(
        texts,
        embeddings,
        top_kmeans,
        top_labels,
        top_k
    )

    topic_tree = {}

    largest_clusters = find_largest_clusters(
        top_labels,
        n=2
    )

    print("\n=== SECOND LEVEL CLUSTERING ===")

    for cluster_id in largest_clusters:

        print(f"\nSub-clustering cluster {cluster_id}")

        cluster_indices = np.where(
            top_labels == cluster_id
        )[0]

        sub_texts = [
            texts[i]
            for i in cluster_indices
        ]

        sub_embeddings = embeddings[cluster_indices]

        sub_kmeans, sub_labels = run_kmeans(
            sub_embeddings,
            k=3
        )

        sub_cluster_names = label_clusters(
            sub_texts,
            sub_embeddings,
            sub_kmeans,
            sub_labels,
            3
        )

        topic_tree[top_cluster_names[cluster_id]] = list(
            sub_cluster_names.values()
        )

    return topic_tree


# -------------------------
# Display tree
# -------------------------

def print_topic_tree(topic_tree):

    print("\n=== TOPIC TREE ===\n")

    for parent, children in topic_tree.items():

        print(parent)

        for child in children:
            print(f"  └── {child}")

        print()


def save_topic_tree(topic_tree):
    """
    Save topic tree as both text and JSON.
    """

    text_path = os.path.join(
        OUTPUT_DIR,
        "topic_tree.txt"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        "topic_tree.json"
    )

    # Save text version
    with open(text_path, "w") as f:

        for parent, children in topic_tree.items():

            f.write(parent + "\n")

            for child in children:
                f.write(f"  └── {child}\n")

            f.write("\n")

    print(f"Saved topic tree to {text_path}")

    # Save JSON version
    with open(json_path, "w") as f:

        json.dump(
            topic_tree,
            f,
            indent=4
        )

    print(f"Saved topic tree to {json_path}")


# -------------------------
# Main
# -------------------------

def run_clustering():

    print("Loading dataset...")

    texts, labels, label_names = load_raw_dataset()

    texts, labels = subsample_dataset(
        texts,
        labels,
        max_samples=10000
    )

    print("Generating embeddings...")

    model = load_embedding_model()

    embeddings = generate_embeddings(
        model,
        texts
    )

    optimal_k = find_optimal_k(embeddings)

    topic_tree = build_topic_tree(
        texts,
        embeddings,
        top_k=optimal_k
    )

    print_topic_tree(topic_tree)

    save_topic_tree(topic_tree)


if __name__ == "__main__":
    run_clustering()