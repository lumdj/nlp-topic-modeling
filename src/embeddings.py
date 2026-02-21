"""
embeddings.py

Centralized embedding utilities using SentenceTransformer.
Used by both classification and clustering pipelines.
"""

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(model_name=DEFAULT_MODEL):

    print(f"Loading embedding model: {model_name}")

    return SentenceTransformer(model_name)


def generate_embeddings(
    model,
    texts,
    batch_size=64,
    normalize_embeddings=True
):

    print(f"Generating embeddings for {len(texts)} documents...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    if normalize_embeddings:
        embeddings = normalize(embeddings)

    return embeddings