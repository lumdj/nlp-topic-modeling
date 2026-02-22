# NLP Topic Classification and Clustering Pipeline

This project builds an end-to-end NLP pipeline using the 20 Newsgroups dataset to demonstrate:

- Multi-class classification using TF-IDF features
- Classification using SentenceTransformer embeddings
- Semantic clustering using KMeans
- Hierarchical topic tree generation using LLM-based topic labeling

The project compares traditional sparse features with modern dense embeddings and produces interpretable topic groupings.

---

# Repository Structure

```
nlp-topic-modeling/
│
├── src/
│   ├── data_loader.py
│   ├── tfidf_pipeline.py
│   ├── embedding_pipeline.py
│   ├── clustering.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── compare_models.py
│   │
│   └── api/
│       └── openai_client.py
│
├── notebooks/
│   └── demo.ipynb
│
├── outputs/
│   ├── tfidf/
│   ├── embeddings/
│   ├── clustering/
│   └── comparisons/
│
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
└── .gitignore
```

---

# Setup Instructions (Fresh Install)

## 1. Clone repository

```
git clone <your-repo-url>
cd nlp-topic-modeling
```

---

## 2. Create virtual environment

```
python3 -m venv venv
```

Activate:

Mac/Linux:

```
source venv/bin/activate
```

Windows:

```
venv\Scripts\activate
```

---

## 3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Install project as package (IMPORTANT)

This ensures imports work correctly everywhere:

```
pip install -e .
```

---

## 5. Set OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

---

# How to Run

Run all components from project root.

---

## Part 1 — TF-IDF Classification

```
python -m src.tfidf_pipeline
```

Outputs saved to:

```
outputs/tfidf/
```

Includes:

- Confusion matrices
- Classification metrics

---

## Part 2 — Embedding Classification

```
python -m src.embedding_pipeline
```

Outputs saved to:

```
outputs/embeddings/
```

---

## Model Comparison

```
python -m src.compare_models
```

Outputs saved to:

```
outputs/comparisons/
```

---

## Part 3 — Clustering and Topic Tree

```
python -m src.clustering
```

Outputs saved to:

```
outputs/clustering/
```

Includes:

- Elbow plot
- Topic tree text file
- Hierarchical topic structure

---

# Demo Notebook

To view results interactively:

```
jupyter notebook notebooks/demo.ipynb
```

---

# Reproducibility (Recommended order)

Run:

```
python -m src.tfidf_pipeline
python -m src.embedding_pipeline
python -m src.compare_models
python -m src.clustering
```

---

# Technologies Used

- Python
- scikit-learn
- SentenceTransformers
- PyTorch
- OpenAI API
- KMeans clustering
- Matplotlib / Seaborn
- Jupyter Notebook

---

# AI Assistance Disclosure

Portions of this project were developed with the assistance of ChatGPT (OpenAI) as a programming and documentation aid. All implementation decisions, testing, and validation were performed by the authors.