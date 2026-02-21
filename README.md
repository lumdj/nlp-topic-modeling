# NLP Topic Classification and Clustering Pipeline

This project builds an end-to-end NLP pipeline using the 20 Newsgroups dataset to demonstrate:

1. Multi-class text classification using TF-IDF features
2. Multi-class classification using SentenceTransformer embeddings
3. Semantic clustering and hierarchical topic tree generation using KMeans and LLM labeling

The system compares sparse and dense feature representations and produces interpretable topic structures using modern NLP techniques.

---

# Dataset

Dataset: 20 Newsgroups (via scikit-learn)

- 20 topic categories
- Subsampled to 10,000 documents for performance
- Multi-class classification problem

---

# Project Structure

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
│   ├── bow/
│   ├── embeddings/
│   ├── clustering/
│   └── comparisons/
│
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

---

# Setup Instructions

## 1. Create virtual environment

```
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

## 2. Install dependencies

```
pip install -r requirements.txt
```

## 3. Set OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

---

# How to Run

All commands should be run from the project root directory.

---

# Part 1 — TF-IDF Classification

Runs supervised classification using TF-IDF features and evaluates:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest

Command:

```
python -m src.tfidf_pipeline
```

Outputs saved to:

```
outputs/bow/
```

Includes:

- Compact confusion matrices
- Classification metrics

---

# Part 2 — SentenceTransformer Embedding Classification

Runs classification using SentenceTransformer embeddings and the same classifiers.

Command:

```
python -m src.embedding_pipeline
```

Outputs saved to:

```
outputs/embeddings/
```

Includes:

- Compact confusion matrices
- Classification metrics

---

# Model Comparison

Compares TF-IDF and embedding classification performance.

Command:

```
python -m src.compare_models
```

Output saved to:

```
outputs/comparisons/model_comparison.csv
```

---

# Part 3 — Topic Clustering and Topic Tree

Performs semantic clustering using SentenceTransformer embeddings and KMeans.

Features:

- Elbow method to determine optimal cluster count
- Top-level clustering (<10 clusters)
- Sub-clustering of 2 largest clusters
- LLM-generated topic labels
- Hierarchical topic tree output

Command:

```
python -m src.clustering
```

Outputs saved to:

```
outputs/clustering/
```

Includes:

```
elbow_plot.png
topic_tree.txt
topic_tree.json
```

---

# Demo Notebook

To view results interactively:

```
notebooks/demo.ipynb
```

The notebook demonstrates:

- Dataset overview
- TF-IDF classification results
- Embedding classification results
- Model comparison
- Elbow method visualization
- Topic tree output

---

# Summary of Results

Key observations:

- TF-IDF achieved highest classification accuracy (~71%)
- SentenceTransformer embeddings provided strong semantic clustering
- Linear SVM performed best overall
- LLM labeling produced interpretable topic hierarchy

---

# Technologies Used

- Python
- scikit-learn
- SentenceTransformers
- OpenAI API
- KMeans clustering
- Matplotlib / Seaborn
- Jupyter Notebook

---

# Reproducibility

To fully reproduce results, run in this order:

```
python -m src.tfidf_pipeline
python -m src.embedding_pipeline
python -m src.compare_models
python -m src.clustering
```

Then open:

```
notebooks/demo.ipynb
```