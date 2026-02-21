# Architecture Overview

This project implements a modular NLP pipeline for classification and clustering using TF-IDF and SentenceTransformer embeddings.

The architecture separates data loading, feature generation, model training, evaluation, clustering, and LLM labeling into independent modules.

---

# High-Level Data Flow

```
Dataset
   ↓
data_loader.py
   ↓
embeddings.py (for embedding pipeline and clustering)
   ↓
tfidf_pipeline.py / embedding_pipeline.py
   ↓
evaluation.py
   ↓
compare_models.py
   ↓
clustering.py
   ↓
openai_client.py
   ↓
outputs/
```

---

# Module Responsibilities

---

## data_loader.py

Responsibilities:

- Load 20 Newsgroups dataset
- Subsample dataset to 10,000 documents
- Return:

```
texts
labels
label_names
```

This module is used by both classification and clustering pipelines.

---

## tfidf_pipeline.py

Responsibilities:

- Convert text to TF-IDF features using TfidfVectorizer
- Train classifiers:

  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
  - Random Forest

- Evaluate models using evaluation.py
- Save confusion matrices and metrics

Output:

```
outputs/bow/
```

---

## embeddings.py

Responsibilities:

- Load SentenceTransformer model
- Generate document embeddings
- Normalize embeddings

Used by:

- embedding_pipeline.py
- clustering.py

Centralizes embedding logic to avoid duplication.

---

## embedding_pipeline.py

Responsibilities:

- Generate embeddings using SentenceTransformer
- Train classifiers on embeddings
- Evaluate models using evaluation.py
- Save results

Output:

```
outputs/embeddings/
```

---

## evaluation.py

Responsibilities:

- Compute accuracy and Macro-F1
- Generate classification reports
- Save confusion matrices
- Save compact confusion matrices
- Manage organized output folders

Used by:

- tfidf_pipeline.py
- embedding_pipeline.py

---

## compare_models.py

Responsibilities:

- Run TF-IDF pipeline
- Run embedding pipeline
- Compare performance metrics
- Save comparison table

Output:

```
outputs/comparisons/model_comparison.csv
```

---

## clustering.py

Responsibilities:

- Generate embeddings
- Determine optimal cluster count using elbow method
- Perform KMeans clustering
- Select representative documents
- Perform second-level clustering
- Build topic tree structure
- Save topic tree output

Output:

```
outputs/clustering/
```

---

## openai_client.py

Responsibilities:

- Interface with OpenAI API
- Generate topic labels from representative documents
- Return concise topic labels

Used by:

- clustering.py

---

# Output Structure

```
outputs/
│
├── bow/
│   TF-IDF classification results
│
├── embeddings/
│   Embedding classification results
│
├── clustering/
│   Elbow plot and topic tree
│
└── comparisons/
    Model comparison results
```

---

# Design Principles

The architecture follows key software engineering principles:

Modularity  
Each module has a single responsibility.

Reusability  
Embedding and evaluation logic is centralized.

Separation of Concerns  
Data loading, modeling, evaluation, and clustering are independent.

Reproducibility  
All outputs are saved and reproducible via command-line execution.

Scalability  
Pipeline can easily be extended with additional models or datasets.

---

# End-to-End Pipeline Summary

Classification Pipeline:

```
data_loader → tfidf_pipeline → evaluation → outputs/bow
```

Embedding Pipeline:

```
data_loader → embeddings → embedding_pipeline → evaluation → outputs/embeddings
```

Clustering Pipeline:

```
data_loader → embeddings → clustering → openai_client → outputs/clustering
```

Comparison Pipeline:

```
tfidf_pipeline + embedding_pipeline → compare_models → outputs/comparisons
```

---

# Demo Integration

Demo notebook loads precomputed outputs from:

```
outputs/
```

This allows fast demonstration without rerunning full pipelines.