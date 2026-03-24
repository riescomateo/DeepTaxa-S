# DeepTaxa-S

High-throughput taxonomic classifier for biological sequences using DNABERT-S embeddings and vector similarity search.

---

## 🚀 Summary

Built a scalable biological sequence classifier processing **1M+ sequences**, using transformer-based embeddings and vector search to perform fast genus-level taxonomic classification.

---

## 📊 Key Results

Evaluated on a held-out 20% test set using Recall@K and Mean Reciprocal Rank (MRR):

| Category | Train Samples | Recall@1 | Recall@5 | MRR |
|----------|-------------|---------|---------|------|
| Common   | 21+         | 90.17%  | 92.70%  | 0.913 |
| Medium   | 6–20        | 70.60%  | 74.66%  | 0.725 |
| Rare     | 1–5         | 46.07%  | 49.95%  | 0.479 |

- Confidence filtering applied using entropy-based thresholding (≥ 0.8)
- Optimized for **genus-level classification** due to biological constraints in COI sequences

---

## 🧠 Overview

DeepTaxa-S is a taxonomic classification pipeline that:

- Encodes COI sequences into **768-dimensional embeddings** using DNABERT-S
- Stores embeddings in a **Milvus vector database**
- Performs classification via **nearest-neighbor search (cosine similarity)**
- Applies **confidence filtering** to improve prediction reliability

The system is designed for **high-throughput classification and scalable data processing**.

---

## ⚙️ System Architecture


Sequences → DNABERT-S → Embeddings (768-dim) → Milvus → Nearest Neighbor Search → Prediction + Confidence Score


---

## 🔑 Key Features

- **Vector Search at Scale** — Fast similarity search using Milvus (local mode, no server required)
- **Confidence Filtering** — Entropy-based scoring to filter low-confidence predictions
- **Flexible Input** — Supports raw sequences or precomputed embeddings
- **Incremental Updates** — Add new sequences with duplicate detection
- **Stratified Evaluation** — Performance measured across rare/medium/common classes

---

## 🛠 Tech Stack

- Python, PyTorch, Hugging Face Transformers  
- DNABERT-S (Transformer encoder)  
- Milvus (vector database)  
- Pandas, NumPy  
- scikit-learn, SciPy  

---

## ▶️ Usage (Quick Start)

### Classify sequences (fast mode)

```bash
python classify_sequences.py \
    --input_embeddings data/test_embeddings.npy \
    --input_csv data/test_sequences.csv \
    --output results/predictions.csv
📌 Notes
Designed for genus-level classification due to limitations in COI sequence resolution
GPU recommended for embedding generation (CUDA)

## 📄 License

MIT License