# DeepTaxa-S

> A high-throughput taxonomic classifier for COI protein sequences using DNABERT-S embeddings and vector similarity search.

---

## Overview

DeepTaxa-S is a taxonomic classification pipeline optimized for Genus-level identification of biological sequences. It leverages **DNABERT-S** to encode COI (Cytochrome c oxidase I) protein sequences into high-dimensional embeddings, and uses **Milvus** as a scalable vector database for fast nearest-neighbor search. The system supports confidence-filtered predictions, dynamic database updates, and batch processing — making it suitable for both research and production-scale workflows.

---

## Key Features

- **Vector Similarity Search** — Sequences are encoded as 768-dimensional embeddings and matched against a Milvus vector store using cosine similarity, enabling sub-second classification at scale. The system uses **Milvus in local file mode** (`MilvusClient` over `milvus.db`), requiring no server infrastructure or external deployment.
- **Confidence Scoring** — Each prediction is assigned a confidence score derived from Shannon entropy, neighbor agreement, or a combined metric. Results below a configurable threshold (e.g., `--min_confidence 0.8`) can be filtered out automatically.
- **Flexible Input Modes** — Classify using pre-computed embeddings (fast) or compute them on-the-fly directly from raw sequences.
- **Dynamic Database Updates** — New sequences can be added to the vector store incrementally, with built-in duplicate detection before insertion.
- **Genus-Size-Aware Evaluation** — Benchmarking is stratified by training sample size (rare / medium / common), providing a realistic view of model performance across data sparsity regimes.

---

## Benchmarks

Evaluated on a held-out 20% test set using Recall@K and Mean Reciprocal Rank (MRR).

| Category | Train Samples | Recall@1 | Recall@5 | MRR |
|----------|--------------|----------|----------|-----|
| Common   | 21+          | **90.17%** | **92.70%** | **0.9132** |
| Medium   | 6–20         | **70.60%** | **74.66%** | **0.7248** |
| Rare     | 1–5          | **46.07%** | **49.95%** | **0.4787** |

> Confidence filtering (`entropy` method, threshold `0.8`) was applied during evaluation runs. Metrics reflect performance on high-certainty predictions only.
>
> **Scope note:** While DeepTaxa-S processes and stores the complete taxonomic hierarchy (Kingdom → Species), the system is primarily optimized and validated at the Genus level.

This decision is rooted in a fundamental biological challenge: the DNA barcoding gap. While the COI (Cytochrome c oxidase I) gene is the gold standard for molecular identification, it often lacks sufficient phylogenetic resolution to distinguish between closely related species. In many taxonomic groups, intraspecific variation can overlap with interspecific divergence, leading to "noise" that complicates species-level assignments.

By focusing on the Genus level, DeepTaxa-S provides a high-confidence classification tool that remains robust against these evolutionary ambiguities, offering a more reliable balance for ecological and bioinformatic research.

---

## Tech Stack

| Component     | Library / Tool                                      |
|---------------|-----------------------------------------------------|
| Language      | Python 3.9+                                         |
| Deep Learning | PyTorch, Hugging Face Transformers                  |
| Encoder Model | [DNABERT-S](https://github.com/MAGICS-LAB/DNABERT_S) (`zhihan1996/DNABERT-S`) |
| Vector Store  | [Milvus](https://milvus.io/) (local file mode via MilvusClient) |
| Data Handling | Pandas, NumPy                                       |
| Evaluation    | scikit-learn, SciPy                                 |

---

## Project Structure

```
.
├── check_duplicates.py         # Detect duplicate sequences before DB insertion
├── classify_sequences.py       # Main classification script
├── embed_sequences_s.py        # Generate and store embeddings in Milvus
├── evaluate.py                 # Recall@K and MRR evaluation by genus size
├── evaluate_confidence.py      # Evaluation with confidence threshold filtering
├── generate_new_embeddings.py  # Generate .npy embeddings for new sequences
├── requirements.txt
├── split_dataset.py            # Split large datasets into chunks for processing
│
├── data/
│   ├── dataset.csv
│   ├── final_dataset.csv
│   └── prepare_data.py
│
├── Flash attention patches/
│   └── flash_patch.py          # ⚠️ Required patch — must be run before model usage
│
└── gpuhub-tmp/
    └── milvus_db/
        └── milvus.db
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DeepTaxa-S.git
cd DeepTaxa-S
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. ⚠️ Apply the Flash Attention patch (Required)

There is a known compatibility issue between DNABERT-S and Triton/Flash Attention. The patch modifies library cache files in-place — but those files are only created the first time the model is loaded. Follow these steps in order:

**3a. Trigger model initialization** by running the classifier once (it will fail, which is expected):

```bash
python classify_sequences.py --compute_embeddings --input_csv data/dataset.csv --output /tmp/test.csv
```

This initializes the Triton/Flash Attention library cache files on disk. Once you see the error, proceed to the next step.

**3b. Apply the patch:**

```bash
python "Flash attention patches/flash_patch.py"
```

The patch will now find and modify the cached library files. After this, all subsequent runs will work correctly. Skipping step 3a means the patch has no files to modify and will have no effect.

> **Hardware note:** A CUDA-capable GPU is strongly recommended for embedding generation. The Triton patch is specifically designed to optimize performance in CUDA environments — on CPU-only setups the patch is not required but inference will be significantly slower.

---

## Usage

### Classify sequences

Using pre-computed embeddings (recommended for speed):

```bash
python classify_sequences.py \
    --input_embeddings data/test_embeddings.npy \
    --input_csv data/test_sequences.csv \
    --output results/predictions.csv
```

Computing embeddings on-the-fly:

```bash
python classify_sequences.py \
    --compute_embeddings \
    --input_csv data/test_sequences.csv \
    --output results/predictions.csv \
    --batch_size 32
```

With a minimum confidence filter:

```bash
python classify_sequences.py \
    --input_embeddings data/test_embeddings.npy \
    --input_csv data/test_sequences.csv \
    --output results/predictions.csv \
    --min_confidence 0.8
```

---

### Add new sequences to the database

**Step 1 — Generate embeddings for new sequences:**

```bash
python generate_new_embeddings.py \
    --input data/new_sequences.csv \
    --output data/new_embeddings.npy
```

**Step 2 — Check for duplicates before insertion:**

```bash
python check_duplicates.py \
    --input_csv data/new_sequences.csv \
    --input_embeddings data/new_embeddings.npy
```

**Step 3 — If the dataset is large, split into chunks first:**

```bash
# Edit INPUT_CSV and CHUNK_SIZE in split_dataset.py, then run:
python split_dataset.py
# A manifest and processing shell script will be generated automatically.
```

---

### Run evaluation

Standard Recall@K evaluation stratified by genus size:

```bash
python evaluate.py
```

Evaluation with confidence threshold filtering:

```bash
python evaluate_confidence.py
```

---

## Data Format

Input CSVs must contain at minimum a `Sequence` column with raw nucleotide/protein sequences. A `Header` column is optional — if absent, sequential IDs (`seq_000001`, `seq_000002`, ...) are generated automatically.

For classification and evaluation, taxonomy columns (`Kingdom`, `Phylum`, `Class`, `Order`, `Family`, `Genus`, `Species`) are expected when ground-truth labels are required.

---

## License

MIT License. See `LICENSE` for details.