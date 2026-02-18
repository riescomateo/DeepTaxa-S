"""
🧬 DNA SEQUENCE CLASSIFICATION WITH CONFIDENCE SCORES
Optimized version for Kaggle Notebooks

Usage modes:
1. With pre-computed embeddings (FAST)
2. Without embeddings (computes on-the-fly)
3. With minimum confidence filter
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import argparse
import os
from collections import Counter
from scipy.stats import entropy
from datetime import datetime


print("="*80)
print("🧬 DNA SEQUENCE CLASSIFICATION WITH CONFIDENCE SCORES")
print("="*80)

# ============================================
# ARGUMENT PARSER
# ============================================
parser = argparse.ArgumentParser(description='Classify DNA sequences with confidence scores')

# Input options
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--input_embeddings', type=str,
                        help='Path to pre-computed embeddings (.npy file)')
input_group.add_argument('--compute_embeddings', action='store_true',
                        help='Compute embeddings on-the-fly (requires --model)')

parser.add_argument('--input_csv', type=str, required=True,
                   help='Path to CSV with sequences to classify')
parser.add_argument('--output', type=str, required=True,
                   help='Path to save classification results (CSV)')

# Model configuration (only needed if computing embeddings)
parser.add_argument('--batch_size', type=int, default=32,
                   help='Batch size for embedding generation')

# Optional filters
parser.add_argument('--min_confidence', type=float, default=0.0,
                   help='Minimum confidence threshold (0-1) to include in output')

args = parser.parse_args()


# ========== CONFIGURATION VARIABLES (from args) ==========
USE_PRECOMPUTED_EMBEDDINGS = True
INPUT_CSV = args.input_csv
OUTPUT_CSV = args.output
INPUT_EMBEDDINGS = args.input_embeddings
COMPUTE_EMBEDDINGS = args.compute_embeddings
MAX_LEN = 768
BATCH_SIZE = args.batch_size
MIN_CONFIDENCE = args.min_confidence

# ========== COMMON CONFIGURATION ==========

# Model
MODEL_NAME = 'zhihan1996/DNABERT-S'

# Milvus database
MILVUS_DB_PATH = 'gpuhub-tmp/milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'

# Classification parameters
K_NEIGHBORS = 10       # Number of neighbors for confidence calculation
K_PREDICT = 1          # Neighbors for prediction (typically 1)
CONFIDENCE_METHOD = 'entropy'  # 'entropy', 'agreement', 'distance', 'combined'

# Taxonomy hierarchy
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

# ============================================
# CONFIGURATION VALIDATION
# ============================================

# Validate operation mode
if not USE_PRECOMPUTED_EMBEDDINGS and not COMPUTE_EMBEDDINGS:
    print("❌ Error: Must enable USE_PRECOMPUTED_EMBEDDINGS or COMPUTE_EMBEDDINGS")
    raise ValueError("Invalid configuration")

if COMPUTE_EMBEDDINGS and not MODEL_NAME:
    print("❌ Error: MODEL_NAME required when COMPUTE_EMBEDDINGS=True")
    raise ValueError("MODEL_NAME not specified")

if USE_PRECOMPUTED_EMBEDDINGS and not INPUT_EMBEDDINGS:
    print("❌ Error: INPUT_EMBEDDINGS required when USE_PRECOMPUTED_EMBEDDINGS=True")
    raise ValueError("INPUT_EMBEDDINGS not specified")

# Display configuration
print(f"\n📋 Configuration:")
print(f"   Mode: {'Pre-computed embeddings' if USE_PRECOMPUTED_EMBEDDINGS else 'Compute on-the-fly'}")
print(f"   Input CSV: {INPUT_CSV}")
if USE_PRECOMPUTED_EMBEDDINGS:
    print(f"   Input embeddings: {INPUT_EMBEDDINGS}")
if COMPUTE_EMBEDDINGS:
    print(f"   Model: {MODEL_NAME}")
    print(f"   Batch size: {BATCH_SIZE}")
print(f"   Output: {OUTPUT_CSV}")
print(f"   K neighbors: {K_NEIGHBORS}")
print(f"   Confidence method: {CONFIDENCE_METHOD}")
if MIN_CONFIDENCE > 0:
    print(f"   Min confidence: {MIN_CONFIDENCE}")

# ============================================
# INITIALIZE DEVICE AND MODEL
# ============================================
device = None
model = None
tokenizer = None

if COMPUTE_EMBEDDINGS:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n📦 Loading model: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        raise

# ============================================
# CONNECT TO MILVUS
# ============================================
print(f"\n🔌 Connecting to Milvus...")
try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    if not client.has_collection(COLLECTION_NAME):
        print(f"   ❌ Collection '{COLLECTION_NAME}' not found")
        available = client.list_collections()
        print(f"   Available collections: {available}")
        raise ValueError(f"Collection {COLLECTION_NAME} not found")
    
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"   ✅ Connected to '{COLLECTION_NAME}'")
    print(f"   📊 Database size: {stats['row_count']:,} sequences")
    
except Exception as e:
    print(f"   ❌ Error connecting to Milvus: {e}")
    raise

# ============================================
# LOAD INPUT DATA
# ============================================
print(f"\n📂 Loading input data...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"   ✅ CSV loaded: {len(df):,} sequences")
    
    # Validate required columns
    if 'Sequence' not in df.columns:
        print(f"   ❌ Error: CSV must have 'Sequence' column")
        print(f"   Available columns: {list(df.columns)}")
        raise ValueError("Missing 'Sequence' column")
    
    # Add Header if missing
    if 'Header' not in df.columns:
        df['Header'] = [f"seq_{i:06d}" for i in range(len(df))]
        print(f"   ℹ️ Generated Header IDs")
    
except Exception as e:
    print(f"   ❌ Error loading CSV: {e}")
    raise

# ============================================
# LOAD OR COMPUTE EMBEDDINGS
# ============================================
if USE_PRECOMPUTED_EMBEDDINGS:
    print(f"\n📥 Loading pre-computed embeddings...")
    try:
        embeddings = np.load(INPUT_EMBEDDINGS)
        print(f"   ✅ Embeddings loaded: {embeddings.shape}")
        
        # Validate alignment
        if len(embeddings) != len(df):
            print(f"   ❌ Error: Embeddings-CSV mismatch")
            print(f"      Embeddings: {len(embeddings)}, CSV: {len(df)}")
            raise ValueError("Embeddings and CSV length mismatch")
            
    except Exception as e:
        print(f"   ❌ Error loading embeddings: {e}")
        raise
        
else:  # COMPUTE_EMBEDDINGS
    print(f"\n🧮 Computing embeddings...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max length: {MAX_LEN}")
    
    def get_embeddings(sequences, batch_size):
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="   Generating"):
            batch = sequences[i:i+batch_size]
            batch = [s.replace('\n', '').strip() for s in batch]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=MAX_LEN)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
            # Clear cache periodically
            if i % (batch_size * 50) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings)
    
    try:
        sequences = df['Sequence'].astype(str).tolist()
        embeddings = get_embeddings(sequences, BATCH_SIZE)
        print(f"   ✅ Embeddings computed: {embeddings.shape}")
        
    except Exception as e:
        print(f"   ❌ Error computing embeddings: {e}")
        raise

# ============================================
# CONFIDENCE FUNCTIONS
# ============================================
def calculate_entropy_confidence(labels):
    """
    Confidence based on Shannon entropy.
    Lower entropy = higher confidence.
    Returns: score 0-1 (1 = perfect agreement)
    """
    if not labels or len(set(labels)) == 1:
        return 1.0
    
    counts = Counter(labels)
    probs = [count / len(labels) for count in counts.values()]
    ent = entropy(probs, base=2)
    
    max_entropy = np.log2(len(labels))
    normalized_entropy = ent / max_entropy if max_entropy > 0 else 0
    
    confidence = 1.0 - normalized_entropy
    return confidence

def calculate_agreement_confidence(labels, prediction):
    """
    Confidence as the fraction of neighbors that agree.
    Returns: score 0-1
    """
    if not labels:
        return 0.0
    
    agreement_count = sum(1 for label in labels if label == prediction)
    return agreement_count / len(labels)

def calculate_distance_confidence(distance):
    """
    Confidence based on distance to nearest neighbor.
    Lower distance = higher confidence.
    For COSINE: distance typically in [0, 2]
    Returns: score 0-1
    """
    confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
    return confidence

def calculate_combined_confidence(entropy_conf, agreement_conf, distance_conf):
    """
    Combines multiple confidence metrics.
    Weighted average.
    """
    w_entropy = 0.4
    w_agreement = 0.4
    w_distance = 0.2
    
    combined = (w_entropy * entropy_conf +
                w_agreement * agreement_conf +
                w_distance * distance_conf)
    
    return combined

# ============================================
# CLASSIFICATION FUNCTION
# ============================================
def classify_sequence(embedding, client):
    """
    Classifies a sequence and calculates confidence scores.
    """
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[embedding.tolist()],
            limit=K_NEIGHBORS,
            output_fields=HIERARCHY
        )
    except Exception as e:
        print(f"   ⚠️ Search error: {e}")
        return None
    
    if not results or not results[0]:
        return None
    
    hits = results[0]
    prediction_hits = hits[:K_PREDICT]
    
    # Majority vote prediction
    prediction = {}
    for level in HIERARCHY:
        labels = [hit['entity'].get(level, 'Unknown') for hit in prediction_hits]
        most_common = Counter(labels).most_common(1)[0][0]
        prediction[level] = most_common
    
    nearest_distance = hits[0]['distance']
    
    # Confidence calculation per taxonomic level
    confidences = {}
    entropies = {}
    agreements = {}
    
    for level in HIERARCHY:
        neighbor_labels = [hit['entity'].get(level, 'Unknown') for hit in hits]
        
        entropy_conf = calculate_entropy_confidence(neighbor_labels)
        entropies[level] = entropy_conf
        
        agreement_conf = calculate_agreement_confidence(neighbor_labels, prediction[level])
        agreements[level] = agreement_conf
        
        distance_conf = calculate_distance_confidence(nearest_distance)
        
        # Select confidence method
        if CONFIDENCE_METHOD == 'entropy':
            confidences[level] = entropy_conf
        elif CONFIDENCE_METHOD == 'agreement':
            confidences[level] = agreement_conf
        elif CONFIDENCE_METHOD == 'distance':
            confidences[level] = distance_conf
        elif CONFIDENCE_METHOD == 'combined':
            confidences[level] = calculate_combined_confidence(
                entropy_conf, agreement_conf, distance_conf
            )
    
    return {
        'prediction': prediction,
        'confidence': confidences,
        'distance': nearest_distance,
        'entropy': entropies,
        'agreement': agreements,
        'num_neighbors': len(hits)
    }

# ============================================
# PROCESS ALL SEQUENCES
# ============================================
print(f"\n🔬 Classifying sequences...")
print(f"   K neighbors: {K_NEIGHBORS}")
print(f"   K predict: {K_PREDICT}")
print(f"   Confidence method: {CONFIDENCE_METHOD}")

results_data = []
failed_count = 0

for i in tqdm(range(len(df)), desc="   Classifying"):
    row = df.iloc[i]
    embedding = embeddings[i]
    
    result = classify_sequence(embedding, client)
    
    if result:
        record = {
            'Header': row['Header'],
            'Sequence': row['Sequence'],
            'Distance_Nearest': result['distance'],
            'Num_Neighbors': result['num_neighbors']
        }
        
        # Add predictions and confidence per level
        for level in HIERARCHY:
            record[f'Pred_{level}'] = result['prediction'][level]
            record[f'Conf_{level}'] = result['confidence'][level]
            record[f'Entropy_{level}'] = result['entropy'][level]
            record[f'Agreement_{level}'] = result['agreement'][level]
        
        # Filter by minimum confidence if configured
        if MIN_CONFIDENCE > 0:
            max_conf = max(result['confidence'].values())
            if max_conf >= MIN_CONFIDENCE:
                results_data.append(record)
        else:
            results_data.append(record)
    else:
        failed_count += 1

# ============================================
# SAVE RESULTS
# ============================================
print(f"\n💾 Saving results...")
try:
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    file_size = os.path.getsize(OUTPUT_CSV) / 1e6
    
    print(f"   ✅ Results saved: {OUTPUT_CSV}")
    print(f"   📊 Records: {len(results_df):,}")
    print(f"   📦 Size: {file_size:.2f} MB")
    
    if failed_count > 0:
        print(f"   ⚠️ Failed: {failed_count}")
    
except Exception as e:
    print(f"   ❌ Error saving results: {e}")
    raise

# ============================================
# SUMMARY STATISTICS
# ============================================
print(f"\n📈 Summary Statistics:")
print(f"   Total sequences: {len(df):,}")
print(f"   Successfully classified: {len(results_df):,}")
print(f"   Failed: {failed_count}")

if MIN_CONFIDENCE > 0:
    filtered_out = len(df) - failed_count - len(results_df)
    print(f"   Filtered by confidence: {filtered_out}")

if len(results_df) > 0:
    print(f"\n   Confidence distribution (Genus level):")
    genus_conf = results_df['Conf_Genus']
    print(f"      Mean: {genus_conf.mean():.3f}")
    print(f"      Median: {genus_conf.median():.3f}")
    print(f"      Min: {genus_conf.min():.3f}")
    print(f"      Max: {genus_conf.max():.3f}")
    
    print(f"\n   Distance distribution:")
    distances = results_df['Distance_Nearest']
    print(f"      Mean: {distances.mean():.4f}")
    print(f"      Median: {distances.median():.4f}")
    print(f"      Min: {distances.min():.4f}")
    print(f"      Max: {distances.max():.4f}")
    
    # Show high and low confidence examples
    print(f"\n   📊 Top 5 highest confidence (Genus):")
    top5 = results_df.nlargest(5, 'Conf_Genus')[['Header', 'Pred_Genus', 'Conf_Genus', 'Distance_Nearest']]
    print(top5.to_string(index=False))
    
    if len(results_df) >= 5:
        print(f"\n   ⚠️ Top 5 lowest confidence (Genus):")
        bottom5 = results_df.nsmallest(5, 'Conf_Genus')[['Header', 'Pred_Genus', 'Conf_Genus', 'Distance_Nearest']]
        print(bottom5.to_string(index=False))

print("\n" + "="*80)
print("✅ CLASSIFICATION COMPLETED")
print("="*80)
print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Clear GPU memory if used
if torch.cuda.is_available() and COMPUTE_EMBEDDINGS:
    torch.cuda.empty_cache()
    print("\n🧹 GPU memory cleared")