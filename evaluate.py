import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from tqdm import tqdm
import os
from collections import defaultdict
import json
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
MILVUS_DB_PATH = 'gpuhub-tmp/milvusmilvus.db'
COLLECTION_NAME = 'dna_sequences_s'
TEST_CACHE_PATH = 'data/test_dataset_cache.csv'
TEST_EMBEDDINGS_PATH = 'data/test_embeddings_s.npy'
ORIGINAL_DATASET_PATH = 'data/final_dataset.csv'
RESULTS_DIR = 'results'

# Evaluation parameters
TOP_K_VALUES = [1, 3, 5, 10, 20]
BATCH_SIZE = 100  # Process queries in batches

# Genus size ranges
RANGES = {
    'rare': (1, 5),                  # 1-5 samples
    'medium': (6, 20),               # 6-20 samples
    'common': (21, float('inf'))     # 21+ samples
}

# ============================================
# SETUP
# ============================================
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("="*80)
print("🧬 RETRIEVAL EVALUATION BY GENUS SIZE")
print("="*80)

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n1️⃣ Loading data...")

# Load test set
if not os.path.exists(TEST_CACHE_PATH):
    print(f"❌ Error: {TEST_CACHE_PATH} not found")
    exit(1)
test_df = pd.read_csv(TEST_CACHE_PATH)
print(f"   ✅ Test set: {len(test_df):,} sequences")

# Load test embeddings
if not os.path.exists(TEST_EMBEDDINGS_PATH):
    print(f"❌ Error: {TEST_EMBEDDINGS_PATH} not found")
    print("   Run first: python generate_test_embeddings.py")
    exit(1)
test_embeddings = np.load(TEST_EMBEDDINGS_PATH)
print(f"   ✅ Test embeddings: {test_embeddings.shape}")

# Verify alignment
if len(test_df) != len(test_embeddings):
    print(f"❌ Error: Data misalignment")
    print(f"   Test DF: {len(test_df)}, Embeddings: {len(test_embeddings)}")
    exit(1)

# Load original dataset to count genus samples
print(f"   📊 Analyzing original dataset...")
original_df = pd.read_csv(ORIGINAL_DATASET_PATH)
print(f"   ✅ Original dataset: {len(original_df):,} sequences")

# ============================================
# STEP 2: ANALYZE GENUS DISTRIBUTION
# ============================================
print("\n2️⃣ Analyzing genus distribution...")

# Count samples per genus in TRAIN set (Milvus collection)
# Filter only train split from original
train_df = original_df[original_df['Header'].isin(
    set(original_df['Header']) - set(test_df['Header'])
)]

genus_counts = train_df['Genus'].value_counts().to_dict()
print(f"   ✅ {len(genus_counts)} unique genera in train set")

# Categorize test samples by genus size
test_df['genus_sample_count'] = test_df['Genus'].map(genus_counts).fillna(0).astype(int)

category_assignment = {}
for genus, count in genus_counts.items():
    if RANGES['rare'][0] <= count <= RANGES['rare'][1]:
        category_assignment[genus] = 'rare'
    elif RANGES['medium'][0] <= count <= RANGES['medium'][1]:
        category_assignment[genus] = 'medium'
    elif count >= RANGES['common'][0]:
        category_assignment[genus] = 'common'

test_df['category'] = test_df['Genus'].map(category_assignment)

# Filter out test samples without valid category (genus not in train)
test_df_valid = test_df[test_df['category'].notna()].copy()
removed = len(test_df) - len(test_df_valid)
if removed > 0:
    print(f"   ⚠️ Removed {removed} samples with no genus in train set")

print("\n   📊 Test sample distribution by category:")
for cat_name, (min_val, max_val) in RANGES.items():
    cat_samples = (test_df_valid['category'] == cat_name).sum()
    cat_genera = test_df_valid[test_df_valid['category'] == cat_name]['Genus'].nunique()
    range_str = f"{min_val}-{int(max_val) if max_val != float('inf') else '∞'}"
    print(f"   • {cat_name.upper():8} ({range_str:8} samples): {cat_samples:6,} queries ({cat_genera:4} genera)")

# ============================================
# STEP 3: INITIALIZE MILVUS
# ============================================
print("\n3️⃣ Connecting to Milvus...")
try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
    if not client.has_collection(COLLECTION_NAME):
        print(f"❌ Error: Collection '{COLLECTION_NAME}' does not exist")
        exit(1)
    
    # Get collection stats
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"   ✅ Collection '{COLLECTION_NAME}' loaded")
    print(f"   📊 Entities in Milvus: {stats['row_count']:,}")
except Exception as e:
    print(f"❌ Error connecting to Milvus: {e}")
    exit(1)

# ============================================
# STEP 4: EVALUATION FUNCTIONS
# ============================================
def calculate_metrics(results_per_query, k_values):
    """Calculate Recall@K and MRR for given results"""
    metrics = {k: {'recall': [], 'mrr': []} for k in k_values}
    
    for query_result in results_per_query:
        true_genus = query_result['true_genus']
        retrieved_genera = query_result['retrieved_genera']
        
        for k in k_values:
            top_k_genera = retrieved_genera[:k]
            
            # Recall@K
            recall = 1.0 if true_genus in top_k_genera else 0.0
            metrics[k]['recall'].append(recall)
            
            # MRR (only for first occurrence)
            if k == max(k_values):  # Calculate MRR once with max K
                try:
                    rank = retrieved_genera.index(true_genus) + 1
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0
                metrics[k]['mrr'].append(mrr)
    
    # Calculate averages
    summary = {}
    for k in k_values:
        summary[k] = {
            'recall@k': np.mean(metrics[k]['recall']) * 100,
            'count': len(metrics[k]['recall'])
        }
        if k == max(k_values):
            summary[k]['mrr'] = np.mean(metrics[k]['mrr'])
    
    return summary

def query_milvus_batch(embeddings, top_k=20):
    """Query Milvus with a batch of embeddings"""
    vectors = [emb.tolist() for emb in embeddings]
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=vectors,
        limit=top_k,
        output_fields=["Genus"],
        search_params={"metric_type": "COSINE"}
    )
    
    return results

# ============================================
# STEP 5: RUN EVALUATION
# ============================================
print("\n4️⃣ Running evaluation...")
print(f"   Top-K values: {TOP_K_VALUES}")
print(f"   Batch size: {BATCH_SIZE}")

# Store results per category
all_results = {cat: [] for cat in RANGES.keys()}

# Process each category
for category in ['rare', 'medium', 'common']:
    print(f"\n   🔍 Evaluating category: {category.upper()}")
    
    # Get test samples for this category
    cat_df = test_df_valid[test_df_valid['category'] == category].copy()
    cat_indices = cat_df.index.tolist()
    cat_embeddings = test_embeddings[cat_indices]
    
    if len(cat_df) == 0:
        print(f"      ⚠️ No samples for this category")
        continue
    
    print(f"      Processing {len(cat_df):,} queries...")
    
    # Process in batches
    for i in tqdm(range(0, len(cat_df), BATCH_SIZE),
                  desc=f"      {category}",
                  leave=False):
        batch_df = cat_df.iloc[i:i+BATCH_SIZE]
        batch_embeddings = cat_embeddings[i:i+BATCH_SIZE]
        
        # Query Milvus
        search_results = query_milvus_batch(batch_embeddings, top_k=max(TOP_K_VALUES))
        
        # Process results
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            true_genus = row['Genus']
            retrieved_genera = [hit['entity']['Genus'] for hit in search_results[idx]]
            
            all_results[category].append({
                'true_genus': true_genus,
                'retrieved_genera': retrieved_genera,
                'query_header': row['Header']
            })

# ============================================
# STEP 6: CALCULATE METRICS
# ============================================
print("\n5️⃣ Calculating metrics...")

final_metrics = {}
for category in ['rare', 'medium', 'common']:
    if len(all_results[category]) > 0:
        final_metrics[category] = calculate_metrics(all_results[category], TOP_K_VALUES)

# ============================================
# STEP 7: DISPLAY RESULTS
# ============================================
print("\n" + "="*80)
print("📊 EVALUATION RESULTS")
print("="*80)

for category in ['rare', 'medium', 'common']:
    cat_range = RANGES[category]
    range_str = f"{cat_range[0]}-{int(cat_range[1]) if cat_range[1] != float('inf') else '∞'}"
    
    print(f"\n🔸 {category.upper()} ({range_str} samples in train)")
    print("-" * 80)
    
    if category not in final_metrics:
        print("   No data for this category")
        continue
    
    metrics = final_metrics[category]
    num_queries = metrics[TOP_K_VALUES[0]]['count']
    print(f"   Queries evaluated: {num_queries:,}")
    print()
    
    # Table header
    print(f"   {'Metric':<15} {'Value':>10}")
    print(f"   {'-'*15} {'-'*10}")
    
    # Recall@K
    for k in TOP_K_VALUES:
        recall = metrics[k]['recall@k']
        print(f"   Recall@{k:<8} {recall:>9.2f}%")
    
    # MRR
    mrr = metrics[max(TOP_K_VALUES)]['mrr']
    print(f"   {'MRR':<15} {mrr:>10.4f}")

# ============================================
# STEP 8: SAVE RESULTS
# ============================================
print("\n6️⃣ Saving results...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(RESULTS_DIR, f"evaluation_by_genus_size_{timestamp}.json")

results_output = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'test_samples_total': len(test_df_valid),
        'milvus_collection': COLLECTION_NAME,
        'top_k_values': TOP_K_VALUES,
        'ranges': {k: list(v) for k, v in RANGES.items()}
    },
    'category_distribution': {
        cat: {
            'num_queries': len(all_results[cat]),
            'num_genera': test_df_valid[test_df_valid['category'] == cat]['Genus'].nunique()
        }
        for cat in ['rare', 'medium', 'common']
    },
    'metrics': final_metrics
}

with open(results_file, 'w') as f:
    json.dump(results_output, f, indent=2)

print(f"   ✅ Results saved to: {results_file}")

# Save detailed CSV
csv_file = os.path.join(RESULTS_DIR, f"detailed_results_{timestamp}.csv")
detailed_rows = []

for category in ['rare', 'medium', 'common']:
    for result in all_results[category]:
        detailed_rows.append({
            'category': category,
            'true_genus': result['true_genus'],
            'query_header': result['query_header'],
            'rank': result['retrieved_genera'].index(result['true_genus']) + 1
                    if result['true_genus'] in result['retrieved_genera'] else -1,
            'top5_retrieved': ','.join(result['retrieved_genera'][:5])
        })

detailed_df = pd.DataFrame(detailed_rows)
detailed_df.to_csv(csv_file, index=False)
print(f"   ✅ Detailed results saved to: {csv_file}")

# ============================================
# STEP 9: SUMMARY STATISTICS
# ============================================
print("\n" + "="*80)
print("📈 COMPARATIVE SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Category': ['RARE (1-5)', 'MEDIUM (6-20)', 'COMMON (21+)'],
    'Queries': [len(all_results['rare']), len(all_results['medium']), len(all_results['common'])],
    'Recall@1': [
        final_metrics['rare'][1]['recall@k'] if 'rare' in final_metrics else 0,
        final_metrics['medium'][1]['recall@k'] if 'medium' in final_metrics else 0,
        final_metrics['common'][1]['recall@k'] if 'common' in final_metrics else 0
    ],
    'Recall@5': [
        final_metrics['rare'][5]['recall@k'] if 'rare' in final_metrics else 0,
        final_metrics['medium'][5]['recall@k'] if 'medium' in final_metrics else 0,
        final_metrics['common'][5]['recall@k'] if 'common' in final_metrics else 0
    ],
    'MRR': [
        final_metrics['rare'][max(TOP_K_VALUES)]['mrr'] if 'rare' in final_metrics else 0,
        final_metrics['medium'][max(TOP_K_VALUES)]['mrr'] if 'medium' in final_metrics else 0,
        final_metrics['common'][max(TOP_K_VALUES)]['mrr'] if 'common' in final_metrics else 0
    ]
})

print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("✅ EVALUATION COMPLETED")
print("="*80)
print(f"\nResults saved to: {RESULTS_DIR}/")
print(f"  • JSON: {os.path.basename(results_file)}")
print(f"  • CSV:  {os.path.basename(csv_file)}")
print("="*80)