import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from tqdm import tqdm
import os
from collections import defaultdict, Counter
from scipy.stats import entropy
import json
from datetime import datetime

print("="*80)
print("🧬 EVALUACIÓN CON FILTRO DE CONFIANZA")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================
# Milvus connection
USE_REMOTE = False  # Set to True for remote server, False for local file



# Local file configuration
MILVUS_DB_PATH = 'gpuhub-tmp/milvus_db/milvus.db'

COLLECTION_NAME = 'dna_sequences_s'
TEST_CACHE_PATH = 'data/test_dataset_cache.csv'
TEST_EMBEDDINGS_PATH = 'data/test_embeddings_s.npy'
ORIGINAL_DATASET_PATH = 'data/final_dataset.csv'
RESULTS_DIR = 'results'

# Evaluation parameters
TOP_K_VALUES = [1, 3, 5, 10, 20]
BATCH_SIZE = 100  # Process queries in batches

# CONFIDENCE FILTERING
MIN_CONFIDENCE = 0.8  # Solo incluir queries con confianza >= 0.8
CONFIDENCE_METHOD = 'entropy'  # 'entropy', 'agreement', 'combined'

# Genus size ranges
RANGES = {
    'rare': (1, 5),      # 1-5 samples
    'medium': (6, 20),   # 6-20 samples
    'common': (21, float('inf'))  # 21+ samples
}

# Taxonomy hierarchy
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

# ============================================
# SETUP
# ============================================
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print(f"\n⚙️ Configuración:")
print(f"   Confianza mínima: {MIN_CONFIDENCE}")
print(f"   Método de confianza: {CONFIDENCE_METHOD}")
print(f"   Top-K valores: {TOP_K_VALUES}")

# ============================================
# CONFIDENCE CALCULATION FUNCTIONS
# ============================================
def calculate_entropy_confidence(labels):
    """
    Calculate confidence based on Shannon entropy.
    Lower entropy = higher confidence
    Returns: confidence score 0-1
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
    Calculate confidence as fraction of neighbors that agree.
    Returns: confidence score 0-1
    """
    if not labels:
        return 0.0
    
    agreement_count = sum(1 for label in labels if label == prediction)
    return agreement_count / len(labels)

def calculate_combined_confidence(entropy_conf, agreement_conf):
    """
    Combine entropy and agreement confidence.
    """
    return (entropy_conf + agreement_conf) / 2.0

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n1️⃣ Cargando datos...")

# Load test set
if not os.path.exists(TEST_CACHE_PATH):
    print(f"❌ Error: {TEST_CACHE_PATH} no encontrado")
    exit(1)
test_df = pd.read_csv(TEST_CACHE_PATH)
print(f"   ✅ Test set: {len(test_df):,} secuencias")

# Load test embeddings
if not os.path.exists(TEST_EMBEDDINGS_PATH):
    print(f"❌ Error: {TEST_EMBEDDINGS_PATH} no encontrado")
    exit(1)
test_embeddings = np.load(TEST_EMBEDDINGS_PATH)
print(f"   ✅ Test embeddings: {test_embeddings.shape}")

# Verify alignment
if len(test_df) != len(test_embeddings):
    print(f"❌ Error: Desalineación de datos")
    print(f"   Test DF: {len(test_df)}, Embeddings: {len(test_embeddings)}")
    exit(1)

# Load original dataset
print(f"   📊 Analizando dataset original...")
original_df = pd.read_csv(ORIGINAL_DATASET_PATH)
print(f"   ✅ Dataset original: {len(original_df):,} secuencias")

# ============================================
# STEP 2: ANALYZE GENUS DISTRIBUTION
# ============================================
print("\n2️⃣ Analizando distribución de géneros...")

# Count samples per genus in TRAIN set
train_df = original_df[original_df['Header'].isin(
    set(original_df['Header']) - set(test_df['Header'])
)]

genus_counts = train_df['Genus'].value_counts().to_dict()
print(f"   ✅ {len(genus_counts)} géneros únicos en train set")

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

# Filter out test samples without valid category
test_df_valid = test_df[test_df['category'].notna()].copy()
removed = len(test_df) - len(test_df_valid)
if removed > 0:
    print(f"   ⚠️ Removidas {removed} muestras sin género en train set")

print("\n   📊 Distribución de muestras de test por categoría:")
for cat_name, (min_val, max_val) in RANGES.items():
    cat_samples = (test_df_valid['category'] == cat_name).sum()
    cat_genera = test_df_valid[test_df_valid['category'] == cat_name]['Genus'].nunique()
    range_str = f"{min_val}-{int(max_val) if max_val != float('inf') else '∞'}"
    print(f"   • {cat_name.upper():8} ({range_str:8} samples): {cat_samples:6,} queries ({cat_genera:4} géneros)")

# ============================================
# STEP 3: INITIALIZE MILVUS
# ============================================
print("\n3️⃣ Conectando a Milvus...")
try:
    if not USE_REMOTE:
        print(f"   💾 Conectando a base de datos local: {MILVUS_DB_PATH}")
        client = MilvusClient(uri=MILVUS_DB_PATH)
    
    print(f"   ✅ Conexión establecida")
    
    if not client.has_collection(COLLECTION_NAME):
        print(f"❌ Error: Colección '{COLLECTION_NAME}' no existe")
        collections = client.list_collections()
        for col in collections:
            print(f"   • {col}")
        exit(1)
    
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"   ✅ Colección '{COLLECTION_NAME}' cargada")
    print(f"   📊 Entidades en Milvus: {stats['row_count']:,}")
    
except Exception as e:
    print(f"❌ Error conectando a Milvus: {e}")
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
            if k == max(k_values):
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
    """Query Milvus with batch of embeddings"""
    vectors = [emb.tolist() for emb in embeddings]
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=vectors,
        limit=top_k,
        output_fields=["Genus"],
        search_params={"metric_type": "COSINE"}
    )
    
    return results

def calculate_confidence_for_query(hits, true_genus):
    """
    Calculate confidence score for a query based on retrieved neighbors.
    Returns confidence score 0-1.
    """
    neighbor_genera = [hit['entity']['Genus'] for hit in hits]
    
    # Prediction (nearest neighbor)
    predicted_genus = neighbor_genera[0] if neighbor_genera else 'Unknown'
    
    # Calculate confidence based on method
    if CONFIDENCE_METHOD == 'entropy':
        confidence = calculate_entropy_confidence(neighbor_genera)
    elif CONFIDENCE_METHOD == 'agreement':
        confidence = calculate_agreement_confidence(neighbor_genera, predicted_genus)
    elif CONFIDENCE_METHOD == 'combined':
        entropy_conf = calculate_entropy_confidence(neighbor_genera)
        agreement_conf = calculate_agreement_confidence(neighbor_genera, predicted_genus)
        confidence = calculate_combined_confidence(entropy_conf, agreement_conf)
    else:
        confidence = 1.0  # Default
    
    return confidence

# ============================================
# STEP 5: RUN EVALUATION WITH CONFIDENCE FILTERING
# ============================================
print("\n4️⃣ Ejecutando evaluación con filtro de confianza...")
print(f"   Top-K valores: {TOP_K_VALUES}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Confianza mínima: {MIN_CONFIDENCE}")

# Store results per category
all_results = {cat: [] for cat in RANGES.keys()}
confidence_stats = {cat: {'total': 0, 'passed': 0, 'failed': 0} for cat in RANGES.keys()}

# Process each category
for category in ['rare', 'medium', 'common']:
    print(f"\n   🔍 Evaluando categoría: {category.upper()}")
    
    # Get test samples for this category
    cat_df = test_df_valid[test_df_valid['category'] == category].copy()
    cat_indices = cat_df.index.tolist()
    cat_embeddings = test_embeddings[cat_indices]
    
    if len(cat_df) == 0:
        print(f"      ⚠️ Sin muestras para esta categoría")
        continue
    
    print(f"      Procesando {len(cat_df):,} queries...")
    
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
            hits = search_results[idx]
            retrieved_genera = [hit['entity']['Genus'] for hit in hits]
            
            # Calculate confidence
            confidence = calculate_confidence_for_query(hits, true_genus)
            
            confidence_stats[category]['total'] += 1
            
            # Only include if confidence >= MIN_CONFIDENCE
            if confidence >= MIN_CONFIDENCE:
                all_results[category].append({
                    'true_genus': true_genus,
                    'retrieved_genera': retrieved_genera,
                    'query_header': row['Header'],
                    'confidence': confidence
                })
                confidence_stats[category]['passed'] += 1
            else:
                confidence_stats[category]['failed'] += 1

# ============================================
# STEP 6: CALCULATE METRICS (only on high-confidence queries)
# ============================================
print("\n5️⃣ Calculando métricas...")

final_metrics = {}
for category in ['rare', 'medium', 'common']:
    if len(all_results[category]) > 0:
        final_metrics[category] = calculate_metrics(all_results[category], TOP_K_VALUES)

# ============================================
# STEP 7: DISPLAY RESULTS
# ============================================
print("\n" + "="*80)
print("📊 RESULTADOS DE EVALUACIÓN (CON FILTRO DE CONFIANZA)")
print("="*80)

# Show confidence filtering stats first
print("\n🔍 Estadísticas de filtrado por confianza:")
for category in ['rare', 'medium', 'common']:
    stats = confidence_stats[category]
    if stats['total'] > 0:
        pass_rate = (stats['passed'] / stats['total']) * 100
        print(f"\n   {category.upper()}:")
        print(f"      Total queries: {stats['total']:,}")
        print(f"      Confianza >= {MIN_CONFIDENCE}: {stats['passed']:,} ({pass_rate:.1f}%)")
        print(f"      Filtradas: {stats['failed']:,} ({100-pass_rate:.1f}%)")

print("\n" + "-"*80)
print("📈 Métricas de retrieval (solo queries con alta confianza):")
print("-"*80)

for category in ['rare', 'medium', 'common']:
    cat_range = RANGES[category]
    range_str = f"{cat_range[0]}-{int(cat_range[1]) if cat_range[1] != float('inf') else '∞'}"
    
    print(f"\n🔸 {category.upper()} ({range_str} samples en train)")
    print("-" * 80)
    
    if category not in final_metrics:
        print("   Sin datos para esta categoría")
        continue
    
    metrics = final_metrics[category]
    num_queries = metrics[TOP_K_VALUES[0]]['count']
    print(f"   Queries evaluadas (alta confianza): {num_queries:,}")
    print()
    
    # Table header
    print(f"   {'Métrica':<15} {'Valor':>10}")
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
print("\n6️⃣ Guardando resultados...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(RESULTS_DIR, f"evaluation_confidence_filtered_{timestamp}.json")

results_output = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'min_confidence': MIN_CONFIDENCE,
        'confidence_method': CONFIDENCE_METHOD,
        'test_samples_total': len(test_df_valid),
        'milvus_collection': COLLECTION_NAME,
        'top_k_values': TOP_K_VALUES,
        'ranges': {k: list(v) for k, v in RANGES.items()}
    },
    'confidence_filtering': confidence_stats,
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

print(f"   ✅ Resultados guardados en: {results_file}")

# Save detailed CSV
csv_file = os.path.join(RESULTS_DIR, f"detailed_results_confidence_{timestamp}.csv")
detailed_rows = []

for category in ['rare', 'medium', 'common']:
    for result in all_results[category]:
        detailed_rows.append({
            'category': category,
            'true_genus': result['true_genus'],
            'query_header': result['query_header'],
            'confidence': result['confidence'],
            'rank': result['retrieved_genera'].index(result['true_genus']) + 1 
                    if result['true_genus'] in result['retrieved_genera'] else -1,
            'top5_retrieved': ','.join(result['retrieved_genera'][:5])
        })

detailed_df = pd.DataFrame(detailed_rows)
detailed_df.to_csv(csv_file, index=False)
print(f"   ✅ Resultados detallados guardados en: {csv_file}")

# ============================================
# STEP 9: SUMMARY STATISTICS
# ============================================
print("\n" + "="*80)
print("📈 RESUMEN COMPARATIVO (ALTA CONFIANZA)")
print("="*80)

comparison_data = []
for cat in ['rare', 'medium', 'common']:
    cat_range = RANGES[cat]
    range_str = f"{cat_range[0]}-{int(cat_range[1]) if cat_range[1] != float('inf') else '∞'}"
    
    row = {
        'Category': f'{cat.upper()} ({range_str})',
        'Total': confidence_stats[cat]['total'],
        'High_Conf': confidence_stats[cat]['passed'],
        'Filtered': confidence_stats[cat]['failed'],
        'Pass_Rate': f"{(confidence_stats[cat]['passed']/confidence_stats[cat]['total']*100):.1f}%" if confidence_stats[cat]['total'] > 0 else "N/A"
    }
    
    if cat in final_metrics and 1 in final_metrics[cat]:
        row['Recall@1'] = f"{final_metrics[cat][1]['recall@k']:.2f}%"
        row['Recall@5'] = f"{final_metrics[cat][5]['recall@k']:.2f}%"
        row['MRR'] = f"{final_metrics[cat][max(TOP_K_VALUES)]['mrr']:.4f}"
    else:
        row['Recall@1'] = "N/A"
        row['Recall@5'] = "N/A"
        row['MRR'] = "N/A"
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("✅ EVALUACIÓN COMPLETADA")
print("="*80)
print(f"\n📌 Nota: Métricas calculadas SOLO sobre queries con confianza >= {MIN_CONFIDENCE}")
print(f"   Esto representa el rendimiento en escenarios de alta certeza.")
print(f"\nResultados guardados en: {RESULTS_DIR}/")
print(f"  • JSON: {os.path.basename(results_file)}")
print(f"  • CSV:  {os.path.basename(csv_file)}")
print("="*80)