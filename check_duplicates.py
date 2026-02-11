import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import hashlib

print("="*80)
print("🔍 VERIFICACIÓN DE DUPLICADOS EN BASE DE DATOS")
print("="*80)

# ============================================
# ARGUMENT PARSER
# ============================================
parser = argparse.ArgumentParser(description='Check for duplicate sequences in Milvus database')

parser.add_argument('--input_csv', type=str, required=True,
                   help='Path to CSV with new sequences to check')
parser.add_argument('--input_embeddings', type=str, required=True,
                   help='Path to embeddings (.npy) corresponding to input_csv')


# Duplicate detection methods

parser.add_argument('--batch_size', type=int, default=100,
                   help='Batch size for processing')


args = parser.parse_args()

# ============================================
# CONFIGURATION FROM ARGUMENTS
# ============================================
INPUT_CSV = args.input_csv
INPUT_EMBEDDINGS = args.input_embeddings
MILVUS_DB_PATH = 'gpuhub-tmp/milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
OUTPUT_CSV = 'duplicates.csv'
METHOD = 'both'
SIMILARITY_THRESHOLD = 0.9999
BATCH_SIZE = args.batch_size
LIMIT = None

HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

# ============================================
# VALIDATION
# ============================================
if not os.path.exists(INPUT_CSV):
    print(f"❌ Error: CSV not found: {INPUT_CSV}")
    exit(1)

if not os.path.exists(INPUT_EMBEDDINGS):
    print(f"❌ Error: Embeddings not found: {INPUT_EMBEDDINGS}")
    exit(1)

print(f"\n📋 Configuración:")
print(f"   Input CSV: {INPUT_CSV}")
print(f"   Input embeddings: {INPUT_EMBEDDINGS}")
print(f"   Database: {MILVUS_DB_PATH}")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   Method: {METHOD}")
if 'embedding' in METHOD:
    print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"      (0.9999 = casi idénticas, 0.95 = muy similares)")
print(f"   Output: {OUTPUT_CSV}")

# ============================================
# LOAD DATA
# ============================================
print(f"\n1️⃣ Cargando datos...")

try:
    df = pd.read_csv(INPUT_CSV)
    print(f"   ✅ CSV loaded: {len(df):,} sequences")
    
    if LIMIT:
        df = df.head(LIMIT)
        print(f"   ⚠️ Limited to first {LIMIT:,} sequences")
    
    embeddings = np.load(INPUT_EMBEDDINGS)
    print(f"   ✅ Embeddings loaded: {embeddings.shape}")
    
    # Validate alignment
    if len(df) != len(embeddings):
        print(f"   ❌ Error: CSV-Embeddings mismatch")
        print(f"      CSV: {len(df)}, Embeddings: {len(embeddings)}")
        exit(1)
        
except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    exit(1)

# Ensure Header column exists
if 'Header' not in df.columns:
    df['Header'] = [f"new_seq_{i:06d}" for i in range(len(df))]
    print(f"   ℹ️ Generated Header IDs")

# ============================================
# CONNECT TO MILVUS
# ============================================
print(f"\n2️⃣ Conectando a Milvus...")

try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    if not client.has_collection(COLLECTION_NAME):
        print(f"   ❌ Collection '{COLLECTION_NAME}' not found")
        exit(1)
    
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"   ✅ Connected to '{COLLECTION_NAME}'")
    print(f"   📊 Database size: {stats['row_count']:,} sequences")
    
except Exception as e:
    print(f"   ❌ Error connecting to Milvus: {e}")
    exit(1)

# ============================================
# HELPER FUNCTIONS
# ============================================
def create_taxonomy_key(row):
    """Create unique key from full taxonomy"""
    taxonomy_parts = [str(row.get(level, 'Unknown')) for level in HIERARCHY]
    return '|'.join(taxonomy_parts)

def check_taxonomy_duplicates(df, client, batch_size=100):
    """
    Method 1: Check by complete taxonomy
    Verifies if same taxonomic classification exists.
    """
    print(f"\n   🌳 Método: Verificación por taxonomía completa")
    print(f"      - Detecta: Misma clasificación taxonómica (Reino→Especie)")
    print(f"      - Uso: Evitar organismos duplicados")
    
    duplicates = []
    
    # Create taxonomy key for input
    df['taxonomy_key'] = df.apply(create_taxonomy_key, axis=1)
    
    print(f"   🔍 Verificando {len(df):,} taxonomías...")
    
    for i in tqdm(range(0, len(df), batch_size), desc="      Verificando"):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            try:
                # Build filter for exact taxonomy match
                filter_parts = []
                for level in HIERARCHY:
                    value = str(row.get(level, 'Unknown'))
                    value = value.replace('"', '\\"')
                    filter_parts.append(f'{level} == "{value}"')
                
                filter_expr = ' and '.join(filter_parts)
                
                result = client.query(
                    collection_name=COLLECTION_NAME,
                    filter=filter_expr,
                    output_fields=["id", "sequence"] + HIERARCHY,
                    limit=1
                )
                
                if result:
                    duplicates.append({
                        'input_index': idx,
                        'input_header': row['Header'],
                        'input_sequence': row['Sequence'][:80] + '...' if len(row['Sequence']) > 80 else row['Sequence'],
                        'duplicate_id': result[0]['id'],
                        'duplicate_sequence': (result[0].get('sequence', '')[:80] + '...') if len(result[0].get('sequence', '')) > 80 else result[0].get('sequence', ''),
                        'match_type': 'taxonomy_exact',
                        'taxonomy': row['taxonomy_key']
                    })
                    
            except Exception as e:
                # If query fails, skip
                continue
    
    return duplicates

def check_embedding_similarity_duplicates(df, embeddings, client, threshold=0.9999, batch_size=100):
    """
    Method 2: Check by embedding similarity
    Finds sequences with very similar embeddings.
    High threshold (0.9999) = almost identical sequences
    Lower threshold (0.95-0.98) = similar sequences
    """
    print(f"\n   🧮 Método: Verificación por similitud de embeddings")
    print(f"      - Detecta: Secuencias con embeddings muy similares")
    print(f"      - Threshold: {threshold}")
    if threshold >= 0.999:
        print(f"      - Interpretación: Secuencias casi idénticas")
    elif threshold >= 0.95:
        print(f"      - Interpretación: Secuencias muy similares")
    else:
        print(f"      - Interpretación: Secuencias moderadamente similares")
    
    duplicates = []
    
    print(f"   🔍 Buscando vecinos muy cercanos para {len(df):,} secuencias...")
    
    for i in tqdm(range(0, len(df), batch_size), desc="      Verificando"):
        batch_df = df.iloc[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        # Search for nearest neighbors
        vectors = [emb.tolist() for emb in batch_embeddings]
        
        try:
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=vectors,
                limit=5,  # Get top 5 most similar
                output_fields=["id", "sequence"] + HIERARCHY,
                search_params={"metric_type": "COSINE"}
            )
            
            # Check each result
            for batch_idx, hits in enumerate(results):
                global_idx = i + batch_idx
                row = batch_df.iloc[batch_idx]
                
                for hit in hits:
                    distance = hit['distance']
                    
                    # For COSINE metric: similarity = 1 - distance
                    similarity = 1.0 - distance
                    
                    if similarity >= threshold:
                        duplicates.append({
                            'input_index': global_idx,
                            'input_header': row['Header'],
                            'input_sequence': row['Sequence'][:80] + '...' if len(row['Sequence']) > 80 else row['Sequence'],
                            'input_genus': row.get('Genus', 'Unknown'),
                            'duplicate_id': hit['entity']['id'],
                            'duplicate_sequence': (hit['entity'].get('sequence', '')[:80] + '...') if len(hit['entity'].get('sequence', '')) > 80 else hit['entity'].get('sequence', ''),
                            'duplicate_genus': hit['entity'].get('Genus', 'Unknown'),
                            'match_type': 'embedding_similarity',
                            'similarity': round(similarity, 6),
                            'distance': round(distance, 6)
                        })
                        break  # Only report first match per query
                        
        except Exception as e:
            print(f"\n      ⚠️ Error in batch {i}: {e}")
            continue
    
    return duplicates

def check_combined_duplicates(df, embeddings, client, threshold=0.9999, batch_size=100):
    """
    Method 3: Combined approach
    First checks embedding similarity, then validates with taxonomy.
    """
    print(f"\n   🔬 Método: Combinado (embedding + taxonomía)")
    print(f"      - Paso 1: Busca similitud de embedding >= {threshold}")
    print(f"      - Paso 2: Valida que taxonomía también coincida")
    print(f"      - Más robusto: evita falsos positivos")
    
    # First get embedding similarity candidates
    embedding_dups = check_embedding_similarity_duplicates(df, embeddings, client, threshold, batch_size)
    
    print(f"\n   ✅ Candidatos por embedding: {len(embedding_dups)}")
    
    if len(embedding_dups) == 0:
        return []
    
    print(f"   🔍 Validando taxonomía...")
    
    validated_duplicates = []
    
    for dup in tqdm(embedding_dups, desc="      Validando"):
        idx = dup['input_index']
        row = df.iloc[idx]
        
        # Check if taxonomy also matches
        try:
            filter_parts = []
            for level in HIERARCHY:
                value = str(row.get(level, 'Unknown'))
                value = value.replace('"', '\\"')
                filter_parts.append(f'{level} == "{value}"')
            
            filter_expr = ' and '.join(filter_parts) + f' and id == "{dup["duplicate_id"]}"'
            
            result = client.query(
                collection_name=COLLECTION_NAME,
                filter=filter_expr,
                output_fields=["id"],
                limit=1
            )
            
            if result:
                dup['match_type'] = 'combined_validated'
                dup['taxonomy_match'] = 'YES'
            else:
                dup['taxonomy_match'] = 'NO'
            
            validated_duplicates.append(dup)
            
        except Exception as e:
            dup['taxonomy_match'] = 'ERROR'
            validated_duplicates.append(dup)
    
    # Count validated matches
    validated_count = sum(1 for d in validated_duplicates if d.get('taxonomy_match') == 'YES')
    print(f"   ✅ Validados con taxonomía: {validated_count}/{len(validated_duplicates)}")
    
    return validated_duplicates

# ============================================
# RUN DUPLICATE CHECK
# ============================================
print(f"\n3️⃣ Ejecutando verificación de duplicados...")

duplicates = []

if METHOD == 'taxonomy':
    duplicates = check_taxonomy_duplicates(df, client, batch_size=BATCH_SIZE)
    
elif METHOD == 'embedding_similarity':
    duplicates = check_embedding_similarity_duplicates(df, embeddings, client, threshold=SIMILARITY_THRESHOLD, batch_size=BATCH_SIZE)
    
elif METHOD == 'combined':
    duplicates = check_combined_duplicates(df, embeddings, client, threshold=SIMILARITY_THRESHOLD, batch_size=BATCH_SIZE)

# ============================================
# GENERATE REPORT
# ============================================
print(f"\n4️⃣ Generando reporte...")

duplicate_indices = set([d['input_index'] for d in duplicates])
num_unique = len(df) - len(duplicate_indices)

if len(duplicates) == 0:
    print(f"\n   ✅ ¡No se encontraron duplicados!")
    print(f"   Todas las {len(df):,} secuencias son nuevas")
    
    # Create summary report
    summary = {
        'total_checked': len(df),
        'duplicates_found': 0,
        'unique_sequences': len(df),
        'method': METHOD,
        'threshold': SIMILARITY_THRESHOLD if 'embedding' in METHOD else 'N/A'
    }
    
    report_df = pd.DataFrame([summary])
    
else:
    print(f"\n   ⚠️ Encontrados {len(duplicates)} duplicados potenciales")
    print(f"   📊 Afectan a {len(duplicate_indices)} secuencias de entrada")
    
    # Create detailed report
    report_df = pd.DataFrame(duplicates)
    
    print(f"\n   📊 Resumen:")
    print(f"      Total secuencias verificadas: {len(df):,}")
    print(f"      Duplicados encontrados: {len(duplicates):,}")
    print(f"      Secuencias únicas: {num_unique:,}")
    print(f"      Secuencias duplicadas: {len(duplicate_indices):,}")

# Save report
try:
    report_df.to_csv(OUTPUT_CSV, index=False)
    file_size = os.path.getsize(OUTPUT_CSV) / 1e3
    print(f"\n   ✅ Reporte guardado: {OUTPUT_CSV}")
    print(f"   📦 Size: {file_size:.1f} KB")
except Exception as e:
    print(f"\n   ❌ Error saving report: {e}")

# ============================================
# DETAILED SUMMARY
# ============================================
if len(duplicates) > 0:
    print(f"\n" + "="*80)
    print("📋 DUPLICADOS DETECTADOS - PRIMEROS 10 EJEMPLOS")
    print("="*80)
    
    for i, dup in enumerate(duplicates[:10]):
        print(f"\n{i+1}. Input: {dup['input_header']}")
        if 'input_genus' in dup:
            print(f"   Genus: {dup['input_genus']}")
        print(f"   Secuencia: {dup['input_sequence']}")
        print(f"   ↓ DUPLICA A ↓")
        print(f"   DB ID: {dup['duplicate_id']}")
        if 'duplicate_genus' in dup:
            print(f"   Genus: {dup['duplicate_genus']}")
        print(f"   Secuencia: {dup['duplicate_sequence']}")
        print(f"   Match: {dup['match_type']}")
        if 'similarity' in dup:
            print(f"   Similitud: {dup['similarity']:.6f} ({dup['similarity']*100:.4f}%)")
        if 'taxonomy_match' in dup:
            print(f"   Taxonomía coincide: {dup['taxonomy_match']}")
    
    if len(duplicates) > 10:
        print(f"\n   ... y {len(duplicates) - 10} más (ver CSV completo)")

print("\n" + "="*80)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*80)
print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================
# ACTIONABLE RECOMMENDATIONS
# ============================================
if len(duplicates) > 0:
    unique_indices = sorted([i for i in range(len(df)) if i not in duplicate_indices])
    
    print(f"\n💡 PASOS SIGUIENTES:")
    print(f"\n   Opción 1 - Insertar solo secuencias únicas:")
    print(f"   ┌─────────────────────────────────────────────")
    print(f"   │ Total a insertar: {len(unique_indices):,} secuencias")
    print(f"   │ Duplicadas omitidas: {len(duplicate_indices):,} secuencias")
    print(f"   └─────────────────────────────────────────────")
    
    # Save list of unique indices
    unique_indices_file = OUTPUT_CSV.replace('.csv', '_unique_indices.txt')
    with open(unique_indices_file, 'w') as f:
        f.write('\n'.join(map(str, unique_indices)))
    print(f"\n   ✅ Índices únicos guardados: {unique_indices_file}")
    
    # Save filtered CSV and embeddings info
    filtered_csv = OUTPUT_CSV.replace('.csv', '_unique_only.csv')
    df_unique = df.iloc[unique_indices]
    df_unique.to_csv(filtered_csv, index=False)
    print(f"   ✅ CSV filtrado guardado: {filtered_csv}")
    
    print(f"\n   Para filtrar embeddings en Python:")
    print(f"   >>> unique_indices = np.loadtxt('{unique_indices_file}', dtype=int)")
    print(f"   >>> unique_embeddings = embeddings[unique_indices]")
    print(f"   >>> np.save('unique_embeddings.npy', unique_embeddings)")
    
    print(f"\n   Opción 2 - Revisar duplicados manualmente:")
    print(f"   Abre: {OUTPUT_CSV}")
    print(f"   Decide caso por caso si son realmente duplicados")

else:
    print(f"\n✅ TODAS LAS SECUENCIAS SON ÚNICAS")
    print(f"   Puedes proceder a insertarlas en la base de datos sin problemas.")