import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import hashlib

print("="*80)
print("🔍 DUPLICATE SEQUENCE VERIFICATION IN DATABASE")
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

print(f"\n📋 Configuration:")
print(f"   Input CSV: {INPUT_CSV}")
print(f"   Input embeddings: {INPUT_EMBEDDINGS}")
print(f"   Database: {MILVUS_DB_PATH}")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   Method: {METHOD}")
if 'embedding' in METHOD:
    print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"      (0.9999 = almost identical, 0.95 = very similar)")
print(f"   Output: {OUTPUT_CSV}")

# ============================================
# LOAD DATA
# ============================================
print(f"\n1️⃣ Loading data...")

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
print(f"\n2️⃣ Connecting to Milvus...")

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
    Method 1: Check by complete taxonomy.
    Verifies if the same taxonomic classification already exists.
    """
    print(f"\n   🌳 Method: Full taxonomy verification")
    print(f"      - Detects: Same taxonomic classification (Kingdom→Species)")
    print(f"      - Use case: Avoid duplicate organisms")
    
    duplicates = []
    
    # Create taxonomy key for input
    df['taxonomy_key'] = df.apply(create_taxonomy_key, axis=1)
    
    print(f"   🔍 Checking {len(df):,} taxonomies...")
    
    for i in tqdm(range(0, len(df), batch_size), desc="      Checking"):
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
    Method 2: Check by embedding similarity.
    Finds sequences with very similar embeddings.
    High threshold (0.9999) = almost identical sequences.
    Lower threshold (0.95-0.98) = similar sequences.
    """
    print(f"\n   🧮 Method: Embedding similarity verification")
    print(f"      - Detects: Sequences with very similar embeddings")
    print(f"      - Threshold: {threshold}")
    if threshold >= 0.999:
        print(f"      - Interpretation: Almost identical sequences")
    elif threshold >= 0.95:
        print(f"      - Interpretation: Very similar sequences")
    else:
        print(f"      - Interpretation: Moderately similar sequences")
    
    duplicates = []
    
    print(f"   🔍 Searching for nearest neighbors for {len(df):,} sequences...")
    
    for i in tqdm(range(0, len(df), batch_size), desc="      Checking"):
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
    Method 3: Combined approach.
    First checks embedding similarity, then validates with taxonomy.
    """
    print(f"\n   🔬 Method: Combined (embedding + taxonomy)")
    print(f"      - Step 1: Find embedding similarity >= {threshold}")
    print(f"      - Step 2: Validate that taxonomy also matches")
    print(f"      - More robust: avoids false positives")
    
    # First get embedding similarity candidates
    embedding_dups = check_embedding_similarity_duplicates(df, embeddings, client, threshold, batch_size)
    
    print(f"\n   ✅ Candidates by embedding: {len(embedding_dups)}")
    
    if len(embedding_dups) == 0:
        return []
    
    print(f"   🔍 Validating taxonomy...")
    
    validated_duplicates = []
    
    for dup in tqdm(embedding_dups, desc="      Validating"):
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
    print(f"   ✅ Validated with taxonomy: {validated_count}/{len(validated_duplicates)}")
    
    return validated_duplicates

# ============================================
# RUN DUPLICATE CHECK
# ============================================
print(f"\n3️⃣ Running duplicate verification...")

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
print(f"\n4️⃣ Generating report...")

duplicate_indices = set([d['input_index'] for d in duplicates])
num_unique = len(df) - len(duplicate_indices)

if len(duplicates) == 0:
    print(f"\n   ✅ No duplicates found!")
    print(f"   All {len(df):,} sequences are new")
    
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
    print(f"\n   ⚠️ Found {len(duplicates)} potential duplicates")
    print(f"   📊 Affecting {len(duplicate_indices)} input sequences")
    
    # Create detailed report
    report_df = pd.DataFrame(duplicates)
    
    print(f"\n   📊 Summary:")
    print(f"      Total sequences checked: {len(df):,}")
    print(f"      Duplicates found: {len(duplicates):,}")
    print(f"      Unique sequences: {num_unique:,}")
    print(f"      Duplicate sequences: {len(duplicate_indices):,}")

# Save report
try:
    report_df.to_csv(OUTPUT_CSV, index=False)
    file_size = os.path.getsize(OUTPUT_CSV) / 1e3
    print(f"\n   ✅ Report saved: {OUTPUT_CSV}")
    print(f"   📦 Size: {file_size:.1f} KB")
except Exception as e:
    print(f"\n   ❌ Error saving report: {e}")

# ============================================
# DETAILED SUMMARY
# ============================================
if len(duplicates) > 0:
    print(f"\n" + "="*80)
    print("📋 DETECTED DUPLICATES - FIRST 10 EXAMPLES")
    print("="*80)
    
    for i, dup in enumerate(duplicates[:10]):
        print(f"\n{i+1}. Input: {dup['input_header']}")
        if 'input_genus' in dup:
            print(f"   Genus: {dup['input_genus']}")
        print(f"   Sequence: {dup['input_sequence']}")
        print(f"   ↓ DUPLICATES ↓")
        print(f"   DB ID: {dup['duplicate_id']}")
        if 'duplicate_genus' in dup:
            print(f"   Genus: {dup['duplicate_genus']}")
        print(f"   Sequence: {dup['duplicate_sequence']}")
        print(f"   Match: {dup['match_type']}")
        if 'similarity' in dup:
            print(f"   Similarity: {dup['similarity']:.6f} ({dup['similarity']*100:.4f}%)")
        if 'taxonomy_match' in dup:
            print(f"   Taxonomy match: {dup['taxonomy_match']}")
    
    if len(duplicates) > 10:
        print(f"\n   ... and {len(duplicates) - 10} more (see full CSV)")

print("\n" + "="*80)
print("✅ VERIFICATION COMPLETED")
print("="*80)
print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================
# ACTIONABLE RECOMMENDATIONS
# ============================================
if len(duplicates) > 0:
    unique_indices = sorted([i for i in range(len(df)) if i not in duplicate_indices])
    
    print(f"\n💡 NEXT STEPS:")
    print(f"\n   Option 1 - Insert only unique sequences:")
    print(f"   ┌─────────────────────────────────────────────")
    print(f"   │ Total to insert: {len(unique_indices):,} sequences")
    print(f"   │ Duplicates skipped: {len(duplicate_indices):,} sequences")
    print(f"   └─────────────────────────────────────────────")
    
    # Save list of unique indices
    unique_indices_file = OUTPUT_CSV.replace('.csv', '_unique_indices.txt')
    with open(unique_indices_file, 'w') as f:
        f.write('\n'.join(map(str, unique_indices)))
    print(f"\n   ✅ Unique indices saved: {unique_indices_file}")
    
    # Save filtered CSV and embeddings info
    filtered_csv = OUTPUT_CSV.replace('.csv', '_unique_only.csv')
    df_unique = df.iloc[unique_indices]
    df_unique.to_csv(filtered_csv, index=False)
    print(f"   ✅ Filtered CSV saved: {filtered_csv}")
    
    print(f"\n   To filter embeddings in Python:")
    print(f"   >>> unique_indices = np.loadtxt('{unique_indices_file}', dtype=int)")
    print(f"   >>> unique_embeddings = embeddings[unique_indices]")
    print(f"   >>> np.save('unique_embeddings.npy', unique_embeddings)")
    
    print(f"\n   Option 2 - Review duplicates manually:")
    print(f"   Open: {OUTPUT_CSV}")
    print(f"   Decide case by case whether they are true duplicates")

else:
    print(f"\n✅ ALL SEQUENCES ARE UNIQUE")
    print(f"   You can proceed to insert them into the database without issues.")