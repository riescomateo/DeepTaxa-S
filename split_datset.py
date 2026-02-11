import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*80)
print("✂️ LARGE DATASET SPLITTER")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================
# Input files
INPUT_CSV = 'data/new_sequences.csv'
INPUT_EMBEDDINGS = 'data/new_embeddings.npy'  # Optional

# Output directory
OUTPUT_DIR = 'data/chunks'

# Split configuration
CHUNK_SIZE = 10000  # Sequences per chunk
# Alternative: set NUM_CHUNKS instead
# NUM_CHUNKS = 10  # Divide into N equal parts

# Naming convention
PREFIX = 'chunk'  # chunk_001.csv, chunk_002.csv, etc.

print(f"\n⚙️ Configuration:")
print(f"   Input CSV: {INPUT_CSV}")
print(f"   Input Embeddings: {INPUT_EMBEDDINGS}")
print(f"   Chunk size: {CHUNK_SIZE:,} sequences")
print(f"   Output directory: {OUTPUT_DIR}")

# ============================================
# LOAD DATA
# ============================================
print(f"\n1️⃣ Loading data...")

try:
    # Load CSV
    df = pd.read_csv(INPUT_CSV)
    print(f"   ✅ CSV loaded: {len(df):,} sequences")
    
    # Load embeddings if they exist
    embeddings = None
    if os.path.exists(INPUT_EMBEDDINGS):
        embeddings = np.load(INPUT_EMBEDDINGS)
        print(f"   ✅ Embeddings loaded: {embeddings.shape}")
        
        # Validate alignment
        if len(embeddings) != len(df):
            print(f"   ❌ Error: Data misalignment")
            print(f"      CSV: {len(df)}, Embeddings: {len(embeddings)}")
            exit(1)
    else:
        print(f"   ⚠️ Embeddings not found (only CSV will be split)")
    
except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    exit(1)

# ============================================
# CALCULATE CHUNKS
# ============================================
print(f"\n2️⃣ Calculating split...")

total_sequences = len(df)
num_chunks = (total_sequences + CHUNK_SIZE - 1) // CHUNK_SIZE

# If using NUM_CHUNKS instead:
# num_chunks = NUM_CHUNKS
# CHUNK_SIZE = (total_sequences + NUM_CHUNKS - 1) // NUM_CHUNKS

print(f"   Total sequences: {total_sequences:,}")
print(f"   Chunk size: {CHUNK_SIZE:,}")
print(f"   Number of chunks: {num_chunks}")

# Calculate actual sizes
chunk_info = []
for i in range(num_chunks):
    start_idx = i * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, total_sequences)
    size = end_idx - start_idx
    chunk_info.append({
        'chunk_id': i + 1,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'size': size
    })

print(f"\n   📊 Chunk distribution:")
for info in chunk_info[:5]:  # Show first 5
    print(f"      Chunk {info['chunk_id']:03d}: {info['size']:,} sequences (indices {info['start_idx']}-{info['end_idx']-1})")
if len(chunk_info) > 5:
    print(f"      ... and {len(chunk_info)-5} more chunks")

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
print(f"\n3️⃣ Creating output directory...")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"   ✅ Directory created: {OUTPUT_DIR}")
except Exception as e:
    print(f"   ❌ Error creating directory: {e}")
    exit(1)

# ============================================
# SPLIT AND SAVE
# ============================================
print(f"\n4️⃣ Splitting and saving chunks...")

csv_files = []
npy_files = []

for info in chunk_info:
    chunk_id = info['chunk_id']
    start_idx = info['start_idx']
    end_idx = info['end_idx']
    
    # Generate filenames
    csv_filename = os.path.join(OUTPUT_DIR, f"{PREFIX}_{chunk_id:03d}.csv")
    
    try:
        # Split CSV
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_df.to_csv(csv_filename, index=False)
        csv_files.append(csv_filename)
        
        # Split embeddings if available
        if embeddings is not None:
            npy_filename = os.path.join(OUTPUT_DIR, f"{PREFIX}_{chunk_id:03d}.npy")
            chunk_embeddings = embeddings[start_idx:end_idx]
            np.save(npy_filename, chunk_embeddings)
            npy_files.append(npy_filename)
        
        # Progress indicator
        if chunk_id % 5 == 0 or chunk_id == num_chunks:
            print(f"   ✅ Chunk {chunk_id:03d}/{num_chunks} saved ({info['size']:,} sequences)")
    
    except Exception as e:
        print(f"   ❌ Error in chunk {chunk_id}: {e}")
        continue

# ============================================
# GENERATE MANIFEST
# ============================================
print(f"\n5️⃣ Generating manifest file...")

manifest_file = os.path.join(OUTPUT_DIR, 'manifest.txt')

manifest_lines = [
    "="*80,
    "📋 CHUNK MANIFEST",
    "="*80,
    f"\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"\n📊 GENERAL INFORMATION:",
    f"   • Original file: {INPUT_CSV}",
    f"   • Total sequences: {total_sequences:,}",
    f"   • Number of chunks: {num_chunks}",
    f"   • Chunk size: ~{CHUNK_SIZE:,} sequences",
    f"   • Output directory: {OUTPUT_DIR}",
    f"\n📁 GENERATED FILES:",
]

for i, (csv_file, info) in enumerate(zip(csv_files, chunk_info), 1):
    rel_csv = os.path.basename(csv_file)
    manifest_lines.append(f"\n   Chunk {i:03d}:")
    manifest_lines.append(f"      • CSV: {rel_csv} ({info['size']:,} sequences)")
    
    if embeddings is not None and i-1 < len(npy_files):
        rel_npy = os.path.basename(npy_files[i-1])
        manifest_lines.append(f"      • NPY: {rel_npy}")

manifest_lines.extend([
    f"\n💡 RECOMMENDED USAGE:",
    f"\n   To check duplicates per chunk:",
    f"   ```bash",
    f"   for i in {{001..{num_chunks:03d}}}; do",
    f"       python check_duplicates.py \\",
    f"           --input_csv {OUTPUT_DIR}/{PREFIX}_$i.csv \\",
    f"           --input_embeddings {OUTPUT_DIR}/{PREFIX}_$i.npy",
    f"   done",
    f"   ```",
    f"\n   Or manually:",
    f"   ```bash",
    f"   python check_duplicates.py \\",
    f"       --input_csv {OUTPUT_DIR}/{PREFIX}_001.csv \\",
    f"       --input_embeddings {OUTPUT_DIR}/{PREFIX}_001.npy",
    f"   ```",
    f"\n   To process all unique chunks:",
    f"   ```bash",
    f"   # After checking duplicates, insert unique chunks",
    f"   for i in {{001..{num_chunks:03d}}}; do",
    f"       python add_to_milvus.py \\",
    f"           --input_csv {OUTPUT_DIR}/{PREFIX}_$i_unique.csv \\",
    f"           --input_embeddings {OUTPUT_DIR}/{PREFIX}_$i_unique.npy",
    f"   done",
    f"   ```",
])

manifest_lines.append("\n" + "="*80)

manifest_text = '\n'.join(manifest_lines)

with open(manifest_file, 'w') as f:
    f.write(manifest_text)

print(f"   ✅ Manifest saved: {manifest_file}")

# ============================================
# GENERATE PROCESSING SCRIPT
# ============================================
print(f"\n6️⃣ Generating processing script...")

# Create a bash script to process all chunks
process_script = os.path.join(OUTPUT_DIR, 'process_all_chunks.sh')

script_lines = [
    "#!/bin/bash",
    "# Script to process all chunks",
    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "echo '================================'",
    "echo 'PROCESSING ALL CHUNKS'",
    "echo '================================'",
    "",
    f"CHUNK_DIR='{OUTPUT_DIR}'",
    f"PREFIX='{PREFIX}'",
    f"NUM_CHUNKS={num_chunks}",
    "",
    "# Check duplicates in each chunk",
    "echo ''",
    "echo '1️⃣ Checking duplicates...'",
    "for i in $(seq -f '%03g' 1 $NUM_CHUNKS); do",
    "    echo \"Processing chunk $i/$NUM_CHUNKS...\"",
    "    python check_duplicates.py \\",
    "        --input_csv \"$CHUNK_DIR/${PREFIX}_$i.csv\" \\",
    "        --input_embeddings \"$CHUNK_DIR/${PREFIX}_$i.npy\"",
    "    ",
    "    if [ $? -ne 0 ]; then",
    "        echo \"Error in chunk $i\"",
    "        exit 1",
    "    fi",
    "done",
    "",
    "echo ''",
    "echo '2️⃣ Inserting unique chunks into Milvus...'",
    "for i in $(seq -f '%03g' 1 $NUM_CHUNKS); do",
    "    UNIQUE_CSV=\"$CHUNK_DIR/${PREFIX}_${i}_unique.csv\"",
    "    UNIQUE_NPY=\"$CHUNK_DIR/${PREFIX}_${i}_unique.npy\"",
    "    ",
    "    if [ -f \"$UNIQUE_CSV\" ]; then",
    "        echo \"Inserting unique chunk $i...\"",
    "        python add_to_milvus.py \\",
    "            --input_csv \"$UNIQUE_CSV\" \\",
    "            --input_embeddings \"$UNIQUE_NPY\"",
    "    else",
    "        echo \"Chunk $i: no duplicates found, using original\"",
    "        python add_to_milvus.py \\",
    "            --input_csv \"$CHUNK_DIR/${PREFIX}_$i.csv\" \\",
    "            --input_embeddings \"$CHUNK_DIR/${PREFIX}_$i.npy\"",
    "    fi",
    "done",
    "",
    "echo ''",
    "echo '✅ Processing completed'",
]

with open(process_script, 'w') as f:
    f.write('\n'.join(script_lines))

# Make script executable
os.chmod(process_script, 0o755)

print(f"   ✅ Processing script saved: {process_script}")
print(f"   To run: bash {process_script}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*80)
print("✅ SPLIT COMPLETED")
print("="*80)

print(f"\n📊 Summary:")
print(f"   • Chunks created: {num_chunks}")
print(f"   • CSV files: {len(csv_files)}")
if embeddings is not None:
    print(f"   • NPY files: {len(npy_files)}")
print(f"   • Directory: {OUTPUT_DIR}/")

# Calculate total size
total_csv_size = sum(os.path.getsize(f) for f in csv_files) / 1e6
print(f"\n💾 Storage used:")
print(f"   • CSV total: {total_csv_size:.1f} MB")
if embeddings is not None:
    total_npy_size = sum(os.path.getsize(f) for f in npy_files) / 1e6
    print(f"   • NPY total: {total_npy_size:.1f} MB")
    print(f"   • Total: {total_csv_size + total_npy_size:.1f} MB")

print(f"\n📋 Helper files:")
print(f"   • {manifest_file}")
print(f"   • {process_script}")

print(f"\n💡 Next steps:")
print(f"   1. Review the manifest: cat {manifest_file}")
print(f"   2. Test a single chunk:")
print(f"      python check_duplicates.py \\")
print(f"          --input_csv {csv_files[0]} \\")
print(f"          --input_embeddings {npy_files[0] if npy_files else 'N/A'}")
print(f"   3. Process all chunks:")
print(f"      bash {process_script}")

print("\n" + "="*80)