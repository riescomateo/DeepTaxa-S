import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import traceback
import json
from datetime import datetime

# Configuration
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MILVUS_DB_PATH = 'gpuhub-tmp/milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
BATCH_SIZE = 64
INSERT_BATCH_SIZE = 2048
MAX_LEN = 768
TEST_CACHE_PATH = 'data/test_dataset_cache.csv'

# ============================================
# CHECKPOINT CONFIGURATION
# ============================================
CHECKPOINT_FILE = 'embedding_checkpoint.json'
CHECKPOINT_INTERVAL = 500  # Save every 500 batches

# Initialize Device (Multi-GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check GPU availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU available")
    num_gpus = 0

# Initialize Model and Tokenizer
print("\nLoading model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    
    if num_gpus > 1:
        print(f"\n🚀 Enabling DataParallel on {num_gpus} GPUs...")
        model = torch.nn.DataParallel(model)
        print(f"   Batch will be split: {BATCH_SIZE} total → {BATCH_SIZE//num_gpus} per GPU")
    
    model.eval()
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize Milvus
print("\nInitializing Milvus...")
try:
    if not os.path.exists(os.path.dirname(MILVUS_DB_PATH)):
        os.makedirs(os.path.dirname(MILVUS_DB_PATH))
    
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=768,
            id_type="string",
            max_length=512,
            metric_type="COSINE",
            auto_id=False
        )
        print(f"Collection '{COLLECTION_NAME}' created (no initial index).")
except Exception as e:
    print(f"Error initializing Milvus: {e}")
    exit(1)

# Function to generate embeddings
def get_embeddings(sequences):
    sequences = [s.replace('\n', '').strip() for s in sequences]
    inputs = tokenizer(sequences, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    return embeddings

# Process CSV in chunks
print(f"\nProcessing {CSV_PATH}...")

if not os.path.exists(CSV_PATH):
    print(f"File not found: {CSV_PATH}")
    exit(1)

print("Sampling 100% of the dataset...")
chunks = []
read_chunk_size = 50000

try:
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Reading and sampling"):
        sampled_chunk = chunk.sample(frac=1.0, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
print(f"Sampled {len(df_sample)} sequences.")

# Train/test split
print("Performing Train/Test Split (80/20)...")
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
train_df['split'] = 'train'
test_df['split'] = 'test'

# ============================================
# SAVE TEST SET
# ============================================
if not os.path.exists(TEST_CACHE_PATH):
    print(f"\n💾 Saving test set to {TEST_CACHE_PATH}...")
    test_df.to_csv(TEST_CACHE_PATH, index=False)
    print(f"   ✅ Test set saved: {len(test_df):,} sequences")
    print(f"   📁 Location: {TEST_CACHE_PATH}")
else:
    print(f"\n✅ Test set already exists at {TEST_CACHE_PATH}")
    # Verify it matches
    existing_test = pd.read_csv(TEST_CACHE_PATH)
    if len(existing_test) != len(test_df):
        print(f"   ⚠️ WARNING: Size mismatch!")
        print(f"   Existing: {len(existing_test):,} | Current: {len(test_df):,}")
        print(f"   Consider deleting {TEST_CACHE_PATH} and re-running")

processing_df = train_df
print(f"\nProcessing training set only ({len(processing_df)} sequences). Test set skipped.")

# ============================================
# LOAD CHECKPOINT
# ============================================
start_batch = 0
if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        
        start_batch = checkpoint.get('last_batch', 0)
        prev_sequences = checkpoint.get('sequences_processed', 0)
        
        print(f"\n{'='*60}")
        print(f"🔄 CHECKPOINT FOUND")
        print(f"{'='*60}")
        print(f"Last completed batch: {start_batch:,}")
        print(f"Sequences already processed: {prev_sequences:,}")
        print(f"Previous progress: {(start_batch * BATCH_SIZE / len(processing_df)) * 100:.1f}%")
        print(f"Resuming from batch {start_batch + 1}...")
        print(f"{'='*60}\n")
        
        # Adjust start_batch to continue from next batch
        start_batch = start_batch + 1
        
    except Exception as e:
        print(f"⚠️ Error reading checkpoint: {e}")
        print("Starting from the beginning...")
        start_batch = 0
else:
    print("\n🆕 No checkpoint found. Starting from the beginning.\n")

# ============================================
# Process in batches
# ============================================
total_rows = len(processing_df)
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"{'='*60}")
print(f"🚀 STARTING EMBEDDING GENERATION")
print(f"{'='*60}")
print(f"Total sequences: {total_rows:,}")
print(f"Batch size: {BATCH_SIZE} (total)")
if num_gpus > 1:
    print(f"Per GPU: {BATCH_SIZE//num_gpus} (batch split automatically)")
print(f"Total batches: {num_batches:,}")
print(f"Starting from batch: {start_batch:,}")
print(f"Remaining batches: {num_batches - start_batch:,}")
print(f"{'='*60}\n")

data_buffer = []
batch_count = 0

for batch_idx in tqdm(range(start_batch, num_batches),
                     desc="Embedding batches",
                     initial=start_batch,
                     total=num_batches):
    
    i = batch_idx * BATCH_SIZE
    batch_df = processing_df.iloc[i : i + BATCH_SIZE]
    
    try:
        sequences = batch_df['Sequence'].astype(str).tolist()
        headers = batch_df['Header'].astype(str).tolist()
        
        metadatas = []
        for _, row in batch_df.iterrows():
            meta = {
                'Kingdom': str(row['Kingdom']) if pd.notna(row['Kingdom']) else "Unknown",
                'Phylum': str(row['Phylum']) if pd.notna(row['Phylum']) else "Unknown",
                'Class': str(row['Class']) if pd.notna(row['Class']) else "Unknown",
                'Order': str(row['Order']) if pd.notna(row['Order']) else "Unknown",
                'Family': str(row['Family']) if pd.notna(row['Family']) else "Unknown",
                'Genus': str(row['Genus']) if pd.notna(row['Genus']) else "Unknown",
                'Species': str(row['Species']) if pd.notna(row['Species']) else "Unknown",
                'split': row['split']
            }
            metadatas.append(meta)
            
        embeddings = get_embeddings(sequences)
        
        for j in range(len(sequences)):
            data_buffer.append({
                "id": headers[j],
                "vector": embeddings[j].tolist(),
                "sequence": sequences[j],
                **metadatas[j]
            })
            
        if len(data_buffer) >= INSERT_BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, data=data_buffer)
            data_buffer = []
        
        batch_count += 1
        
        # ============================================
        # SAVE CHECKPOINT
        # ============================================
        if batch_count % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                'last_batch': batch_idx,
                'sequences_processed': (batch_idx + 1) * BATCH_SIZE,
                'total_sequences': total_rows,
                'progress_percent': ((batch_idx + 1) / num_batches) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            tqdm.write(f"💾 Checkpoint saved: batch {batch_idx:,} " +
                      f"({checkpoint_data['progress_percent']:.1f}% complete)")
        
        # Clear GPU cache periodically
        if batch_count % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        # ============================================
        # SAVE ON INTERRUPT
        # ============================================
        print(f"\n\n⚠️ Process interrupted by user")
        print(f"Saving checkpoint at batch {batch_idx}...")
        
        checkpoint_data = {
            'last_batch': batch_idx - 1,  # Last completed batch
            'sequences_processed': batch_idx * BATCH_SIZE,
            'total_sequences': total_rows,
            'progress_percent': (batch_idx / num_batches) * 100,
            'timestamp': datetime.now().isoformat(),
            'interrupted': True
        }
        
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"✅ Checkpoint saved successfully")
        print(f"To resume, run the script again")
        exit(0)
        
    except Exception as e:
        # ============================================
        # SAVE ON ERROR
        # ============================================
        print(f"\n❌ Error processing batch {batch_idx}: {e}")
        traceback.print_exc()
        
        print(f"Saving checkpoint...")
        checkpoint_data = {
            'last_batch': batch_idx - 1,
            'sequences_processed': batch_idx * BATCH_SIZE,
            'total_sequences': total_rows,
            'progress_percent': (batch_idx / num_batches) * 100,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }
        
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"✅ Checkpoint saved. You can retry from this point.")
        break

# Insert remaining data
if data_buffer:
    print(f"\n💾 Inserting {len(data_buffer)} remaining elements...")
    try:
        client.upsert(collection_name=COLLECTION_NAME, data=data_buffer)
    except Exception as e:
        print(f"Error inserting final batch: {e}")

# ============================================
# CLEANUP CHECKPOINT
# ============================================
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("\n✅ Checkpoint deleted - process completed successfully")

# Create index
print("\nGenerating HNSW index...")
try:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
        params={}
    )
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    print("✅ Index created successfully.")
except Exception as e:
    print(f"⚠️ Note when creating index: {e}")

# Summary
print("\n" + "="*60)
print("✅ PROCESS COMPLETED!")
print("="*60)
print(f"Embeddings stored at: {MILVUS_DB_PATH}")
print(f"Collection: {COLLECTION_NAME}")

if torch.cuda.is_available():
    for i in range(num_gpus):
        mem_allocated = torch.cuda.memory_allocated(i) / 1e9
        mem_cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i} memory used: {mem_allocated:.2f}GB / {mem_cached:.2f}GB cached")

print("="*60)

if torch.cuda.is_available():
    torch.cuda.empty_cache()