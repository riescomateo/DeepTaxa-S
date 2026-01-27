import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
import numpy as np
import json
import traceback
import argparse
from datetime import datetime

# ============================================
# Configuration
# ============================================
parser = argparse.ArgumentParser(description='Add new sequences to existing Milvus database')
parser.add_argument('--input', type=str, required=True, help='Path to CSV with new sequences')
parser.add_argument('--db_path', type=str, default='milvus_db/milvus.db', help='Path to Milvus database')
parser.add_argument('--collection', type=str, default='dna_sequences_s', help='Collection name')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for embeddings (GPU)')
parser.add_argument('--insert_batch', type=int, default=2048, help='Batch size for Milvus insertion')
parser.add_argument('--skip_duplicates', action='store_true', help='Skip sequences with duplicate IDs')
parser.add_argument('--checkpoint_interval', type=int, default=5000, help='Save checkpoint every N sequences')
args = parser.parse_args()

# Settings
NEW_CSV_PATH = args.input
MILVUS_DB_PATH = args.db_path
COLLECTION_NAME = args.collection
MODEL_NAME = 'zhihan1996/DNABERT-S'
BATCH_SIZE = args.batch_size  # GPU optimized
INSERT_BATCH_SIZE = args.insert_batch
MAX_LEN = 768
CHECKPOINT_FILE = f'add_sequences_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
CHECKPOINT_INTERVAL = args.checkpoint_interval
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

# ============================================
# Initialize Device (GPU)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: No GPU detected! This will be very slow.")
    print("Consider using GPU or reducing batch size")

# ============================================
# Initialize Model and Tokenizer
# ============================================
print("\nCargando modelo y tokenizador DNABERT-S...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    exit(1)

# ============================================
# Initialize Milvus
# ============================================
print(f"\nConectando a Milvus: {MILVUS_DB_PATH}")
try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    # Check if collection exists
    if not client.has_collection(COLLECTION_NAME):
        print(f"❌ Error: Colección '{COLLECTION_NAME}' no existe.")
        print("Ejecuta embed_sequences.py primero para crear la base de datos.")
        exit(1)
    
    print(f"✅ Conectado a colección '{COLLECTION_NAME}'")
    
    # Get current stats
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"📊 Vectores actuales en la base de datos: {stats['row_count']}")
    
except Exception as e:
    print(f"❌ Error conectando a Milvus: {e}")
    exit(1)

# ============================================
# Load New Sequences
# ============================================
print(f"\n📂 Cargando nuevas secuencias desde: {NEW_CSV_PATH}")

if not os.path.exists(NEW_CSV_PATH):
    print(f"❌ Archivo no encontrado: {NEW_CSV_PATH}")
    exit(1)

try:
    new_df = pd.read_csv(NEW_CSV_PATH)
    print(f"✅ Cargadas {len(new_df)} secuencias")
except Exception as e:
    print(f"❌ Error leyendo CSV: {e}")
    exit(1)

# Validate required columns
required_cols = ['Header', 'Sequence'] + HIERARCHY
missing_cols = [col for col in required_cols if col not in new_df.columns]
if missing_cols:
    print(f"❌ Error: Columnas faltantes en el CSV: {missing_cols}")
    print(f"Columnas requeridas: {required_cols}")
    exit(1)

# ============================================
# Check for Duplicate IDs
# ============================================
if args.skip_duplicates:
    print("\n🔍 Verificando IDs duplicados...")
    
    # Get all existing IDs from Milvus
    # Note: This loads all IDs into memory - for huge databases, use pagination
    try:
        existing_ids = set()
        limit = 10000
        offset = 0
        
        with tqdm(desc="Cargando IDs existentes", unit=" batch") as pbar:
            while True:
                results = client.query(
                    collection_name=COLLECTION_NAME,
                    filter="",
                    output_fields=["id"],
                    limit=limit,
                    offset=offset
                )
                
                if not results:
                    break
                
                existing_ids.update([r['id'] for r in results])
                offset += len(results)
                pbar.update(1)
                
                if len(results) < limit:
                    break
        
        print(f"   Total IDs en la base de datos: {len(existing_ids)}")
        
        # Filter out duplicates
        original_count = len(new_df)
        new_df = new_df[~new_df['Header'].isin(existing_ids)]
        duplicates_count = original_count - len(new_df)
        
        if duplicates_count > 0:
            print(f"   ⚠️ Encontrados {duplicates_count} IDs duplicados - serán omitidos")
            print(f"   ✅ {len(new_df)} secuencias nuevas para agregar")
        else:
            print(f"   ✅ No se encontraron duplicados")
            
    except Exception as e:
        print(f"   ⚠️ Error verificando duplicados: {e}")
        print(f"   Continuando sin verificación de duplicados...")

if len(new_df) == 0:
    print("\n⚠️ No hay secuencias nuevas para agregar (todas son duplicadas)")
    exit(0)

# ============================================
# Function to Generate Embeddings
# ============================================
def get_embeddings_batched(sequences, batch_size=BATCH_SIZE):
    """Generate embeddings in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generando embeddings"):
        batch = sequences[i:i+batch_size]
        batch = [s.replace('\n', '').strip() for s in batch]
        
        # Tokenize
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_LEN
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract [CLS] token embeddings
        embeddings = outputs[0][:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        
        # Clear GPU cache periodically
        if device.type == "cuda" and i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    return np.vstack(all_embeddings)

# ============================================
# Load Checkpoint if Exists
# ============================================
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
        start_idx = checkpoint['last_processed_idx']
    print(f"\n🔄 Reanudando desde el índice {start_idx}")
else:
    print(f"\n🆕 Iniciando desde el principio")

# ============================================
# Process and Insert New Sequences
# ============================================
total_rows = len(new_df)
print(f"\n🚀 Procesando {total_rows} secuencias nuevas...")
print(f"   Batch size (embeddings): {BATCH_SIZE}")
print(f"   Batch size (inserción): {INSERT_BATCH_SIZE}")
print(f"   Checkpoint cada: {CHECKPOINT_INTERVAL} secuencias")

data_buffer = []
sequences_processed = 0

try:
    for i in tqdm(range(start_idx, total_rows, BATCH_SIZE), desc="Procesando lotes"):
        batch_df = new_df.iloc[i:i+BATCH_SIZE]
        
        try:
            # Extract data
            sequences = batch_df['Sequence'].astype(str).tolist()
            headers = batch_df['Header'].astype(str).tolist()
            
            # Prepare metadata
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
                    'split': 'added'  # Mark as added data
                }
                metadatas.append(meta)
            
            # Generate embeddings
            embeddings = get_embeddings_batched(sequences, batch_size=BATCH_SIZE)
            
            # Add to buffer
            for j in range(len(sequences)):
                data_buffer.append({
                    "id": headers[j],
                    "vector": embeddings[j].tolist(),
                    "sequence": sequences[j],
                    **metadatas[j]
                })
            
            sequences_processed += len(sequences)
            
            # Insert when buffer is full
            if len(data_buffer) >= INSERT_BATCH_SIZE:
                print(f"\n   💾 Insertando {len(data_buffer)} vectores en Milvus...")
                client.insert(
                    collection_name=COLLECTION_NAME,
                    data=data_buffer
                )
                data_buffer = []
            
            # Save checkpoint
            if (i + BATCH_SIZE) % CHECKPOINT_INTERVAL == 0:
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({
                        'last_processed_idx': i + BATCH_SIZE,
                        'sequences_processed': sequences_processed,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                print(f"\n   💾 Checkpoint guardado en índice {i + BATCH_SIZE}")
        
        except Exception as e:
            print(f"\n❌ Error procesando lote en índice {i}: {e}")
            traceback.print_exc()
            
            # Save checkpoint on error
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({
                    'last_processed_idx': i,
                    'sequences_processed': sequences_processed,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"   💾 Checkpoint guardado debido a error")
            break
    
    # Insert remaining data
    if data_buffer:
        print(f"\n💾 Insertando {len(data_buffer)} vectores restantes...")
        try:
            client.insert(
                collection_name=COLLECTION_NAME,
                data=data_buffer
            )
        except Exception as e:
            print(f"❌ Error insertando lote final: {e}")

except KeyboardInterrupt:
    print("\n\n⚠️ Proceso interrumpido por el usuario")
    print(f"   Procesadas: {sequences_processed} secuencias")
    print(f"   Checkpoint guardado en: {CHECKPOINT_FILE}")
    exit(1)

# ============================================
# Cleanup and Summary
# ============================================
# Remove checkpoint file on success
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("\n✅ Checkpoint eliminado - proceso completado exitosamente")

# Get final stats
final_stats = client.get_collection_stats(COLLECTION_NAME)
print(f"\n{'='*60}")
print(f"✅ ¡Proceso completado exitosamente!")
print(f"{'='*60}")
print(f"   Secuencias agregadas: {sequences_processed}")
print(f"   Total vectores en la base de datos: {final_stats['row_count']}")
print(f"   Base de datos: {MILVUS_DB_PATH}")
print(f"   Colección: {COLLECTION_NAME}")
print(f"{'='*60}")

# Clear GPU memory
if device.type == "cuda":
    torch.cuda.empty_cache()
    print("\n🧹 Memoria GPU limpiada")