import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import traceback

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MILVUS_DB_PATH = 'gpuhub-tmp/milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
BATCH_SIZE = 64  # Increased for dual GPU (16 per GPU)
INSERT_BATCH_SIZE = 2048
MAX_LEN = 768

# ============================================
# Inicializar Dispositivo (Multi-GPU)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Check GPU availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs disponibles: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memoria: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU disponible")
    num_gpus = 0

# ============================================
# Inicializar Modelo y Tokenizador
# ============================================
print("\nCargando modelo y tokenizador...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Move to GPU BEFORE wrapping with DataParallel
    model = model.to(device)
    
    # Enable DataParallel for multi-GPU
    if num_gpus > 1:
        print(f"\n🚀 Activando DataParallel en {num_gpus} GPUs...")
        model = torch.nn.DataParallel(model)
        print(f"   Batch será dividido: {BATCH_SIZE} total → {BATCH_SIZE//num_gpus} por GPU")
    
    model.eval()
    print("✅ Modelo cargado exitosamente")
    
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit(1)

# ============================================
# Inicializar Milvus
# ============================================
print("\nInicializando Milvus...")
try:
    if not os.path.exists(os.path.dirname(MILVUS_DB_PATH)):
        os.makedirs(os.path.dirname(MILVUS_DB_PATH))
    
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    if client.has_collection(COLLECTION_NAME):
        print(f"Colección '{COLLECTION_NAME}' ya existe.")
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=768,
            id_type="string",
            max_length=512,
            metric_type="COSINE",
            auto_id=False
        )
        print(f"Colección '{COLLECTION_NAME}' creada (sin índice inicial).")
except Exception as e:
    print(f"Error inicializando Milvus: {e}")
    exit(1)

# ============================================
# Función para generar embeddings
# ============================================
def get_embeddings(sequences):
    """Generate embeddings with multi-GPU support"""
    # Clean sequences
    sequences = [s.replace('\n', '').strip() for s in sequences]
    
    # Tokenize
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, 
                      truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract [CLS] token embeddings
    # Note: DataParallel wraps output, so we access it the same way
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    
    return embeddings

# ============================================
# Procesar CSV en fragmentos
# ============================================
print(f"\nProcesando {CSV_PATH}...")

if not os.path.exists(CSV_PATH):
    print(f"Archivo no encontrado: {CSV_PATH}")
    exit(1)

# Load and sample data
print("Muestreando 100% del conjunto de datos...")
chunks = []
read_chunk_size = 50000 

try:
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Leyendo y muestreando"):
        sampled_chunk = chunk.sample(frac=1.0, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
print(f"Muestreadas {len(df_sample)} secuencias.")

# Train/test split
print("Realizando División Entrenamiento/Prueba (80/20)...")
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
train_df['split'] = 'train'
test_df['split'] = 'test'

processing_df = train_df
print(f"Procesando solo conjunto de entrenamiento ({len(processing_df)} secuencias). Test set ignorado.")

# ============================================
# Process in batches
# ============================================
total_rows = len(processing_df)
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\n{'='*60}")
print(f"🚀 INICIANDO GENERACIÓN DE EMBEDDINGS")
print(f"{'='*60}")
print(f"Total secuencias: {total_rows:,}")
print(f"Batch size: {BATCH_SIZE} (total)")
if num_gpus > 1:
    print(f"Por GPU: {BATCH_SIZE//num_gpus} (batch dividido automáticamente)")
print(f"Total batches: {num_batches:,}")
print(f"{'='*60}\n")

data_buffer = []

for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Lotes de embeddings"):
    batch_df = processing_df.iloc[i : i + BATCH_SIZE]
    
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
                'split': row['split']
            }
            metadatas.append(meta)
            
        # Generate embeddings (automatically uses both GPUs)
        embeddings = get_embeddings(sequences)
        
        # Add to buffer
        for j in range(len(sequences)):
            data_buffer.append({
                "id": headers[j],
                "vector": embeddings[j].tolist(),
                "sequence": sequences[j],
                **metadatas[j]
            })
            
        # Insert to Milvus when buffer is full
        if len(data_buffer) >= INSERT_BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                data=data_buffer
            )
            data_buffer = []
        
        # Clear GPU cache periodically to avoid memory buildup
        if i % (BATCH_SIZE * 50) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ Error procesando lote comenzando en índice {i}: {e}")
        traceback.print_exc()
        break

# Insert remaining data
if data_buffer:
    print(f"\nInsertando {len(data_buffer)} elementos restantes...")
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            data=data_buffer
        )
    except Exception as e:
        print(f"Error insertando lote final: {e}")

# ============================================
# Create index
# ============================================
print("\nGenerando índice HNSW...")
try:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
        params={}
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )
    print("✅ Índice creado exitosamente.")
except Exception as e:
    print(f"⚠️ Nota al crear índice: {e}")

# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("✅ ¡PROCESO COMPLETADO!")
print("="*60)
print(f"Embeddings almacenados en: {MILVUS_DB_PATH}")
print(f"Colección: {COLLECTION_NAME}")
if torch.cuda.is_available():
    for i in range(num_gpus):
        mem_allocated = torch.cuda.memory_allocated(i) / 1e9
        mem_cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i} memoria usada: {mem_allocated:.2f}GB / {mem_cached:.2f}GB cached")
print("="*60)

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()