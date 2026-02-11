import argparse
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ============================================
# Configuration
# ============================================
parser = argparse.ArgumentParser(description='Generate embeddings for DNA sequences')
parser.add_argument('--input', type=str, required=True, 
                    help='Path to input CSV file with sequences')
parser.add_argument('--output', type=str, required=True, 
                    help='Path to save embeddings (.npy file)')
parser.add_argument('--model', type=str, default='zhihan1996/DNABERT-S', 
                    help='Model name or path')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='Batch size for GPU processing')
parser.add_argument('--max_len', type=int, default=768, 
                    help='Maximum sequence length')
parser.add_argument('--checkpoint_interval', type=int, default=50, 
                    help='Clear GPU cache every N batches')
args = parser.parse_args()

# Settings from arguments
INPUT_CSV_PATH = args.input
OUTPUT_EMBEDDINGS_PATH = args.output
MODEL_NAME = args.model
BATCH_SIZE = args.batch_size
MAX_LEN = args.max_len
CHECKPOINT_INTERVAL = args.checkpoint_interval

print("🧬 Generando embeddings para test set...")

# Verificar que existe el test set
if not os.path.exists(INPUT_CSV_PATH):
    print(f"❌ Error: {INPUT_CSV_PATH} no existe")
    print("Ejecuta primero: python recreate_test_set.py")
    exit(1)

# Cargar test set
print(f"\n1️⃣ Cargando test set desde {INPUT_CSV_PATH}...")
test_df = pd.read_csv(INPUT_CSV_PATH)
print(f"   ✅ {len(test_df):,} secuencias")

# Inicializar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n2️⃣ Inicializando modelo en {device}...")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"   GPUs disponibles: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    num_gpus = 0

# Cargar modelo
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    
    # Multi-GPU si está disponible
    if num_gpus > 1:
        print(f"   🚀 Usando DataParallel con {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    model.eval()
    print("   ✅ Modelo cargado")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Función para generar embeddings
def get_embeddings_batched(sequences, batch_size=BATCH_SIZE):
    all_embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generando embeddings"):
        batch = sequences[i:i+batch_size]
        batch = [s.replace('\n', '').strip() for s in batch]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                          truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs[0][:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        
        # Limpiar cache cada checkpoint_interval batches
        if i % (batch_size * CHECKPOINT_INTERVAL) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.vstack(all_embeddings)

# Generar embeddings
print(f"\n3️⃣ Generando embeddings...")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Total batches: {(len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE}")

test_sequences = test_df['Sequence'].astype(str).tolist()
test_embeddings = get_embeddings_batched(test_sequences)

print(f"   ✅ Shape: {test_embeddings.shape}")

# Guardar embeddings
print(f"\n4️⃣ Guardando embeddings en {OUTPUT_EMBEDDINGS_PATH}...")
np.save(OUTPUT_EMBEDDINGS_PATH, test_embeddings)

file_size_mb = os.path.getsize(OUTPUT_EMBEDDINGS_PATH) / 1e6

print(f"\n{'='*60}")
print(f"✅ EMBEDDINGS GENERADOS Y GUARDADOS")
print(f"{'='*60}")
print(f"Ubicación: {OUTPUT_EMBEDDINGS_PATH}")
print(f"Shape: {test_embeddings.shape}")
print(f"Tamaño: {file_size_mb:.1f} MB")
print(f"{'='*60}")

# Limpiar GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("\n🧹 Memoria GPU limpiada")