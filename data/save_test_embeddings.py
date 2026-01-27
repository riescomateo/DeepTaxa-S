import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Configuración
CSV_PATH = 'data/final_dataset.csv'
TEST_CACHE_PATH = 'data/test_dataset_cache.csv'
EMBEDDINGS_SAVE_PATH = 'data/test_embeddings_s.npy'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MAX_LEN = 256
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

# Inicializar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Verificar si ya existen los embeddings
if os.path.exists(EMBEDDINGS_SAVE_PATH):
    print(f"Los embeddings ya existen en {EMBEDDINGS_SAVE_PATH}.")
    # Verificar timestamps
    if os.path.exists(TEST_CACHE_PATH):
        cache_mtime = os.path.getmtime(TEST_CACHE_PATH)
        emb_mtime = os.path.getmtime(EMBEDDINGS_SAVE_PATH)
        if emb_mtime > cache_mtime:
            print("Los embeddings están actualizados. Saliendo.")
            exit(0)
        else:
            print("Los embeddings están desactualizados. Recalculando...")
    else:
        print("No se encontró el archivo de cache de prueba para verificar la antigüedad. Asumiendo que los embeddings son válidos.")
        exit(0)

# Cargar Datos de Prueba
use_cache = False
if os.path.exists(TEST_CACHE_PATH):
    if os.path.exists(CSV_PATH):
        csv_mtime = os.path.getmtime(CSV_PATH)
        cache_mtime = os.path.getmtime(TEST_CACHE_PATH)
        if cache_mtime > csv_mtime:
            use_cache = True
            print(f"Caché es válido (más reciente que CSV).")
        else:
            print("Caché desactualizado (CSV es más reciente). Recargando...")
    else:
        use_cache = True

if use_cache:
    print(f"Cargando conjunto de prueba desde caché: {TEST_CACHE_PATH}")
    test_df = pd.read_csv(TEST_CACHE_PATH)
    print(f"Tamaño del conjunto de prueba cargado: {len(test_df)}")
else:
    print(f"Recargando conjunto de datos para reconstruir la división de prueba...")
    if not os.path.exists(CSV_PATH):
        print(f"Archivo no encontrado: {CSV_PATH}")
        exit(1)

    chunks = []
    read_chunk_size = 50000 
    cols_to_use = ['Sequence'] + HIERARCHY
    
    try:
        for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size, usecols=cols_to_use), desc="Leyendo y muestreando"):
            sampled_chunk = chunk.sample(frac=1.0, random_state=42)
            chunks.append(sampled_chunk)
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        exit(1)

    df_sample = pd.concat(chunks)
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    print(f"Tamaño total del conjunto de prueba generado: {len(test_df)}")
    
    test_df.to_csv(TEST_CACHE_PATH, index=False)
    print(f"Conjunto de prueba guardado en caché: {TEST_CACHE_PATH}")

# Inicializar Modelo y Tokenizador
print("Cargando modelo y tokenizador...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit(1)

# Función para generar embeddings en lotes
def get_embeddings_batched(sequences, batch_size=128):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generando Embeddings"):
        batch = sequences[i:i+batch_size]
        batch = [s.replace('\n', '').strip() for s in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs[0][:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Calcular embeddings
print("Calculando embeddings para el conjunto de prueba...")
test_sequences = test_df['Sequence'].astype(str).tolist()
test_embeddings = get_embeddings_batched(test_sequences, batch_size=128)

# Guardar embeddings
print(f"Guardando embeddings en {EMBEDDINGS_SAVE_PATH}...")
np.save(EMBEDDINGS_SAVE_PATH, test_embeddings)
print("Embeddings guardados exitosamente.")
