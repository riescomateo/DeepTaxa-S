import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
MAX_LEN = 256 
K_NEIGHBORS = 50 
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

# Inicializar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Inicializar Modelo y Tokenizador
print("Cargando modelo y tokenizador...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit(1)

# Inicializar Milvus
print("Conectando a Milvus...")
try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
    print(f"Conectado a Milvus.")
except Exception as e:
    print(f"Error conectando a Milvus: {e}")
    exit(1)

# Recargar Datos de Prueba
TEST_CACHE_PATH = 'data/test_dataset_cache.csv'

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
    read_chunk_size = 50000  # Igualar a embed_sequences.py para consistencia
    cols_to_use = ['Sequence'] + HIERARCHY
    
    try:
        for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size, usecols=cols_to_use), desc="Leyendo y muestreando"):
            sampled_chunk = chunk.sample(frac=1.0, random_state=42) # Igualar a embed_sequences.py (100%)
            chunks.append(sampled_chunk)
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        exit(1)

    df_sample = pd.concat(chunks)
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    print(f"Tamaño total del conjunto de prueba generado: {len(test_df)}")
    
    # Guardar en caché para futuras ejecuciones
    test_df.to_csv(TEST_CACHE_PATH, index=False)
    print(f"Conjunto de prueba guardado en caché: {TEST_CACHE_PATH}")

# Limitar a 30 muestras para prueba
test_df = test_df.head(30)
print(f"Limitando evaluación a {len(test_df)} muestras.")

# Función para generar embeddings en lotes (con soporte de caché)
def get_embeddings_batched(sequences, batch_size=8, cache_path='data/test_embeddings_s.npy'):
    # Intentar cargar embeddings pre-calculados
    if os.path.exists(cache_path):
        print(f"Cargando embeddings pre-calculados desde {cache_path}...")
        all_embeddings = np.load(cache_path)
        
        # Verificar que coincida el número de secuencias
        if len(all_embeddings) == len(sequences):
            print(f"Embeddings cargados correctamente: {all_embeddings.shape}")
            return all_embeddings
        else:
            print(f"Advertencia: Caché tiene {len(all_embeddings)} embeddings pero se necesitan {len(sequences)}.")
            print("Regenerando embeddings...")
    
    # Si no existe caché o no coincide, generar embeddings
    print("Generando embeddings...")
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

# Lógica de Búsqueda Iterativa (Optimizada)
def iterative_search_from_embedding(embedding, client, k=K_NEIGHBORS):
    # print("DEBUG: Starting iterative search for a sample")
    current_filter = {} # Removed "split": "train" as DB only contains train data
    predicted_taxonomy = {}
    
    # Dynamic K: Start at K_NEIGHBORS and decay to 1
    k_values = np.linspace(k, 1, len(HIERARCHY))
    k_values = [int(round(x)) for x in k_values]
    k_values = [max(1, x) for x in k_values]
    
    for i, level in enumerate(HIERARCHY):
        current_k = k_values[i]

        # Construir filtro de Milvus
        filter_expr = ""
        if current_filter:
            conditions = [f'{key} == "{value}"' for key, value in current_filter.items()]
            filter_expr = " and ".join(conditions)

        # Consultar Milvus con filtro actual
        try:
            # print(f"DEBUG: Querying level {level} with filter {filter_expr}")
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[embedding.tolist()],
                limit=current_k,
                filter=filter_expr,
                output_fields=[level]
            )
            # print(f"DEBUG: Query returned for {level}")
        except Exception as e:
            print(f"Error querying Milvus at level {level}: {e}")
            break
        
        if not results or not results[0]:
            break
            
        hits = results[0]
        values = [hit['entity'].get(level, "Unknown") for hit in hits]
        
        if not values:
            break

        counts = Counter(values)
        most_common_val, count = counts.most_common(1)[0]
        confidence = count / len(values)
        
        if most_common_val != "Unknown":
            current_filter[level] = most_common_val
            predicted_taxonomy[level] = {
                "value": most_common_val,
                "confidence": confidence
            }
        else:
            break
            
    # Calcular puntaje de confianza general
    total_weight = 0
    weighted_confidence_sum = 0
    weights = {'Kingdom': 7, 'Phylum': 6, 'Class': 5, 'Order': 4, 'Family': 3, 'Genus': 2, 'Species': 1}
    
    for level, data in predicted_taxonomy.items():
        w = weights.get(level, 1)
        weighted_confidence_sum += data['confidence'] * w
        total_weight += w
        
    overall_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
    
    return predicted_taxonomy, overall_confidence

# Bucle de Evaluación
print(f"Iniciando evaluación en {len(test_df)} muestras...")

# Pre-calcular embeddings
print("Pre-calculando embeddings para el conjunto de prueba...")
test_sequences = test_df['Sequence'].astype(str).tolist()
test_embeddings = get_embeddings_batched(test_sequences, batch_size=8)

if np.isnan(test_embeddings).any():
    print("ERROR: Embeddings contain NaNs!")
    exit(1)
if np.isinf(test_embeddings).any():
    print("ERROR: Embeddings contain Infs!")
    exit(1)
print(f"Embeddings shape: {test_embeddings.shape}")

y_true = {level: [] for level in HIERARCHY}
y_pred = {level: [] for level in HIERARCHY}
confidences = []

def process_single_sample(args):
    embedding, (_, row) = args
    true_tax = {level: str(row[level]) if pd.notna(row[level]) else "Unknown" for level in HIERARCHY}
    
    prediction, overall_score = iterative_search_from_embedding(embedding, client)
    
    pred_result = {}
    for level in HIERARCHY:
        if level in prediction:
            pred_result[level] = prediction[level]['value']
        else:
            pred_result[level] = "Unpredicted"
            
    return true_tax, pred_result, overall_score

print("Ejecutando búsqueda iterativa en paralelo...")
# Preparar argumentos
args_list = list(zip(test_embeddings, test_df.iterrows()))

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_single_sample, args) for args in args_list]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluando"):
        try:
            true_tax, pred_result, overall_score = future.result()
            confidences.append(overall_score)
            
            for level in HIERARCHY:
                y_true[level].append(true_tax[level])
                y_pred[level].append(pred_result[level])
        except Exception as e:
            print(f"Error procesando muestra: {e}")


# Calcular Métricas
print("\n--- Resultados de Evaluación Iterativa (DNABERT-S) ---")
metrics = []

for level in HIERARCHY:
    # ¿Filtrar 'Unknown' en la verdad terreno si se desea?
    # Usualmente queremos evaluar contra la verdad terreno conocida.
    # Pero aquí mantenemos todo.
    
    acc = accuracy_score(y_true[level], y_pred[level])
    
    # Promedio ponderado para multi-clase
    prec = precision_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    rec = recall_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    f1 = f1_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    
    metrics.append({
        "Level": level,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

df_metrics = pd.DataFrame(metrics)
print(df_metrics.to_string(index=False))

print(f"\nConfianza General Promedio: {np.mean(confidences):.4f}")

# Guardar resultados en CSV
df_metrics.to_csv("Evaluations/iterative_search_results_s.csv", index=False)
print("Resultados guardados en Evaluations/iterative_search_results_s.csv")
