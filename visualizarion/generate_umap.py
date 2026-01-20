import pandas as pd
import numpy as np
from pymilvus import MilvusClient
import umap
import plotly.express as px
import os
from tqdm import tqdm

# Configuration
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s' # Using the _s collection as per user preference for simple search results
OUTPUT_CSV = 'umap_results.csv'
SAMPLE_SIZE = 50000  # Number of points to visualize
BATCH_SIZE = 10000   # Fetch in batches

def generate_umap_data():
    print("Connecting to Milvus...")
    try:
        client = MilvusClient(uri=MILVUS_DB_PATH)
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return

    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} not found.")
        return

    print(f"Fetching data from {COLLECTION_NAME}...")
    
    # Fetch data
    # We need vector and metadata
    output_fields = ["vector", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    
    # Since we can't easily random sample from Milvus without IDs, we'll fetch a large chunk and sample locally
    # or fetch sequentially. Given the data is likely shuffled during insertion (train_test_split), 
    # sequential fetch might be random enough.
    
    data = []
    offset = 0
    
    pbar = tqdm(total=SAMPLE_SIZE, desc="Fetching vectors")
    
    while len(data) < SAMPLE_SIZE:
        current_limit = min(BATCH_SIZE, SAMPLE_SIZE - len(data))
        try:
            res = client.query(
                collection_name=COLLECTION_NAME,
                filter="", # Empty filter to get all
                limit=current_limit,
                offset=offset,
                output_fields=output_fields
            )
            
            if not res:
                break
                
            data.extend(res)
            offset += len(res)
            pbar.update(len(res))
            
            if len(res) < current_limit: # End of collection
                break
                
        except Exception as e:
            print(f"Error querying Milvus: {e}")
            break
            
    pbar.close()
    
    if not data:
        print("No data found.")
        return

    print(f"Fetched {len(data)} records.")
    
    # Prepare for UMAP
    print("Preparing data for UMAP...")
    vectors = np.array([d['vector'] for d in data])
    
    # Extract metadata
    metadata = {
        'Phylum': [d.get('Phylum', 'Unknown') for d in data],
        'Class': [d.get('Class', 'Unknown') for d in data],
        'Order': [d.get('Order', 'Unknown') for d in data],
        'Family': [d.get('Family', 'Unknown') for d in data],
        'Genus': [d.get('Genus', 'Unknown') for d in data],
        'Species': [d.get('Species', 'Unknown') for d in data]
    }
    
    df = pd.DataFrame(metadata)
    
    print(f"Running UMAP on {len(vectors)} vectors with 3 components...")
    reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=-1)
    embedding = reducer.fit_transform(vectors)
    
    df['x'] = embedding[:, 0]
    df['y'] = embedding[:, 1]
    df['z'] = embedding[:, 2]
    
    print(f"Saving results to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    generate_umap_data()
