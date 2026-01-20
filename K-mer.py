import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import time
from tqdm import tqdm
import argparse

def get_kmers(sequence, k):
    """Generate k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def find_discriminative_kmers(csv_path, k=8, sample_size=None, target_col='Species'):
    print(f"Loading dataset from {csv_path}...")
    
    # Count k-mers per class and globally
    class_kmer_counts = defaultdict(Counter)
    global_kmer_counts = Counter()
    class_counts = Counter()
    
    print(f"Counting {k}-mers...")
    start_time = time.time()
    
    try:
        # Use chunking to reduce memory usage
        chunk_size = 20000
        if sample_size:
            reader = pd.read_csv(csv_path, nrows=sample_size, chunksize=chunk_size)
            print(f"Analyzing {sample_size} samples (subset). Target column: {target_col}")
        else:
            reader = pd.read_csv(csv_path, chunksize=chunk_size)
            print(f"Analyzing full dataset in chunks. Target column: {target_col}")

        for chunk in tqdm(reader, desc="Processing chunks"):
            # Update class counts
            counts = chunk[target_col].value_counts()
            for label, count in counts.items():
                class_counts[label] += count

            for _, row in chunk.iterrows():
                seq = row['Sequence']
                label = row[target_col]
                
                if not isinstance(seq, str):
                    continue
                    
                kmers = get_kmers(seq, k)
                # Use a set to count presence/absence per sequence (binary feature)
                # or list for frequency. Presence is usually better for short motifs.
                unique_kmers = set(kmers)
                
                for kmer in unique_kmers:
                    class_kmer_counts[label][kmer] += 1
                    global_kmer_counts[kmer] += 1
                    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
            
    print(f"Counting took {time.time() - start_time:.2f} seconds.")
    
    # Score k-mers
    # A simple score: P(kmer | class) * log(P(kmer | class) / P(kmer | global))
    # Or just TF-IDF style: Frequency in class * log(Total classes / Classes containing kmer)
    
    print("Scoring k-mers...")
    kmer_scores = []
    
    # We want k-mers that are very frequent in one class but rare in others.
    # Let's look at the top k-mers for each class.
    
    results = []
    print("Calculating scores for all classes...")
    
    for label, n_class in tqdm(class_counts.items(), desc="Classes"):
        if label not in class_kmer_counts:
            continue
            
        scores = {}
        for kmer, count in class_kmer_counts[label].items():
            # Calculate F1 Score (Harmonic mean of Precision and Recall)
            # Precision: How many sequences with this k-mer are actually this genus?
            # Recall: How many sequences of this genus have this k-mer?
            
            tp = count
            total_kmer_count = global_kmer_counts[kmer]
            
            precision = tp / total_kmer_count
            recall = tp / n_class
            
            # Weighted Score favoring Precision (Uniqueness)
            # Score = Precision^2 * Recall
            # This penalizes k-mers that appear in other genera much more than k-mers that miss some sequences in the target genus.
            score = (precision ** 2) * recall
            
            # Filter out very rare kmers
            if count > 5: 
                scores[kmer] = score
        
        if not scores:
            continue
            
        # Find single best k-mer
        # We prioritize score, then precision, then recall
        best_kmer = max(scores.items(), key=lambda x: x[1])[0]
        max_score = scores[best_kmer]
        
        results.append({
            target_col: label,
            'Kmer': best_kmer,
            'Score': max_score,
            'Frequency_in_class': class_kmer_counts[label][best_kmer],
            'Class_Size': n_class,
            'Precision': class_kmer_counts[label][best_kmer] / global_kmer_counts[best_kmer],
            'Recall': class_kmer_counts[label][best_kmer] / n_class
        })
            
    # Save results
    results_df = pd.DataFrame(results)
    output_file = 'important_subsequences.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} best discriminative k-mers to {output_file}")
    
    # Print a preview
    print("\nPreview of top results:")
    print(results_df.sort_values('Score', ascending=False).head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find important subsequences (k-mers) in DNA data.")
    parser.add_argument("--k", type=int, default=10, help="Length of k-mer")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to analyze (default: all)")
    args = parser.parse_args()
    
    find_discriminative_kmers('data/final_dataset.csv', k=args.k, sample_size=args.samples, target_col='Genus')
