import sqlite3
import cupy as cp
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Parameters
db_file = 'uniref90-10pm.db'
threshold = 0.4
batch_size = 1_000_000  # Number of pairs per batch
checkpoint_file = 'checkpoint.txt'
batch_prefix = 'protein_similarity_batch_'
current_batch = 0

# Function to log messages with timestamps
def log(message):
    print(f"[{datetime.now()}] {message}", flush=True)


# Save results in TSV format
def save_batch_in_tsv(filename, results):
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"{result['Protein_A']}\t{result['Protein_B']}\t{result['Cosine_Similarity']:.6f}\n")
    log(f"Batch saved in TSV format: {filename}")


# Load checkpoint if it exists
start_i, start_j = 0, 1
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        start_i, start_j, current_batch = map(int, f.readline().strip().split())
    log(f"Resuming from i={start_i}, j={start_j}, batch={current_batch}")
else:
    log("Starting from the beginning")


# Connect to the SQLite database
log("Connecting to the database...")
conn = sqlite3.connect(db_file)
cursor = conn.cursor()


# Query to select proteins with sequence length <= 1200 and non-null embeddings
log("Executing SQL query to retrieve protein data...")
query = """
SELECT name, embedding 
FROM uniref90 
WHERE LENGTH(sequence) <= 1200 AND embedding IS NOT NULL
"""
cursor.execute(query)
rows = cursor.fetchall()
log(f"Retrieved {len(rows)} protein entries from the database")


# Process embeddings and ensure shape is 1024
log("Processing embeddings...")
protein_ids = []
embeddings = []

for row in rows:
    protein_id, embedding_blob = row
    
    # Convert BLOB to NumPy array
    embedding = np.frombuffer(embedding_blob, dtype=np.float16)
    
    # Ensure shape is 1024
    if embedding.shape[0] != 1024:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
    if embedding.shape[0] < 1024:
        embedding = np.pad(embedding, (0, 1024 - embedding.shape[0]), mode='constant')
    elif embedding.shape[0] > 1024:
        embedding = embedding[:1024]
    
    protein_ids.append(protein_id)
    embeddings.append(embedding)

log(f"Processed {len(embeddings)} embeddings with shape (1024,)")

# Convert embeddings to CuPy array
log("Converting embeddings to GPU...")
embeddings = cp.array(embeddings)

# Normalize embeddings
log("Normalizing embeddings...")
norms = cp.linalg.norm(embeddings, axis=1, keepdims=True)  # Compute L2 norms
normalized_embeddings = embeddings / norms  # Normalize each embedding

# Compute pairwise cosine similarity matrix
log("Computing pairwise cosine similarity matrix on the GPU...")
similarity_matrix = cp.matmul(normalized_embeddings, normalized_embeddings.T)
similarity_matrix = cp.asnumpy(similarity_matrix)  # Convert back to NumPy for final processing
log("Cosine similarity matrix computation complete")

# Initialize storage for pairs that meet the threshold
results = []

# Main processing loop with resumption capability
num_proteins = len(protein_ids)
for i in range(start_i, num_proteins):
    for j in range(i + 1 if i != start_i else start_j, num_proteins):
        similarity_score = similarity_matrix[i, j]
        if similarity_score >= threshold:
            results.append({
                'Protein_A': protein_ids[i],
                'Protein_B': protein_ids[j],
                'Cosine_Similarity': similarity_score
            })

        # If batch is full, save it and clear results
        if len(results) >= batch_size:
            batch_filename = f'{batch_prefix}{current_batch}.tsv'
            save_batch_in_tsv(batch_filename, results)
            current_batch += 1
            results = []  # Clear the results for the next batch

            # Update the checkpoint file
            with open(checkpoint_file, 'w') as f:
                f.write(f"{i} {j} {current_batch}\n")
            log(f"Checkpoint updated: i={i}, j={j}, batch={current_batch}")

# Save any remaining results in the final batch
if results:
    batch_filename = f'{batch_prefix}{current_batch}.tsv'
    save_batch_in_tsv(batch_filename, results)

    # Update the checkpoint file for the final state
    with open(checkpoint_file, 'w') as f:
        f.write(f"{num_proteins} {num_proteins} {current_batch + 1}\n")
    log("Final checkpoint updated to indicate completion")

log(f"All pairwise similarity scores (≥ {threshold}) saved in TSV format.")