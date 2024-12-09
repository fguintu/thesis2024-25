import numpy as np
import sqlite3
import os
import glob
from datetime import datetime

# Parameters
db_file = '../uniref90-10pm.db'
threshold = 0.77
batch_size = 1_000_000  # Save results in batches of 100k
batch_prefix = 'protein_similarity_batch_'

# SLURM array variables
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
total_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

# Logging function
def log(message):
    print(f"[{datetime.now()}] {message}", flush=True)

# Save results
def save_batch_in_tsv(filename, results):
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"{result['Protein_A']}\t{result['Protein_B']}\t{result['Cosine_Similarity']:.6f}\n")
    log(f"Batch saved in TSV format: {filename}")

# Get the last processed indices
def get_last_indices(protein_ids):
    # Find the highest indexed file
    pattern = f"{batch_prefix}{task_id}_*.tsv"
    batch_files = glob.glob(pattern)
    if not batch_files:
        return 0, 0  # No files found, start from the beginning

    # Sort files numerically by extracting the numeric part after the last underscore
    batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    last_file = batch_files[-1]
    last_i = int(last_file.split('_')[-1].split('.')[0])  # Extract i from filename

    # Read the last line of the file to get the last j protein ID
    with open(last_file, 'r') as f:
        last_line = f.readlines()[-1].strip()
    last_j_protein = last_line.split('\t')[1]  # Get Protein_B ID

    # Find the index of the last j protein in the protein_ids list
    last_j = protein_ids.index(last_j_protein)

    return last_i, last_j + 1  # Resume at i and j + 1

# Load proteins from the database
def load_proteins():
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # get the embeddings
    embeddings = []
    with open('ruseq10000.txt') as f:
        keys = f.read().splitlines()
    protein_ids = keys
    for key in keys:
        c.execute('SELECT embedding FROM uniref90 WHERE name = ?', (key,))
        embedding = c.fetchone()[0]

        # Parse the embedding, default dtype=np.float16
        embedding = np.frombuffer(embedding, dtype=np.float16)

        # If shape is not 1024, re-parse as np.float32
        if embedding.shape[0] != 1024:
            embedding = np.frombuffer(embedding, dtype=np.float32)
        embeddings.append(embedding)

    conn.close()
    # query = """
    # SELECT name, embedding
    # FROM uniref90
    # WHERE LENGTH(sequence) <= 1200 AND embedding IS NOT NULL
    # """
    # cursor.execute(query)
    # rows = cursor.fetchall()
    # conn.close()

    # protein_ids, embeddings = [], []
    # for row in rows:
    #     protein_id, embedding_blob = row

    #     # Parse the embedding, default dtype=np.float16
    #     embedding = np.frombuffer(embedding_blob, dtype=np.float16)

    #     # If shape is not 1024, re-parse as np.float32
    #     if embedding.shape[0] != 1024:
    #         embedding = np.frombuffer(embedding_blob, dtype=np.float32)

    #     protein_ids.append(protein_id)
    #     embeddings.append(embedding)

    return protein_ids, np.array(embeddings)

# Compute pairwise similarities
def compute_similarity(protein_ids, embeddings, start_i, start_j):
    results = []
    num_proteins = len(protein_ids)

    # Resume computation
    for i in range(start_i, num_proteins):
        j_start = start_j + 1 if i == start_i else i + 1  # Start at j+1 for the first i
        norm_i = np.linalg.norm(embeddings[i])
        for j in range(j_start, num_proteins):
            # Compute cosine similarity
            dot_product = np.dot(embeddings[i], embeddings[j])
            # norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            similarity = dot_product / (norm_i * norm_j)

            # Save results if similarity meets threshold
            if similarity >= threshold:
                results.append({
                    'Protein_A': protein_ids[i],
                    'Protein_B': protein_ids[j],
                    'Cosine_Similarity': similarity
                })

            if len(results) >= batch_size:
                save_batch_in_tsv(f"{batch_prefix}{task_id}_{i}.tsv", results)
                results = []  # Clear results

    # Save any remaining results
    if results:
        save_batch_in_tsv(f"{batch_prefix}{task_id}_final.tsv", results)

# Main execution
if __name__ == "__main__":
    log("Loading proteins...")
    protein_ids, embeddings = load_proteins()
    log(f"Loaded {len(protein_ids)} proteins.")

    # Divide work among tasks
    num_proteins = len(protein_ids)
    proteins_per_task = num_proteins // total_tasks
    task_start_idx = task_id * proteins_per_task
    task_end_idx = num_proteins if task_id == total_tasks - 1 else (task_id + 1) * proteins_per_task

    # Get last processed indices
    last_i, last_j = get_last_indices(protein_ids)
    log(f"Resuming from i={last_i}, j={last_j}...")

    log(f"Task {task_id} processing proteins {task_start_idx} to {task_end_idx}.")
    compute_similarity(protein_ids, embeddings, max(task_start_idx, last_i), last_j)
    log(f"Task {task_id} completed.")