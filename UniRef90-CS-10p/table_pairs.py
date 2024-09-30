import sqlite3
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Define functions for Euclidean and Cosine distance
def compute_distances(pair):
    name1, sequence1, embedding1, name2, sequence2, embedding2 = pair
    embedding1_np = np.frombuffer(embedding1, dtype=np.float32)
    embedding2_np = np.frombuffer(embedding2, dtype=np.float32)
    
    euclid_d = np.linalg.norm(embedding1_np - embedding2_np)
    dot_product = np.dot(embedding1_np, embedding2_np)
    norm_a = np.linalg.norm(embedding1_np)
    norm_b = np.linalg.norm(embedding2_np)
    cos_d = 1 - (dot_product / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0
    
    return (name1, sequence1, name2, sequence2, euclid_d, cos_d)

# Connect to the SQLite database
conn = sqlite3.connect('uniref90-10pm.db')
cursor = conn.cursor()

# Fetch pairs from the uniref90 table
cursor.execute('''
    SELECT a.name, a.sequence, a.embedding, b.name, b.sequence, b.embedding
    FROM uniref90 a
    JOIN uniref90 b ON a.name < b.name
    WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL;
''')
pairs = cursor.fetchall()

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor(max_workers=7) as executor:  # Using 7 cores on an 8-core CPU
    results = executor.map(compute_distances, pairs)

# Insert the results into the uniref90_embedding_pairs table
for result in results:
    cursor.execute('''
        INSERT INTO uniref90_embedding_pairs (name1, sequence1, name2, sequence2, euclid_d, cos_d)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', result)

# Commit and close the connection
conn.commit()
conn.close()

print("Task completed successfully with parallel processing.")
