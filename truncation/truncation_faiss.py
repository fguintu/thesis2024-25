import numpy as np
import faiss
import sqlite3
from graph_tool.all import Graph, label_components

faiss.omp_set_num_threads(64)  # Limit number of threads for Faiss

# Step 1: Load your embeddings
# Assume embeddings is a NumPy array of shape (5000000, 1024)
# embeddings = np.load("path_to_embeddings.npy")
def load_proteins():
    conn = sqlite3.connect('../uniref90-10pm.db')
    cursor = conn.cursor()
    query = """
    SELECT name, embedding
    FROM uniref90
    WHERE LENGTH(sequence) <= 1200 AND embedding IS NOT NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    protein_ids, embeddings = [], []
    for row in rows:
        protein_id, embedding_blob = row

        # Parse the embedding, default dtype=np.float16
        embedding = np.frombuffer(embedding_blob, dtype=np.float16)

        # If shape is not 1024, re-parse as np.float32
        if embedding.shape[0] != 1024:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        protein_ids.append(protein_id)
        embeddings.append(embedding)

    protein_ids = np.array(protein_ids)
    return protein_ids, np.array(embeddings)


embeddings, ids = load_proteins()

# Normalize embeddings for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Step 2: Create a Faiss index for inner product (cosine similarity)
d = embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatIP(d)  # Inner product index
index.add(embeddings)  # Add embeddings to the index

# Step 3: Query the index for neighbors
threshold = 0.77  # Cosine similarity threshold
k = 100  # Maximum number of neighbors to consider (adjust based on your data)
distances, indices = index.search(embeddings, k)

# Step 4: Filter neighbors by threshold (use IDs instead of indices)
filtered_edges = []
for i in range(len(distances)):
    for j, dist in zip(indices[i], distances[i]):
        if dist > threshold and i != j:
            filtered_edges.append((ids[i], ids[j]))  # Use IDs instead of indices

# Step 5: Build a graph for clustering
g = Graph(directed=False)
vertex_map = {id_: g.add_vertex() for id_ in ids}  # Map IDs to graph-tool vertices
for id1, id2 in filtered_edges:
    g.add_edge(vertex_map[id1], vertex_map[id2])

# Step 6: Find connected components (clusters)
component_labels, component_count = label_components(g)

# Step 7: Map IDs to their cluster IDs
id_to_cluster = {}
for i, id_ in enumerate(ids):
    id_to_cluster[id_] = int(component_labels[g.vertex(i)])

# Save the cluster mapping if needed
np.save("id_to_cluster_mapping.npy", id_to_cluster)

# Optional: Print cluster statistics
print(f"Number of clusters: {component_count}")