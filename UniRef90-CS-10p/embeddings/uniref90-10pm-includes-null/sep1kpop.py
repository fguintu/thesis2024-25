import sqlite3
import h5py
import numpy as np
import tqdm

# Open your HDF5 file and SQLite connection
with h5py.File('output4.h5', 'r') as h5, sqlite3.connect('sep1000.db') as conn:
    c = conn.cursor()

    # Create a temporary table to hold the name and embedding data
    c.execute('CREATE TABLE temp_embeddings (name TEXT, embedding BLOB)')

    # Fetch the keys (names) from the HDF5 file
    keys = list(h5.keys())

    # Insert embeddings into the temporary table in batches
    BATCH_SIZE = 1000
    for i in tqdm.tqdm(range(0, len(keys), BATCH_SIZE), desc='Inserting into temp table', total=len(keys) // BATCH_SIZE):
        batch_keys = keys[i:i + BATCH_SIZE]
        embeddings = [(key, np.array(h5[key]).tobytes()) for key in batch_keys]
        c.executemany('INSERT INTO temp_embeddings (name, embedding) VALUES (?, ?)', embeddings)

    # Now, perform the bulk update using a JOIN
    # c.execute('''
    #     UPDATE sequences_le_1000
    #     SET embedding = (
    #         SELECT temp_embeddings.embedding
    #         FROM temp_embeddings
    #         WHERE temp_embeddings.name = sequences_le_1000.name
    #     )
    #     WHERE sequences_le_1000.embedding IS NULL
    #     AND EXISTS (
    #         SELECT 1 FROM temp_embeddings
    #         WHERE temp_embeddings.name = sequences_le_1000.name
    #     )
    # ''')

    # Commit all changes
    conn.commit()

    # Drop the temporary table after use
    # c.execute('DROP TABLE temp_embeddings')
