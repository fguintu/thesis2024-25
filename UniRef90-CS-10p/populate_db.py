import sqlite3
import sys
import os
import time
import numpy as np

def populate_db_with_embeddings(db, path):
    """
    Function to populate the database with embeddings from the specified path
    which contains multiple folders in a given range.
    """
    start_time = time.time()  # Start timer for this folder

    # Iterate through each folder in the given directory
    for root, dirs, files in os.walk(path):
        if 'ids.txt' in files and 'protein_embeddings.npy' in files:
            ids_path = os.path.join(root, 'ids.txt')
            embeddings_path = os.path.join(root, 'protein_embeddings.npy')
            
            # Read the IDs from ids.txt
            with open(ids_path) as f:
                ids = f.readlines()

            # Clean the IDs by extracting the portion after 'UA='
            ids = [id.split('UA=')[1].split(' ')[0].strip() for id in ids]

            # Load the protein embeddings from the .npy file
            embeddings = np.load(embeddings_path)

            # Ensure the number of ids matches the number of embeddings
            if len(ids) != embeddings.shape[0]:
                print(f"Warning: Mismatch between ids and embeddings in folder {root}")
                continue

            # Connect to the SQLite database
            conn = sqlite3.connect(db)
            c = conn.cursor()

            for i, id in enumerate(ids):
                # Check if the ID is in the database
                c.execute("SELECT * FROM uniref90 WHERE name=?", (id,))
                result = c.fetchone()

                if result:
                    # If ID is found, update its embedding
                    c.execute("UPDATE uniref90 SET embedding=? WHERE name=?", (embeddings[i].tobytes(), id))
                    conn.commit()

            # Close the connection for this batch
            conn.close()

    end_time = time.time()  # End timer for this folder
    print(f"Time taken to process folder {path}: {end_time - start_time:.2f} seconds")

def main():
    # Parse command-line arguments
    db = sys.argv[1]  # Database path
    base_path = sys.argv[2]  # Base directory containing the folders
    interval = sys.argv[3]  # Interval like '0-69'

    # Parse the interval range (e.g., '0-69')
    start, end = map(int, interval.split('-'))

    # Iterate through each folder in the interval range
    for i in range(start, end + 1):
        folder_name = str(i)
        folder_path = os.path.join(base_path, folder_name)
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_path}")
            populate_db_with_embeddings(db, folder_path)
        else:
            print(f"Folder {folder_name} does not exist. Skipping...")

if __name__ == "__main__":
    main()
