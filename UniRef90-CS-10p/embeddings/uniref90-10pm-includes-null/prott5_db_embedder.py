import sqlite3
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import time as time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
print("Tokenizer loaded")

model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(device)
print("Model loaded")

model.to(torch.float32) if device==torch.device("cpu") else None

def update_embeddings_in_batch(batch, conn, cursor):
    try:
        update_query = "UPDATE sequences_le_1000 SET embedding = ? WHERE name = ?"
        cursor.executemany(update_query, batch)
        conn.commit()
        print("Updated embeddings in batch")
    except Exception as e:
        print(f"Error updating embeddings in batch: {e}")
    return

def batch_generate_embeddings(batch_size=1):
    conn = sqlite3.connect("sep1000.db")
    cursor = conn.cursor()
    print("Connected to database")

    batch_num = 1
    while True:
        print("Batch number: {}".format(batch_num))
        start = time.time()

        try:
            cursor.execute("SELECT COUNT(*) FROM sequences_gt_1000 WHERE embedding IS NULL")
            remaining = cursor.fetchone()[0]
            if remaining == 0:
                print("No remaining sequences to process.")
                break

            print("Remaining: {}".format(remaining))

            cursor.execute(f"SELECT name, sequence FROM sequences_gt_1000 WHERE embedding IS NULL LIMIT {batch_size}")
            rows = cursor.fetchall()

            if not rows:
                print("No rows fetched, stopping.")
                break

            batch = []
            sequences = [row[1] for row in rows]
            lengths = [len(sequence) for sequence in sequences]

            # Clean and tokenize the sequences
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

            ids = tokenizer(sequences, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

            # Process and store the embeddings
            for i, row in enumerate(rows):
                name = row[0]
                embedding = embedding_repr.last_hidden_state[i, :lengths[i]].mean(dim=0)
                batch.append((embedding.cpu().numpy().tobytes(), name))

            print(batch)
            # update_embeddings_in_batch(batch, conn, cursor)

            # Clean up memory if using GPU
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            batch_num += 1

        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            break

        end = time.time()
        print("Time taken: {} seconds".format(end - start))
        print("time per sequence: {} seconds".format((end - start) / batch_size))
        break
        print("estimated time remaining: {} minutes".format((remaining - batch_size) * (end - start) / 60))
    cursor.close()
    conn.close()
    print("Processing complete.")
    return

if __name__ == "__main__":
    print("Generating embeddings in batch")
    batch_generate_embeddings()
