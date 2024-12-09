import pandas as pd

import os
print(os.getcwd())

# Specify the file paths
keys_file_path = './cutoff/rs/10k_0-9.tsv'
# result_pairs_file_path = './tmp/result_pairs.tsv'

# Load the keys file
keys_df = pd.read_csv(keys_file_path, sep='\t', header=None, names=['ID1', 'ID2', 'Score'])

print(keys_df.head())

def check_pairs_exist(keys_df, result_pairs_path):
    # Read the result pairs file
    result_pairs_df = pd.read_csv(result_pairs_path, sep='\t', header=None, names=['ID1', 'ID2'])
    
    # Convert result pairs to sets of frozensets to make comparison order-independent
    result_pairs_set = set(frozenset((row.ID1, row.ID2)) for _, row in result_pairs_df.iterrows())

    # print first 5 rows of result_pairs_set
    print(list(result_pairs_set)[:5])

    # Check each pair in keys_df
    for _, row in keys_df.iterrows():
        pair = frozenset((row.ID1, row.ID2))
        if pair not in result_pairs_set:
            return f"Missing pair: {row.ID1}, {row.ID2}"
    
    return "All pairs exist."

# Run the check
check_result = check_pairs_exist(keys_df, result_pairs_file_path)
print(check_result)
