# whenever logging, use print with flush=True to ensure that the output is printed immediately

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import time

print('Setting random seed...', flush=True)
np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush=True)

# Load the tokenizer
print('Loading tokenizer...', flush=True)
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
print('Tokenizer loaded', flush=True)

# Load the model
print('Loading model...', flush=True)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device).eval()
print('Model loaded', flush=True)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.to(torch.float32) if device==torch.device("cpu") else None

# prepare your protein sequences as a list
# use 1000 randomly generated protein sequences as an example
# generate them by amino acid composition with average length
# Average Amino Acid Composition:
# A: 9.46%
# C: 1.19%
# D: 5.48%
# E: 6.20%
# F: 3.84%
# G: 7.45%
# H: 2.22%
# I: 5.46%
# K: 4.86%
# L: 9.97%
# M: 2.38%
# N: 3.69%
# P: 4.91%
# Q: 3.88%
# R: 5.78%
# S: 6.53%
# T: 5.51%
# V: 6.98%
# W: 1.29%
# Y: 2.92%
# average length: 373.1435522149767
sequence_examples = []
# generate sequences with lengths in the specified ranges, using the letters probability distribution
print('Generating sequences...', flush=True)
probabilities = {'A': 0.0946, 'C': 0.0119, 'D': 0.0548, 'E': 0.062, 'F': 0.0384, 'G': 0.0745, 'H': 0.0222, 'I': 0.0546, 'K': 0.0486, 'L': 0.0997, 'M': 0.0238, 'N': 0.0369, 'P': 0.0491, 'Q': 0.0388, 'R': 0.0578, 'S': 0.0653, 'T': 0.0551, 'V': 0.0698, 'W': 0.0129, 'Y': 0.0292}
for _ in range(1000):
    length = np.random.exponential(373.1435522149767)
    sequence = ''.join(np.random.choice(list(probabilities.keys()), int(length), p=list(probabilities.values())))
    sequence_examples.append(sequence)
sequence_examples = sorted(sequence_examples, key=len, reverse=True)
lengths = [len(sequence) for sequence in sequence_examples]

#store the sequences in a file for later use
with open('sequences.txt', 'w') as f:
    for sequence in sequence_examples:
        f.write(sequence + '\n')
print('Sequences generated', flush=True)

max_residues=4000
max_seq_len=1000
max_batch=100

avg_len = np.mean(lengths)
n_long = sum([1 for l in lengths if l > max_seq_len])

print("Average sequence length: {}".format(avg_len), flush=True)
print("Number of sequences >{}: {}".format(max_seq_len, n_long), flush=True)

start = time.time()
batch = list()
total_embeddings = list()

for seq_idx, sequence in enumerate(sequence_examples, 1):
    seq = seq.replace('U','X').replace('Z','X').replace('O','X')
    seq_len = len(seq)
    seq = ' '.join(list(seq))
    batch.append((seq, seq_len))

    n_res_batch = sum([s_len for _, s_len in batch]) + seq_len
    if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(sequence_examples) or seq_len > max_seq_len:
        seqs, seq_lens = zip(*batch)
        batch = list()

        print('Tokenizing sequences in batch number {}...'.format(seq_idx), flush=True)
        token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
        print('Sequences tokenized', flush=True)

        try:
            with torch.no_grad():
                print('Generating embeddings for batch number {}...'.format(seq_idx), flush=True)
                embedding_repr = model(input_ids, attention_mask=attention_mask)
                print('Embeddings generated for batch number {}'.format(seq_idx), flush=True)
        except RuntimeError:
            print("RuntimeError during embedding for (L={}). Try lowering batch size. ".format(seq_len) +
                      "If single sequence processing does not work, you need more vRAM to process your protein.")
            continue

        print('Processing embeddings for batch number {}...'.format(seq_idx), flush=True)
        for i, seq_len in enumerate(seq_lens):
            emb = embedding_repr.last_hidden_state[i, :seq_len].mean(dim=0)
            total_embeddings.append(emb.detach().cpu().numpy().squeeze())
        print('Embeddings processed for batch number {}'.format(seq_idx), flush=True)

end = time.time()

print('Saving embeddings...', flush=True)
total_embeddings = np.array(total_embeddings)
np.save('embeddings.npy', total_embeddings)
print('Embeddings stored', flush=True)

print('\n############# STATS #############', flush=True)
print('Total number of sequences processed: {}'.format(len(sequence_examples)), flush=True)
print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(sequence_examples), avg_len), flush=True)
print('All done!', flush=True)

