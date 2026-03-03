import nltk
import numpy as np
from collections import Counter
from scipy.sparse.linalg import svds
from numpy.linalg import norm
import os
import torch

nltk.download('brown', quiet=True)
from nltk.corpus import brown

def is_word(token):
    return token.isalpha()


def load_and_preprocess(min_count):
    sentences = brown.sents()

    preprocessed_corpus = []
    for sent in sentences:
        preprocessed_sentence = []
        for word in sent:
            word = word.lower()
            if is_word(word):
                preprocessed_sentence.append(word)
        preprocessed_corpus.append(preprocessed_sentence)

    counts = {}
    for sent in preprocessed_corpus:
        for word in sent:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

    vocab = []
    for word in sorted(counts.keys()):
        if counts[word] >= min_count:
            vocab.append(word)

    vocab_set = set(vocab)

    tokens = []
    for sent in preprocessed_corpus:
        for word in sent:
            if word in vocab_set:
                tokens.append(word)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return tokens, word2idx, idx2word


# ===== CORE SVD LOGIC (unchanged) =====

def build_cooccurrence(tokens, word2idx, context_window):
    V = len(word2idx)
    matrix = np.zeros((V, V), dtype=np.float32)

    for i, word in enumerate(tokens):
        if word not in word2idx:
            continue
        target_idx = word2idx[word]
        window = context_window
        while window > 0:
            if i - window >= 0:
                context_word = tokens[i - window]
                if context_word in word2idx:
                    matrix[target_idx][word2idx[context_word]] += 1
            if i + window < len(tokens):
                context_word = tokens[i + window]
                if context_word in word2idx:
                    matrix[target_idx][word2idx[context_word]] += 1
            window -= 1

    return matrix


def compute_ppmi(matrix):
    total_sum = matrix.sum()
    row_sum = matrix.sum(axis=1, keepdims=True)
    col_sum = matrix.sum(axis=0, keepdims=True)
    denominator = row_sum @ col_sum

    PMI = np.zeros_like(matrix, dtype=float)
    mask = (matrix > 0) & (denominator > 0)
    PMI[mask] = np.log2((matrix[mask] * total_sum) / denominator[mask])
    C_PPMI = np.maximum(PMI, 0)
    return C_PPMI


def compute_svd_embeddings(C_PPMI, k_components):
    U, s, Vt = svds(C_PPMI, k=k_components)
    reverse_s = s[::-1]
    reverse_U = U[:, ::-1]
    reverse_Vt = Vt[::-1, :]
    embeddings = reverse_U @ np.diag(reverse_s)
    return embeddings, reverse_U, reverse_s, reverse_Vt


def reconstruction_error(C_PPMI, U, s, Vt):
    reconstructed = U @ np.diag(s) @ Vt
    frob_norm = np.linalg.norm(C_PPMI, 'fro')
    if frob_norm == 0:
        return float('inf')
    return np.linalg.norm(C_PPMI - reconstructed, 'fro') / frob_norm


# ===== HYPERPARAMETER TUNING =====

min_counts        = [4]
context_windows   = [2, 4]
k_components_list = [100, 200, 300]

print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

best_config     = None
best_error      = float('inf')
tuning_results  = []

for min_count in min_counts:
    tokens, word2idx, idx2word = load_and_preprocess(min_count)
    print(f"\nmin_count={min_count} | vocab={len(word2idx)} | tokens={len(tokens)}")

    for context_window in context_windows:
        print(f"  Building co-occurrence matrix (window={context_window})...")
        matrix = build_cooccurrence(tokens, word2idx, context_window)
        print(f"  Computing PPMI...")
        C_PPMI = compute_ppmi(matrix)

        for k in k_components_list:
            print(f"  Running SVD (k={k})...")
            embeddings, U, s, Vt = compute_svd_embeddings(C_PPMI, k)
            err = reconstruction_error(C_PPMI, U, s, Vt)
            print(f"  window={context_window}, k={k} -> rel_recon_error={err:.4f}")

            config = {
                'min_count': min_count,
                'context_window': context_window,
                'k_components': k,
                'rel_recon_error': err
            }
            tuning_results.append(config)

            if err < best_error:
                best_error      = err
                best_config     = config
                best_embeddings = embeddings
                best_word2idx   = word2idx
                best_idx2word   = idx2word

print("\nBest config:", best_config)

# ===== USE BEST EMBEDDINGS FOR EVALUATION =====

embeddings = best_embeddings
word2idx   = best_word2idx
idx2word   = best_idx2word

os.makedirs('embeddings', exist_ok=True)
embeddings_dict = {
    'embeddings': torch.FloatTensor(embeddings),
    'word2idx': word2idx,
    'idx2word': idx2word,
    'best_hyperparameters': best_config,
    'tuning_results': tuning_results
}

output_path = "embeddings/svd.pt"
torch.save(embeddings_dict, output_path)
print(f"\nEmbeddings saved to {output_path}")

# Best config: {'min_count': 4, 'context_window': 2, 'k_components': 300, 'rel_recon_error': np.float64(0.9080691800671397)}