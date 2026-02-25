import nltk
import numpy as np
from collections import Counter
from scipy.sparse.linalg import svds
from numpy.linalg import norm
nltk.download('brown')
from nltk.corpus import brown
import re

words = [w.lower() for w in brown.words()]
print(type(words))
print(words[:10])

def is_word(token):
    return re.search(r"[a-z]", token)

# Step 1: lowercase
tokens = [w.lower() for w in brown.words()]

# Step 2: remove punctuation-only tokens
tokens = [w for w in tokens if is_word(w)]

# Step 3: remove rare words
counts = Counter(tokens)
tokens = [w for w in tokens if counts[w] >= 2]

# Step 4: vocabulary cutoff
# vocab = [w for w in counts]
vocab_set = set(tokens)

# Step 5: final filtered token stream
sorted(vocab_set)
tokens = [w for w in tokens if w in vocab_set]
word2idx = {w: i for i, w in enumerate(vocab_set)}
idx2word = {i: w for w, i in word2idx.items()}

print(len(tokens))
print(len(vocab_set))

matrix = []

for i in range(len(vocab_set)):
  row_i = []
  for j in range(len(vocab_set)):
    row_i.append(0)
  matrix.append(row_i)

matrix = np.array(matrix)
print(matrix.shape)

context_window = 2

for i, word in enumerate(tokens):
  if word not in (word2idx):
    continue
  target_idx = word2idx[word]
  window = context_window
  while window > 0:
    if i-window >= 0:
      context_word = tokens[i-window]
      if context_word not in (word2idx):
        continue
      context_idx = word2idx[context_word]
      matrix[target_idx][context_idx] += 1

    if i+window < len(tokens):
      context_word = tokens[i+window]
      if context_word not in (word2idx):
        continue
      context_idx = word2idx[context_word]
      matrix[target_idx][context_idx] += 1
    window -= 1

print(matrix.shape)
print(sum(matrix[word2idx['the']]))
print(sum(matrix[word2idx['trial']]))

total_sum = matrix.sum()

row_sum = matrix.sum(axis=1, keepdims=True)
col_sum = matrix.sum(axis=0, keepdims=True)

denominator = row_sum @ col_sum

PMI = np.zeros_like(matrix, dtype=float)

# Only compute where both matrix and denominator are non-zero
mask = (matrix > 0) & (denominator > 0)
PMI[mask] = np.log2((matrix[mask] * total_sum) / denominator[mask])
C_PPMI = np.maximum(PMI, 0)

w = word2idx['jury']
c = word2idx['said']
print("pair -->", idx2word[w], idx2word[c])
print("Matrix value:", C_PPMI[w][c], matrix[w][c])

w = word2idx['the']
c = word2idx['trial']
print("pair -->", idx2word[w], idx2word[c])
print("Matrix value:", C_PPMI[w][c], matrix[w][c])


k_components = 200
U, s, Vt = svds(C_PPMI, k=k_components)

print("Shape of U (Left singular vectors):", U.shape)
print("Shape of s (Singular values):", s.shape)
print("Shape of Vt (Right singular vectors):", Vt.shape)

reverse_s = s[::-1]
reverse_U = U[:, ::-1]
reverse_Vt = Vt[::-1, :]

embeddings = reverse_U @ np.diag(reverse_s)
print(embeddings.shape)

def cosine_sim(w1, w2):
    v1 = embeddings[word2idx[w1]]
    v2 = embeddings[word2idx[w2]]
    
    # Handle zero vectors to avoid division by zero
    norm_v1 = norm(v1)
    norm_v2 = norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return np.dot(v1, v2) / (norm_v1 * norm_v2)


def most_similar(word, topk=5):
    target = embeddings[word2idx[word]]
    target_norm = norm(target)
    
    if target_norm == 0:
        return []  # Return empty list if target has zero norm
    
    sims = []
    for w, idx in word2idx.items():
        embedding_norm = norm(embeddings[idx])
        if embedding_norm == 0:
            sim = 0.0
        else:
            sim = np.dot(target, embeddings[idx]) / (target_norm * embedding_norm)
        sims.append((w, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[1:topk+1]

most_similar("trial")

def analogy(a, b, c, topk=5):
    vec = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]
    vec_norm = norm(vec)
    
    if vec_norm == 0:
        return []  # Return empty list if vec has zero norm
    
    sims = []
    for w, idx in word2idx.items():
        embedding_norm = norm(embeddings[idx])
        if embedding_norm == 0:
            sim = 0.0
        else:
            sim = np.dot(vec, embeddings[idx]) / (vec_norm * embedding_norm)
        sims.append((w, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topk]

# ===== COMPREHENSIVE FUNCTION TESTS =====
print("\n" + "="*60)
print("COMPREHENSIVE FUNCTION TESTS")
print("="*60)

# Test 1: cosine_sim function
print("\n--- Test 1: cosine_sim(w1, w2) ---")
test_pairs = [
    ("jury", "trial"),
    ("jury", "said"),
    ("the", "a"),
    ("court", "case"),
    ("said", "said")
]

print("Testing cosine similarity between word pairs:")
for w1, w2 in test_pairs:
    try:
        sim = cosine_sim(w1, w2)
        print(f"  cosine_sim('{w1}', '{w2}') = {sim:.4f}")
    except KeyError as e:
        print(f"  cosine_sim('{w1}', '{w2}') = ERROR: word '{e.args[0]}' not in vocabulary")

# Test 2: most_similar function
print("\n--- Test 2: most_similar(word, topk=5) ---")
test_words = ["trial", "jury", "said", "court", "case"]

for word in test_words:
    try:
        similar = most_similar(word, topk=5)
        print(f"\nMost similar to '{word}':")
        for w, sim in similar:
            print(f"  {w}: {sim:.4f}")
    except KeyError:
        print(f"  '{word}' not in vocabulary")

# Test 3: most_similar with different topk values
print("\n--- Test 3: most_similar(word, topk) with different k values ---")
word = "trial"
for k in [3, 5, 10]:
    try:
        similar = most_similar(word, topk=k)
        print(f"\nTop {k} most similar to '{word}':")
        for w, sim in similar:
            print(f"  {w}: {sim:.4f}")
    except KeyError:
        print(f"  '{word}' not in vocabulary")

# Test 4: analogy function
print("\n--- Test 4: analogy(a, b, c, topk=5) ---")
test_analogies = [
    ("jury", "trial", "court"),
    ("said", "asked", "told"),
]

for a, b, c in test_analogies:
    try:
        result = analogy(a, b, c, topk=5)
        print(f"\nAnalogy: {a}:{b} :: {c}:?")
        for w, sim in result:
            print(f"  {w}: {sim:.4f}")
    except KeyError as e:
        print(f"  Analogy: {a}:{b} :: {c}:? - ERROR: word '{e.args[0]}' not in vocabulary")

# Test 5: analogy with different topk values
print("\n--- Test 5: analogy(a, b, c, topk) with different k values ---")
a, b, c = "jury", "trial", "court"
for k in [3, 5, 10]:
    try:
        result = analogy(a, b, c, topk=k)
        print(f"\nAnalogy: {a}:{b} :: {c}:? (Top {k})")
        for w, sim in result:
            print(f"  {w}: {sim:.4f}")
    except KeyError as e:
        print(f"  ERROR: word '{e.args[0]}' not in vocabulary")

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)

import os
import torch
os.makedirs('embeddings', exist_ok=True)
embeddings_dict = {
    'embeddings': torch.FloatTensor(embeddings),
    'word2idx': word2idx,
    'idx2word': idx2word
}

output_path = "embeddings/svd.pt"
torch.save(embeddings_dict, output_path)
print(f"\nEmbeddings saved to {output_path}")