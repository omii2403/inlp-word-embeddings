import nltk
from nltk.corpus import brown
import numpy as np
from scipy.sparse.linalg import svds
import torch
import os
import string

nltk.download('brown')

sentences = brown.sents()
print(f"Total sentences: {len(sentences)}")

def is_valid_word(word):
    return word.isalpha()

sentences = [[word.lower() for word in sent if is_valid_word(word)] for sent in sentences]
sentences = [sent for sent in sentences if len(sent) > 0]

print(f"Sentences after cleaning: {len(sentences)}")

all_words = [word for sentence in sentences for word in sentence]
print(f"Total tokens after cleaning: {len(all_words)}")

print("\nPreprocessing Filtering vocabulary")

from collections import Counter
word_counts = Counter(all_words)
print(f"Unique word types before filtering: {len(word_counts)}")

minimum_count = 2
vocab = {word for word, count in word_counts.items() if count >= minimum_count}
print(f"Vocabulary size after filtering, minimum_count={minimum_count}: {len(vocab)}")

filtered_sentences = [[word for word in sent if word in vocab] for sent in sentences]
print(f"Sentences with filtered vocab: {len(filtered_sentences)}")

word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

cooccurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)

for sentence in filtered_sentences:
    for i in range(len(sentence)):
        target_word = sentence[i]
        if target_word not in word_to_index:
            continue
        target_index = word_to_index[target_word]
        
        for j in range(max(0, i - 2), min(len(sentence), i + 3)):
            if j != i:
                context_word = sentence[j]
                if context_word not in word_to_index:
                    continue
                context_index = word_to_index[context_word]
                cooccurrence_matrix[target_index][context_index] += 1

print("Co-occurrence matrix created.")
print(f"Co-occurrence matrix shape: {cooccurrence_matrix.shape}")
print(f"Non-zero entries: {np.count_nonzero(cooccurrence_matrix)}")
print(f"Matrix sparsity: {(1 - np.count_nonzero(cooccurrence_matrix) / cooccurrence_matrix.size) * 100:.2f}%")

if 'dog' in word_to_index:
    dog_idx = word_to_index['dog']
    dog_context_counts = cooccurrence_matrix[dog_idx, :]
    top_contexts_idx = np.argsort(-dog_context_counts)[:5]
    print(f"\nTop 5 context words for 'dog':")
    for rank, ctx_idx in enumerate(top_contexts_idx, 1):
        ctx_word = index_to_word[ctx_idx]
        count = dog_context_counts[ctx_idx]
        if count > 0:
            print(f"  {rank}. {ctx_word:15} (count: {int(count)})")
else:
    print("'dog' not in vocabulary")

total = cooccurrence_matrix.sum()
pwc = cooccurrence_matrix / total
pw = cooccurrence_matrix.sum(axis=1, keepdims=True) / total  # Shape: (vocab_size, 1)
pc = cooccurrence_matrix.sum(axis=0, keepdims=True) / total  # Shape: (1, vocab_size)

print("\nApplying PPMI Weighting")
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log(pwc / (pw * pc + 1e-10) + 1e-10)
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)

pmi_matrix = np.maximum(pmi, 0)  # PPMI

print(f"Non-zero PPMI entries: {np.count_nonzero(pmi_matrix)}")
print(f"Max PPMI value: {pmi_matrix.max():.4f}")
print(f"Mean PPMI (non-zero): {pmi_matrix[pmi_matrix > 0].mean():.4f}")
print(f"PPMI matrix shape: {pmi_matrix.shape}")

print("\nPerforming SVD")
U, S, Vt = svds(pmi_matrix, k=300)
idx = np.argsort(S)[::-1]
S_sorted = S[idx]
U_sorted = U[:, idx].copy()

embeddings = U_sorted
explained_variance = S_sorted / S_sorted.sum()
print(f"Embeddings shape: {embeddings.shape}")
print(f"Explained variance ratio: {explained_variance.sum():.4f}")


os.makedirs('embeddings', exist_ok=True)
embeddings_dict = {
    'embeddings': torch.FloatTensor(embeddings),
    'word2idx': word_to_index,
    'idx2word': index_to_word
}

output_path = "embeddings/svd.pt"
torch.save(embeddings_dict, output_path)
print(f"\nEmbeddings saved to {output_path}")