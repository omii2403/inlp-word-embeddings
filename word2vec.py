import copy
import itertools
import json
import os
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
from nltk.corpus import brown
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_word(token):
    return token.isalpha()


def load_and_preprocess(min_count):
    nltk.download("brown", quiet=True)
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

    filtered_sentences = []
    for sent in preprocessed_corpus:
        filtered_sent = [w for w in sent if w in vocab_set]
        filtered_sentences.append(filtered_sent)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    word_counts = np.array([counts[w] for w in vocab], dtype=np.float32)

    dist = word_counts ** 0.75
    dist = dist / dist.sum()
    neg_sampling_dist = torch.tensor(dist, dtype=torch.float)

    indexed_sentences = []
    for sent in filtered_sentences:
        indexed_sent = [word2idx[w] for w in sent]
        if len(indexed_sent) > 1:
            indexed_sentences.append(indexed_sent)

    return indexed_sentences, word2idx, idx2word, neg_sampling_dist


def train_val_split(sentences, val_ratio, seed):
    ids = list(range(len(sentences)))
    random.Random(seed).shuffle(ids)
    val_size = int(len(ids) * val_ratio)
    val_ids = set(ids[:val_size])

    train_sentences = [s for i, s in enumerate(sentences) if i not in val_ids]
    val_sentences = [s for i, s in enumerate(sentences) if i in val_ids]
    return train_sentences, val_sentences


class SkipGramDataset(Dataset):
    def __init__(self, sentences, window_size):
        self.sentences = sentences
        self.window_size = window_size
        self.positions = []
        for sent_id, sent in enumerate(sentences):
            for word_pos in range(len(sent)):
                self.positions.append((sent_id, word_pos))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        sent_id, word_pos = self.positions[idx]
        sentence = self.sentences[sent_id]
        center_word = sentence[word_pos]

        start = max(0, word_pos - self.window_size)
        end = min(len(sentence), word_pos + self.window_size + 1)
        context_candidates = [sentence[i] for i in range(start, end) if i != word_pos]
        context_word = random.choice(context_candidates)

        return (
            torch.tensor(center_word, dtype=torch.long),
            torch.tensor(context_word, dtype=torch.long),
        )


class StaticPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c, p = self.pairs[idx]
        return torch.tensor(c, dtype=torch.long), torch.tensor(p, dtype=torch.long)


def build_all_pairs(sentences, window_size):
    pairs = []
    for sent in sentences:
        for i, center in enumerate(sent):
            start = max(0, i - window_size)
            end = min(len(sent), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                pairs.append((center, sent[j]))
    return pairs


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, pos_context_words, neg_context_words):
        center_emb = self.in_embeddings(center_words)
        pos_emb = self.out_embeddings(pos_context_words)
        neg_emb = self.out_embeddings(neg_context_words)

        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        return pos_score, neg_score


def sample_negative(batch_size, k, neg_sampling_dist, device):
    neg_samples = torch.multinomial(
        neg_sampling_dist,
        num_samples=batch_size * k,
        replacement=True,
    )
    neg_samples = neg_samples.view(batch_size, k)
    return neg_samples.to(device)


def run_epoch(model, data_loader, optimizer, criterion, neg_sampling_dist, k, device, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.set_grad_enabled(train_mode):
        for center_batch, context_batch in data_loader:
            center_batch = center_batch.to(device)
            context_batch = context_batch.to(device)
            batch_size = center_batch.size(0)

            neg_batch = sample_negative(batch_size, k, neg_sampling_dist, device)
            pos_score, neg_score = model(center_batch, context_batch, neg_batch)

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            loss = criterion(pos_score, pos_labels) + criterion(neg_score, neg_labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def train_with_early_stopping(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    neg_sampling_dist,
    k,
    device,
    max_epochs,
    patience,
):
    best_val = float("inf")
    best_state = None
    wait = 0
    train_history = []
    val_history = []

    for epoch in range(max_epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            neg_sampling_dist,
            k,
            device,
            train_mode=True,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            neg_sampling_dist,
            k,
            device,
            train_mode=False,
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({
                "in_embeddings": model.in_embeddings.state_dict(),
                "out_embeddings": model.out_embeddings.state_dict(),
            })
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.in_embeddings.load_state_dict(best_state["in_embeddings"])
        model.out_embeddings.load_state_dict(best_state["out_embeddings"])

    return train_history, val_history, best_val, len(train_history)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_sentences, word2idx, idx2word, neg_sampling_dist = load_and_preprocess(4)
    print("Total indexed sentences:", len(all_sentences))
    print("Vocab size:", len(word2idx))

    train_sentences, val_sentences = train_val_split(all_sentences, 0.1, 42)
    print("Train sentences:", len(train_sentences), "| Val sentences:", len(val_sentences))

    embedding_dims = [100, 200, 300]
    window_sizes = [2, 4]
    negatives = [5, 10]
    learning_rates = [0.001, 0.003]
    batch_sizes = [256, 512]

    criterion = nn.BCEWithLogitsLoss()
    tuning_results = []
    best = None

    for embedding_dim, window_size, k, lr, batch_size in itertools.product(
        embedding_dims, window_sizes, negatives, learning_rates, batch_sizes
    ):
        print(
            f"\nTuning config -> dim={embedding_dim}, window={window_size}, "
            f"k={k}, lr={lr}, batch_size={batch_size}"
        )

        train_pairs = build_all_pairs(train_sentences, window_size=window_size)
        val_pairs = build_all_pairs(val_sentences, window_size=window_size)

        train_loader = DataLoader(
            StaticPairDataset(train_pairs),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            StaticPairDataset(val_pairs),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = SkipGramModel(len(word2idx), embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_hist, val_hist, best_val, used_epochs = train_with_early_stopping(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            neg_sampling_dist=neg_sampling_dist,
            k=k,
            device=device,
            max_epochs=8,
            patience=3
        )

        print(
            f"  done in {used_epochs} epochs | "
            f"last_train={train_hist[-1]:.4f} | best_val={best_val:.4f}"
        )

        result = {
            "embedding_dim": embedding_dim,
            "window_size": window_size,
            "negative_k": k,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs_run": used_epochs,
            "train_loss_last": train_hist[-1],
            "val_loss_best": best_val,
            "train_loss_history": train_hist,
            "val_loss_history": val_hist,
        }
        tuning_results.append(result)

        if best is None or result["val_loss_best"] < best["val_loss_best"]:
            best = result

    print("\nBest config selected:")
    print(best)

    final_window = best["window_size"]
    final_dim = best["embedding_dim"]
    final_k = best["negative_k"]
    final_lr = best["learning_rate"]
    final_batch_size = best["batch_size"]

    final_train_pairs = build_all_pairs(all_sentences, window_size=final_window)
    final_val_pairs = build_all_pairs(val_sentences, window_size=final_window)

    final_train_loader = DataLoader(
        StaticPairDataset(final_train_pairs),
        batch_size=final_batch_size,
        shuffle=True,
        drop_last=True,
    )
    final_val_loader = DataLoader(
        StaticPairDataset(final_val_pairs),
        batch_size=final_batch_size,
        shuffle=False,
        drop_last=False,
    )

    final_model = SkipGramModel(len(word2idx), final_dim).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=final_lr)

    final_train_hist, final_val_hist, final_best_val, final_epochs_run = train_with_early_stopping(
        model=final_model,
        optimizer=final_optimizer,
        criterion=criterion,
        train_loader=final_train_loader,
        val_loader=final_val_loader,
        neg_sampling_dist=neg_sampling_dist,
        k=final_k,
        device=device,
        max_epochs=15,
        patience=3,
    )

    os.makedirs(os.path.dirname('embeddings/skipgram.pt'), exist_ok=True)
    vectors = final_model.in_embeddings.weight.detach().cpu()

    payload = {
        "embeddings": vectors,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "best_hyperparameters": best,
        "final_train_loss_history": final_train_hist,
        "final_val_loss_history": final_val_hist,
        "final_best_val_loss": final_best_val,
        "final_epochs_run": final_epochs_run,
        "meta": {
            "model": "skipgram_negative_sampling",
            "min_count": 5,
            "val_ratio": 0.1,
            "seed": 42,
            "preprocessing": "matched_to_word2vec_embedding_notebook",
        },
    }

    torch.save(payload, 'embeddings/skipgram.pt')
    print(f"Saved embeddings to embeddings/skipgram.pt")

    with open('word2vec_tuning_results.json', "w", encoding="utf-8") as f:
        json.dump({"best": best, "all_results": tuning_results}, f, indent=2)
    print(f"Saved tuning results to word2vec_tuning_results.json")


if __name__ == "__main__":
    main()
