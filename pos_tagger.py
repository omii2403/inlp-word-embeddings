import argparse
import copy
import itertools
import json
import os
import random

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


config = {
    "seed": 42,
    "svd_path": "embeddings/svd.pt",
    "skipgram_path": "embeddings/skipgram.pt",
    "glove_txt_path": "embeddings/glove.6B.300d.txt",
    "glove_pt_path": "embeddings/glove.pt",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "window_sizes": [1, 2],
    "hidden_sizes": [256, 512],
    "learning_rates": [0.001, 0.003],
    "batch_sizes": [256, 512],
    "dropout": 0.3,
    "tune_epochs": 10,
    "final_epochs": 20,
    "early_stop_patience": 3,
    "output_dir": "embeddings",
    "results_json": "pos_tuning_results.json",
    "checkpoint_json": "pos_tuning_checkpoint.json",
    "resume_from_checkpoint": True,
}


class POSDataset(Dataset):
    def __init__(self, windows, tags):
        self.windows = windows
        self.tags = tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.windows[idx], self.tags[idx]


class MLPTagger(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_tags),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_glove_txt_to_pt(txt_path, pt_path):
    if os.path.exists(pt_path):
        print(f"glove.pt already exists at {pt_path}, skipping conversion")
        return

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing GloVe text file: {txt_path}")

    print(f"Reading {txt_path} ...")
    raw = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            raw[parts[0]] = np.array(parts[1:], dtype=np.float32)

    emb_dim = next(iter(raw.values())).shape[0]
    word2idx = {w: i for i, w in enumerate(sorted(raw.keys()))}
    idx2word = {i: w for w, i in word2idx.items()}

    matrix = np.zeros((len(word2idx), emb_dim), dtype=np.float32)
    for word, idx in word2idx.items():
        matrix[idx] = raw[word]

    os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)
    torch.save(
        {
            "embeddings": torch.FloatTensor(matrix),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "meta": {"source": txt_path, "dim": emb_dim, "vocab": len(word2idx)},
        },
        pt_path,
    )
    print(f"Saved -> {pt_path} (vocab={len(word2idx)}, dim={emb_dim})")


def load_pos_data(seed, train_ratio, val_ratio):
    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    from nltk.corpus import brown

    tagged_sents = brown.tagged_sents(tagset="universal")

    filtered_sents = []
    for sent in tagged_sents:
        filtered = [(w.lower(), t) for w, t in sent if w.isalpha()]
        if filtered:
            filtered_sents.append(filtered)

    all_tags = sorted(set(t for sent in filtered_sents for _, t in sent))
    tag2idx = {t: i for i, t in enumerate(all_tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    ids = list(range(len(filtered_sents)))
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_sents = [filtered_sents[i] for i in ids[:n_train]]
    val_sents = [filtered_sents[i] for i in ids[n_train:n_train + n_val]]
    test_sents = [filtered_sents[i] for i in ids[n_train + n_val:]]

    print(f"Sentences -> train: {len(train_sents)} | val: {len(val_sents)} | test: {len(test_sents)}")
    print(f"Tags ({len(all_tags)}): {all_tags}")

    return train_sents, val_sents, test_sents, tag2idx, idx2tag


def load_embedding_pt(path, name):
    data = torch.load(path, map_location="cpu", weights_only=False)
    vectors = data["embeddings"].numpy()
    word2idx = data["word2idx"]
    print(f"{name:<10}: vocab={len(word2idx):>7}, dim={vectors.shape[1]}")
    return vectors, word2idx


def build_embedding_matrix(vectors, word2idx, device):
    pad_row = np.zeros((1, vectors.shape[1]), dtype=np.float32)
    full = np.vstack([vectors, pad_row])
    mat = torch.FloatTensor(full).to(device)
    mat.requires_grad_(False)
    return mat


def build_windows(sents, word2idx, tag2idx, window_size):
    pad_idx = len(word2idx)
    all_windows = []
    all_tags = []

    for sent in sents:
        words = [w for w, _ in sent]
        tags = [t for _, t in sent]

        for i in range(len(words)):
            ctx = []
            for offset in range(-window_size, window_size + 1):
                j = i + offset
                if j < 0 or j >= len(words):
                    ctx.append(pad_idx)
                else:
                    ctx.append(word2idx.get(words[j], pad_idx))
            all_windows.append(ctx)
            all_tags.append(tag2idx[tags[i]])

    return torch.LongTensor(all_windows), torch.LongTensor(all_tags)


def run_epoch(model, loader, emb_matrix, optimizer, criterion, device, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.set_grad_enabled(train_mode):
        for windows, tags in loader:
            windows = windows.to(device)
            tags = tags.to(device)

            x = emb_matrix[windows]
            x = x.view(windows.size(0), -1)

            logits = model(x)
            loss = criterion(logits, tags)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def train_with_early_stopping(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    emb_matrix,
    device,
    max_epochs,
    patience,
):
    best_val = float("inf")
    best_state = None
    wait = 0
    train_hist = []
    val_hist = []

    for epoch in range(max_epochs):
        train_loss = run_epoch(model, train_loader, emb_matrix, optimizer, criterion, device, True)
        val_loss = run_epoch(model, val_loader, emb_matrix, optimizer, criterion, device, False)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        print(f"  Epoch {epoch + 1:>2}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_hist, val_hist, best_val, len(train_hist)


def evaluate(model, loader, emb_matrix, device):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for windows, tags in loader:
            windows = windows.to(device)
            x = emb_matrix[windows].view(windows.size(0), -1)
            preds = model(x).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(tags.tolist())

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")
    return acc, f1, all_true, all_preds


def cfg_key(emb_name, window_size, hidden_size, lr, batch_size):
    return f"{emb_name}|win={window_size}|h={hidden_size}|lr={lr}|bs={batch_size}"


def save_checkpoint(path, results, best):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"all_results": results, "best": best}, f, indent=2)


def load_checkpoint(path):
    if not os.path.exists(path):
        return [], None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("all_results", []), data.get("best", None)


def recompute_best(results):
    if not results:
        return None
    return min(results, key=lambda x: x["val_loss_best"])


def train_full_pipeline():
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_sents, val_sents, test_sents, tag2idx, idx2tag = load_pos_data(
        config["seed"], config["train_ratio"], config["val_ratio"]
    )
    num_tags = len(tag2idx)

    convert_glove_txt_to_pt(config["glove_txt_path"], config["glove_pt_path"])

    svd_vectors, svd_word2idx = load_embedding_pt(config["svd_path"], "SVD")
    sg_vectors, sg_word2idx = load_embedding_pt(config["skipgram_path"], "SkipGram")
    glove_vectors, glove_word2idx = load_embedding_pt(config["glove_pt_path"], "GloVe")

    embeddings = {
        "glove": (glove_vectors, glove_word2idx),
        "svd": (svd_vectors, svd_word2idx),
        "skipgram": (sg_vectors, sg_word2idx),
    }

    criterion = nn.CrossEntropyLoss()

    if config["resume_from_checkpoint"]:
        tuning_results, _ = load_checkpoint(config["checkpoint_json"])
        if tuning_results:
            print(f"Loaded checkpoint: {len(tuning_results)} completed configs")
        else:
            print("No checkpoint found, starting fresh")
    else:
        tuning_results = []

    completed_keys = {
        cfg_key(r["emb_name"], r["window_size"], r["hidden_size"], r["lr"], r["batch_size"])
        for r in tuning_results
    }

    all_combos = list(
        itertools.product(
            config["window_sizes"],
            config["hidden_sizes"],
            config["learning_rates"],
            config["batch_sizes"],
        )
    )

    print(f"Configs per embedding: {len(all_combos)} | Total: {len(all_combos) * len(embeddings)}")

    for emb_name, (vectors, word2idx) in embeddings.items():
        emb_matrix = build_embedding_matrix(vectors, word2idx, device)
        emb_dim = vectors.shape[1]

        print("\n" + "=" * 55)
        print(f"Tuning [{emb_name}] dim={emb_dim}")
        print("=" * 55)

        for window_size, hidden_size, lr, batch_size in all_combos:
            key = cfg_key(emb_name, window_size, hidden_size, lr, batch_size)
            if key in completed_keys:
                print(f"  Skipping: {key}")
                continue

            print(f"\n  --- {key} ---")
            train_w, train_t = build_windows(train_sents, word2idx, tag2idx, window_size)
            val_w, val_t = build_windows(val_sents, word2idx, tag2idx, window_size)

            train_loader = DataLoader(
                POSDataset(train_w, train_t),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                POSDataset(val_w, val_t),
                batch_size=batch_size,
                shuffle=False,
            )

            input_size = (2 * window_size + 1) * emb_dim
            model = MLPTagger(input_size, hidden_size, num_tags, config["dropout"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_hist, val_hist, best_val, epochs_run = train_with_early_stopping(
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                emb_matrix,
                device,
                config["tune_epochs"],
                config["early_stop_patience"],
            )

            result = {
                "emb_name": emb_name,
                "window_size": window_size,
                "hidden_size": hidden_size,
                "lr": lr,
                "batch_size": batch_size,
                "dropout": config["dropout"],
                "epochs_run": epochs_run,
                "val_loss_best": best_val,
                "train_loss_last": train_hist[-1],
                "train_hist": train_hist,
                "val_hist": val_hist,
            }
            tuning_results.append(result)
            completed_keys.add(key)

            best_per_emb = {
                name: recompute_best([r for r in tuning_results if r["emb_name"] == name])
                for name in embeddings
            }
            save_checkpoint(config["checkpoint_json"], tuning_results, best_per_emb)
            with open(config["results_json"], "w", encoding="utf-8") as f:
                json.dump({"best_per_emb": best_per_emb, "all_results": tuning_results}, f, indent=2)

    best_per_emb = {
        name: recompute_best([r for r in tuning_results if r["emb_name"] == name])
        for name in embeddings
    }

    print("\nBest config per embedding:")
    for name, b in best_per_emb.items():
        print(
            f"  [{name}] window={b['window_size']} hidden={b['hidden_size']} "
            f"lr={b['lr']} bs={b['batch_size']} val_loss={b['val_loss_best']:.4f}"
        )

    os.makedirs(config["output_dir"], exist_ok=True)

    final_models = {}
    final_results = {}

    for emb_name, (vectors, word2idx) in embeddings.items():
        best = best_per_emb[emb_name]

        best_window = best["window_size"]
        best_hidden = best["hidden_size"]
        best_lr = best["lr"]
        best_bs = best["batch_size"]

        print("\n" + "=" * 50)
        print(f"Final training [{emb_name}] dim={vectors.shape[1]}")
        print(f"  window={best_window} | hidden={best_hidden} | lr={best_lr} | bs={best_bs}")
        print("=" * 50)

        emb_matrix = build_embedding_matrix(vectors, word2idx, device)
        emb_dim = vectors.shape[1]
        input_size = (2 * best_window + 1) * emb_dim

        train_w, train_t = build_windows(train_sents, word2idx, tag2idx, best_window)
        val_w, val_t = build_windows(val_sents, word2idx, tag2idx, best_window)
        test_w, test_t = build_windows(test_sents, word2idx, tag2idx, best_window)

        train_loader = DataLoader(POSDataset(train_w, train_t), batch_size=best_bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(POSDataset(val_w, val_t), batch_size=best_bs, shuffle=False)
        test_loader = DataLoader(POSDataset(test_w, test_t), batch_size=512, shuffle=False)

        model = MLPTagger(input_size, best_hidden, num_tags, config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

        train_hist, val_hist, best_val, epochs_run = train_with_early_stopping(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            emb_matrix,
            device,
            config["final_epochs"],
            config["early_stop_patience"],
        )

        final_models[emb_name] = (model, emb_matrix, test_loader)
        final_results[emb_name] = {
            "val_loss": best_val,
            "epochs_run": epochs_run,
            "train_hist": train_hist,
            "val_hist": val_hist,
        }

        out_path = os.path.join(config["output_dir"], f"pos_tagger_{emb_name}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "tag2idx": tag2idx,
                "idx2tag": idx2tag,
                "params": {
                    "window_size": best_window,
                    "hidden_size": best_hidden,
                    "dropout": config["dropout"],
                    "lr": best_lr,
                    "batch_size": best_bs,
                    "emb_dim": emb_dim,
                    "input_size": input_size,
                    "num_tags": num_tags,
                },
            },
            out_path,
        )
        print(f"Saved -> {out_path}")

    print(f"{'Embedding':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 35)

    best_acc = -1.0
    best_emb_name = None

    for emb_name, (model, emb_matrix, test_loader) in final_models.items():
        acc, f1, all_true, all_preds = evaluate(model, test_loader, emb_matrix, device)
        final_results[emb_name]["accuracy"] = acc
        final_results[emb_name]["macro_f1"] = f1
        final_results[emb_name]["all_true"] = all_true
        final_results[emb_name]["all_preds"] = all_preds
        print(f"{emb_name:<12} {acc:>10.4f} {f1:>10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_emb_name = emb_name

    print(f"\nBest model: {best_emb_name} (accuracy={best_acc:.4f})")

    cm = confusion_matrix(
        final_results[best_emb_name]["all_true"],
        final_results[best_emb_name]["all_preds"],
    )
    tag_names = [idx2tag[i] for i in range(num_tags)]

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=tag_names, yticklabels=tag_names, cmap="Blues")
    plt.title(f"Confusion Matrix - {best_emb_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fig_path = os.path.join("figures", f"confusion_matrix_{best_emb_name}.png")
    plt.savefig(fig_path, dpi=180)
    print(f"Saved -> {fig_path}")

    summary = {
        "best_hyperparameters_per_emb": best_per_emb,
        "test_results": {
            emb: {k: v for k, v in r.items() if k not in ("all_true", "all_preds", "train_hist", "val_hist")}
            for emb, r in final_results.items()
        },
    }

    with open("final_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved -> final_results.json")


def predict_sentence(model, emb_matrix, word2idx, words, window_size, device):
    pad_idx = len(word2idx)
    windows = []

    for i in range(len(words)):
        ctx = []
        for offset in range(-window_size, window_size + 1):
            j = i + offset
            if j < 0 or j >= len(words):
                ctx.append(pad_idx)
            else:
                ctx.append(word2idx.get(words[j], pad_idx))
        windows.append(ctx)

    w_tensor = torch.LongTensor(windows).to(device)
    with torch.no_grad():
        x = emb_matrix[w_tensor].view(w_tensor.size(0), -1)
        pred_ids = model(x).argmax(dim=1).cpu().tolist()
    return pred_ids


def error_reason(true_tag, pred_tag, is_oov):
    if is_oov:
        return "likely unseen/OOV token"
    if {true_tag, pred_tag}.issubset({"NOUN", "VERB"}):
        return "noun-verb ambiguity"
    if {true_tag, pred_tag}.issubset({"ADJ", "NOUN"}):
        return "adjective-noun ambiguity"
    if {true_tag, pred_tag}.issubset({"ADV", "ADJ"}):
        return "adverb-adjective ambiguity"
    if true_tag == "PRT" and pred_tag == "ADP":
        return "particle vs preposition confusion"
    return "short context window can miss syntax cues"


def collect_error_examples(model, emb_matrix, word2idx, idx2tag, test_sents, window_size, device, max_examples):
    examples = []

    for sent in test_sents:
        words = [w for w, _ in sent]
        gold_tags = [t for _, t in sent]
        pred_ids = predict_sentence(model, emb_matrix, word2idx, words, window_size, device)
        pred_tags = [idx2tag[i] for i in pred_ids]

        mismatches = []
        for word, gold, pred in zip(words, gold_tags, pred_tags):
            if gold != pred:
                is_oov = word not in word2idx
                mismatches.append(
                    {
                        "word": word,
                        "gold": gold,
                        "pred": pred,
                        "is_oov": is_oov,
                        "reason": error_reason(gold, pred, is_oov),
                    }
                )

        if mismatches:
            examples.append({"sentence": " ".join(words), "mismatches": mismatches})

        if len(examples) >= max_examples:
            break

    return examples


def evaluate_pretrained(max_examples=5, report_path="pos_error_analysis.md"):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    convert_glove_txt_to_pt(config["glove_txt_path"], config["glove_pt_path"])

    _, _, test_sents, _, _ = load_pos_data(config["seed"], config["train_ratio"], config["val_ratio"])

    model_specs = {
        "glove": {
            "model_path": "embeddings/pos_tagger_glove.pt",
            "embedding_path": config["glove_pt_path"],
        },
        "skipgram": {
            "model_path": "embeddings/pos_tagger_skipgram.pt",
            "embedding_path": config["skipgram_path"],
        },
        "svd": {
            "model_path": "embeddings/pos_tagger_svd.pt",
            "embedding_path": config["svd_path"],
        },
    }

    summary_rows = []
    all_examples = {}

    for emb_name in ["glove", "skipgram", "svd"]:
        spec = model_specs[emb_name]
        if not os.path.exists(spec["model_path"]):
            print(f"Skipping {emb_name}: {spec['model_path']} not found")
            continue
        if not os.path.exists(spec["embedding_path"]):
            print(f"Skipping {emb_name}: {spec['embedding_path']} not found")
            continue

        ckpt = torch.load(spec["model_path"], map_location="cpu", weights_only=False)
        emb_data = torch.load(spec["embedding_path"], map_location="cpu", weights_only=False)

        vectors = emb_data["embeddings"].numpy()
        word2idx = emb_data["word2idx"]
        emb_matrix = build_embedding_matrix(vectors, word2idx, device)

        params = ckpt["params"]
        idx2tag = {int(k): v for k, v in ckpt["idx2tag"].items()}
        tag2idx = {v: k for k, v in idx2tag.items()}

        model = MLPTagger(
            int(params["input_size"]),
            int(params["hidden_size"]),
            int(params["num_tags"]),
            float(params["dropout"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])

        test_w, test_t = build_windows(test_sents, word2idx, tag2idx, int(params["window_size"]))
        test_loader = DataLoader(POSDataset(test_w, test_t), batch_size=512, shuffle=False)

        acc, f1, _, _ = evaluate(model, test_loader, emb_matrix, device)
        summary_rows.append((emb_name, acc, f1))

        examples = collect_error_examples(
            model,
            emb_matrix,
            word2idx,
            idx2tag,
            test_sents,
            int(params["window_size"]),
            device,
            max_examples,
        )
        all_examples[emb_name] = examples

    print(f"\n{'Embedding':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 36)
    for emb_name, acc, f1 in summary_rows:
        print(f"{emb_name:<12} {acc:>10.4f} {f1:>10.4f}")

    lines = [
        "# POS Tagger Error Analysis (All Models)",
        "",
        "## Test Metrics",
        "",
        "| Embedding | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]

    for emb_name, acc, f1 in summary_rows:
        lines.append(f"| {emb_name} | {acc:.4f} | {f1:.4f} |")

    for emb_name in ["glove", "skipgram", "svd"]:
        if emb_name not in all_examples:
            continue
        lines.append("")
        lines.append(f"## {emb_name.upper()} Error Examples")
        lines.append("")

        examples = all_examples[emb_name]
        if not examples:
            lines.append("No sentence-level mistakes found in sampled data.")
            lines.append("")
            continue

        for i, ex in enumerate(examples, start=1):
            lines.append(f"### Example {i}")
            lines.append("")
            lines.append(f"Sentence: `{ex['sentence']}`")
            lines.append("")
            lines.append("Incorrect tags:")
            for mm in ex["mismatches"]:
                oov_note = " (OOV)" if mm["is_oov"] else ""
                lines.append(
                    f"- `{mm['word']}`: gold=`{mm['gold']}`, pred=`{mm['pred']}`{oov_note}; why: {mm['reason']}"
                )
            lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    with open("pos_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": summary_rows, "examples": all_examples}, f, indent=2)

    print(f"Saved report -> {report_path}")
    print("Saved json   -> pos_error_analysis.json")


def parse_args():
    parser = argparse.ArgumentParser(description="POS tagger: train or evaluate pretrained models")
    parser.add_argument("--mode", choices=["train", "eval-pretrained"], default="eval-pretrained")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--report-path", type=str, default="pos_error_analysis.md")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train_full_pipeline()
    else:
        evaluate_pretrained(max_examples=args.max_examples, report_path=args.report_path)


if __name__ == "__main__":
    main()
