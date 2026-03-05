import argparse
import json
import os
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


config = {
    "seed": 42,
    "svd_path": "embeddings/svd.pt",
    "skipgram_path": "embeddings/skipgram.pt",
    "glove_txt_path": "embeddings/glove.6B.300d.txt",
    "glove_pt_path": "embeddings/glove.pt",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
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


def load_pos_data(seed, train_ratio, val_ratio):
    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    from nltk.corpus import brown

    tagged_sents = brown.tagged_sents(tagset="universal")

    filtered_sents = []
    for sent in tagged_sents:
        filtered = []
        for w, t in sent:
            if w.isalpha():
                filtered.append((w.lower(), t))
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
    val_sents = [filtered_sents[i] for i in ids[n_train : n_train + n_val]]
    test_sents = [filtered_sents[i] for i in ids[n_train + n_val :]]

    print(f"Sentences -> train: {len(train_sents)} | val: {len(val_sents)} | test: {len(test_sents)}")
    print(f"Tags ({len(all_tags)}): {all_tags}")

    return train_sents, val_sents, test_sents, tag2idx, idx2tag


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
    return acc, f1


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
            input_size=int(params["input_size"]),
            hidden_size=int(params["hidden_size"]),
            num_tags=int(params["num_tags"]),
            dropout=float(params["dropout"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])

        test_w, test_t = build_windows(test_sents, word2idx, tag2idx, int(params["window_size"]))
        test_loader = DataLoader(POSDataset(test_w, test_t), batch_size=512, shuffle=False)

        acc, f1 = evaluate(model, test_loader, emb_matrix, device)
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

    json_out = {
        "metrics": [{"embedding": e, "accuracy": a, "macro_f1": f} for e, a, f in summary_rows],
        "examples": all_examples,
    }
    with open("pos_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    print(f"Saved report -> {report_path}")
    print("Saved json   -> pos_error_analysis.json")


def parse_args():
    parser = argparse.ArgumentParser(description="POS pretrained evaluation + error analysis")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--report-path", type=str, default="pos_error_analysis.md")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_pretrained(max_examples=args.max_examples, report_path=args.report_path)


if __name__ == "__main__":
    main()
