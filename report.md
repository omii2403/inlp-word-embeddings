# Assignment 2 Report

Name: `INLP - Spring 2026`

Date: `2026-03-05`

## 1. Overview
This report covers all required tasks:
1. Training word embeddings with SVD and Word2Vec (Skip-Gram + negative sampling)
2. Geometric/ethical analysis of SVD, Word2Vec, and pre-trained GloVe embeddings
3. POS tagging with MLP using the three embedding variants

All metrics and hyperparameters below are taken from the code and saved artifacts in this workspace.

## 2. Task 1: Training Word Vectors

### 2.1 SVD Embeddings (`svd_embeddings.py`)

#### Method
- Corpus: Brown (`nltk.corpus.brown`)
- Preprocessing:
  - lowercase
  - keep only alphabetic tokens (`token.isalpha()`)
- Build co-occurrence matrix with symmetric context window
- Convert co-occurrence matrix to PPMI
- Apply truncated SVD (`svds`) and use `U @ diag(s)` as embeddings
- Save artifacts to `embeddings/svd.pt`

#### Hyperparameters searched in code
- `min_count`: `[4]`
- `context_window`: `[2, 4]`
- `k_components`: `[100, 200, 300]`

#### Best saved hyperparameters (from `embeddings/svd.pt`)
- `min_count=4`
- `context_window=4`
- `k_components=300`
- Relative reconstruction error: `0.9153`

#### Justification
- `min_count=4` removes very rare words, reducing noise and making co-occurrence estimates more reliable.
- `context_window=4` gives broader semantic context than window 2, which usually helps global/topic-level relations in count-based models.
- `k=300` is a common embedding size and preserves more information than 100/200 while still compressing the matrix.
- Selection criterion is reconstruction error on the PPMI matrix.

## 2.2 Neural Embeddings: Word2Vec Skip-Gram (`word2vec.py`)

#### Method
- Model: Skip-Gram with negative sampling
- Negative sampling distribution: unigram^0.75
- Loss: `BCEWithLogitsLoss` over positive + negative pairs
- Optimizer: Adam
- Early stopping with patience 3

#### Hyperparameter search space
- `embedding_dim`: `[100, 200, 300]`
- `window_size`: `[2, 4]`
- `negative_k`: `[5, 10]`
- `learning_rate`: `[0.001, 0.003]`
- `batch_size`: `[256, 512]`
- Total configs: `48`

#### Best config
- `embedding_dim=100`
- `window_size=4`
- `negative_k=10`
- `learning_rate=0.003`
- `batch_size=512`
- `epochs_run=6` (early stopped)
- `best_val_loss=1.4230`

#### Justification
- `window_size=4` consistently appears in top configurations, indicating larger context helps this corpus.
- `negative_k=10` slightly improves discrimination by providing more contrastive negatives.
- `lr=0.003` converges faster and reached lower validation loss before early stopping.
- `batch_size=512` stabilizes updates and was present in top-performing settings.
- `embedding_dim=100` winning over larger dims suggests better bias-variance tradeoff for this data/training budget.

Note: `word2vec.py` preprocesses with `min_count=4`, but the saved metadata in `skipgram.pt` mentions `min_count=5`. The effective tuning results in `word2vec_tuning_results.json` are still valid and used in this report.

## 3. Task 2: Are the Embeddings Fishy?

### 3.1 Where `comparison.py` belongs
`comparison.py` is the analysis script for Section 2. It should be reported under:
- Task 2.1 Analogy tests
- Task 2.2 Bias check

It is not part of embedding training. It evaluates already trained embeddings (`svd.pt`, `skipgram.pt`) and pre-trained GloVe (`glove.pt`).

### 3.2 Analogy test (Top-5) for required three analogies
(From `embedding_analysis_results.md`, generated via `comparison.py`)

#### Paris : France :: Delhi : ?
- SVD: `treaty, britain, germany, holland, europe`
- Word2Vec: `pratt, airports, woonsocket, prevot, maximization`
- GloVe: `india, pakistan, indian, punjab, kashmir`

#### King : Man :: Queen : ?
- SVD: `a, he, young, him, woman`
- Word2Vec: `cement, rich, ellen, mary, club`
- GloVe: `woman, girl, person, she, lady`

#### Swim : Swimming :: Run : ?
- SVD: `hit, gamut, ritchie, poked, triple`
- Word2Vec: `supports, liberalism, spirits, offering, little`
- GloVe: `running, runs, three, two, four`

#### Interpretation
- GloVe shows strongest semantic/syntactic consistency on all three required analogies.
- SVD and Word2Vec capture some local associations but often miss the intended relation.
- The likely reason is corpus size/domain (Brown) and limited training setup compared with large pre-trained corpora.

### 3.3 Bias check (GloVe only)
(Required by assignment, from `comparison.py` output)

| Profession | cos(profession, man) | cos(profession, woman) |
|---|---:|---:|
| doctor | 0.4012 | 0.4691 |
| nurse | 0.2373 | 0.4496 |
| homemaker | 0.0529 | 0.2857 |

#### Interpretation
- In this snapshot, all three professions are closer to `woman` than `man`.
- This indicates the embedding reflects gendered association patterns from training data, matching known embedding bias behavior.

## 4. Task 3: POS Tagging with MLP

### 4.1 Data and setup
- Brown corpus with universal tagset (`tagset='universal'`)
- Preprocessing: lowercase + alphabetic tokens only
- Split: 80% train / 10% val / 10% test (seed 42)
- Input representation: sliding window + concatenation
- Boundary handling: PAD index with zero vector
- Embeddings kept frozen during MLP training

### 4.2 MLP architecture
- Input: `(2C+1) * emb_dim`
- Hidden layer 1: `hidden_size`
- Hidden layer 2: `hidden_size // 2`
- Activations: ReLU
- Regularization: Dropout `0.3`
- Output: logits over POS tags

### 4.3 POS hyperparameter tuning (`pos_tuning_results.json`)
Search space (per embedding type):
- `window_size`: `[1, 2]`
- `hidden_size`: `[256, 512]`
- `lr`: `[0.001, 0.003]`
- `batch_size`: `[256, 512]`
- Total: `16` per embedding, `48` overall

Best per embedding:
- GloVe: `window=1, hidden=512, lr=0.001, batch=256, dropout=0.3, best_val_loss=0.0860`
- SkipGram: `window=1, hidden=512, lr=0.001, batch=512, dropout=0.3, best_val_loss=0.1187`
- SVD: `window=1, hidden=512, lr=0.001, batch=512, dropout=0.3, best_val_loss=0.1461`

### 4.4 Test performance (`final_results.json`)
| Embedding | Accuracy | Macro-F1 |
|---|---:|---:|
| GloVe | 0.9743 | 0.9409 |
| SkipGram | 0.9629 | 0.9106 |
| SVD | 0.9527 | 0.8941 |

### 4.5 Confusion matrix (best model: GloVe)
Tag order:
`[ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRON, PRT, VERB, X]`

```text
7320 3 101 0 0 295 0 0 2 61 0
5 13950 64 10 32 5 0 6 84 14 2
129 85 5057 13 15 40 0 1 38 36 0
0 1 3 3759 2 1 0 0 0 0 0
0 28 3 2 13539 4 0 32 0 1 0
268 2 11 0 8 25301 18 1 4 265 23
0 0 0 0 0 7 681 0 0 0 0
0 20 1 0 55 2 0 4774 1 2 0
0 117 18 0 0 15 0 0 2581 5 0
83 10 14 0 0 371 0 0 0 17352 1
4 0 0 0 0 42 0 0 0 2 62
```

Main confusion trends:
- `ADJ <-> NOUN`
- `PRT -> ADP`
- `VERB -> NOUN`

These are linguistically plausible confusions under short context windows.

## 5. Error Analysis (All POS Models)
Source: `pos_error_analysis.md` (generated from `pos_tagger.py`)

### 5.1 GloVe model examples
1. `and dauntless by greentree adios in`
- `dauntless`: ADJ -> NOUN
- `adios`: NOUN -> ADV
- Reason: adjective/noun ambiguity and sparse context

2. `these responses are explicable in terms of characteristics inherent in the crisis`
- `characteristics`: NOUN -> ADJ
- `inherent`: ADJ -> NOUN
- Reason: nominal/adjectival ambiguity

3. `... called him in as a private lawyer ...`
- `in`: PRT -> ADP
- Reason: particle/preposition ambiguity

### 5.2 SkipGram model examples
1. `merciful god julia`
- `merciful`: ADJ -> NOUN (OOV)
- Reason: unseen token for this vocabulary

2. `... a chartered electrical engineer ...`
- `chartered`: VERB -> NOUN
- Reason: verb/noun ambiguity

3. `... set about maximizing ...`
- `about`: ADV -> ADP
- `maximizing`: VERB -> NOUN
- Reason: function-word ambiguity + noun bias

### 5.3 SVD model examples
1. `merciful god julia`
- `merciful`: ADJ -> NOUN (OOV)
- Reason: unseen/rare lexical item

2. `... an unwillingness to generalize and to search ...`
- `to`: PRT -> ADP
- `generalize`: VERB -> NOUN
- `search`: VERB -> NOUN
- Reason: infinitival marker ambiguity and verbal noun-like contexts

3. `these responses are explicable in terms of characteristics inherent in the crisis`
- `explicable`: ADJ -> NOUN
- Reason: adjective/noun confusion

## 6. Embedding Comparison Summary
- Pre-trained GloVe significantly outperformed SVD and SkipGram in both analogy quality and POS tagging.
- SkipGram is clearly better than SVD on downstream POS metrics.
- SVD still captures useful structure, but it is weaker for fine-grained semantic and syntactic regularities in this setup.

## 7. How to place files in final submission
Use assignment structure:
- `embeddings/svd.pt`
- `embeddings/skipgram.pt`
- `svd_embeddings.py`
- `word2vec.py`
- `pos_tagger.py`
- `README.md`
- `report.pdf` (convert this markdown to PDF)

Recommended note in README:
- `comparison.py` is the Section 2 analysis script and should be run after embeddings are available.

## 8. Reproducibility commands
```bash
python svd_embeddings.py
python word2vec.py
python comparison.py
python pos_tagger.py --max-examples 5 --report-path pos_error_analysis.md
```
