# Word2Vec Hyperparameter Justification

## Chosen configuration
From the tuning run, the selected setting is:

- embedding_dim = **100**
- window_size = **2**
- negative_k = **10**
- learning_rate = **0.001**
- batch_size = **256**

The embedding exported in `embeddings/skipgram.pt` stores this under `best_hyperparameters`.

---

## Empirical evidence from tuning (concise)
Conceptual reasoning is primary, but the final choice is also supported by your measured validation loss.

- Total configurations evaluated: **36**
- Selection metric: **minimum validation loss** (`val_loss_best`)
- Best validation loss: **1.8826**
- Second-best validation loss: **1.9486**
- Absolute improvement over second-best: **0.0660**
- Relative improvement over second-best: **~3.39%**

Local consistency check (same dim/window/k, different batch size):
- bs=256 -> **1.8826**
- bs=512 -> **2.1879**
- bs=1024 -> **2.5850**

This confirms the chosen setting is not arbitrary; it is both conceptually sensible and empirically strongest in your sweep.

---

## Conceptual justification of each parameter

### 1) Embedding dimension = 100
Embedding dimension controls representational capacity.

- If the dimension is too small, the model cannot capture enough semantic/syntactic structure.
- If the dimension is too large, parameter count grows, optimization becomes harder, and the model can spend capacity on noise.

Choosing **100** is a balanced setting for Brown-corpus-scale training with SGNS: enough capacity for useful relations, without unnecessary complexity.

### 2) Context window size = 2
Window size controls what “context” means.

- Smaller windows emphasize local syntactic signal (nearby words, grammar-like relations).
- Larger windows include broader topical signal but also introduce more weak/noisy co-occurrences.

Choosing **2** biases learning toward cleaner local dependencies, which is often effective for SGNS when the objective is stable word-context prediction.

### 3) Negative samples (`k`) = 10
Negative sampling defines contrastive strength.

- Very small `k` may provide weak separation between true and noise contexts.
- Larger `k` usually improves discrimination by showing more incorrect contexts per update.
- Excessively large `k` increases cost and can over-focus training on negatives.

Choosing **10** gives stronger contrastive learning than low-`k` settings while remaining computationally practical.

### 4) Learning rate = 0.001
Learning rate controls update magnitude.

- Too high: unstable updates and oscillation.
- Too low: slow convergence and under-training within epoch budget.

Choosing **0.001** is a conservative, stable Adam setting for SGNS training and supports smooth convergence with early stopping.

### 5) Batch size = 256
Batch size affects gradient noise and generalization behavior.

- Smaller batches add stochasticity that can help escape sharp minima and improve generalization.
- Larger batches produce smoother gradients but may converge to less generalizable solutions under fixed training budget.

Choosing **256** gives a practical trade-off between stability, memory use, and generalization-oriented optimization dynamics.

### 6) Early stopping + tuning/final epoch split
Early stopping avoids unnecessary training once validation no longer improves.

- `tune_epochs` gives each trial a bounded budget for fair search.
- `final_epochs` gives the selected configuration a larger budget.

This setup is conceptually sound because it separates **model selection** from **final training** while controlling overfitting.

---

## Short report-ready justification
The selected SGNS configuration (100-dim embeddings, context window 2, 10 negative samples, learning rate 0.001, batch size 256) is justified by standard Word2Vec trade-offs: moderate representational capacity, low-noise local context modeling, sufficiently strong contrastive supervision, stable optimization steps, and mini-batch dynamics that support better generalization. Early stopping further ensures the model is selected at a point of best validation behavior rather than over-trained epochs. Together, these choices form a coherent and theoretically grounded parameter set for Brown-corpus Word2Vec training.
