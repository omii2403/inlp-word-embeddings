# Embedding Analysis Results (Custom + PDF Examples)

Generated from running `python .\comparison.py` on March 3, 2026.

## 1) Cosine Similarity (Custom Pairs)

| Pair | SVD | Word2Vec | GloVe |
|---|---:|---:|---:|
| man - woman | 0.4791 | 0.4258 | 0.6999 |
| good - bad | 0.4964 | 0.4741 | 0.6445 |
| day - night | 0.4236 | 0.3178 | 0.6383 |
| law - court | 0.3557 | 0.3483 | 0.5095 |
| city - town | 0.3879 | 0.3780 | 0.6429 |

## 2) Analogy Results (Top-1)

| Analogy | SVD | Word2Vec | GloVe |
|---|---|---|---|
| brother:sister::son:? | daughter | brett | daughter |
| walk:walking::swim:? | johns | beverly | swimming |
| good:better::bad:? | waste | freezing | worse |
| city:country::village:? | conducting | along | villages |
| paris:france::delhi:? | treaty | pratt | india |
| king:man::queen:? | a | cement | woman |
| swim:swimming::run:? | hit | supports | running |

## 3) Task 2.1 Analogy (Top-5, Custom)

### Brother : Sister :: Son : ?
- SVD: daughter (0.50), jastrow (0.49), mary (0.48), stratford (0.47), sisters (0.47)
- Word2Vec: brett (0.52), vernon (0.47), barnumville (0.46), explorer (0.44), honors (0.44)
- GloVe: daughter (0.84), mother (0.76), wife (0.70), daughters (0.68), niece (0.65)

### Walk : Walking :: Swim : ?
- SVD: johns (0.36), swimming (0.34), hopkins (0.33), riverside (0.31), thinkers (0.31)
- Word2Vec: beverly (0.43), cincinnati (0.42), debut (0.42), camera (0.42), waters (0.41)
- GloVe: swimming (0.69), swam (0.59), swimmers (0.52), swims (0.52), biking (0.51)

### Good : Better :: Bad : ?
- SVD: waste (0.36), bigger (0.34), learn (0.30), thornburg (0.30), anything (0.30)
- Word2Vec: freezing (0.37), knows (0.36), remember (0.36), behind (0.35), intertwined (0.35)
- GloVe: worse (0.73), things (0.57), gotten (0.56), actually (0.55), getting (0.55)

## 4) Task 2.1 Analogy (PDF Examples)

### Paris : France :: Delhi : ?
- SVD: treaty (0.31), britain (0.29), germany (0.29), holland (0.29), europe (0.28)
- Word2Vec: pratt (0.46), airports (0.45), woonsocket (0.43), prevot (0.43), maximization (0.43)
- GloVe: india (0.74), pakistan (0.63), indian (0.51), punjab (0.51), kashmir (0.50)

### King : Man :: Queen : ?
- SVD: a (0.48), he (0.46), young (0.45), him (0.44), woman (0.44)
- Word2Vec: cement (0.36), rich (0.35), ellen (0.34), mary (0.33), club (0.33)
- GloVe: woman (0.66), girl (0.54), person (0.47), she (0.47), lady (0.45)

### Swim : Swimming :: Run : ?
- SVD: hit (0.42), gamut (0.42), ritchie (0.39), poked (0.38), triple (0.38)
- Word2Vec: supports (0.33), liberalism (0.31), spirits (0.31), offering (0.31), little (0.30)
- GloVe: running (0.55), runs (0.51), three (0.46), two (0.45), four (0.45)

## 5) Task 2.2 Bias Check (GloVe only)

| Profession | cos(profession, man) | cos(profession, woman) |
|---|---:|---:|
| doctor | 0.4012 | 0.4691 |
| nurse | 0.2373 | 0.4496 |
| homemaker | 0.0529 | 0.2857 |

## 6) Brief Notes

- This report now includes both your custom examples and the assignment PDF examples.
- In this run, GloVe shows more semantically coherent analogy outputs on both sets.
- Bias scores show stronger similarity with `woman` than `man` for all three tested professions in this embedding snapshot.
