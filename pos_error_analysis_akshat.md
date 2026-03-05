# POS Tagger Error Analysis (All Models)

## Test Metrics

| Embedding | Accuracy | Macro-F1 |
|---|---:|---:|
| glove | 0.9743 | 0.9409 |
| skipgram | 0.9629 | 0.9106 |
| svd | 0.9527 | 0.8941 |

## GLOVE Error Examples

### Example 1

Sentence: `and dauntless by greentree adios in`

Incorrect tags:
- `dauntless`: gold=`ADJ`, pred=`NOUN`; why: adjective-noun ambiguity
- `adios`: gold=`NOUN`, pred=`ADV`; why: short context window can miss syntax cues

### Example 2

Sentence: `these responses are explicable in terms of characteristics inherent in the crisis`

Incorrect tags:
- `characteristics`: gold=`NOUN`, pred=`ADJ`; why: adjective-noun ambiguity
- `inherent`: gold=`ADJ`, pred=`NOUN`; why: adjective-noun ambiguity

### Example 3

Sentence: `ernest gross leaned back in his chair and told peter marshall how secretary general dag hammarskjold had on december called him in as a private lawyer to review conduct relating to his association with the special committee on the problem of hungary`

Incorrect tags:
- `in`: gold=`PRT`, pred=`ADP`; why: particle vs preposition confusion


## SKIPGRAM Error Examples

### Example 1

Sentence: `merciful god julia`

Incorrect tags:
- `merciful`: gold=`ADJ`, pred=`NOUN` (OOV); why: likely unseen/OOV token

### Example 2

Sentence: `he is a member of the institution of electrical engineers london a registered professional engineer in connecticut and ohio and a chartered electrical engineer in great britain`

Incorrect tags:
- `chartered`: gold=`VERB`, pred=`NOUN`; why: noun-verb ambiguity

### Example 3

Sentence: `and dauntless by greentree adios in`

Incorrect tags:
- `dauntless`: gold=`ADJ`, pred=`VERB` (OOV); why: likely unseen/OOV token


## SVD Error Examples

### Example 1

Sentence: `merciful god julia`

Incorrect tags:
- `merciful`: gold=`ADJ`, pred=`NOUN` (OOV); why: likely unseen/OOV token

### Example 2

Sentence: `he is a member of the institution of electrical engineers london a registered professional engineer in connecticut and ohio and a chartered electrical engineer in great britain`

Incorrect tags:
- `chartered`: gold=`VERB`, pred=`NOUN`; why: noun-verb ambiguity

### Example 3

Sentence: `impartiality to him meant an unwillingness to generalize and to search for a synthesis`

Incorrect tags:
- `to`: gold=`PRT`, pred=`ADP`; why: particle vs preposition confusion
- `generalize`: gold=`VERB`, pred=`NOUN`; why: noun-verb ambiguity
- `to`: gold=`PRT`, pred=`ADP`; why: particle vs preposition confusion
- `search`: gold=`VERB`, pred=`NOUN`; why: noun-verb ambiguity
