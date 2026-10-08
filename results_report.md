# Zero-Data Robustness Auditing: Results Report

## 1. Experimental Setup

- Seed sentences: 600; metamorphic test pairs: 2110
- Pairs per relation: MR-1: 662, MR-2: 276, MR-3: 642, MR-4: 530

| Model | Backend | Labels |
|---|---|---|
| distilbert-sst2 | huggingface | NEGATIVE, POSITIVE |
| bert-sst2 | huggingface | LABEL_0, LABEL_1 |
| roberta-twitter-sentiment | huggingface | negative, neutral, positive |

| Relation | Description |
|---|---|
| MR-1 | Lexical equivalence (synonym / intensifier substitution) |
| MR-2 | Syntactic invariance (voice conversion, punctuation / whitespace padding) |
| MR-3 | Entity swapping (names and locations) |
| MR-4 | Negation inversion (litotes / double negation) |

## 2. Results

### 2.1 Metamorphic Violation Rate (95% Wilson CI)

| Model | MR-1 | MR-2 | MR-3 | MR-4 | All |
|---|---|---|---|---|---|
| distilbert-sst2 | 0.020 [0.01, 0.03] (n=662) | 0.011 [0.00, 0.03] (n=276) | 0.034 [0.02, 0.05] (n=642) | 0.064 [0.05, 0.09] (n=530) | 0.034 [0.03, 0.04] (n=2110) |
| bert-sst2 | 0.009 [0.00, 0.02] (n=662) | 0.011 [0.00, 0.03] (n=276) | 0.020 [0.01, 0.03] (n=642) | 0.036 [0.02, 0.06] (n=530) | 0.019 [0.01, 0.03] (n=2110) |
| roberta-twitter-sentiment | 0.009 [0.00, 0.02] (n=662) | 0.004 [0.00, 0.02] (n=276) | 0.002 [0.00, 0.01] (n=642) | 0.296 [0.26, 0.34] (n=530) | 0.078 [0.07, 0.09] (n=2110) |

### 2.2 Mean Prediction Divergence (JSD, bits) and Confidence Degradation

| Model | MR-1 | MR-2 | MR-3 | MR-4 | All |
|---|---|---|---|---|---|
| distilbert-sst2 | 0.0160 / 0.003 | 0.0050 / 0.009 | 0.0098 / 0.008 | 0.0556 / 0.013 | 0.0226 / 0.008 |
| bert-sst2 | 0.0068 / 0.005 | 0.0034 / 0.008 | 0.0038 / 0.009 | 0.0300 / 0.015 | 0.0113 / 0.009 |
| roberta-twitter-sentiment | 0.0061 / 0.022 | 0.0024 / 0.009 | 0.0013 / 0.008 | 0.1634 / 0.100 | 0.0437 / 0.031 |

Cells show `MPD / mean confidence degradation` (degradation computed on non-flipped pairs).

### 2.3 MVR by Operator

| Model | adverbial_reposition | complementizer_insertion | double_negation | entity_swap | punctuation_padding | synonym_substitution | voice_conversion |
|---|---|---|---|---|---|---|---|
| distilbert-sst2 | 0.032 | 0.000 | 0.064 | 0.034 | 0.000 | 0.020 | 0.016 |
| bert-sst2 | 0.000 | 0.046 | 0.036 | 0.020 | 0.000 | 0.009 | 0.000 |
| roberta-twitter-sentiment | 0.000 | 0.000 | 0.296 | 0.002 | 0.012 | 0.009 | 0.000 |

### 2.4 MVR by Seed Category

| Model | affirmation | critique | description | neutral |
|---|---|---|---|---|
| distilbert-sst2 | 0.044 | 0.000 | 0.045 | 0.072 |
| bert-sst2 | 0.020 | 0.002 | 0.034 | 0.034 |
| roberta-twitter-sentiment | 0.184 | 0.012 | 0.069 | 0.000 |

### 2.5 Observations (auto-generated from the data)

- **distilbert-sst2**: overall MVR 0.034 (95% CI [0.03, 0.04]); highest on MR-4 (0.064), lowest on MR-2 (0.011).
- **bert-sst2**: overall MVR 0.019 (95% CI [0.01, 0.03]); highest on MR-4 (0.036), lowest on MR-1 (0.009).
- **roberta-twitter-sentiment**: overall MVR 0.078 (95% CI [0.07, 0.09]); highest on MR-4 (0.296), lowest on MR-3 (0.002).
- Ranking by overall MVR (lowest = most invariant): bert-sst2 < distilbert-sst2 < roberta-twitter-sentiment.

## 3. Figures

![MVR per model and metamorphic relation](figures/fig_mvr_heatmap.png)
*MVR per model and metamorphic relation*

![MVR per relation with 95% Wilson intervals](figures/fig_mvr_bars.png)
*MVR per relation with 95% Wilson intervals*

![Confidence drop on non-flipped pairs](figures/fig_confidence_drop.png)
*Confidence drop on non-flipped pairs*

![Prediction divergence vs. violation rate](figures/fig_mpd_vs_mvr.png)
*Prediction divergence vs. violation rate*

## 5. Violation Case Studies (highest-divergence examples)

- **distilbert-sst2 / MR-1**: "The restaurant in Seoul is clean." -> "The restaurant in Seoul is spotless." (POSITIVE 1.00 -> NEGATIVE 1.00)
- **distilbert-sst2 / MR-1**: "The hotel in Sydney is clean." -> "The hotel in Sydney is spotless." (POSITIVE 1.00 -> NEGATIVE 1.00)
- **distilbert-sst2 / MR-2**: "Mei prepared the soup." -> "The soup was prepared by Mei." (POSITIVE 0.97 -> NEGATIVE 0.79)
- **distilbert-sst2 / MR-2**: "On Monday, Fatima walked from the station to the library." -> "Fatima walked from the station to the library on Monday." (POSITIVE 0.72 -> NEGATIVE 0.51)
- **distilbert-sst2 / MR-3**: "The report was written by Priya." -> "The report was written by Mei." (POSITIVE 0.99 -> NEGATIVE 0.88)
- **distilbert-sst2 / MR-3**: "Diana sent a message to Jamal about the game." -> "Ethan sent a message to Priya about the game." (NEGATIVE 0.77 -> POSITIVE 0.99)
- **distilbert-sst2 / MR-4**: "The novel seemed satisfying to everyone." -> "The novel seemed not unsatisfying to everyone." (POSITIVE 1.00 -> NEGATIVE 1.00)
- **distilbert-sst2 / MR-4**: "The novel seemed extremely impressive to everyone." -> "The novel seemed not unremarkable to everyone." (POSITIVE 1.00 -> NEGATIVE 1.00)
- **bert-sst2 / MR-1**: "The airline in Berlin is clean." -> "The airline in Berlin is tidy." (LABEL_0 0.92 -> LABEL_1 0.96)
- **bert-sst2 / MR-1**: "The guides at the hotel were truly rude." -> "The guides at the hotel were truly discourteous." (LABEL_0 1.00 -> LABEL_1 0.70)
- **bert-sst2 / MR-2**: "I think the restaurant opens at noon." -> "I think that the restaurant opens at noon." (LABEL_1 0.58 -> LABEL_0 0.71)
- **bert-sst2 / MR-2**: "I think the hotel opens at noon." -> "I think that the hotel opens at noon." (LABEL_1 0.74 -> LABEL_0 0.54)
- **bert-sst2 / MR-3**: "On Monday, Jamal walked from the station to the clinic." -> "On Monday, Mei walked from the station to the clinic." (LABEL_0 0.65 -> LABEL_1 0.77)
- **bert-sst2 / MR-3**: "The cinema in Seoul looked clean from outside." -> "The cinema in London looked clean from outside." (LABEL_1 0.64 -> LABEL_0 0.74)
- **bert-sst2 / MR-4**: "The novel seemed satisfying to everyone." -> "The novel seemed not unsatisfying to everyone." (LABEL_1 1.00 -> LABEL_0 1.00)
- **bert-sst2 / MR-4**: "The novel seemed extremely impressive to everyone." -> "The novel seemed not unremarkable to everyone." (LABEL_1 1.00 -> LABEL_0 1.00)
- **roberta-twitter-sentiment / MR-1**: "According to Fatima, the album in Seoul was quite mediocre." -> "According to Fatima, the album in Seoul was fairly unremarkable." (negative 0.85 -> neutral 0.79)
- **roberta-twitter-sentiment / MR-1**: "Priya described the lecture as quite mediocre." -> "Priya described the lecture as fairly unremarkable." (negative 0.77 -> neutral 0.78)
- **roberta-twitter-sentiment / MR-2**: "The airline in Cairo is small." -> "The airline in Cairo is small ." (neutral 0.59 -> negative 0.49)
- **roberta-twitter-sentiment / MR-3**: "The airline in Berlin is quiet." -> "The airline in Lagos is quiet." (neutral 0.68 -> negative 0.50)
- **roberta-twitter-sentiment / MR-4**: "I thought the performance was incredibly impressive." -> "I thought the performance was not unimpressive." (positive 0.98 -> negative 0.61)
- **roberta-twitter-sentiment / MR-4**: "In my opinion, the lecture was extremely enjoyable." -> "In my opinion, the lecture was not unenjoyable." (positive 0.98 -> negative 0.71)

## 6. Threats to Validity

- Seed sentences are template-generated; coverage of natural language phenomena is limited and MVR values are not estimates of failure rates on real-world text.
- Metamorphic pairs are produced by dictionary-based operators and were not human-validated; some pairs (e.g. 'good' -> 'fine', litotes such as 'not bad') may differ slightly in intensity, so MR-1 and MR-4 test label-level invariance only.
- Neutral seeds have no intrinsic polarity; for a polarity classifier, their MVR measures output instability near the decision boundary rather than a semantic error.
- MVR depends on the seed set and master seed; see the random-seed stability analysis for its spread.
- Entity-swap pools are small and culturally uneven; MR-3 detects sensitivity to the listed tokens, not a full bias audit.

## 7. Reproducibility

- Master random seed: `1234`
- Config SHA-256 (first 12 hex chars): `041755d0ef7c`
- Generated at (UTC): 2026-10-07T03:52:17+00:00
