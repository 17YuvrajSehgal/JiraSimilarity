# Evaluation Metrics Guide (Simple Version)

This project ranks Jira issues to answer:

1. "Which old issues are most similar to this new issue?"
2. "Is this issue a duplicate of another issue?"

All ranking metrics are between `0` and `1`. Higher is better.

## Full forms of all metrics

| Short Name | Full Form |
|---|---|
| `MRR` | Mean Reciprocal Rank |
| `MAP@k` | Mean Average Precision at k |
| `Recall@k` | Recall at k |
| `Precision@k` | Precision at k |
| `NDCG@k` | Normalized Discounted Cumulative Gain at k |
| `HitRate@k` | Hit Rate at k |
| `F1` | F1 Score (harmonic mean of precision and recall) |

## What does `k` mean in `@k`?

`k` means how many top results we look at.

- `@1`: only first result
- `@3`: top 3 results
- `@5`: top 5 results
- `@10`: top 10 results

Example:

- `MAP@5` means: "How good is the ranking quality inside the first 5 results?"
- `Recall@10` means: "How many true matches did we recover in the first 10 results?"

Why we use `1, 3, 5, 10`:

- `1`: strict best-result quality
- `3`: realistic shortlist for quick triage
- `5`: moderate analyst review
- `10`: deeper search when recall matters

## Metric meaning in Jira terms

### MRR

Full form: **Mean Reciprocal Rank**.

Focus: how early the first correct match appears.

High MRR means:

- the first true duplicate/similar issue is usually very near the top.

Low MRR means:

- users need to scroll more before seeing the first useful issue.

### MAP@k

Full form: **Mean Average Precision at k**.

Focus: ranking quality across multiple relevant issues in top `k`.

High `MAP@k` means:

- in the top `k`, true related issues are not only present, but usually placed early.
- for Jira triage, this means less manual filtering and faster analyst decisions.

Low `MAP@k` means:

- relevant issues are missing or buried below irrelevant ones.

### Recall@k

Full form: **Recall at k**.

Focus: coverage.

High `Recall@k` means:

- by the first `k` results, the model found most true related/duplicate issues.

Low `Recall@k` means:

- many true matches are still missed within top `k`.

### Precision@k

Full form: **Precision at k**.

Focus: cleanliness of top `k`.

High `Precision@k` means:

- most of top `k` are actually relevant.

Low `Precision@k` means:

- many top results are noise.

Note:

- If each query has only one true duplicate, `Precision@10` cannot be very high by design.

### NDCG@k

Full form: **Normalized Discounted Cumulative Gain at k**.

Focus: ordering quality, with extra reward for early correct results.

High `NDCG@k` means:

- the ranking is close to the ideal order for user-facing lists.

Low `NDCG@k` means:

- useful items may exist, but ordering is poor.

### HitRate@k

Full form: **Hit Rate at k**.

Focus: quick success.

High `HitRate@k` means:

- for most queries, at least one useful issue appears in top `k`.

Low `HitRate@k` means:

- many queries show no useful result in top `k`.

### Threshold metrics (duplicates mode)

For duplicate classification, we pick a score cutoff (for example `0.45`, `0.55`, `0.65`):

- Precision: of predicted duplicates, how many are correct
- Recall: of true duplicates, how many were found
- F1 (F1 Score): balance between precision and recall

Tradeoff:

- lower threshold: more recall, less precision
- higher threshold: more precision, less recall

## High/medium/low quick bands (rule of thumb)

Use these as practical guidance for Jira-style retrieval:

| Metric | Low | Medium | High |
|---|---:|---:|---:|
| MRR | `< 0.10` | `0.10 - 0.25` | `> 0.25` |
| MAP@10 | `< 0.08` | `0.08 - 0.20` | `> 0.20` |
| Recall@10 | `< 0.40` | `0.40 - 0.75` | `> 0.75` |
| NDCG@10 | `< 0.20` | `0.20 - 0.35` | `> 0.35` |
| HitRate@10 | `< 0.60` | `0.60 - 0.90` | `> 0.90` |

## Ideal values for a "good" model

Theoretical ideal for all ranking metrics is `1.0`.

In practice for Jira duplicate/similarity work, use these targets:

| Metric | Practical "Good" Target | Theoretical Ideal |
|---|---:|---:|
| `MRR` | `>= 0.25` | `1.00` |
| `MAP@10` | `>= 0.20` | `1.00` |
| `Recall@10` | `>= 0.75` | `1.00` |
| `NDCG@10` | `>= 0.35` | `1.00` |
| `HitRate@10` | `>= 0.90` | `1.00` |
| `F1` (duplicates) | `>= 0.70` (at chosen threshold) | `1.00` |

Notes:

- These practical targets are dataset-dependent, so treat them as strong guidance, not fixed laws.
- For `Precision@k`, no universal target exists because it depends on how many true matches exist per query.

Always compare against:

- same dataset version
- same task (`similarity` or `duplicates`)
- same `k`
- same sample/config

Do not directly compare `MAP@5` with `MAP@10`.

## Practical reading examples

- High `MAP@5` + low `Recall@10`:
  - Top results are clean, but model misses many true matches overall.
- High `Recall@10` + low `MAP@5`:
  - Model finds many true matches, but ranking is messy.
- High `MRR` + high `HitRate@3`:
  - Good for fast triage workflows where analysts inspect only a few results.
