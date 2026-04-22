# Comparative Benchmark Analysis (Latest Synthetic Run)

This report compares model performance from the latest benchmark outputs:

- `results/benchmark/2026-04-22_01-26-47_ad-hoc.json` (task: `similarity`)
- `results/benchmark/2026-04-22_01-28-14_ad-hoc.json` (task: `duplicates`)

Dataset context:

- Source: `datasets/synthetic/synthetic_jira_issues.json`
- Sample size requested: `100`
- Queries evaluated: `90` (similarity), `60` (duplicates)
- Top-k: `1, 3, 5, 10`

## Executive summary

1. `logreg-engineered` is the strongest overall model in this run across both tasks.
2. `pairwise-neural-mlp` is now competitive and no longer collapsed to zero, but it has threshold calibration issues in duplicate mode.
3. `graph-metadata-aware` is a strong, stable alternative with excellent recall and hit-rate.
4. Sparse baselines (`bm25`, `tfidf-cosine`, `lexical`) remain reliable and competitive, especially in duplicate retrieval coverage.
5. `random-indexing-dense` remains weakest overall in this synthetic setting.

## Similarity task comparison

Sorted by `MAP@10` (higher is better):

| Rank | Model | MRR | MAP@10 | Recall@10 | NDCG@10 | Hit@10 |
|---|---|---:|---:|---:|---:|---:|
| 1 | logreg-engineered | 0.4977 | 0.3982 | 0.9020 | 0.5703 | 1.0000 |
| 2 | pairwise-neural-mlp | 0.4123 | 0.3051 | 0.7313 | 0.4533 | 0.9778 |
| 3 | graph-metadata-aware | 0.2655 | 0.2358 | 0.8483 | 0.4186 | 1.0000 |
| 4 | tfidf-cosine | 0.2410 | 0.1766 | 0.7233 | 0.3450 | 0.9778 |
| 5 | bm25 | 0.2467 | 0.1640 | 0.6544 | 0.3209 | 0.9556 |
| 6 | rag-hybrid-judge | 0.2410 | 0.1602 | 0.6807 | 0.3254 | 0.9889 |
| 7 | lexical | 0.2356 | 0.1594 | 0.6544 | 0.3161 | 0.9556 |
| 8 | bm25-plus | 0.2123 | 0.1533 | 0.6789 | 0.3135 | 0.9444 |
| 9 | hybrid-sparse-dense | 0.2489 | 0.1511 | 0.6030 | 0.3022 | 0.9778 |
| 10 | random-indexing-dense | 0.2023 | 0.1107 | 0.5446 | 0.2524 | 0.9778 |

Key takeaways:

- `logreg-engineered` leads clearly on ranking quality (`MRR`, `MAP@10`, `NDCG@10`).
- `pairwise-neural-mlp` is second best and materially better than hybrid/sparse baselines on ranking metrics.
- `graph-metadata-aware` provides very strong recall and perfect `Hit@10` while maintaining good ranking quality.

## Duplicates task comparison

Sorted by `MAP@10`:

| Rank | Model | MRR | MAP@10 | Recall@10 | NDCG@10 | Hit@10 | Best F1 (threshold) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | logreg-engineered | 0.2860 | 0.2860 | 0.9833 | 0.4506 | 0.9833 | 0.1948 (0.45) |
| 2 | pairwise-neural-mlp | 0.2814 | 0.2814 | 0.8833 | 0.4220 | 0.8833 | 0.0000 (none at 0.45/0.55/0.65) |
| 3 | bm25 | 0.2213 | 0.2213 | 1.0000 | 0.4030 | 1.0000 | 0.1818 (0.45/0.55/0.65) |
| 4 | tfidf-cosine | 0.2211 | 0.2211 | 1.0000 | 0.4031 | 1.0000 | 0.0662 (0.45) |
| 5 | hybrid-sparse-dense | 0.2172 | 0.2172 | 0.9833 | 0.3954 | 0.9833 | 0.1943 (0.55) |
| 6 | graph-metadata-aware | 0.2170 | 0.2170 | 1.0000 | 0.4006 | 1.0000 | 0.1863 (0.65) |
| 7 | lexical | 0.2162 | 0.2162 | 1.0000 | 0.3985 | 1.0000 | 0.1818 (0.45/0.55/0.65) |
| 8 | rag-hybrid-judge | 0.2075 | 0.2075 | 0.9833 | 0.3872 | 0.9833 | 0.1788 (0.45/0.55/0.65) |
| 9 | bm25-plus | 0.1922 | 0.1922 | 1.0000 | 0.3793 | 1.0000 | 0.1818 (0.45/0.55/0.65) |
| 10 | random-indexing-dense | 0.1695 | 0.1695 | 0.9833 | 0.3519 | 0.9833 | 0.1830 (0.45) |

Key takeaways:

- `logreg-engineered` and `pairwise-neural-mlp` now lead ranking metrics for duplicate retrieval.
- `pairwise-neural-mlp` shows strong rank metrics but poor threshold behavior at current cutoffs (all F1 = 0 at `0.45/0.55/0.65`), indicating score calibration mismatch.
- Several models (`bm25`, `graph-metadata-aware`, `lexical`) achieve perfect `Recall@10` and `Hit@10`, making them strong retrieval-first choices.

## Cross-task overall ranking

Using normalized aggregate score across both tasks and metrics (`MRR`, `MAP@10`, `Recall@10`, `NDCG@10`, `Hit@10`):

1. `logreg-engineered` (9.714)
2. `graph-metadata-aware` (6.330)
3. `pairwise-neural-mlp` (5.772)
4. `tfidf-cosine` (5.156)
5. `bm25` (4.466)
6. `rag-hybrid-judge` (4.438)
7. `lexical` (4.263)
8. `hybrid-sparse-dense` (4.193)
9. `bm25-plus` (3.418)
10. `random-indexing-dense` (2.314)

## What to improve next

1. Calibrate duplicate thresholds per model, not fixed global values.
- Especially for `pairwise-neural-mlp`, sweep thresholds below `0.45` (for example `0.05` to `0.45`) and pick threshold by best validation F1.

2. Add score calibration for supervised duplicate models.
- Apply Platt scaling or isotonic calibration for `logreg-engineered` and `pairwise-neural-mlp` before thresholding.

3. Keep two deployment profiles.
- Retrieval-first: `graph-metadata-aware` or `bm25` (high `Recall@10`/`Hit@10`).
- Ranking-quality-first: `logreg-engineered` (best overall rank quality in this synthetic run).

4. Stress-test generalization.
- Repeat with multiple random seeds and larger synthetic sets to check metric stability.
- Confirm trends on real Jira data before final model decisions.
