# Comparative Benchmark Analysis (Updated with 2026-04-22/23 Runs)

This update uses the latest benchmark outputs:

- `results/benchmark/2026-04-22_23-56-55_similarity_ad-hoc.json` (task: `similarity`)
- `results/benchmark/2026-04-23_00-02-39_duplicate_ad-hoc.json` (task: `duplicates`)

Previous comparison baseline (from the earlier report revision):

- `results/benchmark/2026-04-22_22-52-06_ad-hoc.json` (similarity)
- `results/benchmark/2026-04-22_22-55-11_ad-hoc.json` (duplicates)

Run context:

- Source: `datasets/synthetic/synthetic_jira_issues.json`
- Requested models: 13
- Evaluated models: 13
- `sample_size=100`
- Queries evaluated: `90` (similarity), `60` (duplicates)
- `top_k_values = [1, 3, 5, 10]`

## Improvement check (short answer)

Overall, **no**.  
The latest run shows a major regression in supervised rerankers (`logreg-engineered`, `pairwise-neural-mlp`).  
Graph/lexical/LLM families are mostly stable, with a few small improvements.

## Executive summary (k=5 and k=10)

1. `graph-metadata-aware` is now the strongest similarity model at both `k=5` and `k=10`.
2. For duplicates, `tfidf-cosine`/`bm25` are strongest at `k=5`, while `llm-e5-large` is best at `k=10`.
3. `sbert-dense` was added successfully and is competitive mid-tier, but not top-ranked yet.
4. `logreg-engineered` and `pairwise-neural-mlp` regressed sharply and are currently underperforming.

---

## Similarity task (latest)

### Top models at k=5 (ranked by MAP@5)

| Rank | Model | MAP@5 | Recall@5 | NDCG@5 | Hit@5 |
|---|---|---:|---:|---:|---:|
| 1 | graph-metadata-aware | 0.1441 | 0.4822 | 0.2721 | 0.9222 |
| 2 | tfidf-cosine | 0.1147 | 0.4239 | 0.2290 | 0.8222 |
| 3 | llm-e5-cross-reranker | 0.1145 | 0.4041 | 0.2263 | 0.8111 |
| 4 | bm25-plus | 0.1115 | 0.4128 | 0.2212 | 0.7778 |
| 5 | lexical | 0.1112 | 0.4072 | 0.2194 | 0.7667 |
| 6 | bm25 | 0.1100 | 0.4128 | 0.2207 | 0.7889 |

### Top models at k=10 (ranked by MAP@10)

| Rank | Model | MAP@10 | Recall@10 | NDCG@10 | Hit@10 |
|---|---|---:|---:|---:|---:|
| 1 | graph-metadata-aware | 0.2388 | 0.8539 | 0.4227 | 1.0000 |
| 2 | tfidf-cosine | 0.1722 | 0.6974 | 0.3350 | 0.9556 |
| 3 | llm-e5-large | 0.1719 | 0.7150 | 0.3410 | 0.9667 |
| 4 | llm-e5-cross-reranker | 0.1696 | 0.6733 | 0.3322 | 1.0000 |
| 5 | rag-hybrid-judge | 0.1659 | 0.7030 | 0.3344 | 0.9889 |
| 6 | lexical | 0.1648 | 0.6789 | 0.3241 | 0.9556 |

---

## Duplicates task (latest)

### Top models at k=5 (ranked by MAP@5)

| Rank | Model | MAP@5 | Recall@5 | NDCG@5 | Hit@5 |
|---|---|---:|---:|---:|---:|
| 1 | tfidf-cosine | 0.2067 | 0.9000 | 0.3698 | 0.9000 |
| 2 | bm25 | 0.2056 | 0.8833 | 0.3652 | 0.8833 |
| 3 | bm25-plus | 0.1992 | 0.8667 | 0.3561 | 0.8667 |
| 4 | llm-e5-large | 0.1989 | 0.8167 | 0.3449 | 0.8167 |
| 5 | lexical | 0.1942 | 0.8500 | 0.3482 | 0.8500 |
| 6 | graph-metadata-aware | 0.1925 | 0.8000 | 0.3372 | 0.8000 |

### Top models at k=10 (ranked by MAP@10)

| Rank | Model | MAP@10 | Recall@10 | NDCG@10 | Hit@10 |
|---|---|---:|---:|---:|---:|
| 1 | llm-e5-large | 0.2255 | 1.0000 | 0.4063 | 1.0000 |
| 2 | graph-metadata-aware | 0.2215 | 1.0000 | 0.4042 | 1.0000 |
| 3 | bm25 | 0.2215 | 1.0000 | 0.4033 | 1.0000 |
| 4 | tfidf-cosine | 0.2203 | 1.0000 | 0.4024 | 1.0000 |
| 5 | llm-e5-cross-reranker | 0.2196 | 1.0000 | 0.4010 | 1.0000 |
| 6 | bm25-plus | 0.2176 | 1.0000 | 0.3999 | 1.0000 |

---

## Change vs previous run (k=5 and k=10 MAP)

### Similarity deltas

| Model | MAP@5 (old -> new) | Delta | MAP@10 (old -> new) | Delta |
|---|---:|---:|---:|---:|
| logreg-engineered | 0.3180 -> 0.0006 | **-0.3174** | 0.3982 -> 0.0013 | **-0.3969** |
| pairwise-neural-mlp | 0.2479 -> 0.1053 | **-0.1426** | 0.3051 -> 0.1392 | **-0.1659** |
| graph-metadata-aware | 0.1456 -> 0.1441 | -0.0015 | 0.2358 -> 0.2388 | **+0.0030** |
| tfidf-cosine | 0.1175 -> 0.1147 | -0.0028 | 0.1766 -> 0.1722 | -0.0044 |
| llm-e5-large | n/a -> 0.1088 | n/a | 0.1719 -> 0.1719 | ~0.0000 |
| llm-e5-cross-reranker | 0.1145 -> 0.1145 | ~0.0000 | 0.1696 -> 0.1696 | ~0.0000 |

### Duplicates deltas

| Model | MAP@5 (old -> new) | Delta | MAP@10 (old -> new) | Delta |
|---|---:|---:|---:|---:|
| logreg-engineered | 0.2428 -> 0.0000 | **-0.2428** | 0.2860 -> 0.0021 | **-0.2839** |
| pairwise-neural-mlp | 0.2453 -> 0.0172 | **-0.2281** | 0.2814 -> 0.0410 | **-0.2404** |
| bm25 | 0.1875 -> 0.2056 | **+0.0181** | 0.2213 -> 0.2215 | +0.0002 |
| tfidf-cosine | 0.2100 -> 0.2067 | -0.0033 | 0.2211 -> 0.2203 | -0.0008 |
| graph-metadata-aware | 0.1900 -> 0.1925 | **+0.0025** | n/a -> 0.2215 | n/a |
| llm-e5-large | 0.1989 -> 0.1989 | ~0.0000 | 0.2255 -> 0.2255 | ~0.0000 |
| llm-e5-cross-reranker | n/a -> 0.1856 | n/a | 0.2196 -> 0.2196 | ~0.0000 |

---

## Interpretation

1. The model-family upgrades did not produce a net gain in this run because the supervised rerankers regressed heavily.
2. Non-supervised families are stable and in several cases improved slightly (`graph-metadata-aware`, `bm25` on duplicates@5).
3. LLM baselines (`llm-e5-large`, `llm-e5-cross-reranker`) are stable and currently among the most reliable top performers.
4. `sbert-dense` is functional but currently below `llm-e5-large` and leading lexical/graph models on this synthetic dataset.

## Likely cause of regression

The timing of the drop aligns with recent trainer changes in:

- `logreg-engineered`
- `pairwise-neural-mlp`

Given both collapsed together, this is likely a training-distribution/calibration issue (negative sampling, class weighting, or scoring calibration), not a retrieval-index issue.

## Recommended next steps

1. Revert supervised trainers to the last known-good hyperparameters and re-run.
2. Then re-introduce new trainer changes one block at a time (ablation style) to isolate the breaking change.
3. Keep `graph-metadata-aware` + `llm-e5-large` as temporary primary baselines until supervised regressions are fixed.
