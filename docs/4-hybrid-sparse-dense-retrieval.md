# Hybrid Sparse-Dense Retrieval

This document describes the fourth model family implemented in the project: hybrid sparse-dense retrieval.

## Why this family matters

Sparse and dense models fail in different ways.

- Sparse retrieval is strong for exact terms, identifiers, stack traces, and repeated wording.
- Dense retrieval is stronger when the same issue is described with different words.

Hybrid retrieval matters because it combines both behaviors in one pipeline, which is often the most practical production design for issue intelligence systems.

## Implemented model

The current runnable hybrid model is:

- `hybrid-sparse-dense`

It combines:

- BM25+ sparse retrieval
- dense semantic retrieval from the random-indexing embedding model
- reciprocal rank fusion (RRF) for candidate generation
- a hybrid reranker that uses both sparse and dense features

## How it works

The implementation lives in:

- [src/jira_similarity/model_families/hybrid_sparse_dense.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/hybrid_sparse_dense.py)

The hybrid pipeline works like this:

1. BM25+ generates one candidate list.
2. The dense semantic model generates another candidate list.
3. Reciprocal Rank Fusion merges the two rankings into one candidate pool.
4. The reranker scores candidates using both sparse and dense signals, including `bm25_plus`, `bm25`, `dense_cosine`, title similarity, description overlap, and fused seed score.

## Why RRF was chosen

RRF is a good first hybrid strategy because it is:

- simple
- stable
- easy to interpret
- less sensitive than raw score fusion when the component models use different score scales

That makes it a strong engineering baseline before experimenting with more advanced hybrid fusion.

## Logging and visibility

This family logs:

- shared dense-space training
- hybrid pipeline construction
- hybrid candidate-fusion progress at debug level

Use `--log-level INFO` for normal progress updates and `--log-level DEBUG` when you want detailed fusion visibility.

## Recommended commands

List the hybrid model:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO models --models hybrid-sparse-dense
```

Run the hybrid similarity benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite hybrid-sparse-dense-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the hybrid duplicate benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite hybrid-sparse-dense-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run one hybrid query directly:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level DEBUG similar `
  --model hybrid-sparse-dense `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10
```

## Why this organization is useful

This family is implemented as a separate module so that:

- sparse and dense submodels remain reusable on their own
- fusion logic stays isolated and testable
- future hybrid variants such as weighted-score fusion or cascade reranking can be added cleanly
