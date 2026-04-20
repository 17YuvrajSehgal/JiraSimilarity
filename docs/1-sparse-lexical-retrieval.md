# Sparse Lexical Retrieval

This document describes the first model family implemented in the project: sparse lexical retrieval.

## Why this family matters

Sparse lexical retrieval is the correct first family to build because it provides:

- the strongest traditional information-retrieval baseline
- fast retrieval on large issue corpora
- high transparency and easy debugging
- a robust comparison point for all later dense, transformer, graph, and LLM approaches

It is also directly aligned with the approach family described in [various-ml-solutions.md](/C:/workplace/JiraSimilarity/docs/various-ml-solutions.md).

## Implemented models

The sparse lexical family is now implemented as explicit runnable models:

- `tfidf-cosine`
- `bm25`
- `bm25-plus`

There is also a compatibility alias:

- `lexical`

The alias is kept so older commands continue to work, but new comparisons should use the explicit sparse model names above.

## Code location

The family is implemented in:

- [src/jira_similarity/model_families/sparse_lexical.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/sparse_lexical.py)

The shared sparse statistics are built in:

- [src/jira_similarity/pipeline.py](/C:/workplace/JiraSimilarity/src/jira_similarity/pipeline.py)

## What each sparse model does

### `tfidf-cosine`

Uses TF-IDF weighted vectors and cosine similarity to compare the query issue against historical issues.

Best used as:

- a classical sparse baseline
- a sanity-check model
- a lightweight benchmark for exact and near-exact word overlap

### `bm25`

Uses BM25-style probabilistic lexical ranking.

Best used as:

- the main sparse retrieval baseline
- the standard IR benchmark for issue retrieval
- the baseline that later dense or hybrid methods must beat

### `bm25-plus`

Uses BM25+ as a variant of BM25.

Best used as:

- a comparison against plain BM25
- a retrieval method that may behave better for some document-length distributions

## How to benchmark the family

Named benchmark suites have been added:

- `sparse-lexical-similarity`
- `sparse-lexical-duplicates`

These let us compare only the sparse lexical family before introducing other approach families.

## Recommended commands

List model families:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity models --models tfidf-cosine bm25 bm25-plus
```

Run sparse lexical similarity benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity benchmark `
  --suite sparse-lexical-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run sparse lexical duplicate benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity benchmark `
  --suite sparse-lexical-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run one sparse model directly:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity similar `
  --model bm25 `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10
```

## Why this organization is useful

The sparse family is separated from other families so that:

- future dense models do not overwrite sparse logic
- model comparisons remain explicit and reproducible
- family-specific benchmarks stay clean
- later hybrid models can reuse sparse retrieval as one component instead of duplicating it

This is the first of the seven approach families, and it now has a clean, research-ready structure.
