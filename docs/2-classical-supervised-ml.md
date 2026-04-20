# Classical Supervised ML With Engineered Features

This document describes the second model family implemented in the project: classical supervised machine learning over engineered Jira issue-pair features.

## Why this family matters

This family sits between pure lexical retrieval and heavier neural models.

It is useful because it:

- learns from historical labeled links instead of relying only on fixed sparse weights
- combines text similarity with structured Jira metadata
- stays lightweight and interpretable
- gives us a strong non-neural supervised baseline before we move to dense encoders and transformers

## Implemented model

The current runnable model in this family is:

- `logreg-engineered`

It uses:

- BM25 candidate generation
- engineered pairwise features
- a trained logistic regression reranker

## What features it uses

The engineered feature set includes:

- sparse text similarity scores such as `bm25`, `bm25_plus`, and `tfidf_cosine`
- title token overlap
- title character n-gram overlap
- description overlap
- weighted-term overlap
- component overlap
- affected-version overlap
- fix-version overlap
- metadata matches for project, issue type, priority, and status
- issue-length similarity
- candidate seed score from first-stage retrieval

This design keeps the model transparent: every score comes from an understandable signal.

## How training works

The family is implemented in:

- [src/jira_similarity/model_families/classical_supervised.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/classical_supervised.py)

Training works like this:

1. Historical linked and duplicate issue pairs are treated as positive training examples.
2. BM25 candidate generation is used to mine hard negatives that look somewhat similar but are not labeled matches.
3. A logistic regression model is trained over the engineered feature vectors.
4. At query time, BM25 fetches candidates and the learned model reranks them by predicted match probability.

## Evaluation behavior

During offline evaluation, the implementation retrains the supervised model with the current query issue held out from the training pairs.

That is important because it reduces direct label leakage and gives a more realistic signal than training on every pair and testing on the same issue.

## Recommended commands

List the model:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity models --models logreg-engineered
```

Run the classical ML similarity benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity benchmark `
  --suite classical-ml-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the classical ML duplicate benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity benchmark `
  --suite classical-ml-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the model directly on one query:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity similar `
  --model logreg-engineered `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10
```

## Why this organization is useful

This family is implemented as its own module and training pipeline so that:

- feature engineering stays separate from data loading
- learned reranking stays separate from sparse retrieval
- future SVM, random-forest, or boosted-tree variants can reuse the same feature-building path
- later neural families can be compared against a clean supervised baseline instead of a one-off experiment
