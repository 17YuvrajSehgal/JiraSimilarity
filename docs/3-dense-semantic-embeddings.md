# Dense Semantic Embedding Models

This document describes the third model family implemented in the project: dense semantic embeddings.

## Why this family matters

Sparse methods are strong at exact word overlap, but they struggle when two Jira issues describe the same problem with different wording.

Dense semantic embeddings matter because they:

- move retrieval from pure term overlap toward meaning-level similarity
- help recover paraphrases and vocabulary mismatch
- provide the foundation for future sentence-transformer and bi-encoder models
- let us compare lexical, supervised, and dense approaches in a clean progression

## Implemented model

The current runnable dense model is:

- `random-indexing-dense`

This is a dependency-free dense semantic baseline built for the current environment, where transformer libraries are not installed locally.

It is not meant to be the final state-of-the-art dense model. It is meant to establish the dense family cleanly so we can later plug in stronger encoders such as SBERT or MPNet without changing the rest of the application architecture.

## How it works

The implementation lives in:

- [src/jira_similarity/model_families/dense_semantic.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/dense_semantic.py)

The current dense pipeline works like this:

1. Build deterministic sparse random index vectors for each token.
2. Train token semantic vectors by accumulating neighboring token signals across Jira issue text.
3. Mix the learned semantic vectors with a small lexical anchor.
4. Encode each Jira issue into a dense document embedding.
5. Encode the incoming query the same way.
6. Retrieve nearest issues by cosine similarity in dense vector space.

This gives us a real dense retrieval path without pulling in external ML frameworks.

## Logging and progress visibility

This family logs its internal progress through the standard application logger.

When you run with `--log-level INFO` or `--log-level DEBUG`, you will see:

- search-index build start
- sparse/classical/dense pipeline initialization
- dense semantic training pass progress
- logistic-regression training progress
- evaluation progress by model

Use `DEBUG` when you want the most visibility during experimentation.

## Recommended commands

List the dense model:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO models --models random-indexing-dense
```

Run the dense similarity benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite dense-semantic-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the dense duplicate benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite dense-semantic-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run one dense query directly:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level DEBUG similar `
  --model random-indexing-dense `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10
```

## Why this organization is useful

The dense family is isolated in its own module so that:

- future transformer encoders can reuse the same dense family slot
- candidate generation stays separate from storage and input adapters
- benchmarking remains consistent across lexical, supervised, and dense models
- logging remains consistent when stronger dense models are added later
