# Deep Neural Architectures for Pairwise Duplicate Classification

This document describes the fifth model family implemented in the project: deep neural architectures for pairwise duplicate classification.

## Why this family matters

The earlier families focus mainly on retrieval and classical scoring.

This family matters because duplicate detection is often best handled as a pairwise classification problem:

- first retrieve a manageable candidate set
- then score each query-candidate pair with a more expressive model

That is the role of this family.

## Implemented model

The current runnable model is:

- `pairwise-neural-mlp`

It is a dependency-free neural duplicate classifier built for the current environment, where transformer libraries are not installed locally.

It is designed as a strong architectural placeholder for later Siamese encoders or cross-encoder transformers.

## How it works

The implementation lives in:

- [src/jira_similarity/model_families/deep_pairwise.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/deep_pairwise.py)

The current pipeline works like this:

1. Hybrid sparse-dense retrieval generates candidate issues.
2. Engineered pairwise features are built for each query-candidate pair.
3. The feature set includes lexical, metadata, and dense semantic signals.
4. A small multilayer perceptron (MLP) is trained on labeled positive and hard-negative issue pairs.
5. The MLP outputs a duplicate probability used for reranking and thresholding.

## Why this is useful now

Even though this is not yet a transformer-based classifier, it gives us the right pipeline shape:

- candidate retrieval
- pairwise neural scoring
- holdout-safe training during evaluation
- detailed logging

That means we can later replace the MLP with a true Siamese or cross-encoder model without rewriting the application shell.

## Logging and progress

This family logs:

- pairwise training-set construction
- neural classifier training start and completion
- training loss progress at debug level
- evaluation progress through the engine

Use `--log-level DEBUG` when you want detailed training visibility.

## Recommended commands

List the model:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO models --models pairwise-neural-mlp
```

Run the duplicate benchmark for this family:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite deep-pairwise-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the similarity-oriented comparison:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite deep-pairwise-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the model directly:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level DEBUG duplicates `
  --model pairwise-neural-mlp `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10 `
  --threshold 0.55
```

## Why this organization is useful

This family is kept separate so that:

- pairwise neural scoring does not get mixed into retrieval-only baselines
- later transformer duplicate classifiers can reuse the same slot in the architecture
- comparisons across classical and neural rerankers stay clean
