# LLM-Based and RAG-Style Approaches

This document describes the sixth model family implemented in the project: LLM-based and RAG-style approaches.

## Why this family matters

This family adds a reasoning layer on top of retrieval.

Instead of only returning a score, it tries to answer a more practical question:

- given the retrieved candidate issues, what is the best judgment about similarity or duplication, and why?

That is important for real Jira workflows because analysts usually want:

- a ranked result
- an explanation
- a confidence-oriented judgment

## Implemented model

The current runnable model is:

- `rag-hybrid-judge`

It is implemented as a local RAG-style architecture because this environment does not currently have a real hosted LLM or local transformer model wired into the project.

That means the current system is honest about its behavior:

- retrieval is real
- reasoning is real
- explanations are real
- the "LLM-style" reasoning component is a local fallback designed to be replaced later with an actual LLM judge

## How it works

The implementation lives in:

- [src/jira_similarity/model_families/llm_rag.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/llm_rag.py)

The pipeline works like this:

1. Hybrid sparse-dense retrieval brings back the best candidate issues.
2. Pairwise lexical, dense, and metadata features are computed for each query-candidate pair.
3. A local RAG-style judge combines the retrieved evidence into a final score.
4. The judge also produces natural-language reasons explaining why the candidate looks duplicate, related, or weak.

## Why this still counts as the right sixth family

The key architectural idea of the LLM/RAG family is not only "call a remote model."

It is:

- retrieve context
- reason over that context
- produce an interpretable judgment

The current implementation already does that.

Later, when you want a true LLM-backed version, the same family can be extended so that the local judge is replaced by:

- an OpenAI or other hosted LLM duplicate judge
- an LLM-based explanation generator
- a true RAG flow that cites retrieved Jira evidence in a prompt

## Logging and visibility

This family logs:

- pipeline creation
- shared retrieval-model preparation
- evaluation progress through the engine

Use `--log-level DEBUG` when you want to inspect the reasoning-oriented workflow in more detail.

## Recommended commands

List the model:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO models --models rag-hybrid-judge
```

Run the RAG-style duplicate benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite llm-rag-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the similarity-oriented RAG comparison:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite llm-rag-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the model directly:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level DEBUG duplicates `
  --model rag-hybrid-judge `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10 `
  --threshold 0.55
```

## Why this organization is useful

This family is isolated so that:

- later hosted LLM integration does not disturb the earlier baselines
- retrieval and reasoning remain cleanly separated
- explanation-oriented evaluation can evolve independently from pure retrieval evaluation
