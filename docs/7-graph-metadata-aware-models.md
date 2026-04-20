# Graph- and Metadata-Aware Models

This document describes the seventh model family implemented in the project: graph- and metadata-aware models.

## Why this family matters

Issue trackers are not just text collections.

They also contain structure:

- explicit links between issues
- duplicate relationships
- projects
- components
- versions
- priorities
- statuses

This family matters because those structures often reveal relationships that text-only models miss.

## Implemented model

The current runnable model is:

- `graph-metadata-aware`

It combines:

- hybrid sparse-dense seed retrieval
- metadata-driven seed expansion
- graph propagation over explicit issue links
- graph-aware reranking with metadata alignment features

## How it works

The implementation lives in:

- [src/jira_similarity/model_families/graph_metadata.py](/C:/workplace/JiraSimilarity/src/jira_similarity/model_families/graph_metadata.py)

The pipeline works like this:

1. Build a graph-and-metadata space from the Jira corpus.
2. Use hybrid sparse-dense retrieval to get strong initial seed candidates.
3. Add metadata-based seed scores from project, component, issue type, versions, priority, and status.
4. Propagate those seed scores through explicit issue-link and duplicate-link graph edges.
5. Rerank with a combination of sparse, dense, metadata-alignment, and graph-context signals.

## What graph signals are used

The current implementation uses:

- explicit issue links
- duplicate links
- shared project membership
- shared components
- shared affected versions
- shared fix versions
- issue type alignment
- priority alignment
- status alignment

This is intentionally practical and grounded in the data that real Jira systems often already have.

## Logging and visibility

This family logs:

- graph-space construction
- candidate generation
- graph-aware pipeline creation

Use `--log-level DEBUG` when you want to inspect graph-aware ranking behavior in more detail.

## Recommended commands

List the model:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO models --models graph-metadata-aware
```

Run the graph-metadata similarity benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite graph-metadata-similarity `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the graph-metadata duplicate benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level INFO benchmark `
  --suite graph-metadata-duplicates `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run the model directly:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity --log-level DEBUG similar `
  --model graph-metadata-aware `
  --title "Null pointer exception in payment service" `
  --description "Checkout fails when customer profile is missing an address." `
  --project APP `
  --top-k 10
```

## Why this family is a good final comparative baseline

This family gives the project a realistic structured-data baseline.

That matters because many industrial Jira datasets contain:

- richer link structure than public datasets
- reliable component and team metadata
- product and subsystem organization that should influence similarity

So this family is often one of the most practically useful comparisons when moving from public research data to real company data.
