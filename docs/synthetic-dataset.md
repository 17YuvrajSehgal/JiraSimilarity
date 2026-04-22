# Synthetic Dataset (JSON Source)

This project now includes a synthetic Jira-like dataset designed for fast experimentation without MySQL.

## What it is

- Generator script: `scripts/generate_synthetic_jira_dataset.py`
- Default generated output: `datasets/synthetic/synthetic_jira_issues.json`
- Issue records follow the same nested pattern as `datasets/synthetic/original.json`:
  - top-level `jira_id`
  - nested `metadata` object with Jira-like fields (`summary`, `project_*`, `issue_type`, `status`, `priority`, `description`, `history`, `activity`, etc.)
- Input format is directly compatible with the existing JSON source adapter in `jira_similarity`.

## Record pattern

Each synthetic issue record looks like:

```json
{
  "jira_id": "PAY-10000",
  "metadata": {
    "issue_id": 10000,
    "summary": "...",
    "project_key": "PAY",
    "issue_type": "Bug",
    "status": "Open",
    "priority": "High",
    "affects_versions": ["2026.1"],
    "components": ["payments"],
    "fix_versions": ["2026.2"],
    "description": "...",
    "related_issues": ["PAY-10001"],
    "duplicate_issues": ["PAY-10001"],
    "comments_id": ["1000001", "1000002"],
    "comments_body": "...",
    "history": [],
    "activity": [],
    "submissions": []
  }
}
```

Research extension fields remain available in `metadata.synthetic_profile` and top-level `pair_labels`.

## Why this dataset exists

The synthetic dataset is intended for:

- fast local testing when TAWOS is too large for quick iterations,
- controlled experiments across model families,
- reproducible benchmark runs with deterministic data generation.

It includes:

- high-similarity duplicate pairs,
- medium-similarity linked/related pairs,
- low-similarity hard negatives with lexical overlap,
- paraphrased duplicate variants,
- cross-project noise candidates,
- graph-relevant links for graph/metadata-aware pipelines.

## Research extension fields

In addition to core Jira-like fields used by the system, the dataset includes optional research metadata:

- `synthetic_profile` per issue (e.g., semantic cluster id and role),
- top-level `pair_labels` with explicit pairwise labels and similarity bands.

These extension fields are intentionally ignored by the current repository loader, so they are safe for experiments.

## Generate a dataset

```powershell
python scripts\generate_synthetic_jira_dataset.py `
  --output datasets\synthetic\synthetic_jira_issues.json `
  --cluster-count 30 `
  --seed 20260421
```

Notes:

- Each cluster creates 5 issues.
- `cluster-count=30` produces 150 issues.
- Generation is deterministic for a fixed seed.

## Run benchmarks against synthetic data

```powershell
.\.venv\Scripts\python.exe -m jira_similarity `
  --source json `
  --json-path datasets\synthetic\synthetic_jira_issues.json `
  --compute-device auto `
  --log-level INFO `
  benchmark `
  --task similarity `
  --models bm25 bm25-plus logreg-engineered random-indexing-dense hybrid-sparse-dense pairwise-neural-mlp rag-hybrid-judge graph-metadata-aware `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

For faster debug loops:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity `
  --source json `
  --json-path datasets\synthetic\synthetic_jira_issues.json `
  --compute-device auto `
  --load-limit 120 `
  --candidate-pool-size 80 `
  --log-level DEBUG `
  benchmark `
  --task duplicates `
  --models random-indexing-dense pairwise-neural-mlp `
  --sample-size 20 `
  --top-k-values 1 3
```
