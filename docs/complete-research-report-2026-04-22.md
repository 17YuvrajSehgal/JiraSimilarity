# Complete Research Report: Jira Issue Similarity and Duplicate Detection

Date: 2026-04-22  
Project: `JiraSimilarity`  
Prepared for: Professor review

## 1) Executive summary

This research builds and evaluates a modular system that helps teams:

1. find historical Jira issues similar to a new issue, and
2. detect likely duplicate issues early in triage.

We implemented and benchmarked **10 runnable models** across **7 model families**, with a shared evaluation framework and reproducible benchmarking pipeline.

Latest benchmark highlights on the synthetic Jira dataset:

- Best overall model in this run: **`logreg-engineered`**
- Strong alternatives: **`graph-metadata-aware`**, **`pairwise-neural-mlp`**
- Reliable lexical baselines: **`bm25`**, **`tfidf-cosine`**, **`lexical`**
- Key next improvement: **threshold calibration for `pairwise-neural-mlp` in duplicate mode**

## 2) Problem and motivation

Large Jira repositories accumulate repeated and related issues. Manual triage becomes expensive and inconsistent.

This work addresses two linked research tasks:

1. **Similarity retrieval**: Given a new issue, rank historical issues by relevance.
2. **Duplicate detection**: Determine whether the new issue is likely redundant with an existing issue.

Why this matters:

- reduces triage effort,
- reduces backlog noise from duplicates,
- improves issue assignment and prioritization,
- helps teams reuse prior investigation knowledge.

## 3) Research objectives

This project is designed as a comparative research platform, not a single-model experiment.

Primary objectives:

1. Build a clean, extensible architecture for multiple model families.
2. Compare sparse, supervised, dense, hybrid, neural pairwise, RAG-style, and graph-aware approaches under one evaluation pipeline.
3. Support practical deployment constraints:
- runs on CPU and GPU,
- supports multiple data sources (MySQL and JSON),
- logs progress for long-running experiments.
4. Prepare for scale-up from local GPU (RTX laptop) to H100-class infrastructure later.

## 4) System architecture (what was built)

The system is modular:

1. **Repository layer** (input adapters): MySQL and JSON.
2. **Preprocessing/index layer**: tokenization, sparse stats, feature preparation.
3. **Candidate generation**: retrieve likely matches quickly.
4. **Feature extraction**: lexical, dense, metadata, and graph-aware signals.
5. **Reranking/classification**: model-specific scoring.
6. **Evaluation engine**: unified metrics for similarity and duplicate tasks.

This design allows model families to share infrastructure while keeping comparisons fair.

## 5) Datasets used

### 5.1 Real dataset path

The project supports real Jira data from TAWOS through:

- MySQL adapter (`--source mysql`)
- JSON adapter (`--source json`)

### 5.2 Synthetic research dataset

To enable fast and reproducible experiments without full MySQL load, we created:

- `datasets/synthetic/synthetic_jira_issues.json`

Dataset profile in latest run:

- `cluster_count=30`
- `issue_count=150`
- directed duplicate edges: 60
- directed related/link edges: 191

Design goals of synthetic data:

- high/medium/low similarity examples,
- paraphrased duplicate variants,
- hard negatives with lexical overlap,
- metadata-rich records (project, component, versions, status, priority),
- link structure for graph-aware models.

Schema is Jira-like (`jira_id` + nested `metadata`) and compatible with loader.

## 6) Model families and implemented models

### 6.1 Sparse lexical retrieval

- `tfidf-cosine`
- `bm25`
- `bm25-plus`
- `lexical` (compatibility alias)

Purpose: fast, transparent baselines; strong exact-token retrieval.

### 6.2 Classical supervised ML (engineered features)

- `logreg-engineered`

Purpose: learn from labeled pairs using lexical + metadata features with logistic reranking.

### 6.3 Dense semantic embeddings

- `random-indexing-dense`

Purpose: semantic matching under vocabulary mismatch, lightweight dense baseline.

### 6.4 Hybrid sparse+dense retrieval

- `hybrid-sparse-dense`

Purpose: combine lexical precision and semantic recall via RRF-based fusion.

### 6.5 Deep pairwise duplicate classification

- `pairwise-neural-mlp`

Purpose: nonlinear pairwise duplicate scoring over lexical+dense+metadata features.

### 6.6 LLM/RAG-style reasoning

- `rag-hybrid-judge`

Purpose: retrieval + local reasoning style with explanation output (LLM-style architecture placeholder).

### 6.7 Graph/metadata-aware retrieval

- `graph-metadata-aware`

Purpose: exploit Jira link graph and metadata structure beyond text.

## 7) Compute, GPU, and runtime engineering

Compute-device options:

- `--compute-device auto`: CUDA if available, else CPU
- `--compute-device cuda`: request CUDA (fallback to CPU if unavailable)
- `--compute-device cpu`: force CPU

Important practical finding:

- Not all models are GPU-accelerated in current implementation.
- Sparse lexical models are CPU-only by design.
- Torch-backed paths are used where applicable.

Recent fix completed in this project:

- Supervised rerankers (`logreg-engineered`, `pairwise-neural-mlp`) were previously vulnerable to train/inference candidate-pool mismatch, which caused collapse to near-zero metrics.
- We aligned training and inference candidate pool behavior and added safeguards/tests.
- This materially improved supervised/neural model rankings in latest results.

## 8) Evaluation methodology

Tasks evaluated:

1. `similarity`
2. `duplicates`

Top-k settings:

- `k = 1, 3, 5, 10`

Meaning of `k`:

- `@k` evaluates only the first `k` ranked results.

Metrics used:

- `MRR` (Mean Reciprocal Rank)
- `MAP@k` (Mean Average Precision at k)
- `Recall@k`
- `Precision@k`
- `NDCG@k` (Normalized Discounted Cumulative Gain at k)
- `HitRate@k`
- Duplicate thresholds (`0.45`, `0.55`, `0.65`) with Precision/Recall/F1

Evaluation files (latest):

- `results/benchmark/2026-04-22_01-26-47_ad-hoc.json` (similarity)
- `results/benchmark/2026-04-22_01-28-14_ad-hoc.json` (duplicates)

## 9) Results summary (latest run)

### 9.1 Similarity task (top models by MAP@10)

1. `logreg-engineered`: MAP@10 `0.3982`, MRR `0.4977`, Recall@10 `0.9020`, NDCG@10 `0.5703`
2. `pairwise-neural-mlp`: MAP@10 `0.3051`, MRR `0.4123`, Recall@10 `0.7313`
3. `graph-metadata-aware`: MAP@10 `0.2358`, Recall@10 `0.8483`, Hit@10 `1.0000`

Observation:

- Supervised reranking currently provides the strongest ranking quality on this synthetic setup.

### 9.2 Duplicates task (top models by MAP@10)

1. `logreg-engineered`: MAP@10 `0.2860`, MRR `0.2860`, Recall@10 `0.9833`, NDCG@10 `0.4506`
2. `pairwise-neural-mlp`: MAP@10 `0.2814`, MRR `0.2814`, Recall@10 `0.8833`
3. `bm25`: MAP@10 `0.2213`, Recall@10 `1.0000`

Important nuance:

- `pairwise-neural-mlp` ranks candidates well but has threshold-calibration issues at fixed thresholds (`0.45/0.55/0.65`), where F1 is currently `0.0`.
- This indicates score scale mismatch, not ranking failure.

### 9.3 Cross-task composite ranking (normalized aggregate)

1. `logreg-engineered`
2. `graph-metadata-aware`
3. `pairwise-neural-mlp`
4. `tfidf-cosine`
5. `bm25`

## 10) Interpretation for a non-specialist reviewer

What these results mean in plain language:

1. The system is already capable of finding useful related and duplicate issues at high rates in top-10 lists.
2. Different models are good at different things:
- supervised models are strongest for ranking quality,
- graph/metadata models are strong for broad coverage,
- sparse baselines remain robust and useful.
3. The project has moved beyond proof-of-concept and now supports systematic comparative research.

## 11) Limitations

1. Current headline results are from synthetic data, not yet final real-world TAWOS-scale conclusions.
2. Some model families are still lightweight approximations (for example local dense baseline, local RAG-style judge).
3. Threshold metrics depend on score calibration; one global threshold is not optimal for all models.
4. Scaling behavior on full industrial volumes still needs dedicated H100 experiments.

## 12) Next research steps (proposed)

1. **Threshold calibration** per model (especially `pairwise-neural-mlp`):
- sweep lower thresholds and select via validation F1.
2. **Probability calibration**:
- Platt or isotonic calibration for supervised/neural duplicate scores.
3. **Real-data validation**:
- run full comparative suite on TAWOS MySQL data with fixed seeds and repeated trials.
4. **Scale-up experiments on H100**:
- measure throughput, training time, memory, and quality changes vs laptop GPU.
5. **Stronger dense/neural backbones**:
- replace random-indexing baseline with transformer encoders in same pipeline slot.

## 13) Reproducibility checklist

Environment setup:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Generate synthetic dataset:

```powershell
python scripts\generate_synthetic_jira_dataset.py `
  --output datasets\synthetic\synthetic_jira_issues.json `
  --cluster-count 30 `
  --seed 20260421
```

Run full similarity benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity `
  --source json `
  --json-path datasets\synthetic\synthetic_jira_issues.json `
  --compute-device cuda `
  --log-level INFO `
  benchmark `
  --task similarity `
  --models all `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

Run full duplicates benchmark:

```powershell
.\.venv\Scripts\python.exe -m jira_similarity `
  --source json `
  --json-path datasets\synthetic\synthetic_jira_issues.json `
  --compute-device cuda `
  --log-level INFO `
  benchmark `
  --task duplicates `
  --models all `
  --sample-size 100 `
  --top-k-values 1 3 5 10
```

## 14) Conclusion

This research successfully established a complete comparative framework for Jira issue intelligence:

- multiple model families implemented end-to-end,
- reproducible benchmarks across similarity and duplicate tasks,
- GPU/CPU-compatible runtime,
- improved supervised/neural behavior after engineering fixes,
- clear roadmap to real-data and large-GPU validation.

In its current state, the work is strong enough for academic reporting and ready for the next phase of empirical validation on full-scale real Jira data.
