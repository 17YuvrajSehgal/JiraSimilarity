"""results.py — Persist search, duplicate, evaluation, and benchmark results to disk.

Each command writes two complementary files:
  - A machine-readable JSON file with every field preserved
  - A human-readable Markdown summary for quick inspection

The directory layout is:
  results/
    similar/           ← 'similar' command
    duplicates/        ← 'duplicates' command
    evaluate/          ← 'evaluate' command
    benchmark/         ← 'benchmark' command
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Iterable

from .domain import IssueQuery, ModelEvaluation, SearchResult

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def _now() -> str:
    return datetime.now().strftime(_TIMESTAMP_FORMAT)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _search_result_to_dict(result: SearchResult) -> dict:
    return {
        "rank": result.rank,
        "issue_key": result.issue.issue_key,
        "project_key": result.issue.project_key,
        "issue_type": result.issue.issue_type,
        "priority": result.issue.priority,
        "status": result.issue.status,
        "title": result.issue.title,
        "score": result.score,
        "model_name": result.model_name,
        "feature_scores": result.feature_scores,
        "reasons": list(result.reasons),
    }


def _evaluation_to_dict(report: ModelEvaluation) -> dict:
    return {
        "model_name": report.model_name,
        "task": report.task,
        "queries_evaluated": report.queries_evaluated,
        "mrr": report.mrr,
        "map_at_k": report.map_at_k,
        "recall_at_k": report.recall_at_k,
        "precision_at_k": report.precision_at_k,
        "ndcg_at_k": report.ndcg_at_k,
        "hit_rate_at_k": report.hit_rate_at_k,
        "threshold_metrics": report.threshold_metrics,
    }


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_search_table(results: list[SearchResult]) -> str:
    if not results:
        return "_No results matched the query._\n"
    lines = [
        "| Rank | Key | Project | Score | Title |",
        "|------|-----|---------|-------|-------|",
    ]
    for r in results:
        title = r.issue.title[:60] + ("…" if len(r.issue.title) > 60 else "")
        lines.append(
            f"| {r.rank} | {r.issue.issue_key} | {r.issue.project_key or '-'} "
            f"| {r.score:.4f} | {title} |"
        )
    return "\n".join(lines) + "\n"


def _md_reasons(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for r in results:
        reasons = "; ".join(r.reasons) if r.reasons else "_no explanation_"
        lines.append(f"**{r.rank}. {r.issue.issue_key}** — {reasons}")
    return "\n".join(lines) + "\n"


def _md_evaluation_table(reports: list[ModelEvaluation]) -> str:
    if not reports:
        return "_No evaluations._\n"
    k_values = sorted({k for r in reports for k in r.map_at_k})

    # ── Table 1: Ranking metrics ──────────────────────────────────────
    sep = " | ".join(["---"] * len(k_values))
    map_hdr = " | ".join(f"MAP@{k}" for k in k_values)
    ndcg_hdr = " | ".join(f"NDCG@{k}" for k in k_values)
    ranking_lines = [
        f"| Model | Queries | MRR | {map_hdr} | {ndcg_hdr} |",
        f"|-------|---------|-----|{sep}|{sep}|",
    ]
    for r in reports:
        maps = " | ".join(f"{r.map_at_k.get(k, 0):.4f}" for k in k_values)
        ndcgs = " | ".join(f"{r.ndcg_at_k.get(k, 0):.4f}" for k in k_values)
        ranking_lines.append(
            f"| {r.model_name} | {r.queries_evaluated} | {r.mrr:.4f} | {maps} | {ndcgs} |"
        )

    # ── Table 2: Retrieval quality metrics ────────────────────────────
    p_hdr = " | ".join(f"P@{k}" for k in k_values)
    r_hdr = " | ".join(f"Recall@{k}" for k in k_values)
    hit_hdr = " | ".join(f"Hit@{k}" for k in k_values)
    quality_lines = [
        f"| Model | {p_hdr} | {r_hdr} | {hit_hdr} |",
        f"|-------|{sep}|{sep}|{sep}|",
    ]
    for r in reports:
        ps = " | ".join(f"{r.precision_at_k.get(k, 0):.4f}" for k in k_values)
        rs = " | ".join(f"{r.recall_at_k.get(k, 0):.4f}" for k in k_values)
        hits = " | ".join(f"{r.hit_rate_at_k.get(k, 0):.4f}" for k in k_values)
        quality_lines.append(f"| {r.model_name} | {ps} | {rs} | {hits} |")

    return (
        "### Ranking Metrics\n\n"
        + "\n".join(ranking_lines)
        + "\n\n### Retrieval Quality Metrics\n\n"
        + "\n".join(quality_lines)
        + "\n"
    )


def _md_threshold_table(reports: list[ModelEvaluation]) -> str:
    lines: list[str] = []
    for r in reports:
        if not r.threshold_metrics:
            continue
        lines.append(f"\n### `{r.model_name}` threshold metrics\n")
        lines.append("| Threshold | Precision | Recall | F1 |")
        lines.append("|-----------|-----------|--------|----|")
        for threshold, metrics in sorted(r.threshold_metrics.items()):
            lines.append(
                f"| {threshold} | {metrics.get('precision', 0):.4f} "
                f"| {metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
            )
    return "\n".join(lines) + "\n" if lines else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ResultsWriter:
    """Writes results to <results_dir>/<command>/ as JSON + Markdown files."""

    def __init__(self, results_dir: str | Path = "results"):
        self._base = Path(results_dir)
        logger.info("ResultsWriter initialised: results_dir=%s", self._base.resolve())

    # ------------------------------------------------------------------
    # similar / duplicates
    # ------------------------------------------------------------------

    def save_search(
        self,
        results: list[SearchResult],
        *,
        command: str,
        model_name: str,
        query_title: str,
    ) -> Path:
        """Save results from the 'similar' or 'duplicates' command."""
        subdir = _ensure_dir(self._base / command)
        stem = f"{_now()}_{model_name}"
        json_path = subdir / f"{stem}.json"
        md_path = subdir / f"{stem}.md"

        payload = {
            "command": command,
            "model_name": model_name,
            "query_title": query_title,
            "result_count": len(results),
            "results": [_search_result_to_dict(r) for r in results],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        md = [
            f"# {command.capitalize()} Results\n",
            f"**Model:** `{model_name}`  \n**Query:** _{query_title}_  \n"
            f"**Results returned:** {len(results)}\n",
            "## Ranked Results\n",
            _md_search_table(results),
            "## Explanations\n",
            _md_reasons(results),
        ]
        md_path.write_text("\n".join(md), encoding="utf-8")

        logger.info(
            "Results saved: command=%s model=%s results=%s path=%s",
            command,
            model_name,
            len(results),
            json_path,
        )
        return json_path

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def save_evaluation(
        self,
        reports: list[ModelEvaluation],
        *,
        task: str,
        sample_size: int | None,
    ) -> Path:
        """Save results from the 'evaluate' command."""
        subdir = _ensure_dir(self._base / "evaluate")
        stem = f"{_now()}_{task}"
        json_path = subdir / f"{stem}.json"
        md_path = subdir / f"{stem}.md"

        payload = {
            "task": task,
            "sample_size": sample_size,
            "model_count": len(reports),
            "evaluations": [_evaluation_to_dict(r) for r in reports],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        md = [
            f"# Evaluation Results — {task}\n",
            f"**Sample size:** {sample_size or 'all'}  \n"
            f"**Models evaluated:** {len(reports)}\n",
            "## Performance Summary\n",
            _md_evaluation_table(reports),
        ]
        threshold_section = _md_threshold_table(reports)
        if threshold_section:
            md.append("## Threshold Metrics\n")
            md.append(threshold_section)
        md_path.write_text("\n".join(md), encoding="utf-8")

        logger.info(
            "Evaluation results saved: task=%s models=%s path=%s",
            task,
            [r.model_name for r in reports],
            json_path,
        )
        return json_path

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------

    def save_benchmark(self, result, *, suite_name: str, sample_size: int | None) -> Path:
        """Save results from the 'benchmark' command."""
        subdir = _ensure_dir(self._base / "benchmark")
        stem = f"{_now()}_{suite_name}"
        json_path = subdir / f"{stem}.json"
        md_path = subdir / f"{stem}.md"

        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

        reports = list(result.evaluations)
        md = [
            f"# Benchmark Results — {suite_name}\n",
            f"**Task:** {result.task}  \n"
            f"**Sample size:** {sample_size or 'all'}  \n"
            f"**Models:** {', '.join(result.model_names)}\n",
            "## Performance Summary\n",
            _md_evaluation_table(reports),
        ]
        threshold_section = _md_threshold_table(reports)
        if threshold_section:
            md.append("## Threshold Metrics\n")
            md.append(threshold_section)
        md_path.write_text("\n".join(md), encoding="utf-8")

        logger.info(
            "Benchmark results saved: suite=%s task=%s models=%s path=%s",
            suite_name,
            result.task,
            list(result.model_names),
            json_path,
        )
        return json_path
