"""Jira similarity and duplicate-detection baselines."""

from .benchmarking import BenchmarkRunner
from .bootstrap import ApplicationBuilder, SimilarityApplication
from .config import DatabaseConfig, RuntimeConfig, SourceConfig
from .domain import IssueDocument, IssueQuery, SearchResult
from .engine import JiraSimilarityEngine

__all__ = [
    "ApplicationBuilder",
    "BenchmarkRunner",
    "DatabaseConfig",
    "IssueDocument",
    "IssueQuery",
    "JiraSimilarityEngine",
    "RuntimeConfig",
    "SearchResult",
    "SimilarityApplication",
    "SourceConfig",
]
