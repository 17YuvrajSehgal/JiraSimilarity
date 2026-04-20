from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(slots=True)
class IssueDocument:
    issue_id: int
    issue_key: str
    project_key: str | None
    title: str
    description_text: str = ""
    issue_type: str | None = None
    priority: str | None = None
    status: str | None = None
    resolution: str | None = None
    components: tuple[str, ...] = ()
    affected_versions: tuple[str, ...] = ()
    fix_versions: tuple[str, ...] = ()
    comments: tuple[str, ...] = ()
    linked_issue_ids: tuple[int, ...] = ()
    duplicate_issue_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class IssueQuery:
    title: str
    description_text: str = ""
    project_key: str | None = None
    issue_type: str | None = None
    priority: str | None = None
    status: str | None = None
    components: tuple[str, ...] = ()
    affected_versions: tuple[str, ...] = ()
    fix_versions: tuple[str, ...] = ()
    comments: tuple[str, ...] = ()
    issue_id: int | None = None
    issue_key: str | None = None
    linked_issue_ids: tuple[int, ...] = ()
    duplicate_issue_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class PreparedIssue:
    issue_id: int | None
    issue_key: str | None
    project_key: str | None
    issue_type: str | None
    priority: str | None
    status: str | None
    weighted_terms: tuple[str, ...]
    term_frequency: Counter[str]
    document_length: int
    title_terms: frozenset[str]
    description_terms: frozenset[str]
    title_ngrams: frozenset[str]
    component_terms: frozenset[str]
    affected_version_terms: frozenset[str]
    fix_version_terms: frozenset[str]
    linked_issue_ids: frozenset[int]
    duplicate_issue_ids: frozenset[int]


@dataclass(slots=True)
class SearchResult:
    rank: int
    issue: IssueDocument
    score: float
    model_name: str
    feature_scores: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class ModelEvaluation:
    model_name: str
    task: str
    queries_evaluated: int
    mrr: float
    map_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    threshold_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

