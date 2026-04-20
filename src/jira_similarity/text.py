from __future__ import annotations

from collections import Counter
import re

from .domain import IssueDocument, IssueQuery, PreparedIssue

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "with",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.lower().replace("\r", "\n")
    normalized = re.sub(r"[^a-z0-9_\-\s]", " ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def tokenize(value: str | None, *, keep_stopwords: bool = False) -> tuple[str, ...]:
    normalized = normalize_text(value)
    if not normalized:
        return ()

    tokens: list[str] = []
    for match in TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if not keep_stopwords and token in STOPWORDS:
            continue
        tokens.append(token)
    return tuple(tokens)


def normalize_label(value: str | None) -> str | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    return normalized.replace(" ", "_")


def char_ngrams(value: str | None, n: int = 3) -> frozenset[str]:
    normalized = normalize_text(value)
    if not normalized:
        return frozenset()
    if len(normalized) <= n:
        return frozenset({normalized})
    return frozenset(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def overlap_ratio(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _weighted_terms(record: IssueDocument | IssueQuery) -> tuple[str, ...]:
    title_tokens = list(tokenize(record.title))
    description_tokens = list(tokenize(record.description_text))
    component_tokens = [label for component in record.components if (label := normalize_label(component))]
    affected_tokens = [label for version in record.affected_versions if (label := normalize_label(version))]
    fix_tokens = [label for version in record.fix_versions if (label := normalize_label(version))]
    comment_tokens: list[str] = []
    for comment in record.comments:
        comment_tokens.extend(tokenize(comment))

    metadata_tokens = [
        label
        for raw_value in (record.issue_type, record.priority, record.status, record.project_key)
        if (label := normalize_label(raw_value))
    ]

    weighted = (
        title_tokens * 3
        + description_tokens
        + comment_tokens
        + component_tokens * 2
        + affected_tokens
        + fix_tokens
        + metadata_tokens * 2
    )
    return tuple(weighted)


def prepare_issue(record: IssueDocument | IssueQuery) -> PreparedIssue:
    weighted_terms = _weighted_terms(record)
    description_terms = frozenset(tokenize(record.description_text))
    component_terms = frozenset(
        label for component in record.components if (label := normalize_label(component))
    )
    affected_version_terms = frozenset(
        label for version in record.affected_versions if (label := normalize_label(version))
    )
    fix_version_terms = frozenset(
        label for version in record.fix_versions if (label := normalize_label(version))
    )

    return PreparedIssue(
        issue_id=record.issue_id,
        issue_key=record.issue_key,
        project_key=record.project_key,
        issue_type=record.issue_type,
        priority=record.priority,
        status=record.status,
        weighted_terms=weighted_terms,
        term_frequency=Counter(weighted_terms),
        document_length=max(len(weighted_terms), 1),
        title_terms=frozenset(tokenize(record.title)),
        description_terms=description_terms,
        title_ngrams=char_ngrams(record.title),
        component_terms=component_terms,
        affected_version_terms=affected_version_terms,
        fix_version_terms=fix_version_terms,
        linked_issue_ids=frozenset(record.linked_issue_ids),
        duplicate_issue_ids=frozenset(record.duplicate_issue_ids),
    )

