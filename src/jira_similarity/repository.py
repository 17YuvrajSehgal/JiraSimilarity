from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Any

from .config import DatabaseConfig, RuntimeConfig, SourceConfig
from .domain import IssueDocument

logger = logging.getLogger(__name__)


def _split_grouped_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.split("||") if part)


def _split_grouped_ints(value: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in _split_grouped_values(value))


def _read_text_tuple(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        if "||" in raw_value:
            return _split_grouped_values(raw_value)
        stripped = raw_value.strip()
        return (stripped,) if stripped else ()
    if isinstance(raw_value, (list, tuple, set)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
        return tuple(values)
    raise TypeError(f"Unsupported text collection value: {type(raw_value)!r}")


def _read_int_tuple(raw_value: Any) -> tuple[int, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        return _split_grouped_ints(raw_value)
    if isinstance(raw_value, (list, tuple, set)):
        values = [int(item) for item in raw_value]
        return tuple(values)
    raise TypeError(f"Unsupported integer collection value: {type(raw_value)!r}")


def _normalize_document(document: IssueDocument) -> IssueDocument:
    return replace(
        document,
        components=tuple(sorted(set(document.components))),
        affected_versions=tuple(sorted(set(document.affected_versions))),
        fix_versions=tuple(sorted(set(document.fix_versions))),
        comments=tuple(document.comments),
        linked_issue_ids=tuple(sorted(set(document.linked_issue_ids))),
        duplicate_issue_ids=tuple(sorted(set(document.duplicate_issue_ids))),
    )


def _expand_reverse_links(documents: list[IssueDocument]) -> list[IssueDocument]:
    logger.debug("Expanding reverse issue links across %s documents", len(documents))
    duplicate_neighbors: dict[int, set[int]] = defaultdict(set)
    linked_neighbors: dict[int, set[int]] = defaultdict(set)
    for document in documents:
        for issue_id in document.linked_issue_ids:
            linked_neighbors[document.issue_id].add(issue_id)
            linked_neighbors[issue_id].add(document.issue_id)
        for issue_id in document.duplicate_issue_ids:
            duplicate_neighbors[document.issue_id].add(issue_id)
            duplicate_neighbors[issue_id].add(document.issue_id)

    total_linked = sum(len(v) for v in linked_neighbors.values())
    total_dupes = sum(len(v) for v in duplicate_neighbors.values())
    logger.debug(
        "Reverse-link expansion complete: linked_pairs=%s duplicate_pairs=%s",
        total_linked // 2,
        total_dupes // 2,
    )

    normalized_documents: list[IssueDocument] = []
    for document in documents:
        normalized_documents.append(
            _normalize_document(
                replace(
                    document,
                    linked_issue_ids=tuple(
                        sorted(linked_neighbors.get(document.issue_id, set(document.linked_issue_ids)))
                    ),
                    duplicate_issue_ids=tuple(
                        sorted(duplicate_neighbors.get(document.issue_id, set(document.duplicate_issue_ids)))
                    ),
                )
            )
        )
    return normalized_documents


def _document_from_mapping(payload: dict[str, Any]) -> IssueDocument:
    issue_id = payload.get("issue_id", payload.get("id"))
    issue_key = payload.get("issue_key", payload.get("key"))
    title = payload.get("title")

    if issue_id is None:
        raise ValueError("Each issue record must contain 'issue_id' or 'id'.")
    if issue_key is None:
        raise ValueError("Each issue record must contain 'issue_key' or 'key'.")
    if title is None:
        raise ValueError("Each issue record must contain 'title'.")

    return _normalize_document(
        IssueDocument(
            issue_id=int(issue_id),
            issue_key=str(issue_key),
            project_key=payload.get("project_key"),
            title=str(title),
            description_text=str(payload.get("description_text", payload.get("description", ""))),
            issue_type=payload.get("issue_type"),
            priority=payload.get("priority"),
            status=payload.get("status"),
            resolution=payload.get("resolution"),
            components=_read_text_tuple(payload.get("components")),
            affected_versions=_read_text_tuple(payload.get("affected_versions")),
            fix_versions=_read_text_tuple(payload.get("fix_versions")),
            comments=_read_text_tuple(payload.get("comments")),
            linked_issue_ids=_read_int_tuple(payload.get("linked_issue_ids")),
            duplicate_issue_ids=_read_int_tuple(payload.get("duplicate_issue_ids")),
        )
    )


class BaseIssueRepository(ABC):
    @abstractmethod
    def load_issues(self, runtime: RuntimeConfig) -> list[IssueDocument]:
        raise NotImplementedError


class JsonIssueRepository(BaseIssueRepository):
    def __init__(self, json_path: str):
        self._json_path = Path(json_path)
        logger.debug("JsonIssueRepository initialised: path=%s", json_path)

    def load_issues(self, runtime: RuntimeConfig) -> list[IssueDocument]:
        logger.info("Loading issues from JSON file: %s", self._json_path)
        if not self._json_path.exists():
            logger.error("JSON source file not found: %s", self._json_path)
            raise FileNotFoundError(f"JSON source file does not exist: {self._json_path}")

        payload = json.loads(self._json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_issues = payload.get("issues")
        else:
            raw_issues = payload

        if not isinstance(raw_issues, list):
            raise ValueError("JSON source must be a list of issues or an object with an 'issues' list.")

        logger.debug("JSON payload contains %s raw issue records", len(raw_issues))
        documents = [_document_from_mapping(item) for item in raw_issues]
        if runtime.load_limit is not None:
            logger.info("Applying load_limit=%s (original count=%s)", runtime.load_limit, len(documents))
            documents = documents[: runtime.load_limit]
        logger.info("JSON repository loaded %s issues", len(documents))
        return _expand_reverse_links(documents)


class MySQLIssueRepository(BaseIssueRepository):
    def __init__(self, config: DatabaseConfig):
        self._config = config
        logger.debug(
            "MySQLIssueRepository initialised: host=%s port=%s database=%s user=%s",
            config.host,
            config.port,
            config.database,
            config.user,
        )

    def _connect(self):
        logger.debug(
            "Opening MySQL connection: host=%s port=%s database=%s",
            self._config.host,
            self._config.port,
            self._config.database,
        )
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError(
                "PyMySQL is not installed. Install it with 'pip install -e .[mysql]' before using the MySQL adapter."
            ) from exc

        return pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            charset=self._config.charset,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def load_issues(self, runtime: RuntimeConfig) -> list[IssueDocument]:
        logger.info(
            "Loading issues from MySQL: host=%s database=%s load_limit=%s include_comments=%s",
            self._config.host,
            self._config.database,
            runtime.load_limit,
            runtime.include_comments,
        )
        documents = self._load_base_issues(runtime.load_limit)
        if runtime.include_comments and documents:
            logger.info(
                "Loading comments for %s issues (max_per_issue=%s)",
                len(documents),
                runtime.max_comments_per_issue,
            )
            comment_map = self._load_comments(runtime.max_comments_per_issue)
            logger.info("Comments loaded for %s issues", len(comment_map))
            documents = [
                replace(issue, comments=comment_map.get(issue.issue_id, ()))
                for issue in documents
            ]
        logger.info("MySQL repository loaded %s issues total", len(documents))
        return _expand_reverse_links(documents)

    def _load_base_issues(self, load_limit: int | None) -> list[IssueDocument]:
        logger.debug("Querying base issues from MySQL: load_limit=%s", load_limit)
        limit_clause = "LIMIT %s" if load_limit else ""
        params: list[object] = [load_limit] if load_limit else []
        query = f"""
            SELECT
                i.ID AS issue_id,
                i.Issue_Key AS issue_key,
                p.Project_Key AS project_key,
                COALESCE(i.Title, '') AS title,
                COALESCE(i.Description_Text, i.Description, '') AS description_text,
                i.Type AS issue_type,
                i.Priority AS priority,
                i.Status AS status,
                i.Resolution AS resolution,
                component_data.components AS components,
                affected_data.affected_versions AS affected_versions,
                fix_data.fix_versions AS fix_versions,
                link_data.linked_issue_ids AS linked_issue_ids,
                link_data.duplicate_issue_ids AS duplicate_issue_ids
            FROM Issue i
            JOIN Project p ON p.ID = i.Project_ID
            LEFT JOIN (
                SELECT
                    ic.Issue_ID,
                    GROUP_CONCAT(DISTINCT c.Name ORDER BY c.Name SEPARATOR '||') AS components
                FROM Issue_Component ic
                JOIN Component c ON c.ID = ic.Component_ID
                GROUP BY ic.Issue_ID
            ) AS component_data ON component_data.Issue_ID = i.ID
            LEFT JOIN (
                SELECT
                    av.Issue_ID,
                    GROUP_CONCAT(DISTINCT v.Name ORDER BY v.Name SEPARATOR '||') AS affected_versions
                FROM Affected_Version av
                JOIN Version v ON v.ID = av.Affected_Version_ID
                GROUP BY av.Issue_ID
            ) AS affected_data ON affected_data.Issue_ID = i.ID
            LEFT JOIN (
                SELECT
                    fv.Issue_ID,
                    GROUP_CONCAT(DISTINCT v.Name ORDER BY v.Name SEPARATOR '||') AS fix_versions
                FROM Fix_Version fv
                JOIN Version v ON v.ID = fv.Fix_Version_ID
                GROUP BY fv.Issue_ID
            ) AS fix_data ON fix_data.Issue_ID = i.ID
            LEFT JOIN (
                SELECT
                    Issue_ID,
                    GROUP_CONCAT(DISTINCT Target_Issue_ID ORDER BY Target_Issue_ID SEPARATOR '||') AS linked_issue_ids,
                    GROUP_CONCAT(
                        DISTINCT CASE
                            WHEN LOWER(Name) = 'duplicate' THEN Target_Issue_ID
                            ELSE NULL
                        END ORDER BY Target_Issue_ID SEPARATOR '||'
                    ) AS duplicate_issue_ids
                FROM Issue_Link
                GROUP BY Issue_ID
            ) AS link_data ON link_data.Issue_ID = i.ID
            ORDER BY i.ID
            {limit_clause}
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

        logger.info("MySQL query returned %s issue rows", len(rows))
        return [
            _normalize_document(
                IssueDocument(
                    issue_id=row["issue_id"],
                    issue_key=row["issue_key"],
                    project_key=row["project_key"],
                    title=row["title"],
                    description_text=row["description_text"],
                    issue_type=row["issue_type"],
                    priority=row["priority"],
                    status=row["status"],
                    resolution=row["resolution"],
                    components=_split_grouped_values(row["components"]),
                    affected_versions=_split_grouped_values(row["affected_versions"]),
                    fix_versions=_split_grouped_values(row["fix_versions"]),
                    linked_issue_ids=_split_grouped_ints(row["linked_issue_ids"]),
                    duplicate_issue_ids=_split_grouped_ints(row["duplicate_issue_ids"]),
                )
            )
            for row in rows
        ]

    def _load_comments(self, max_comments_per_issue: int) -> dict[int, tuple[str, ...]]:
        logger.debug("Querying comments from MySQL: max_per_issue=%s", max_comments_per_issue)
        query = """
            SELECT Issue_ID AS issue_id, COALESCE(Comment_Text, Comment, '') AS comment_text
            FROM Comment
            WHERE Comment_Text IS NOT NULL OR Comment IS NOT NULL
            ORDER BY Issue_ID, Creation_Date
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        logger.debug("Fetched %s raw comment rows from MySQL", len(rows))
        grouped: dict[int, list[str]] = defaultdict(list)
        for row in rows:
            comments = grouped[row["issue_id"]]
            if len(comments) < max_comments_per_issue:
                comments.append(row["comment_text"])
        result = {issue_id: tuple(comments) for issue_id, comments in grouped.items()}
        logger.debug(
            "Comments grouped: %s issues have at least one comment (cap=%s per issue)",
            len(result),
            max_comments_per_issue,
        )
        return result


class JiraApiIssueRepository(BaseIssueRepository):
    def load_issues(self, runtime: RuntimeConfig) -> list[IssueDocument]:
        logger.warning(
            "JiraApiIssueRepository.load_issues() called but Jira API adapter is not yet implemented."
        )
        raise NotImplementedError(
            "A Jira API adapter has not been implemented yet, but the repository interface is designed to support it."
        )


class IssueRepositoryFactory:
    @staticmethod
    def create(
        source_config: SourceConfig,
        *,
        database_config: DatabaseConfig | None = None,
    ) -> BaseIssueRepository:
        logger.info("Creating issue repository: source_kind=%s", source_config.kind)
        if source_config.kind == "mysql":
            repo = MySQLIssueRepository(database_config or DatabaseConfig.from_env())
            logger.info("Using MySQLIssueRepository")
            return repo
        if source_config.kind == "json":
            if not source_config.json_path:
                raise ValueError("JSON source selected but no JSON path was provided.")
            repo = JsonIssueRepository(source_config.json_path)
            logger.info("Using JsonIssueRepository: path=%s", source_config.json_path)
            return repo
        if source_config.kind == "jira_api":
            logger.warning("Using JiraApiIssueRepository (not yet implemented).")
            return JiraApiIssueRepository()
        raise ValueError(
            f"Unsupported source kind '{source_config.kind}'. Supported values: mysql, json, jira_api."
        )
