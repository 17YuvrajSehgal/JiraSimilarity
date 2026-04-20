from __future__ import annotations

from dataclasses import dataclass
import os


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _read_optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True, slots=True)
class SourceConfig:
    kind: str = "mysql"
    json_path: str | None = None

    @classmethod
    def from_env(cls) -> "SourceConfig":
        return cls(
            kind=os.getenv("JIRA_SOURCE_KIND", "mysql").strip().lower(),
            json_path=_read_optional("JIRA_JSON_PATH"),
        )

    def with_overrides(
        self,
        *,
        kind: str | None = None,
        json_path: str | None = None,
    ) -> "SourceConfig":
        return SourceConfig(
            kind=(kind or self.kind).strip().lower(),
            json_path=json_path if json_path is not None else self.json_path,
        )


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "TAWOS"
    user: str = "root"
    password: str = "root"
    charset: str = "utf8mb4"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("JIRA_DB_HOST", "127.0.0.1"),
            port=int(os.getenv("JIRA_DB_PORT", "3306")),
            database=os.getenv("JIRA_DB_NAME", "TAWOS"),
            user=os.getenv("JIRA_DB_USER", "root"),
            password=os.getenv("JIRA_DB_PASSWORD", "root"),
            charset=os.getenv("JIRA_DB_CHARSET", "utf8mb4"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    load_limit: int | None = None
    include_comments: bool = False
    max_comments_per_issue: int = 3
    candidate_pool_size: int = 250
    default_top_k: int = 10

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            load_limit=_read_int("JIRA_LOAD_LIMIT", None),
            include_comments=_read_bool("JIRA_INCLUDE_COMMENTS", False),
            max_comments_per_issue=int(os.getenv("JIRA_MAX_COMMENTS", "3")),
            candidate_pool_size=int(os.getenv("JIRA_CANDIDATE_POOL", "250")),
            default_top_k=int(os.getenv("JIRA_TOP_K", "10")),
        )
