from __future__ import annotations

from dataclasses import dataclass

from .config import DatabaseConfig, RuntimeConfig, SourceConfig
from .engine import JiraSimilarityEngine
from .repository import BaseIssueRepository, IssueRepositoryFactory


@dataclass(slots=True)
class SimilarityApplication:
    repository: BaseIssueRepository
    runtime: RuntimeConfig

    def build_engine(self) -> JiraSimilarityEngine:
        documents = self.repository.load_issues(self.runtime)
        return JiraSimilarityEngine(
            documents,
            candidate_pool_size=self.runtime.candidate_pool_size,
            compute_device=self.runtime.compute_device,
        )


class ApplicationBuilder:
    def __init__(
        self,
        *,
        source_config: SourceConfig,
        runtime_config: RuntimeConfig,
        database_config: DatabaseConfig | None = None,
    ):
        self._source_config = source_config
        self._runtime_config = runtime_config
        self._database_config = database_config

    def build(self) -> SimilarityApplication:
        repository = IssueRepositoryFactory.create(
            self._source_config,
            database_config=self._database_config,
        )
        return SimilarityApplication(repository=repository, runtime=self._runtime_config)
