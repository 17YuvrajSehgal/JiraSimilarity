from __future__ import annotations

from dataclasses import dataclass
import logging

from .config import DatabaseConfig, RuntimeConfig, SourceConfig
from .engine import JiraSimilarityEngine
from .repository import BaseIssueRepository, IssueRepositoryFactory

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimilarityApplication:
    repository: BaseIssueRepository
    runtime: RuntimeConfig

    def build_engine(self) -> JiraSimilarityEngine:
        logger.info(
            "Loading issues from repository: candidate_pool_size=%s compute_device=%s",
            self.runtime.candidate_pool_size,
            self.runtime.compute_device,
        )
        documents = self.repository.load_issues(self.runtime)
        logger.info("Building JiraSimilarityEngine with %s documents", len(documents))
        engine = JiraSimilarityEngine(
            documents,
            candidate_pool_size=self.runtime.candidate_pool_size,
            compute_device=self.runtime.compute_device,
        )
        logger.info("JiraSimilarityEngine is ready")
        return engine


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
        logger.info(
            "ApplicationBuilder.build(): source_kind=%s compute_device=%s load_limit=%s",
            self._source_config.kind,
            self._runtime_config.compute_device,
            self._runtime_config.load_limit,
        )
        repository = IssueRepositoryFactory.create(
            self._source_config,
            database_config=self._database_config,
        )
        logger.debug("SimilarityApplication assembled")
        return SimilarityApplication(repository=repository, runtime=self._runtime_config)
