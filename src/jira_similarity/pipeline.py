from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
import math

from .domain import IssueDocument, PreparedIssue
from .text import jaccard_similarity, prepare_issue

logger = logging.getLogger(__name__)


def _squash(score: float, scale: float = 8.0) -> float:
    if score <= 0:
        return 0.0
    return score / (score + scale)


def _normalize_seed_score(seed_score: float) -> float:
    if seed_score <= 0:
        if seed_score < 0:
            return max(0.0, min(1.0, (seed_score + 1.0) / 2.0))
        return 0.0
    if seed_score <= 1.0:
        return seed_score
    return _squash(seed_score, scale=12.0)


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    issue_id: int
    seed_score: float


@dataclass(slots=True)
class SearchIndex:
    documents: dict[int, IssueDocument]
    prepared: dict[int, PreparedIssue]
    doc_count: int
    avg_doc_length: float
    document_frequency: Counter[str]
    postings: dict[str, dict[int, int]]
    project_index: dict[str, list[int]]
    tfidf_vectors: dict[int, dict[str, float]]
    tfidf_norms: dict[int, float]

    @classmethod
    def build(cls, documents: list[IssueDocument]) -> "SearchIndex":
        logger.info("Building search index for %s Jira issues", len(documents))
        document_map = {document.issue_id: document for document in documents}
        prepared = {document.issue_id: prepare_issue(document) for document in documents}
        doc_count = max(len(documents), 1)
        avg_doc_length = sum(item.document_length for item in prepared.values()) / doc_count
        document_frequency: Counter[str] = Counter()
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        project_index: dict[str, list[int]] = defaultdict(list)

        for issue_id, prepared_issue in prepared.items():
            for token in prepared_issue.term_frequency:
                document_frequency[token] += 1
                postings[token][issue_id] = prepared_issue.term_frequency[token]
            if prepared_issue.project_key:
                project_index[prepared_issue.project_key].append(issue_id)

        tfidf_vectors: dict[int, dict[str, float]] = {}
        tfidf_norms: dict[int, float] = {}
        for issue_id, prepared_issue in prepared.items():
            vector = cls._build_tfidf_vector(prepared_issue.term_frequency, document_frequency, doc_count)
            tfidf_vectors[issue_id] = vector
            tfidf_norms[issue_id] = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0

        index = cls(
            documents=document_map,
            prepared=prepared,
            doc_count=doc_count,
            avg_doc_length=avg_doc_length,
            document_frequency=document_frequency,
            postings=dict(postings),
            project_index=dict(project_index),
            tfidf_vectors=tfidf_vectors,
            tfidf_norms=tfidf_norms,
        )
        logger.info(
            "SearchIndex built: docs=%s vocab=%s avg_doc_len=%.1f projects=%s",
            index.doc_count,
            len(index.document_frequency),
            index.avg_doc_length,
            len(index.project_index),
        )
        return index


    @staticmethod
    def _build_tfidf_vector(
        term_frequency: Counter[str],
        document_frequency: Counter[str],
        doc_count: int,
    ) -> dict[str, float]:
        vector: dict[str, float] = {}
        for token, tf in term_frequency.items():
            if tf <= 0:
                continue
            idf = math.log((1 + doc_count) / (1 + document_frequency[token])) + 1.0
            vector[token] = (1.0 + math.log(tf)) * idf
        return vector

    def idf(self, token: str) -> float:
        df = self.document_frequency[token]
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def bm25(self, query: PreparedIssue, candidate: PreparedIssue) -> float:
        score = 0.0
        k1 = 1.2
        b = 0.75
        for token, query_count in query.term_frequency.items():
            candidate_tf = candidate.term_frequency.get(token)
            if not candidate_tf:
                continue
            idf = self.idf(token)
            numerator = candidate_tf * (k1 + 1)
            denominator = candidate_tf + k1 * (1 - b + b * candidate.document_length / self.avg_doc_length)
            score += idf * min(query_count, 3) * (numerator / denominator)
        return score

    def bm25_plus(self, query: PreparedIssue, candidate: PreparedIssue, *, delta: float = 1.0) -> float:
        score = 0.0
        k1 = 1.2
        b = 0.75
        for token, query_count in query.term_frequency.items():
            candidate_tf = candidate.term_frequency.get(token)
            if not candidate_tf:
                continue
            idf = self.idf(token)
            denominator = candidate_tf + k1 * (1 - b + b * candidate.document_length / self.avg_doc_length)
            score += idf * min(query_count, 3) * (((candidate_tf * (k1 + 1)) / denominator) + delta)
        return score

    def tfidf_cosine(self, query: PreparedIssue, candidate: PreparedIssue) -> float:
        query_vector = self._build_tfidf_vector(query.term_frequency, self.document_frequency, self.doc_count)
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0
        candidate_vector = self.tfidf_vectors.get(candidate.issue_id or -1, {})
        dot = 0.0
        if len(query_vector) <= len(candidate_vector):
            for token, weight in query_vector.items():
                dot += weight * candidate_vector.get(token, 0.0)
        else:
            for token, weight in candidate_vector.items():
                dot += weight * query_vector.get(token, 0.0)
        return dot / (query_norm * self.tfidf_norms.get(candidate.issue_id or -1, 1.0))


class CandidateGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        query: PreparedIssue,
        index: SearchIndex,
        *,
        pool_size: int,
    ) -> list[CandidateMatch]:
        raise NotImplementedError


class CandidateGeneratorFallbackMixin:
    @staticmethod
    def fallback(query: PreparedIssue, index: SearchIndex, pool_size: int) -> list[CandidateMatch]:
        if query.project_key and query.project_key in index.project_index:
            return [
                CandidateMatch(issue_id=issue_id, seed_score=0.0)
                for issue_id in index.project_index[query.project_key][:pool_size]
            ]
        return [
            CandidateMatch(issue_id=issue_id, seed_score=0.0)
            for issue_id in list(index.documents)[:pool_size]
        ]


class BM25CandidateGenerator(CandidateGenerator, CandidateGeneratorFallbackMixin):
    def generate(
        self,
        query: PreparedIssue,
        index: SearchIndex,
        *,
        pool_size: int,
    ) -> list[CandidateMatch]:
        candidate_scores: Counter[int] = Counter()
        for token, query_tf in query.term_frequency.items():
            postings = index.postings.get(token)
            if not postings:
                continue
            idf = index.idf(token)
            for issue_id, candidate_tf in postings.items():
                candidate_scores[issue_id] += idf * min(query_tf, 3) * min(candidate_tf, 3)

        if not candidate_scores:
            logger.debug(
                "BM25CandidateGenerator: no matches found, falling back to project/corpus index (pool_size=%s)",
                pool_size,
            )
            return self.fallback(query, index, pool_size)

        results = [
            CandidateMatch(issue_id=issue_id, seed_score=score)
            for issue_id, score in candidate_scores.most_common(pool_size)
        ]
        logger.debug(
            "BM25CandidateGenerator: scored=%s returning=%s top_score=%.4f",
            len(candidate_scores),
            len(results),
            results[0].seed_score if results else 0.0,
        )
        return results


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        query: PreparedIssue,
        candidate: PreparedIssue,
        index: SearchIndex,
        *,
        seed_score: float,
    ) -> dict[str, float]:
        raise NotImplementedError


class StandardFeatureExtractor(FeatureExtractor):
    def extract(
        self,
        query: PreparedIssue,
        candidate: PreparedIssue,
        index: SearchIndex,
        *,
        seed_score: float,
    ) -> dict[str, float]:
        return {
            "bm25": round(_squash(index.bm25(query, candidate)), 4),
            "bm25_plus": round(_squash(index.bm25_plus(query, candidate)), 4),
            "tfidf_cosine": round(index.tfidf_cosine(query, candidate), 4),
            "title_ngram": round(jaccard_similarity(query.title_ngrams, candidate.title_ngrams), 4),
            "description_overlap": round(
                jaccard_similarity(query.description_terms, candidate.description_terms),
                4,
            ),
            "candidate_seed": round(_normalize_seed_score(seed_score), 4),
        }


@dataclass(frozen=True, slots=True)
class RerankResult:
    score: float
    feature_scores: dict[str, float]
    reasons: tuple[str, ...]


class Reranker(ABC):
    @abstractmethod
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        raise NotImplementedError


class WeightedLinearReranker(Reranker):
    def __init__(
        self,
        *,
        weights: dict[str, float],
        clamp_zero_one: bool = True,
        explanation_threshold: float = 0.08,
    ):
        self._weights = weights
        self._clamp_zero_one = clamp_zero_one
        self._explanation_threshold = explanation_threshold

    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        score = 0.0
        for name, weight in self._weights.items():
            score += weight * feature_scores.get(name, 0.0)

        if self._clamp_zero_one:
            score = max(0.0, min(1.0, score))

        result = RerankResult(
            score=round(score, 6),
            feature_scores=feature_scores,
            reasons=self._build_reasons(feature_scores),
        )
        logger.debug(
            "WeightedLinearReranker: final_score=%.4f active_features=%s",
            result.score,
            [k for k, v in feature_scores.items() if v > 0],
        )
        return result

    def _build_reasons(self, feature_scores: dict[str, float]) -> tuple[str, ...]:
        templates = {
            "bm25": "strong lexical overlap across issue text",
            "bm25_plus": "bm25-plus found strong sparse text similarity",
            "tfidf_cosine": "tf-idf cosine similarity is high",
            "title_ngram": "title wording is very similar",
            "description_overlap": "description tokens overlap heavily",
        }
        ordered = sorted(
            (
                (name, feature_scores.get(name, 0.0), self._weights.get(name, 0.0))
                for name in feature_scores
                if self._weights.get(name, 0.0) > 0
            ),
            key=lambda item: item[1] * item[2],
            reverse=True,
        )
        reasons = [
            templates[name]
            for name, value, _ in ordered
            if value >= self._explanation_threshold and name in templates
        ]
        return tuple(reasons[:4])


@dataclass(frozen=True, slots=True)
class RetrievalPipeline:
    name: str
    candidate_generator: CandidateGenerator
    feature_extractor: FeatureExtractor
    reranker: Reranker
    max_candidate_pool_size: int | None = None


def build_pipeline_registry(
    index: SearchIndex,
    *,
    holdout_issue_ids: frozenset[int] = frozenset(),
    compute_device: str = "auto",
    requested_models: frozenset[str] | None = None,
) -> dict[str, RetrievalPipeline]:
    from .model_families.classical_supervised import build_classical_supervised_pipelines
    from .model_families.deep_pairwise import build_deep_pairwise_pipelines
    from .model_families.dense_semantic import build_dense_embedding_space, build_dense_semantic_pipelines
    from .model_families.graph_metadata import build_graph_metadata_pipelines
    from .model_families.hybrid_sparse_dense import build_hybrid_sparse_dense_pipelines
    from .model_families.language_models import build_language_model_pipelines
    from .model_families.llm_rag import build_llm_rag_pipelines
    from .model_families.sparse_lexical import build_sparse_lexical_pipelines

    def needs(any_of: frozenset[str]) -> bool:
        return requested_models is None or bool(requested_models & any_of)

    logger.info(
        "Building pipeline registry: requested_models=%s holdout_count=%s compute_device=%s",
        sorted(requested_models) if requested_models is not None else "all",
        len(holdout_issue_ids),
        compute_device,
    )

    sparse_models = frozenset({"tfidf-cosine", "bm25", "bm25-plus", "lexical"})
    classical_models = frozenset({"logreg-engineered"})
    random_dense_models = frozenset({"random-indexing-dense"})
    transformer_dense_models = frozenset({"sbert-dense"})
    dense_models = random_dense_models | transformer_dense_models
    hybrid_models = frozenset({"hybrid-sparse-dense"})
    pairwise_models = frozenset({"pairwise-neural-mlp"})
    rag_models = frozenset({"rag-hybrid-judge"})
    graph_models = frozenset({"graph-metadata-aware"})
    llm_models = frozenset({"llm-e5-large", "llm-e5-cross-reranker"})

    pipelines: dict[str, RetrievalPipeline] = {}
    if needs(sparse_models):
        logger.info("Building sparse lexical pipelines")
        pipelines.update(build_sparse_lexical_pipelines())

    if needs(classical_models):
        logger.info("Building classical supervised pipelines")
        pipelines.update(
            build_classical_supervised_pipelines(
                index,
                holdout_issue_ids=holdout_issue_ids,
                compute_device=compute_device,
            )
        )

    dense_space = None
    needs_random_dense_space = needs(random_dense_models | hybrid_models | pairwise_models | rag_models | graph_models)
    if needs_random_dense_space:
        logger.info("Training shared dense embedding space")
        dense_space = build_dense_embedding_space(index, compute_device=compute_device)

    if needs(dense_models):
        logger.info("Building dense semantic pipelines")
        pipelines.update(
            build_dense_semantic_pipelines(
                index,
                dense_space=dense_space,
                compute_device=compute_device,
                requested_models=requested_models,
            )
        )
    if needs(hybrid_models):
        logger.info("Building hybrid sparse-dense pipelines")
        pipelines.update(
            build_hybrid_sparse_dense_pipelines(index, dense_space=dense_space, compute_device=compute_device)
        )
    if needs(pairwise_models):
        logger.info("Building deep pairwise duplicate classification pipelines")
        pipelines.update(
            build_deep_pairwise_pipelines(
                index,
                dense_space=dense_space,
                holdout_issue_ids=holdout_issue_ids,
                compute_device=compute_device,
            )
        )
    if needs(rag_models):
        logger.info("Building LLM-style RAG pipelines")
        pipelines.update(build_llm_rag_pipelines(index, dense_space=dense_space, compute_device=compute_device))
    if needs(graph_models):
        logger.info("Building graph and metadata aware pipelines")
        pipelines.update(
            build_graph_metadata_pipelines(index, dense_space=dense_space, compute_device=compute_device)
        )
    if needs(llm_models):
        logger.info("Building transformer language-model pipelines")
        pipelines.update(
            build_language_model_pipelines(
                index,
                compute_device=compute_device,
                requested_models=requested_models,
            )
        )

    logger.info("Pipeline registry ready: %s", ", ".join(sorted(pipelines)))
    return pipelines
