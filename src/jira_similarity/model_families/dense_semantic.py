from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import logging
import math
from typing import Any

from ..compute import resolve_torch_runtime
from ..pipeline import (
    CandidateGenerator,
    CandidateGeneratorFallbackMixin,
    CandidateMatch,
    FeatureExtractor,
    RetrievalPipeline,
    RerankResult,
    Reranker,
    SearchIndex,
    StandardFeatureExtractor,
)

logger = logging.getLogger(__name__)


def _normalize_vector(values: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _dense_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


@dataclass(frozen=True, slots=True)
class DenseEmbeddingSpace:
    dimensions: int
    token_vectors: dict[str, tuple[float, ...]]
    document_vectors: dict[int, tuple[float, ...]]
    lexical_mix: float
    doc_count: int

    def encode(self, term_frequency: Counter[str], idf_lookup) -> tuple[float, ...]:
        accumulator = [0.0] * self.dimensions
        for token, frequency in term_frequency.items():
            weight = (1.0 + math.log(frequency)) * idf_lookup(token)
            token_vector = self.token_vectors.get(token)
            if token_vector is None:
                token_vector = _normalize_vector(_dense_index_vector(token, self.dimensions))
            for index, value in enumerate(token_vector):
                accumulator[index] += value * weight
        return _normalize_vector(accumulator)


@dataclass(slots=True)
class TorchDenseIndex:
    issue_ids: tuple[int, ...]
    matrix: Any
    torch: Any
    device: str

    @classmethod
    def build(
        cls,
        dense_space: DenseEmbeddingSpace,
        *,
        compute_device: str,
    ) -> "TorchDenseIndex | None":
        runtime = resolve_torch_runtime(compute_device)
        if not runtime.enabled or runtime.torch is None:
            return None
        if not dense_space.document_vectors:
            return None

        issue_ids = tuple(sorted(dense_space.document_vectors))
        matrix_values = [dense_space.document_vectors[issue_id] for issue_id in issue_ids]
        torch = runtime.torch
        matrix = torch.tensor(matrix_values, dtype=torch.float32, device=runtime.device)
        return cls(
            issue_ids=issue_ids,
            matrix=matrix,
            torch=torch,
            device=runtime.device,
        )

    def top_candidates(self, query_vector: tuple[float, ...], *, pool_size: int) -> list[CandidateMatch]:
        with self.torch.no_grad():
            query_tensor = self.torch.tensor(query_vector, dtype=self.torch.float32, device=self.device)
            scores = self.matrix @ query_tensor
            positive_mask = scores > 0
            if not positive_mask.any():
                return []

            positive_indices = self.torch.nonzero(positive_mask, as_tuple=False).squeeze(1)
            positive_scores = scores[positive_mask]
            top_count = min(pool_size, int(positive_scores.shape[0]))
            top_scores, top_positions = self.torch.topk(positive_scores, k=top_count)
            selected_indices = positive_indices[top_positions].tolist()
            selected_scores = top_scores.tolist()

        return [
            CandidateMatch(issue_id=self.issue_ids[int(index)], seed_score=float(score))
            for index, score in zip(selected_indices, selected_scores)
        ]


def _dense_index_vector(token: str, dimensions: int, active_dimensions: int = 6) -> list[float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=32).digest()
    vector = [0.0] * dimensions
    cursor = 0
    used_positions: set[int] = set()

    while len(used_positions) < active_dimensions:
        if cursor + 2 >= len(digest):
            digest = hashlib.blake2b(digest, digest_size=32).digest()
            cursor = 0
        position = int.from_bytes(digest[cursor : cursor + 2], "big") % dimensions
        sign = 1.0 if digest[cursor + 2] % 2 == 0 else -1.0
        cursor += 3
        if position in used_positions:
            continue
        vector[position] = sign
        used_positions.add(position)
    return vector


class RandomIndexingTrainer:
    def __init__(
        self,
        *,
        dimensions: int = 96,
        active_dimensions: int = 6,
        context_window: int = 2,
        passes: int = 2,
        lexical_mix: float = 0.25,
    ):
        self._dimensions = dimensions
        self._active_dimensions = active_dimensions
        self._context_window = context_window
        self._passes = passes
        self._lexical_mix = lexical_mix

    def fit(self, index: SearchIndex) -> DenseEmbeddingSpace:
        vocabulary = sorted(index.document_frequency)
        logger.info(
            "Training dense semantic embedding space with random indexing: docs=%s vocab=%s dims=%s passes=%s",
            index.doc_count,
            len(vocabulary),
            self._dimensions,
            self._passes,
        )

        index_vectors = {
            token: _dense_index_vector(token, self._dimensions, self._active_dimensions)
            for token in vocabulary
        }
        semantic_vectors = {token: [0.0] * self._dimensions for token in vocabulary}

        prepared_issues = list(index.prepared.values())
        for pass_index in range(self._passes):
            logger.info("Dense semantic training pass %s/%s started", pass_index + 1, self._passes)
            for document_position, prepared_issue in enumerate(prepared_issues, start=1):
                terms = prepared_issue.weighted_terms
                for term_index, token in enumerate(terms):
                    token_semantics = semantic_vectors[token]
                    left_bound = max(0, term_index - self._context_window)
                    right_bound = min(len(terms), term_index + self._context_window + 1)
                    for neighbor_index in range(left_bound, right_bound):
                        if neighbor_index == term_index:
                            continue
                        neighbor = terms[neighbor_index]
                        distance = abs(neighbor_index - term_index)
                        weight = 1.0 / max(distance, 1)
                        neighbor_index_vector = index_vectors[neighbor]
                        for dimension, value in enumerate(neighbor_index_vector):
                            token_semantics[dimension] += value * weight

                if document_position % 5000 == 0:
                    logger.debug(
                        "Dense semantic training progress: pass=%s documents=%s/%s",
                        pass_index + 1,
                        document_position,
                        len(prepared_issues),
                    )
            logger.info("Dense semantic training pass %s/%s finished", pass_index + 1, self._passes)

        token_vectors: dict[str, tuple[float, ...]] = {}
        for token in vocabulary:
            mixed = [
                semantic_vectors[token][dimension] + (self._lexical_mix * index_vectors[token][dimension])
                for dimension in range(self._dimensions)
            ]
            token_vectors[token] = _normalize_vector(mixed)

        dense_space = DenseEmbeddingSpace(
            dimensions=self._dimensions,
            token_vectors=token_vectors,
            document_vectors={},
            lexical_mix=self._lexical_mix,
            doc_count=index.doc_count,
        )
        document_vectors = {
            issue_id: dense_space.encode(prepared_issue.term_frequency, index.idf)
            for issue_id, prepared_issue in index.prepared.items()
        }
        logger.info("Dense semantic embedding space ready: document_vectors=%s", len(document_vectors))
        return DenseEmbeddingSpace(
            dimensions=self._dimensions,
            token_vectors=token_vectors,
            document_vectors=document_vectors,
            lexical_mix=self._lexical_mix,
            doc_count=index.doc_count,
        )


class DenseSemanticCandidateGenerator(CandidateGenerator, CandidateGeneratorFallbackMixin):
    def __init__(
        self,
        dense_space: DenseEmbeddingSpace,
        *,
        compute_device: str = "auto",
    ):
        self._dense_space = dense_space
        self._torch_index = TorchDenseIndex.build(dense_space, compute_device=compute_device)
        if self._torch_index is not None:
            logger.info("Dense semantic candidate scoring using torch on %s", self._torch_index.device)

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        query_vector = self._dense_space.encode(query.term_frequency, index.idf)
        if self._torch_index is not None:
            top_candidates = self._torch_index.top_candidates(query_vector, pool_size=pool_size)
        else:
            candidate_scores: list[CandidateMatch] = []
            for issue_id, document_vector in self._dense_space.document_vectors.items():
                score = _dense_cosine(query_vector, document_vector)
                if score > 0:
                    candidate_scores.append(CandidateMatch(issue_id=issue_id, seed_score=score))
            candidate_scores.sort(key=lambda item: item.seed_score, reverse=True)
            top_candidates = candidate_scores[:pool_size]

        if not top_candidates:
            return self.fallback(query, index, pool_size)
        logger.debug(
            "Dense semantic candidate generation complete: query_terms=%s candidates=%s top_score=%.4f",
            len(query.term_frequency),
            len(top_candidates),
            top_candidates[0].seed_score if top_candidates else 0.0,
        )
        return top_candidates


class DenseSemanticFeatureExtractor(StandardFeatureExtractor):
    def __init__(self, dense_space: DenseEmbeddingSpace):
        self._dense_space = dense_space

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        query_vector = self._dense_space.encode(query.term_frequency, index.idf)
        candidate_vector = self._dense_space.document_vectors.get(candidate.issue_id or -1)
        dense_score = _dense_cosine(query_vector, candidate_vector) if candidate_vector else 0.0
        feature_scores["dense_cosine"] = round(max(0.0, dense_score), 4)
        return feature_scores


class DenseCosineReranker(Reranker):
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        dense_score = max(0.0, min(1.0, feature_scores.get("dense_cosine", 0.0)))
        reasons: list[str] = []
        if dense_score >= 0.12:
            reasons.append("dense semantic embeddings found a close meaning-level match")
        if feature_scores.get("title_ngram", 0.0) >= 0.1:
            reasons.append("title wording is also similar")
        if feature_scores.get("description_overlap", 0.0) >= 0.08:
            reasons.append("description text overlaps as well")
        if not reasons:
            reasons.append("dense semantic similarity was the primary ranking signal")
        return RerankResult(
            score=round(dense_score, 6),
            feature_scores=feature_scores,
            reasons=tuple(reasons[:4]),
        )


def build_dense_embedding_space(
    index: SearchIndex,
    *,
    compute_device: str = "auto",
) -> DenseEmbeddingSpace:
    _ = compute_device
    return RandomIndexingTrainer().fit(index)


def build_dense_semantic_pipelines(
    index: SearchIndex,
    *,
    dense_space: DenseEmbeddingSpace | None = None,
    compute_device: str = "auto",
) -> dict[str, RetrievalPipeline]:
    effective_dense_space = dense_space or build_dense_embedding_space(index, compute_device=compute_device)
    return {
        "random-indexing-dense": RetrievalPipeline(
            name="random-indexing-dense",
            candidate_generator=DenseSemanticCandidateGenerator(
                effective_dense_space,
                compute_device=compute_device,
            ),
            feature_extractor=DenseSemanticFeatureExtractor(effective_dense_space),
            reranker=DenseCosineReranker(),
        )
    }
