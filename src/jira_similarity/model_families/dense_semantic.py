from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import logging
import math
import os
import time
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
TokenIndexEntries = tuple[tuple[int, float], ...]
_sbert_missing_warned = False

RANDOM_INDEX_DIMENSIONS = int(os.getenv("JIRA_RANDOM_INDEX_DIMS", "192"))
RANDOM_INDEX_ACTIVE_DIMS = int(os.getenv("JIRA_RANDOM_INDEX_ACTIVE_DIMS", "8"))
RANDOM_INDEX_CONTEXT_WINDOW = int(os.getenv("JIRA_RANDOM_INDEX_CONTEXT_WINDOW", "3"))
RANDOM_INDEX_PASSES = int(os.getenv("JIRA_RANDOM_INDEX_PASSES", "2"))
RANDOM_INDEX_LEXICAL_MIX = float(os.getenv("JIRA_RANDOM_INDEX_LEXICAL_MIX", "0.35"))
RANDOM_INDEX_PROGRESS_EVERY = int(os.getenv("JIRA_RANDOM_INDEX_PROGRESS_EVERY", "25000"))

SBERT_MODEL_NAME = "sbert-dense"
DEFAULT_SBERT_MODEL_ID = os.getenv("JIRA_SBERT_MODEL", "sentence-transformers/all-mpnet-base-v2")
DEFAULT_SBERT_BATCH_SIZE = int(os.getenv("JIRA_SBERT_BATCH_SIZE", "64"))
DEFAULT_SBERT_MAX_WEIGHTED_TOKENS = int(os.getenv("JIRA_SBERT_MAX_WEIGHTED_TOKENS", "320"))


def _normalize_vector(values: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _dense_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _query_cache_key(query) -> int | tuple[tuple[str, int], ...]:
    if query.issue_id is not None:
        return query.issue_id
    return tuple(sorted(query.term_frequency.items()))


def _prepared_to_text(prepared_issue, *, max_weighted_terms: int = DEFAULT_SBERT_MAX_WEIGHTED_TOKENS) -> str:
    if not prepared_issue.weighted_terms:
        return "empty issue"
    return " ".join(prepared_issue.weighted_terms[:max_weighted_terms])


def _resolve_sentence_transformer_cls() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    return SentenceTransformer


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


def _dense_index_entries(
    token: str,
    dimensions: int,
    active_dimensions: int = 6,
) -> TokenIndexEntries:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=32).digest()
    cursor = 0
    entries: dict[int, float] = {}

    while len(entries) < active_dimensions:
        if cursor + 2 >= len(digest):
            digest = hashlib.blake2b(digest, digest_size=32).digest()
            cursor = 0
        position = int.from_bytes(digest[cursor : cursor + 2], "big") % dimensions
        sign = 1.0 if digest[cursor + 2] % 2 == 0 else -1.0
        cursor += 3
        if position in entries:
            continue
        entries[position] = sign
    return tuple(entries.items())


def _dense_index_vector(token: str, dimensions: int, active_dimensions: int = 6) -> list[float]:
    vector = [0.0] * dimensions
    for position, sign in _dense_index_entries(token, dimensions, active_dimensions):
        vector[position] = sign
    return vector


class RandomIndexingTrainer:
    def __init__(
        self,
        *,
        dimensions: int = RANDOM_INDEX_DIMENSIONS,
        active_dimensions: int = RANDOM_INDEX_ACTIVE_DIMS,
        context_window: int = RANDOM_INDEX_CONTEXT_WINDOW,
        passes: int = RANDOM_INDEX_PASSES,
        lexical_mix: float = RANDOM_INDEX_LEXICAL_MIX,
        compute_device: str = "auto",
    ):
        self._dimensions = dimensions
        self._active_dimensions = active_dimensions
        self._context_window = context_window
        self._passes = passes
        self._lexical_mix = lexical_mix
        self._compute_device = compute_device

    def fit(self, index: SearchIndex) -> DenseEmbeddingSpace:
        vocabulary = sorted(index.document_frequency)
        logger.info(
            "Training dense semantic embedding space with random indexing: docs=%s vocab=%s dims=%s "
            "active_dims=%s context_window=%s passes=%s lexical_mix=%.2f",
            index.doc_count,
            len(vocabulary),
            self._dimensions,
            self._active_dimensions,
            self._context_window,
            self._passes,
            self._lexical_mix,
        )
        runtime = resolve_torch_runtime(self._compute_device)
        if self._compute_device == "cuda":
            logger.info(
                "Random indexing training is CPU-bound in this implementation; "
                "CUDA=%s (resolved_device=%s) will be used for dense scoring stages.",
                runtime.enabled,
                runtime.device,
            )

        index_entries = {
            token: _dense_index_entries(token, self._dimensions, self._active_dimensions)
            for token in vocabulary
        }
        semantic_vectors = {token: [0.0] * self._dimensions for token in vocabulary}

        prepared_issues = list(index.prepared.values())
        for pass_index in range(self._passes):
            pass_start = time.perf_counter()
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
                        neighbor_index_entries = index_entries[neighbor]
                        for dimension, value in neighbor_index_entries:
                            token_semantics[dimension] += value * weight

                if document_position % RANDOM_INDEX_PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - pass_start
                    logger.info(
                        "Dense semantic training progress: pass=%s docs=%s/%s elapsed=%.1fs rate=%.1f docs/s",
                        pass_index + 1,
                        document_position,
                        len(prepared_issues),
                        elapsed,
                        document_position / max(elapsed, 1e-6),
                    )
            logger.info("Dense semantic training pass %s/%s finished", pass_index + 1, self._passes)

        token_vectors: dict[str, tuple[float, ...]] = {}
        for token in vocabulary:
            mixed = list(semantic_vectors[token])
            for dimension, value in index_entries[token]:
                mixed[dimension] += self._lexical_mix * value
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
        self._query_vector_cache: dict[int | tuple[tuple[str, int], ...], tuple[float, ...]] = {}
        if self._torch_index is not None:
            logger.info("Dense semantic candidate scoring using torch on %s", self._torch_index.device)

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        key = _query_cache_key(query)
        query_vector = self._query_vector_cache.get(key)
        if query_vector is None:
            query_vector = self._dense_space.encode(query.term_frequency, index.idf)
            self._query_vector_cache[key] = query_vector

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
        self._query_vector_cache: dict[int | tuple[tuple[str, int], ...], tuple[float, ...]] = {}

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        key = _query_cache_key(query)
        query_vector = self._query_vector_cache.get(key)
        if query_vector is None:
            query_vector = self._dense_space.encode(query.term_frequency, index.idf)
            self._query_vector_cache[key] = query_vector
        candidate_vector = self._dense_space.document_vectors.get(candidate.issue_id or -1)
        dense_score = _dense_cosine(query_vector, candidate_vector) if candidate_vector else 0.0
        feature_scores["dense_cosine"] = round(max(0.0, dense_score), 4)
        return feature_scores


class DenseCosineReranker(Reranker):
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        dense_score = max(0.0, min(1.0, feature_scores.get("dense_cosine", 0.0)))
        score = (
            (0.84 * dense_score)
            + (0.10 * feature_scores.get("title_ngram", 0.0))
            + (0.06 * feature_scores.get("description_overlap", 0.0))
        )
        bounded_score = max(0.0, min(1.0, score))
        reasons: list[str] = []
        if dense_score >= 0.16:
            reasons.append("dense semantic embeddings found a close meaning-level match")
        if feature_scores.get("title_ngram", 0.0) >= 0.10:
            reasons.append("title wording is also similar")
        if feature_scores.get("description_overlap", 0.0) >= 0.08:
            reasons.append("description text overlaps as well")
        if not reasons:
            reasons.append("dense semantic similarity was the primary ranking signal")
        return RerankResult(
            score=round(bounded_score, 6),
            feature_scores=feature_scores,
            reasons=tuple(reasons[:4]),
        )


@dataclass(slots=True)
class SBERTEmbeddingSpace:
    model_id: str
    issue_ids: tuple[int, ...]
    issue_index: dict[int, int]
    embeddings: Any
    encoder: Any
    torch: Any | None
    device: str

    @classmethod
    def build(
        cls,
        index: SearchIndex,
        *,
        model_id: str,
        sentence_transformer_cls: Any,
        compute_device: str,
    ) -> "SBERTEmbeddingSpace":
        runtime = resolve_torch_runtime(compute_device)
        device = runtime.device if runtime.enabled and runtime.torch is not None else "cpu"
        encoder = sentence_transformer_cls(model_id, device=device)
        issue_ids = tuple(sorted(index.prepared))
        texts = [_prepared_to_text(index.prepared[issue_id]) for issue_id in issue_ids]
        logger.info(
            "Building SBERT dense embedding space: model=%s docs=%s batch=%s device=%s",
            model_id,
            len(issue_ids),
            DEFAULT_SBERT_BATCH_SIZE,
            device,
        )
        embeddings = encoder.encode(
            texts,
            batch_size=DEFAULT_SBERT_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        torch_module = runtime.torch if runtime.enabled else None
        if torch_module is not None:
            logger.info("SBERT dense candidate scoring will use torch on %s", runtime.device)
        return cls(
            model_id=model_id,
            issue_ids=issue_ids,
            issue_index={issue_id: idx for idx, issue_id in enumerate(issue_ids)},
            embeddings=embeddings,
            encoder=encoder,
            torch=torch_module,
            device=device,
        )

    def encode_query(self, query) -> list[float]:
        text = _prepared_to_text(query)
        encoded = self.encoder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return [float(value) for value in encoded]

    def candidate_embedding(self, issue_id: int) -> list[float] | None:
        index_position = self.issue_index.get(issue_id)
        if index_position is None:
            return None
        values = self.embeddings[index_position]
        return [float(value) for value in values]


class SBERTCandidateGenerator(CandidateGenerator, CandidateGeneratorFallbackMixin):
    def __init__(self, embedding_space: SBERTEmbeddingSpace):
        self._space = embedding_space
        self._score_matrix = None
        if self._space.torch is not None:
            self._score_matrix = self._space.torch.tensor(
                self._space.embeddings,
                dtype=self._space.torch.float32,
                device=self._space.device,
            )

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        query_vector = self._space.encode_query(query)
        if self._score_matrix is not None:
            with self._space.torch.no_grad():
                query_tensor = self._space.torch.tensor(
                    query_vector,
                    dtype=self._space.torch.float32,
                    device=self._space.device,
                )
                scores = self._score_matrix @ query_tensor
                top_count = min(pool_size, int(scores.shape[0]))
                if top_count <= 0:
                    return self.fallback(query, index, pool_size)
                top_scores, top_indices = self._space.torch.topk(scores, k=top_count)
                selected_scores = top_scores.tolist()
                selected_indices = top_indices.tolist()
            return [
                CandidateMatch(
                    issue_id=self._space.issue_ids[int(index_position)],
                    seed_score=float(score),
                )
                for index_position, score in zip(selected_indices, selected_scores)
            ]

        scored: list[CandidateMatch] = []
        for issue_id in self._space.issue_ids:
            embedding = self._space.candidate_embedding(issue_id)
            if embedding is None:
                continue
            score = float(sum(left * right for left, right in zip(query_vector, embedding)))
            scored.append(CandidateMatch(issue_id=issue_id, seed_score=score))
        scored.sort(key=lambda item: item.seed_score, reverse=True)
        if not scored:
            return self.fallback(query, index, pool_size)
        return scored[:pool_size]


class SBERTFeatureExtractor(StandardFeatureExtractor):
    def __init__(self, embedding_space: SBERTEmbeddingSpace):
        self._space = embedding_space
        self._query_vector_cache: dict[int | tuple[tuple[str, int], ...], list[float]] = {}

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        key = _query_cache_key(query)
        query_vector = self._query_vector_cache.get(key)
        if query_vector is None:
            query_vector = self._space.encode_query(query)
            self._query_vector_cache[key] = query_vector
        candidate_vector = self._space.candidate_embedding(candidate.issue_id or -1)
        semantic_score = 0.0
        if candidate_vector is not None:
            raw_cosine = float(sum(left * right for left, right in zip(query_vector, candidate_vector)))
            semantic_score = max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))
        feature_scores["sbert_cosine"] = round(semantic_score, 4)
        return feature_scores


class SBERTReranker(Reranker):
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        sbert_score = feature_scores.get("sbert_cosine", 0.0)
        score = (
            (0.80 * sbert_score)
            + (0.08 * feature_scores.get("bm25_plus", 0.0))
            + (0.06 * feature_scores.get("title_ngram", 0.0))
            + (0.06 * feature_scores.get("description_overlap", 0.0))
        )
        bounded_score = max(0.0, min(1.0, score))
        reasons: list[str] = []
        if sbert_score >= 0.72:
            reasons.append("SBERT embeddings found a strong semantic duplicate signal")
        elif sbert_score >= 0.58:
            reasons.append("SBERT embeddings found moderate semantic similarity")
        if feature_scores.get("bm25_plus", 0.0) >= 0.22:
            reasons.append("sparse lexical evidence also supports the pair")
        if feature_scores.get("title_ngram", 0.0) >= 0.10:
            reasons.append("title wording is aligned")
        if not reasons:
            reasons.append("SBERT semantic similarity was the main ranking signal")
        return RerankResult(
            score=round(bounded_score, 6),
            feature_scores=feature_scores,
            reasons=tuple(reasons[:4]),
        )


def build_dense_embedding_space(
    index: SearchIndex,
    *,
    compute_device: str = "auto",
) -> DenseEmbeddingSpace:
    return RandomIndexingTrainer(compute_device=compute_device).fit(index)


def _build_sbert_dense_pipeline(
    index: SearchIndex,
    *,
    compute_device: str,
    requested_models: frozenset[str] | None,
) -> dict[str, RetrievalPipeline]:
    global _sbert_missing_warned
    sentence_transformer_cls = _resolve_sentence_transformer_cls()
    explicitly_requested = requested_models is not None and SBERT_MODEL_NAME in requested_models
    requested_only_sbert = explicitly_requested and requested_models == frozenset({SBERT_MODEL_NAME})
    if sentence_transformer_cls is None:
        install_hint = (
            "SBERT dense model requires optional dependencies. "
            "Install with: pip install -e .[llm,gpu]"
        )
        if requested_only_sbert:
            raise RuntimeError(install_hint)
        if explicitly_requested and not _sbert_missing_warned:
            logger.warning("%s Skipping SBERT dense pipeline for this run.", install_hint)
            _sbert_missing_warned = True
        return {}

    try:
        embedding_space = SBERTEmbeddingSpace.build(
            index,
            model_id=DEFAULT_SBERT_MODEL_ID,
            sentence_transformer_cls=sentence_transformer_cls,
            compute_device=compute_device,
        )
    except Exception as exc:
        model_init_hint = (
            "Unable to load SBERT dense model. "
            "Ensure model artifacts are available (internet access or local cache)."
        )
        if requested_only_sbert:
            raise RuntimeError(model_init_hint) from exc
        logger.warning("%s Skipping SBERT dense pipeline for this run: %s", model_init_hint, exc)
        return {}

    return {
        SBERT_MODEL_NAME: RetrievalPipeline(
            name=SBERT_MODEL_NAME,
            candidate_generator=SBERTCandidateGenerator(embedding_space),
            feature_extractor=SBERTFeatureExtractor(embedding_space),
            reranker=SBERTReranker(),
        )
    }


def build_dense_semantic_pipelines(
    index: SearchIndex,
    *,
    dense_space: DenseEmbeddingSpace | None = None,
    compute_device: str = "auto",
    requested_models: frozenset[str] | None = None,
) -> dict[str, RetrievalPipeline]:
    include_random_indexing = requested_models is None or "random-indexing-dense" in requested_models
    include_sbert = requested_models is None or SBERT_MODEL_NAME in requested_models

    pipelines: dict[str, RetrievalPipeline] = {}
    if include_random_indexing:
        effective_dense_space = dense_space or build_dense_embedding_space(index, compute_device=compute_device)
        pipelines["random-indexing-dense"] = RetrievalPipeline(
            name="random-indexing-dense",
            candidate_generator=DenseSemanticCandidateGenerator(
                effective_dense_space,
                compute_device=compute_device,
            ),
            feature_extractor=DenseSemanticFeatureExtractor(effective_dense_space),
            reranker=DenseCosineReranker(),
        )

    if include_sbert:
        pipelines.update(
            _build_sbert_dense_pipeline(
                index,
                compute_device=compute_device,
                requested_models=requested_models,
            )
        )
    return pipelines
