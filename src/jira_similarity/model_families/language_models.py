from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Any

from ..compute import resolve_torch_runtime
from ..pipeline import (
    CandidateGenerator,
    CandidateMatch,
    FeatureExtractor,
    RetrievalPipeline,
    RerankResult,
    Reranker,
    SearchIndex,
    StandardFeatureExtractor,
)

logger = logging.getLogger(__name__)
_llm_missing_warned = False

LLM_E5_MODEL_NAME = "llm-e5-large"
LLM_E5_CROSS_MODEL_NAME = "llm-e5-cross-reranker"
LLM_MODEL_NAMES = frozenset({LLM_E5_MODEL_NAME, LLM_E5_CROSS_MODEL_NAME})

DEFAULT_EMBEDDING_MODEL_ID = os.getenv("JIRA_LLM_EMBED_MODEL", "intfloat/e5-large-v2")
DEFAULT_CROSS_ENCODER_MODEL_ID = os.getenv("JIRA_LLM_CROSS_MODEL", "BAAI/bge-reranker-base")
DEFAULT_EMBED_BATCH_SIZE = int(os.getenv("JIRA_LLM_EMBED_BATCH_SIZE", "64"))
DEFAULT_MAX_WEIGHTED_TOKENS = int(os.getenv("JIRA_LLM_MAX_WEIGHTED_TOKENS", "320"))
LLM_CROSS_RERANK_POOL_SIZE = int(os.getenv("JIRA_LLM_CROSS_POOL_SIZE", "80"))


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _prepared_to_text(prepared, *, max_tokens: int = DEFAULT_MAX_WEIGHTED_TOKENS) -> str:
    if not prepared.weighted_terms:
        return "empty issue"
    return " ".join(prepared.weighted_terms[:max_tokens])


def _to_unit_interval_from_cosine(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _resolve_sentence_transformers():
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore
    except Exception:
        return None, None
    return SentenceTransformer, CrossEncoder


def _load_sentence_transformers_or_handle(
    *,
    llm_models_were_only_request: bool,
) -> tuple[Any | None, Any | None]:
    global _llm_missing_warned
    sentence_transformer_cls, cross_encoder_cls = _resolve_sentence_transformers()
    if sentence_transformer_cls is not None and cross_encoder_cls is not None:
        return sentence_transformer_cls, cross_encoder_cls

    install_hint = (
        "Sentence-transformer models require optional dependencies. "
        "Install with: pip install -e .[llm,gpu]"
    )
    if llm_models_were_only_request:
        raise RuntimeError(install_hint)

    if not _llm_missing_warned:
        logger.warning("%s Skipping LLM pipelines for this run.", install_hint)
        _llm_missing_warned = True
    return None, None


@dataclass(slots=True)
class LLMEmbeddingSpace:
    model_name: str
    issue_ids: tuple[int, ...]
    embeddings: Any
    encoder: Any
    device: str
    torch: Any | None

    @classmethod
    def build(
        cls,
        index: SearchIndex,
        *,
        embedding_model_id: str,
        sentence_transformer_cls: Any,
        compute_device: str,
    ) -> "LLMEmbeddingSpace":
        runtime = resolve_torch_runtime(compute_device)
        device = runtime.device if runtime.enabled and runtime.torch is not None else "cpu"
        encoder = sentence_transformer_cls(embedding_model_id, device=device)
        issue_ids = tuple(sorted(index.prepared))
        passage_texts = [
            f"passage: {_prepared_to_text(index.prepared[issue_id])}"
            for issue_id in issue_ids
        ]

        logger.info(
            "Building sentence-transformer embedding space: model=%s docs=%s batch=%s device=%s",
            embedding_model_id,
            len(issue_ids),
            DEFAULT_EMBED_BATCH_SIZE,
            device,
        )
        embeddings = encoder.encode(
            passage_texts,
            batch_size=DEFAULT_EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        torch_module = runtime.torch if runtime.enabled else None
        if torch_module is not None:
            logger.info("LLM embedding scoring will use torch on %s", runtime.device)
        return cls(
            model_name=embedding_model_id,
            issue_ids=issue_ids,
            embeddings=embeddings,
            encoder=encoder,
            device=device,
            torch=torch_module,
        )

    def encode_query(self, prepared_query) -> Any:
        query_text = f"query: {_prepared_to_text(prepared_query)}"
        return self.encoder.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]


class LLME5CandidateGenerator(CandidateGenerator):
    def __init__(self, embedding_space: LLMEmbeddingSpace):
        self._space = embedding_space
        self._score_matrix = None
        if self._space.torch is not None:
            self._score_matrix = self._space.torch.tensor(
                self._space.embeddings,
                dtype=self._space.torch.float32,
                device=self._space.device,
            )

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        _ = index
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

        # Fallback scoring path for CPU-only environments without torch acceleration.
        scored: list[CandidateMatch] = []
        for issue_id, embedding in zip(self._space.issue_ids, self._space.embeddings):
            score = float(sum(left * right for left, right in zip(query_vector, embedding)))
            scored.append(CandidateMatch(issue_id=issue_id, seed_score=score))
        scored.sort(key=lambda item: item.seed_score, reverse=True)
        return scored[:pool_size]


class LLME5FeatureExtractor(StandardFeatureExtractor):
    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        feature_scores["llm_embedding_cosine"] = round(_to_unit_interval_from_cosine(seed_score), 4)
        return feature_scores


class LLME5Reranker(Reranker):
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        embedding_score = feature_scores.get("llm_embedding_cosine", 0.0)
        score = (
            (0.78 * embedding_score)
            + (0.12 * feature_scores.get("title_ngram", 0.0))
            + (0.10 * feature_scores.get("description_overlap", 0.0))
        )
        bounded = max(0.0, min(1.0, score))
        reasons: list[str] = []
        if embedding_score >= 0.65:
            reasons.append("transformer embeddings found a strong semantic match")
        if feature_scores.get("title_ngram", 0.0) >= 0.12:
            reasons.append("title wording is also closely aligned")
        if feature_scores.get("description_overlap", 0.0) >= 0.08:
            reasons.append("description overlap further supports the match")
        if not reasons:
            reasons.append("transformer semantic similarity was the main ranking signal")
        return RerankResult(
            score=round(bounded, 6),
            feature_scores=feature_scores,
            reasons=tuple(reasons[:4]),
        )


class LLMCrossFeatureExtractor(StandardFeatureExtractor):
    def __init__(self, cross_encoder):
        self._cross_encoder = cross_encoder

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        query_text = _prepared_to_text(query)
        candidate_text = _prepared_to_text(candidate)
        raw_score = float(self._cross_encoder.predict([(query_text, candidate_text)])[0])
        feature_scores["llm_cross_score"] = round(_sigmoid(raw_score), 4)
        feature_scores["llm_embedding_cosine"] = round(_to_unit_interval_from_cosine(seed_score), 4)
        return feature_scores


class LLMCrossReranker(Reranker):
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        cross_score = feature_scores.get("llm_cross_score", 0.0)
        embedding_prior = feature_scores.get("llm_embedding_cosine", 0.0)
        score = max(0.0, min(1.0, (0.90 * cross_score) + (0.10 * embedding_prior)))
        reasons: list[str] = []
        if cross_score >= 0.70:
            reasons.append("cross-encoder judged this pair as a strong duplicate candidate")
        elif cross_score >= 0.55:
            reasons.append("cross-encoder found moderate duplicate evidence")
        else:
            reasons.append("cross-encoder found weak duplicate evidence")

        if embedding_prior >= 0.65:
            reasons.append("embedding retrieval also supports this candidate")

        return RerankResult(
            score=round(score, 6),
            feature_scores=feature_scores,
            reasons=tuple(reasons[:4]),
        )


def build_language_model_pipelines(
    index: SearchIndex,
    *,
    compute_device: str = "auto",
    requested_models: frozenset[str] | None = None,
) -> dict[str, RetrievalPipeline]:
    requested_llm_models = (
        frozenset(model_name for model_name in requested_models if model_name in LLM_MODEL_NAMES)
        if requested_models is not None
        else None
    )
    llm_models_were_only_request = (
        requested_models is not None
        and bool(requested_llm_models)
        and not (requested_models - LLM_MODEL_NAMES)
    )
    sentence_transformer_cls, cross_encoder_cls = _load_sentence_transformers_or_handle(
        llm_models_were_only_request=llm_models_were_only_request,
    )
    if sentence_transformer_cls is None or cross_encoder_cls is None:
        return {}

    logger.info(
        "Building transformer language-model pipelines: embedding_model=%s cross_encoder=%s",
        DEFAULT_EMBEDDING_MODEL_ID,
        DEFAULT_CROSS_ENCODER_MODEL_ID,
    )
    try:
        embedding_space = LLMEmbeddingSpace.build(
            index,
            embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID,
            sentence_transformer_cls=sentence_transformer_cls,
            compute_device=compute_device,
        )
        candidate_generator = LLME5CandidateGenerator(embedding_space)

        runtime = resolve_torch_runtime(compute_device)
        cross_encoder_device = runtime.device if runtime.enabled else "cpu"
        cross_encoder = cross_encoder_cls(DEFAULT_CROSS_ENCODER_MODEL_ID, device=cross_encoder_device)
    except Exception as exc:
        model_init_hint = (
            "Unable to load transformer language models. "
            "Ensure required model artifacts are available (internet access or local cache)."
        )
        if llm_models_were_only_request:
            raise RuntimeError(model_init_hint) from exc
        logger.warning("%s Skipping LLM pipelines for this run: %s", model_init_hint, exc)
        return {}

    pipelines = {
        LLM_E5_MODEL_NAME: RetrievalPipeline(
            name=LLM_E5_MODEL_NAME,
            candidate_generator=candidate_generator,
            feature_extractor=LLME5FeatureExtractor(),
            reranker=LLME5Reranker(),
        ),
        LLM_E5_CROSS_MODEL_NAME: RetrievalPipeline(
            name=LLM_E5_CROSS_MODEL_NAME,
            candidate_generator=candidate_generator,
            feature_extractor=LLMCrossFeatureExtractor(cross_encoder),
            reranker=LLMCrossReranker(),
            max_candidate_pool_size=LLM_CROSS_RERANK_POOL_SIZE,
        ),
    }
    logger.info("Transformer language-model pipelines ready: %s", ", ".join(sorted(pipelines)))
    return pipelines
