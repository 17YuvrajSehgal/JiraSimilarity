from __future__ import annotations

from dataclasses import dataclass
import logging

from ..pipeline import FeatureExtractor, RetrievalPipeline, RerankResult, Reranker, SearchIndex
from .deep_pairwise import NeuralPairFeatureExtractor
from .dense_semantic import DenseEmbeddingSpace
from .hybrid_sparse_dense import ReciprocalRankFusionCandidateGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGJudgment:
    score: float
    reasons: tuple[str, ...]


class RAGFeatureExtractor(FeatureExtractor):
    def __init__(self, dense_space: DenseEmbeddingSpace):
        self._pairwise_extractor = NeuralPairFeatureExtractor(dense_space)

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = self._pairwise_extractor.extract(query, candidate, index, seed_score=seed_score)
        feature_scores["semantic_lexical_agreement"] = round(
            (feature_scores.get("dense_cosine", 0.0) + feature_scores.get("bm25_plus", 0.0)) / 2.0,
            4,
        )
        feature_scores["metadata_alignment"] = round(
            (
                feature_scores.get("project_match", 0.0)
                + feature_scores.get("component_overlap", 0.0)
                + feature_scores.get("issue_type_match", 0.0)
            )
            / 3.0,
            4,
        )
        feature_scores["retrieval_confidence"] = round(
            (
                feature_scores.get("candidate_seed", 0.0)
                + feature_scores.get("dense_cosine", 0.0)
                + feature_scores.get("bm25_plus", 0.0)
            )
            / 3.0,
            4,
        )
        return feature_scores


class LocalRAGJudge(Reranker):
    def __init__(
        self,
        *,
        contradiction_penalty: float = 0.08,
    ):
        self._contradiction_penalty = contradiction_penalty

    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        judgment = self._judge(feature_scores)
        return RerankResult(
            score=round(judgment.score, 6),
            feature_scores=feature_scores,
            reasons=judgment.reasons,
        )

    def _judge(self, feature_scores: dict[str, float]) -> RAGJudgment:
        dense_cosine = feature_scores.get("dense_cosine", 0.0)
        bm25_plus = feature_scores.get("bm25_plus", 0.0)
        title_ngram = feature_scores.get("title_ngram", 0.0)
        description_overlap = feature_scores.get("description_overlap", 0.0)
        component_overlap = feature_scores.get("component_overlap", 0.0)
        project_match = feature_scores.get("project_match", 0.0)
        issue_type_match = feature_scores.get("issue_type_match", 0.0)
        priority_match = feature_scores.get("priority_match", 0.0)
        candidate_seed = feature_scores.get("candidate_seed", 0.0)
        semantic_lexical_agreement = feature_scores.get("semantic_lexical_agreement", 0.0)
        metadata_alignment = feature_scores.get("metadata_alignment", 0.0)
        retrieval_confidence = feature_scores.get("retrieval_confidence", 0.0)

        score = (
            (0.24 * dense_cosine)
            + (0.16 * bm25_plus)
            + (0.09 * title_ngram)
            + (0.08 * description_overlap)
            + (0.10 * component_overlap)
            + (0.08 * project_match)
            + (0.06 * issue_type_match)
            + (0.04 * priority_match)
            + (0.05 * candidate_seed)
            + (0.05 * semantic_lexical_agreement)
            + (0.05 * metadata_alignment)
        )

        if dense_cosine >= 0.22 and bm25_plus >= 0.28:
            score += 0.10
        if semantic_lexical_agreement >= 0.28 and metadata_alignment >= 0.40:
            score += 0.08
        if retrieval_confidence >= 0.40 and metadata_alignment >= 0.30:
            score += 0.04
        if project_match == 0.0 and component_overlap == 0.0 and dense_cosine < 0.14:
            score -= self._contradiction_penalty
        if issue_type_match == 0.0 and retrieval_confidence < 0.16:
            score -= 0.04

        bounded_score = max(0.0, min(1.0, score))
        reasons = self._build_reasoning(
            bounded_score,
            dense_cosine=dense_cosine,
            bm25_plus=bm25_plus,
            title_ngram=title_ngram,
            description_overlap=description_overlap,
            component_overlap=component_overlap,
            project_match=project_match,
            metadata_alignment=metadata_alignment,
            retrieval_confidence=retrieval_confidence,
        )
        return RAGJudgment(score=bounded_score, reasons=reasons)

    def _build_reasoning(
        self,
        score: float,
        *,
        dense_cosine: float,
        bm25_plus: float,
        title_ngram: float,
        description_overlap: float,
        component_overlap: float,
        project_match: float,
        metadata_alignment: float,
        retrieval_confidence: float,
    ) -> tuple[str, ...]:
        reasons: list[str] = []

        if dense_cosine >= 0.20 and bm25_plus >= 0.25:
            reasons.append("rag-style reasoning found both semantic and lexical evidence for the same issue")
        elif dense_cosine >= 0.20:
            reasons.append("rag-style reasoning found a strong meaning-level match")
        elif bm25_plus >= 0.25:
            reasons.append("rag-style reasoning found strong textual overlap")

        if project_match >= 1.0 or component_overlap >= 0.2 or metadata_alignment >= 0.35:
            reasons.append("retrieved project and component context supports the match")

        if title_ngram >= 0.12 or description_overlap >= 0.10:
            reasons.append("the retrieved issue language closely matches the new report")

        if retrieval_confidence < 0.12 and score < 0.45:
            reasons.append("retrieved evidence is thin, so the match should be reviewed carefully")

        if not reasons:
            if score >= 0.55:
                reasons.append("rag-style reasoning found moderate duplicate evidence across the retrieved context")
            elif score >= 0.35:
                reasons.append("rag-style reasoning found this issue related, but not strongly confirmed as a duplicate")
            else:
                reasons.append("rag-style reasoning found only weak support for this candidate")

        return tuple(reasons[:4])


def build_llm_rag_pipelines(
    index: SearchIndex,
    *,
    dense_space: DenseEmbeddingSpace,
    compute_device: str = "auto",
) -> dict[str, RetrievalPipeline]:
    _ = index
    logger.info("Building LLM-style RAG pipelines with local reasoning fallback")
    return {
        "rag-hybrid-judge": RetrievalPipeline(
            name="rag-hybrid-judge",
            candidate_generator=ReciprocalRankFusionCandidateGenerator(
                dense_space=dense_space,
                compute_device=compute_device,
            ),
            feature_extractor=RAGFeatureExtractor(dense_space),
            reranker=LocalRAGJudge(),
        )
    }
