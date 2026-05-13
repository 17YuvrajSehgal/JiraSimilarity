from __future__ import annotations

from collections import Counter
import logging

from ..pipeline import (
    CandidateGenerator,
    CandidateMatch,
    FeatureExtractor,
    RetrievalPipeline,
    SearchIndex,
    WeightedLinearReranker,
)
from .dense_semantic import DenseEmbeddingSpace, DenseSemanticCandidateGenerator, DenseSemanticFeatureExtractor
from .sparse_lexical import SparseScoreCandidateGenerator

logger = logging.getLogger(__name__)


class ReciprocalRankFusionCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        *,
        dense_space: DenseEmbeddingSpace,
        compute_device: str = "auto",
        sparse_weight: float = 0.52,
        dense_weight: float = 0.48,
        rank_constant: int = 25,
        score_mix: float = 0.18,
    ):
        self._sparse_generator = SparseScoreCandidateGenerator("bm25_plus")
        self._dense_generator = DenseSemanticCandidateGenerator(
            dense_space,
            compute_device=compute_device,
        )
        self._sparse_weight = sparse_weight
        self._dense_weight = dense_weight
        self._rank_constant = rank_constant
        self._score_mix = score_mix

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        logger.debug(
            "Hybrid RRF candidate generation started: sparse_weight=%.2f dense_weight=%.2f pool=%s",
            self._sparse_weight,
            self._dense_weight,
            pool_size,
        )
        sparse_matches = self._sparse_generator.generate(query, index, pool_size=pool_size)
        dense_matches = self._dense_generator.generate(query, index, pool_size=pool_size)

        fused_scores: Counter[int] = Counter()
        self._accumulate_rrf(fused_scores, sparse_matches, self._sparse_weight, source="sparse")
        self._accumulate_rrf(fused_scores, dense_matches, self._dense_weight, source="dense")

        fused_matches = [
            CandidateMatch(issue_id=issue_id, seed_score=score)
            for issue_id, score in fused_scores.most_common(pool_size)
        ]
        logger.debug(
            "Hybrid RRF candidate generation complete: sparse=%s dense=%s fused=%s",
            len(sparse_matches),
            len(dense_matches),
            len(fused_matches),
        )
        return fused_matches

    def _accumulate_rrf(
        self,
        fused_scores: Counter[int],
        matches: list[CandidateMatch],
        weight: float,
        *,
        source: str,
    ) -> None:
        for rank, match in enumerate(matches, start=1):
            fused_scores[match.issue_id] += weight / (self._rank_constant + rank)
            fused_scores[match.issue_id] += self._score_mix * self._normalized_source_score(
                source,
                match.seed_score,
            )

    @staticmethod
    def _normalized_source_score(source: str, raw_score: float) -> float:
        if raw_score <= 0:
            if raw_score < 0 and source == "dense":
                return max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            return 0.0
        if source == "sparse":
            return raw_score / (raw_score + 8.0)
        return max(0.0, min(1.0, raw_score))


class HybridSparseDenseFeatureExtractor(FeatureExtractor):
    def __init__(self, dense_space: DenseEmbeddingSpace):
        self._dense_feature_extractor = DenseSemanticFeatureExtractor(dense_space)

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = self._dense_feature_extractor.extract(query, candidate, index, seed_score=seed_score)
        feature_scores["rrf_seed"] = round(max(0.0, min(1.0, seed_score)), 4)
        return feature_scores


def build_hybrid_sparse_dense_pipelines(
    index: SearchIndex,
    *,
    dense_space: DenseEmbeddingSpace,
    compute_device: str = "auto",
) -> dict[str, RetrievalPipeline]:
    _ = index
    logger.info("Building hybrid sparse-dense pipelines")
    return {
        "hybrid-sparse-dense": RetrievalPipeline(
            name="hybrid-sparse-dense",
            candidate_generator=ReciprocalRankFusionCandidateGenerator(
                dense_space=dense_space,
                compute_device=compute_device,
            ),
            feature_extractor=HybridSparseDenseFeatureExtractor(dense_space),
            reranker=WeightedLinearReranker(
                weights={
                    "dense_cosine": 0.36,
                    "bm25_plus": 0.24,
                    "bm25": 0.10,
                    "tfidf_cosine": 0.06,
                    "title_ngram": 0.08,
                    "description_overlap": 0.06,
                    "candidate_seed": 0.06,
                    "rrf_seed": 0.04,
                },
                explanation_threshold=0.05,
            ),
        )
    }
