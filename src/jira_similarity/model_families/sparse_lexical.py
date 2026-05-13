from __future__ import annotations

from collections import Counter
import logging

from ..pipeline import (
    BM25CandidateGenerator,
    CandidateGenerator,
    CandidateMatch,
    FeatureExtractor,
    RetrievalPipeline,
    SearchIndex,
    StandardFeatureExtractor,
    WeightedLinearReranker,
)

logger = logging.getLogger(__name__)


def _normalize_sparse_score(value: float) -> float:
    if value <= 0:
        return 0.0
    return value / (value + 8.0)


class SparseScoreCandidateGenerator(CandidateGenerator):
    def __init__(self, score_name: str):
        self._score_name = score_name
        logger.debug("SparseScoreCandidateGenerator initialised: score=%s", score_name)

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        candidate_scores: Counter[int] = Counter()
        candidate_ids: set[int] = set()
        for token in query.term_frequency:
            candidate_ids.update(index.postings.get(token, {}).keys())

        if not candidate_ids:
            logger.debug(
                "SparseScoreCandidateGenerator: no postings found for query terms, falling back to BM25"
            )
            return BM25CandidateGenerator().generate(query, index, pool_size=pool_size)

        for issue_id in candidate_ids:
            candidate = index.prepared[issue_id]
            if self._score_name == "bm25":
                score = index.bm25(query, candidate)
            elif self._score_name == "bm25_plus":
                score = index.bm25_plus(query, candidate)
            elif self._score_name == "tfidf_cosine":
                score = index.tfidf_cosine(query, candidate)
            else:
                raise ValueError(f"Unsupported sparse score '{self._score_name}'")
            if score > 0:
                candidate_scores[issue_id] = score

        if not candidate_scores:
            logger.debug(
                "SparseScoreCandidateGenerator: all candidates scored zero for score=%s, falling back to BM25",
                self._score_name,
            )
            return BM25CandidateGenerator().generate(query, index, pool_size=pool_size)

        results = [
            CandidateMatch(issue_id=issue_id, seed_score=score)
            for issue_id, score in candidate_scores.most_common(pool_size)
        ]
        logger.debug(
            "SparseScoreCandidateGenerator: score=%s candidates_scored=%s returning=%s",
            self._score_name,
            len(candidate_scores),
            len(results),
        )
        return results


class SparseLexicalFeatureExtractor(StandardFeatureExtractor):
    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        feature_scores["bm25_plus"] = round(_normalize_sparse_score(index.bm25_plus(query, candidate)), 4)
        feature_scores["tfidf_cosine"] = round(index.tfidf_cosine(query, candidate), 4)
        return feature_scores


def build_sparse_lexical_pipelines() -> dict[str, RetrievalPipeline]:
    logger.info("Building sparse lexical pipelines: bm25, bm25-plus, tfidf-cosine, lexical")
    sparse_feature_extractor: FeatureExtractor = SparseLexicalFeatureExtractor()

    bm25 = RetrievalPipeline(
        name="bm25",
        candidate_generator=SparseScoreCandidateGenerator("bm25"),
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "bm25": 0.70,
                "bm25_plus": 0.12,
                "title_ngram": 0.10,
                "description_overlap": 0.08,
            }
        ),
    )
    logger.debug("Built bm25 pipeline")
    bm25_plus = RetrievalPipeline(
        name="bm25-plus",
        candidate_generator=SparseScoreCandidateGenerator("bm25_plus"),
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "bm25_plus": 0.62,
                "bm25": 0.18,
                "tfidf_cosine": 0.10,
                "title_ngram": 0.06,
                "description_overlap": 0.04,
            }
        ),
    )
    logger.debug("Built bm25-plus pipeline")
    tfidf_cosine = RetrievalPipeline(
        name="tfidf-cosine",
        candidate_generator=SparseScoreCandidateGenerator("tfidf_cosine"),
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "tfidf_cosine": 0.62,
                "bm25_plus": 0.18,
                "bm25": 0.10,
                "title_ngram": 0.06,
                "description_overlap": 0.04,
            }
        ),
    )
    logger.debug("Built tfidf-cosine pipeline")
    lexical = RetrievalPipeline(
        name="lexical",
        candidate_generator=bm25.candidate_generator,
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "bm25": 0.62,
                "bm25_plus": 0.18,
                "tfidf_cosine": 0.10,
                "title_ngram": 0.06,
                "description_overlap": 0.04,
            }
        ),
    )
    logger.debug("Built lexical (alias) pipeline")
    logger.info("Sparse lexical pipelines ready")
    return {
        pipeline.name: pipeline
        for pipeline in (lexical, bm25, bm25_plus, tfidf_cosine)
    }
