from __future__ import annotations

from collections import Counter

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


class SparseScoreCandidateGenerator(CandidateGenerator):
    def __init__(self, score_name: str):
        self._score_name = score_name

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        candidate_scores: Counter[int] = Counter()
        candidate_ids: set[int] = set()
        for token in query.term_frequency:
            candidate_ids.update(index.postings.get(token, {}).keys())

        if not candidate_ids:
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
            return BM25CandidateGenerator().generate(query, index, pool_size=pool_size)

        return [
            CandidateMatch(issue_id=issue_id, seed_score=score)
            for issue_id, score in candidate_scores.most_common(pool_size)
        ]


class SparseLexicalFeatureExtractor(StandardFeatureExtractor):
    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        feature_scores["bm25_plus"] = round(index.bm25_plus(query, candidate), 4)
        feature_scores["tfidf_cosine"] = round(index.tfidf_cosine(query, candidate), 4)
        return feature_scores


def build_sparse_lexical_pipelines() -> dict[str, RetrievalPipeline]:
    sparse_feature_extractor: FeatureExtractor = SparseLexicalFeatureExtractor()

    bm25 = RetrievalPipeline(
        name="bm25",
        candidate_generator=SparseScoreCandidateGenerator("bm25"),
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "bm25": 0.86,
                "title_ngram": 0.14,
            }
        ),
    )
    bm25_plus = RetrievalPipeline(
        name="bm25-plus",
        candidate_generator=SparseScoreCandidateGenerator("bm25_plus"),
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "bm25_plus": 0.80,
                "bm25": 0.10,
                "title_ngram": 0.10,
            }
        ),
    )
    tfidf_cosine = RetrievalPipeline(
        name="tfidf-cosine",
        candidate_generator=SparseScoreCandidateGenerator("tfidf_cosine"),
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "tfidf_cosine": 0.82,
                "title_ngram": 0.10,
                "description_overlap": 0.08,
            }
        ),
    )
    lexical = RetrievalPipeline(
        name="lexical",
        candidate_generator=bm25.candidate_generator,
        feature_extractor=sparse_feature_extractor,
        reranker=WeightedLinearReranker(
            weights={
                "bm25": 0.88,
                "title_ngram": 0.12,
            }
        ),
    )
    return {
        pipeline.name: pipeline
        for pipeline in (lexical, bm25, bm25_plus, tfidf_cosine)
    }
