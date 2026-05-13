from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import logging

from ..pipeline import (
    CandidateGenerator,
    CandidateMatch,
    FeatureExtractor,
    RetrievalPipeline,
    RerankResult,
    Reranker,
    SearchIndex,
)
from ..text import jaccard_similarity
from .dense_semantic import DenseEmbeddingSpace, DenseSemanticFeatureExtractor
from .hybrid_sparse_dense import ReciprocalRankFusionCandidateGenerator

logger = logging.getLogger(__name__)


def _binary_match(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    return 1.0 if left == right else 0.0


@dataclass(frozen=True, slots=True)
class GraphMetadataSpace:
    issue_neighbors: dict[int, dict[int, float]]
    component_index: dict[str, frozenset[int]]
    affected_version_index: dict[str, frozenset[int]]
    fix_version_index: dict[str, frozenset[int]]
    project_index: dict[str, frozenset[int]]
    issue_type_index: dict[str, frozenset[int]]
    priority_index: dict[str, frozenset[int]]
    status_index: dict[str, frozenset[int]]

    @classmethod
    def build(cls, index: SearchIndex) -> "GraphMetadataSpace":
        logger.info("Building graph and metadata space")
        neighbors: dict[int, dict[int, float]] = defaultdict(dict)
        component_index: dict[str, set[int]] = defaultdict(set)
        affected_version_index: dict[str, set[int]] = defaultdict(set)
        fix_version_index: dict[str, set[int]] = defaultdict(set)
        project_index: dict[str, set[int]] = defaultdict(set)
        issue_type_index: dict[str, set[int]] = defaultdict(set)
        priority_index: dict[str, set[int]] = defaultdict(set)
        status_index: dict[str, set[int]] = defaultdict(set)

        for issue_id, prepared_issue in index.prepared.items():
            if prepared_issue.project_key:
                project_index[prepared_issue.project_key].add(issue_id)
            if prepared_issue.issue_type:
                issue_type_index[prepared_issue.issue_type].add(issue_id)
            if prepared_issue.priority:
                priority_index[prepared_issue.priority].add(issue_id)
            if prepared_issue.status:
                status_index[prepared_issue.status].add(issue_id)
            for component in prepared_issue.component_terms:
                component_index[component].add(issue_id)
            for version in prepared_issue.affected_version_terms:
                affected_version_index[version].add(issue_id)
            for version in prepared_issue.fix_version_terms:
                fix_version_index[version].add(issue_id)

            for neighbor_id in prepared_issue.linked_issue_ids:
                if neighbor_id in index.prepared:
                    neighbors[issue_id][neighbor_id] = max(neighbors[issue_id].get(neighbor_id, 0.0), 0.7)
                    neighbors[neighbor_id][issue_id] = max(neighbors[neighbor_id].get(issue_id, 0.0), 0.7)
            for neighbor_id in prepared_issue.duplicate_issue_ids:
                if neighbor_id in index.prepared:
                    neighbors[issue_id][neighbor_id] = max(neighbors[issue_id].get(neighbor_id, 0.0), 1.0)
                    neighbors[neighbor_id][issue_id] = max(neighbors[neighbor_id].get(issue_id, 0.0), 1.0)

        logger.info("Graph and metadata space ready: graph_nodes=%s", len(index.prepared))
        return cls(
            issue_neighbors={issue_id: dict(edges) for issue_id, edges in neighbors.items()},
            component_index={key: frozenset(value) for key, value in component_index.items()},
            affected_version_index={key: frozenset(value) for key, value in affected_version_index.items()},
            fix_version_index={key: frozenset(value) for key, value in fix_version_index.items()},
            project_index={key: frozenset(value) for key, value in project_index.items()},
            issue_type_index={key: frozenset(value) for key, value in issue_type_index.items()},
            priority_index={key: frozenset(value) for key, value in priority_index.items()},
            status_index={key: frozenset(value) for key, value in status_index.items()},
        )

    def metadata_seed_scores(self, query) -> Counter[int]:
        scores: Counter[int] = Counter()
        if query.project_key and query.project_key in self.project_index:
            for issue_id in self.project_index[query.project_key]:
                scores[issue_id] += 0.14
        if query.issue_type and query.issue_type in self.issue_type_index:
            for issue_id in self.issue_type_index[query.issue_type]:
                scores[issue_id] += 0.10
        if query.priority and query.priority in self.priority_index:
            for issue_id in self.priority_index[query.priority]:
                scores[issue_id] += 0.07
        if query.status and query.status in self.status_index:
            for issue_id in self.status_index[query.status]:
                scores[issue_id] += 0.05
        for component in query.component_terms:
            for issue_id in self.component_index.get(component, ()):
                scores[issue_id] += 0.20
        for version in query.affected_version_terms:
            for issue_id in self.affected_version_index.get(version, ()):
                scores[issue_id] += 0.12
        for version in query.fix_version_terms:
            for issue_id in self.fix_version_index.get(version, ()):
                scores[issue_id] += 0.12
        for issue_id, score in list(scores.items()):
            scores[issue_id] = min(score, 1.0)
        return scores

    def propagate(
        self,
        seed_scores: Counter[int],
        *,
        max_seed_nodes: int = 50,
        decay: float = 0.65,
        second_hop_decay: float = 0.28,
    ) -> Counter[int]:
        propagated: Counter[int] = Counter()
        for issue_id, seed_score in seed_scores.most_common(max_seed_nodes):
            propagated[issue_id] += seed_score
            for neighbor_id, edge_weight in self.issue_neighbors.get(issue_id, {}).items():
                hop_score = seed_score * edge_weight * decay
                propagated[neighbor_id] += hop_score
                for second_neighbor_id, second_edge_weight in self.issue_neighbors.get(neighbor_id, {}).items():
                    if second_neighbor_id == issue_id:
                        continue
                    propagated[second_neighbor_id] += hop_score * second_edge_weight * second_hop_decay
        return propagated

    def graph_context_score(self, query, candidate_id: int, index: SearchIndex) -> float:
        neighbors = self.issue_neighbors.get(candidate_id)
        if not neighbors:
            return 0.0

        total_weight = 0.0
        support = 0.0
        for neighbor_id, edge_weight in neighbors.items():
            prepared_neighbor = index.prepared.get(neighbor_id)
            if prepared_neighbor is None:
                continue
            neighbor_support = 0.0
            if query.project_key and prepared_neighbor.project_key == query.project_key:
                neighbor_support += 0.25
            neighbor_support += 0.35 * jaccard_similarity(query.component_terms, prepared_neighbor.component_terms)
            neighbor_support += 0.15 * jaccard_similarity(
                query.affected_version_terms,
                prepared_neighbor.affected_version_terms,
            )
            neighbor_support += 0.15 * jaccard_similarity(
                query.fix_version_terms,
                prepared_neighbor.fix_version_terms,
            )
            neighbor_support += 0.10 * _binary_match(query.issue_type, prepared_neighbor.issue_type)
            neighbor_support += 0.12 * jaccard_similarity(query.title_ngrams, prepared_neighbor.title_ngrams)
            support += neighbor_support * edge_weight
            total_weight += edge_weight

        if total_weight <= 0:
            return 0.0
        return min(1.0, support / total_weight)


class GraphMetadataCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        *,
        graph_space: GraphMetadataSpace,
        dense_space: DenseEmbeddingSpace,
        compute_device: str = "auto",
    ):
        self._graph_space = graph_space
        self._base_generator = ReciprocalRankFusionCandidateGenerator(
            dense_space=dense_space,
            compute_device=compute_device,
        )

    def generate(self, query, index: SearchIndex, *, pool_size: int) -> list[CandidateMatch]:
        logger.debug("Graph-metadata candidate generation started: pool=%s", pool_size)
        base_matches = self._base_generator.generate(query, index, pool_size=pool_size)
        seed_scores: Counter[int] = Counter()
        for rank, match in enumerate(base_matches, start=1):
            seed_scores[match.issue_id] += 1.0 / (8 + rank)
            seed_scores[match.issue_id] += min(max(match.seed_score, 0.0), 1.0)

        seed_scores.update(self._graph_space.metadata_seed_scores(query))
        propagated = self._graph_space.propagate(seed_scores)

        combined: Counter[int] = Counter()
        for issue_id, score in seed_scores.items():
            combined[issue_id] += score
        for issue_id, score in propagated.items():
            combined[issue_id] += score

        matches = [
            CandidateMatch(issue_id=issue_id, seed_score=score)
            for issue_id, score in combined.most_common(pool_size)
        ]
        logger.debug(
            "Graph-metadata candidate generation complete: base=%s combined=%s",
            len(base_matches),
            len(matches),
        )
        return matches


class GraphMetadataFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        *,
        graph_space: GraphMetadataSpace,
        dense_space: DenseEmbeddingSpace,
    ):
        self._graph_space = graph_space
        self._dense_feature_extractor = DenseSemanticFeatureExtractor(dense_space)

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = self._dense_feature_extractor.extract(query, candidate, index, seed_score=seed_score)
        feature_scores["component_overlap"] = round(
            jaccard_similarity(query.component_terms, candidate.component_terms),
            4,
        )
        feature_scores["affected_version_overlap"] = round(
            jaccard_similarity(query.affected_version_terms, candidate.affected_version_terms),
            4,
        )
        feature_scores["fix_version_overlap"] = round(
            jaccard_similarity(query.fix_version_terms, candidate.fix_version_terms),
            4,
        )
        feature_scores["project_match"] = _binary_match(query.project_key, candidate.project_key)
        feature_scores["issue_type_match"] = _binary_match(query.issue_type, candidate.issue_type)
        feature_scores["priority_match"] = _binary_match(query.priority, candidate.priority)
        feature_scores["status_match"] = _binary_match(query.status, candidate.status)
        feature_scores["metadata_alignment"] = round(
            (
                feature_scores["project_match"]
                + feature_scores["issue_type_match"]
                + feature_scores["priority_match"]
                + feature_scores["status_match"]
                + feature_scores["component_overlap"]
                + feature_scores["affected_version_overlap"]
                + feature_scores["fix_version_overlap"]
            )
            / 7.0,
            4,
        )
        feature_scores["graph_context"] = round(
            self._graph_space.graph_context_score(query, candidate.issue_id or -1, index),
            4,
        )
        feature_scores["graph_seed"] = round(min(1.0, seed_score), 4)
        return feature_scores


class GraphMetadataReranker(Reranker):
    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        score = (
            (0.16 * feature_scores.get("bm25_plus", 0.0))
            + (0.16 * feature_scores.get("dense_cosine", 0.0))
            + (0.20 * feature_scores.get("metadata_alignment", 0.0))
            + (0.20 * feature_scores.get("graph_context", 0.0))
            + (0.08 * feature_scores.get("graph_seed", 0.0))
            + (0.08 * feature_scores.get("title_ngram", 0.0))
            + (0.06 * feature_scores.get("description_overlap", 0.0))
            + (0.06 * feature_scores.get("component_overlap", 0.0))
        )
        score = max(0.0, min(1.0, score))
        reasons = self._build_reasons(feature_scores)
        return RerankResult(
            score=round(score, 6),
            feature_scores=feature_scores,
            reasons=reasons,
        )

    def _build_reasons(self, feature_scores: dict[str, float]) -> tuple[str, ...]:
        reasons: list[str] = []
        if feature_scores.get("graph_context", 0.0) >= 0.12:
            reasons.append("graph neighbors connected to similar issues support this match")
        if feature_scores.get("metadata_alignment", 0.0) >= 0.22:
            reasons.append("project, component, and issue metadata align closely")
        if feature_scores.get("dense_cosine", 0.0) >= 0.16:
            reasons.append("semantic embeddings also place the issues close together")
        if feature_scores.get("bm25_plus", 0.0) >= 0.22:
            reasons.append("sparse lexical evidence is still strong")
        if not reasons:
            reasons.append("graph and metadata evidence suggests only a weak connection")
        return tuple(reasons[:4])


def build_graph_metadata_pipelines(
    index: SearchIndex,
    *,
    dense_space: DenseEmbeddingSpace,
    compute_device: str = "auto",
) -> dict[str, RetrievalPipeline]:
    graph_space = GraphMetadataSpace.build(index)
    return {
        "graph-metadata-aware": RetrievalPipeline(
            name="graph-metadata-aware",
            candidate_generator=GraphMetadataCandidateGenerator(
                graph_space=graph_space,
                dense_space=dense_space,
                compute_device=compute_device,
            ),
            feature_extractor=GraphMetadataFeatureExtractor(graph_space=graph_space, dense_space=dense_space),
            reranker=GraphMetadataReranker(),
        )
    }
