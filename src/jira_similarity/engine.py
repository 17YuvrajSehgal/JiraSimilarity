from __future__ import annotations

import logging

from .domain import IssueDocument, IssueQuery, ModelEvaluation, SearchResult
from .model_registry import build_model_catalog, build_runnable_pipeline_registry, resolve_model_names
from .pipeline import RetrievalPipeline, SearchIndex
from .text import prepare_issue

logger = logging.getLogger(__name__)


class JiraSimilarityEngine:
    def __init__(
        self,
        documents: list[IssueDocument],
        *,
        candidate_pool_size: int = 250,
        compute_device: str = "auto",
    ):
        logger.info(
            "Initialising JiraSimilarityEngine: documents=%s candidate_pool_size=%s compute_device=%s",
            len(documents),
            candidate_pool_size,
            compute_device,
        )
        self._index = SearchIndex.build(documents)
        self._candidate_pool_size = candidate_pool_size
        self._compute_device = compute_device
        self._pipelines = build_runnable_pipeline_registry(self._index, compute_device=compute_device)
        self._model_catalog = build_model_catalog()
        logger.info("Jira similarity engine ready with pipelines: %s", ", ".join(sorted(self._pipelines)))

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self._pipelines))

    @property
    def pipelines(self) -> tuple[str, ...]:
        return self.strategies

    @property
    def model_catalog(self) -> dict[str, dict[str, object]]:
        return {name: spec.to_dict() for name, spec in self._model_catalog.items()}

    def search(
        self,
        query: IssueQuery | IssueDocument,
        *,
        model_name: str = "bm25",
        top_k: int = 10,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        pipeline = self._resolve_pipeline(model_name)
        logger.debug("Running search with model=%s top_k=%s", model_name, top_k)
        return self._run_pipeline(
            query,
            pipeline,
            model_name=model_name,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )

    def _run_pipeline(
        self,
        query: IssueQuery | IssueDocument,
        pipeline: RetrievalPipeline,
        *,
        model_name: str,
        top_k: int,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        prepared_query = prepare_issue(query)
        pool_size = candidate_pool_size or self._candidate_pool_size
        candidate_matches = pipeline.candidate_generator.generate(
            prepared_query,
            self._index,
            pool_size=pool_size,
        )
        logger.debug("Candidate generation complete for model=%s candidates=%s", model_name, len(candidate_matches))

        results: list[SearchResult] = []
        for match in candidate_matches:
            candidate_document = self._index.documents[match.issue_id]
            candidate_prepared = self._index.prepared[match.issue_id]
            if prepared_query.issue_id is not None and prepared_query.issue_id == match.issue_id:
                continue
            if prepared_query.issue_key and prepared_query.issue_key == candidate_document.issue_key:
                continue

            feature_scores = pipeline.feature_extractor.extract(
                prepared_query,
                candidate_prepared,
                self._index,
                seed_score=match.seed_score,
            )
            rerank_result = pipeline.reranker.rerank(feature_scores)
            results.append(
                SearchResult(
                    rank=0,
                    issue=candidate_document,
                    score=rerank_result.score,
                    model_name=model_name,
                    feature_scores=rerank_result.feature_scores,
                    reasons=rerank_result.reasons,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        for index, result in enumerate(results[:top_k], start=1):
            result.rank = index
        logger.debug("Search complete for model=%s returned=%s", model_name, min(top_k, len(results)))
        return results[:top_k]

    def find_duplicates(
        self,
        query: IssueQuery | IssueDocument,
        *,
        model_name: str = "bm25",
        threshold: float = 0.55,
        top_k: int = 10,
    ) -> list[SearchResult]:
        logger.debug(
            "find_duplicates: model=%s threshold=%.2f top_k=%s", model_name, threshold, top_k
        )
        results = self.search(query, model_name=model_name, top_k=top_k)
        filtered = [result for result in results if result.score >= threshold]
        logger.debug(
            "find_duplicates: candidates=%s above_threshold=%s (threshold=%.2f)",
            len(results),
            len(filtered),
            threshold,
        )
        return filtered

    def compare_models(
        self,
        query: IssueQuery | IssueDocument,
        *,
        model_names: list[str] | tuple[str, ...],
        top_k: int = 10,
    ) -> dict[str, list[SearchResult]]:
        resolved_model_names = resolve_model_names(model_names)
        logger.info(
            "compare_models: running %s models top_k=%s: %s",
            len(resolved_model_names),
            top_k,
            ", ".join(resolved_model_names),
        )
        results = {
            model_name: self.search(query, model_name=model_name, top_k=top_k)
            for model_name in resolved_model_names
            if model_name in self._pipelines
        }
        logger.debug("compare_models: done, returned results for %s models", len(results))
        return results

    def evaluate(
        self,
        *,
        task: str,
        model_names: list[str] | tuple[str, ...],
        sample_size: int | None = None,
        top_k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> list[ModelEvaluation]:
        if task not in {"similarity", "duplicates"}:
            raise ValueError("task must be 'similarity' or 'duplicates'")

        label_map: dict[int, set[int]] = {}
        for document in self._index.documents.values():
            labels = set(document.linked_issue_ids if task == "similarity" else document.duplicate_issue_ids)
            if labels:
                label_map[document.issue_id] = labels

        query_ids = sorted(label_map)
        if sample_size is not None:
            query_ids = query_ids[:sample_size]

        max_k = max(top_k_values)
        reports: list[ModelEvaluation] = []
        for model_name in resolve_model_names(model_names):
            if model_name not in self._pipelines:
                continue
            logger.info("Evaluating model=%s task=%s queries=%s", model_name, task, len(query_ids))
            reciprocal_ranks: list[float] = []
            average_precisions: dict[int, list[float]] = {k: [] for k in top_k_values}
            recalls: dict[int, list[float]] = {k: [] for k in top_k_values}
            threshold_counts = {
                "0.45": {"tp": 0, "fp": 0, "fn": 0},
                "0.55": {"tp": 0, "fp": 0, "fn": 0},
                "0.65": {"tp": 0, "fp": 0, "fn": 0},
            }

            for query_id in query_ids:
                query_document = self._index.documents[query_id]
                relevant = label_map[query_id]
                results = self._search_for_evaluation(query_document, model_name=model_name, top_k=max_k)
                predicted_ids = [result.issue.issue_id for result in results]
                reciprocal_ranks.append(self._reciprocal_rank(predicted_ids, relevant))

                for k in top_k_values:
                    top_ids = predicted_ids[:k]
                    recalls[k].append(self._recall_at_k(top_ids, relevant))
                    average_precisions[k].append(self._average_precision(top_ids, relevant))

                if task == "duplicates":
                    for threshold, counts in threshold_counts.items():
                        predicted_positive = {
                            result.issue.issue_id for result in results if result.score >= float(threshold)
                        }
                        counts["tp"] += len(predicted_positive & relevant)
                        counts["fp"] += len(predicted_positive - relevant)
                        counts["fn"] += len(relevant - predicted_positive)

                if len(reciprocal_ranks) % 25 == 0:
                    logger.debug(
                        "Evaluation progress model=%s task=%s processed=%s/%s",
                        model_name,
                        task,
                        len(reciprocal_ranks),
                        len(query_ids),
                    )

            reports.append(
                ModelEvaluation(
                    model_name=model_name,
                    task=task,
                    queries_evaluated=len(query_ids),
                    mrr=round(self._mean(reciprocal_ranks), 6),
                    map_at_k={k: round(self._mean(values), 6) for k, values in average_precisions.items()},
                    recall_at_k={k: round(self._mean(values), 6) for k, values in recalls.items()},
                    threshold_metrics=self._threshold_metrics(threshold_counts) if task == "duplicates" else {},
                )
            )
            logger.info(
                "Evaluation complete: model=%s task=%s queries=%s mrr=%.4f map@10=%.4f recall@10=%.4f",
                model_name,
                task,
                len(query_ids),
                self._mean(reciprocal_ranks),
                self._mean(average_precisions.get(10, average_precisions.get(max(top_k_values), []))),
                self._mean(recalls.get(10, recalls.get(max(top_k_values), []))),
            )
            logger.info("Finished evaluating model=%s task=%s", model_name, task)
        return reports

    def _search_for_evaluation(
        self,
        query_document: IssueDocument,
        *,
        model_name: str,
        top_k: int,
    ) -> list[SearchResult]:
        model_spec = self._model_catalog[model_name]
        if model_spec.family not in {"classical_supervised_ml", "deep_pairwise_duplicate_classification"}:
            return self.search(query_document, model_name=model_name, top_k=top_k)

        logger.debug(
            "_search_for_evaluation: using holdout mode for model=%s issue_id=%s",
            model_name,
            query_document.issue_id,
        )
        pipeline = build_runnable_pipeline_registry(
            self._index,
            holdout_issue_ids=frozenset({query_document.issue_id}),
            compute_device=self._compute_device,
        )[model_name]
        return self._run_pipeline(
            query_document,
            pipeline,
            model_name=model_name,
            top_k=top_k,
        )

    def _resolve_pipeline(self, model_name: str):
        try:
            return self._pipelines[model_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._model_catalog))
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}") from exc

    @staticmethod
    def _reciprocal_rank(predicted_ids: list[int], relevant: set[int]) -> float:
        for index, issue_id in enumerate(predicted_ids, start=1):
            if issue_id in relevant:
                return 1.0 / index
        return 0.0

    @staticmethod
    def _recall_at_k(predicted_ids: list[int], relevant: set[int]) -> float:
        if not relevant:
            return 0.0
        return len(set(predicted_ids) & relevant) / len(relevant)

    @staticmethod
    def _average_precision(predicted_ids: list[int], relevant: set[int]) -> float:
        if not relevant:
            return 0.0
        hit_count = 0
        precision_sum = 0.0
        for index, issue_id in enumerate(predicted_ids, start=1):
            if issue_id in relevant:
                hit_count += 1
                precision_sum += hit_count / index
        return precision_sum / len(relevant)

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _threshold_metrics(counts_by_threshold: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for threshold, counts in counts_by_threshold.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
            metrics[threshold] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        return metrics
