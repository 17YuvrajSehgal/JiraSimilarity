from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

from .domain import ModelEvaluation
from .engine import JiraSimilarityEngine
from .model_registry import resolve_model_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    name: str
    description: str
    task: str
    model_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    suite_name: str
    task: str
    model_names: tuple[str, ...]
    sample_size: int | None
    top_k_values: tuple[int, ...]
    evaluations: tuple[ModelEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_name": self.suite_name,
            "task": self.task,
            "model_names": list(self.model_names),
            "sample_size": self.sample_size,
            "top_k_values": list(self.top_k_values),
            "evaluations": [
                {
                    "model_name": evaluation.model_name,
                    "task": evaluation.task,
                    "queries_evaluated": evaluation.queries_evaluated,
                    "mrr": evaluation.mrr,
                    "map_at_k": evaluation.map_at_k,
                    "recall_at_k": evaluation.recall_at_k,
                    "precision_at_k": evaluation.precision_at_k,
                    "ndcg_at_k": evaluation.ndcg_at_k,
                    "hit_rate_at_k": evaluation.hit_rate_at_k,
                    "threshold_metrics": evaluation.threshold_metrics,
                }
                for evaluation in self.evaluations
            ],
        }


def build_benchmark_suites() -> dict[str, BenchmarkSuite]:
    sparse_models = ("tfidf-cosine", "bm25", "bm25-plus")
    return {
        "sparse-lexical-similarity": BenchmarkSuite(
            name="sparse-lexical-similarity",
            description="Compare sparse lexical retrieval methods for similar-issue search.",
            task="similarity",
            model_names=sparse_models,
        ),
        "sparse-lexical-duplicates": BenchmarkSuite(
            name="sparse-lexical-duplicates",
            description="Compare sparse lexical retrieval methods for duplicate-oriented ranking.",
            task="duplicates",
            model_names=sparse_models,
        ),
        "classical-ml-similarity": BenchmarkSuite(
            name="classical-ml-similarity",
            description="Compare sparse lexical retrieval against the engineered-feature logistic model.",
            task="similarity",
            model_names=("bm25", "bm25-plus", "logreg-engineered"),
        ),
        "classical-ml-duplicates": BenchmarkSuite(
            name="classical-ml-duplicates",
            description="Compare duplicate-oriented ranking against the engineered-feature logistic model.",
            task="duplicates",
            model_names=("bm25", "bm25-plus", "logreg-engineered"),
        ),
        "dense-semantic-similarity": BenchmarkSuite(
            name="dense-semantic-similarity",
            description="Compare sparse, supervised, and dense semantic retrieval models.",
            task="similarity",
            model_names=("bm25-plus", "logreg-engineered", "random-indexing-dense"),
        ),
        "dense-semantic-duplicates": BenchmarkSuite(
            name="dense-semantic-duplicates",
            description="Compare duplicate-oriented ranking across sparse, supervised, and dense semantic models.",
            task="duplicates",
            model_names=("bm25-plus", "logreg-engineered", "random-indexing-dense"),
        ),
        "hybrid-sparse-dense-similarity": BenchmarkSuite(
            name="hybrid-sparse-dense-similarity",
            description="Compare sparse, dense, and hybrid retrieval for similar-issue search.",
            task="similarity",
            model_names=("bm25-plus", "random-indexing-dense", "hybrid-sparse-dense"),
        ),
        "hybrid-sparse-dense-duplicates": BenchmarkSuite(
            name="hybrid-sparse-dense-duplicates",
            description="Compare sparse, dense, and hybrid retrieval for duplicate-oriented ranking.",
            task="duplicates",
            model_names=("bm25-plus", "random-indexing-dense", "hybrid-sparse-dense"),
        ),
        "deep-pairwise-duplicates": BenchmarkSuite(
            name="deep-pairwise-duplicates",
            description="Compare hybrid retrieval and neural pairwise duplicate classification.",
            task="duplicates",
            model_names=("hybrid-sparse-dense", "logreg-engineered", "pairwise-neural-mlp"),
        ),
        "deep-pairwise-similarity": BenchmarkSuite(
            name="deep-pairwise-similarity",
            description="Compare hybrid retrieval against the neural pairwise reranker for similar-issue ranking.",
            task="similarity",
            model_names=("hybrid-sparse-dense", "pairwise-neural-mlp"),
        ),
        "llm-rag-duplicates": BenchmarkSuite(
            name="llm-rag-duplicates",
            description="Compare hybrid retrieval, neural reranking, and RAG-style reasoning for duplicates.",
            task="duplicates",
            model_names=("hybrid-sparse-dense", "pairwise-neural-mlp", "rag-hybrid-judge"),
        ),
        "llm-rag-similarity": BenchmarkSuite(
            name="llm-rag-similarity",
            description="Compare hybrid retrieval and RAG-style reasoning for similar-issue ranking.",
            task="similarity",
            model_names=("hybrid-sparse-dense", "rag-hybrid-judge"),
        ),
        "graph-metadata-similarity": BenchmarkSuite(
            name="graph-metadata-similarity",
            description="Compare hybrid, RAG-style, and graph-metadata-aware retrieval.",
            task="similarity",
            model_names=("hybrid-sparse-dense", "rag-hybrid-judge", "graph-metadata-aware"),
        ),
        "graph-metadata-duplicates": BenchmarkSuite(
            name="graph-metadata-duplicates",
            description="Compare graph-metadata-aware retrieval against hybrid and RAG-style duplicate ranking.",
            task="duplicates",
            model_names=("hybrid-sparse-dense", "rag-hybrid-judge", "graph-metadata-aware"),
        ),
    }


class BenchmarkRunner:
    def __init__(self, engine: JiraSimilarityEngine):
        self._engine = engine
        self._suites = build_benchmark_suites()

    def run(
        self,
        *,
        task: str,
        model_names: list[str] | tuple[str, ...],
        sample_size: int | None = None,
        top_k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> BenchmarkRunResult:
        resolved_model_names = tuple(resolve_model_names(model_names))
        logger.info(
            "Running ad-hoc benchmark: task=%s models=%s sample_size=%s top_k_values=%s",
            task,
            ", ".join(resolved_model_names),
            sample_size,
            top_k_values,
        )
        evaluations = tuple(
            self._engine.evaluate(
                task=task,
                model_names=resolved_model_names,
                sample_size=sample_size,
                top_k_values=top_k_values,
            )
        )
        result = BenchmarkRunResult(
            suite_name="ad-hoc",
            task=task,
            model_names=resolved_model_names,
            sample_size=sample_size,
            top_k_values=top_k_values,
            evaluations=evaluations,
        )
        logger.info("Ad-hoc benchmark finished: evaluations=%s", len(evaluations))
        return result

    def run_suite(
        self,
        suite_name: str,
        *,
        sample_size: int | None = None,
        top_k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> BenchmarkRunResult:
        try:
            suite = self._suites[suite_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._suites))
            raise ValueError(f"Unknown benchmark suite '{suite_name}'. Available suites: {available}") from exc

        logger.info(
            "Running benchmark suite: suite=%s task=%s models=%s sample_size=%s top_k_values=%s",
            suite.name,
            suite.task,
            ", ".join(suite.model_names),
            sample_size,
            top_k_values,
        )
        evaluations = tuple(
            self._engine.evaluate(
                task=suite.task,
                model_names=suite.model_names,
                sample_size=sample_size,
                top_k_values=top_k_values,
            )
        )
        result = BenchmarkRunResult(
            suite_name=suite.name,
            task=suite.task,
            model_names=suite.model_names,
            sample_size=sample_size,
            top_k_values=top_k_values,
            evaluations=evaluations,
        )
        logger.info("Benchmark suite finished: suite=%s evaluations=%s", suite.name, len(evaluations))
        return result
