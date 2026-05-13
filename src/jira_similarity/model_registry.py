from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

from .pipeline import RetrievalPipeline, build_pipeline_registry

logger = logging.getLogger(__name__)

GPU_ACCELERATED_MODELS = frozenset(
    {
        "logreg-engineered",
        "random-indexing-dense",
        "sbert-dense",
        "hybrid-sparse-dense",
        "pairwise-neural-mlp",
        "rag-hybrid-judge",
        "graph-metadata-aware",
        "llm-e5-large",
        "llm-e5-cross-reranker",
    }
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    family: str
    stage: str
    runnable: bool
    description: str
    strengths: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_model_catalog() -> dict[str, ModelSpec]:
    logger.debug("Building model catalog")
    return {
        "tfidf-cosine": ModelSpec(
            name="tfidf-cosine",
            family="sparse_lexical",
            stage="retrieval",
            runnable=True,
            description="TF-IDF cosine sparse retrieval baseline over prepared Jira issue text.",
            strengths=("simple", "interpretable", "important classical benchmark"),
            limitations=("lexical only", "often weaker than BM25 on ranking tasks"),
        ),
        "bm25": ModelSpec(
            name="bm25",
            family="sparse_lexical",
            stage="retrieval",
            runnable=True,
            description="Standard BM25 sparse lexical retrieval baseline.",
            strengths=("strong IR baseline", "fast", "good for exact identifiers"),
            limitations=("surface-form heavy", "limited semantic generalization"),
        ),
        "bm25-plus": ModelSpec(
            name="bm25-plus",
            family="sparse_lexical",
            stage="retrieval",
            runnable=True,
            description="BM25+ sparse lexical retrieval baseline.",
            strengths=("robust BM25 variant", "useful lexical comparison point"),
            limitations=("lexical only", "needs empirical comparison against BM25"),
        ),
        "lexical": ModelSpec(
            name="lexical",
            family="sparse_lexical",
            stage="retrieval",
            runnable=True,
            description="Compatibility alias for the BM25 sparse lexical baseline.",
            strengths=("backward compatible", "simple baseline alias"),
            limitations=("prefer bm25 for explicit experiments",),
        ),
        "logreg-engineered": ModelSpec(
            name="logreg-engineered",
            family="classical_supervised_ml",
            stage="reranking",
            runnable=True,
            description="Logistic regression over engineered lexical and Jira metadata pair features.",
            strengths=("interpretable learned weights", "uses structured Jira fields", "lightweight supervised model"),
            limitations=("needs labeled linked pairs", "weaker semantic generalization than dense models"),
        ),
        "random-indexing-dense": ModelSpec(
            name="random-indexing-dense",
            family="dense_semantic_embeddings",
            stage="retrieval",
            runnable=True,
            description="Random-indexing dense semantic embedding baseline with cosine retrieval.",
            strengths=("dense local embeddings", "captures some co-occurrence semantics", "no external ML dependency"),
            limitations=("weaker than transformer encoders", "still an early dense baseline"),
        ),
        "sbert-dense": ModelSpec(
            name="sbert-dense",
            family="dense_semantic_embeddings",
            stage="retrieval",
            runnable=True,
            description="Sentence-transformer SBERT dense retrieval baseline with cosine candidate ranking.",
            strengths=("strong semantic retrieval", "robust to paraphrases", "optional CUDA acceleration"),
            limitations=("requires optional sentence-transformers dependency", "higher compute/memory cost"),
        ),
        "hybrid-sparse-dense": ModelSpec(
            name="hybrid-sparse-dense",
            family="hybrid_sparse_dense_retrieval",
            stage="retrieval",
            runnable=True,
            description="Hybrid sparse-dense retrieval using BM25+ and dense cosine candidate fusion with RRF.",
            strengths=("balances exact lexical matches with semantic similarity", "robust fused candidate generation"),
            limitations=("more moving parts than single-family baselines", "fusion weights still need empirical tuning"),
        ),
        "pairwise-neural-mlp": ModelSpec(
            name="pairwise-neural-mlp",
            family="deep_pairwise_duplicate_classification",
            stage="classification",
            runnable=True,
            description="Dependency-free neural pairwise duplicate classifier with hybrid candidate generation.",
            strengths=("nonlinear pairwise scoring", "uses lexical, metadata, and dense features", "good bridge to later transformer classifiers"),
            limitations=("still lighter than transformer architectures", "training is slower than linear models"),
        ),
        "rag-hybrid-judge": ModelSpec(
            name="rag-hybrid-judge",
            family="llm_rag_style",
            stage="reasoning",
            runnable=True,
            description="Local RAG-style judge over hybrid retrieval with natural-language duplicate reasoning.",
            strengths=("retrieval plus reasoning architecture", "human-readable explanations", "easy to swap for a real LLM later"),
            limitations=("uses a local reasoning fallback instead of a hosted LLM", "not a full generative model yet"),
        ),
        "graph-metadata-aware": ModelSpec(
            name="graph-metadata-aware",
            family="graph_metadata_aware",
            stage="retrieval",
            runnable=True,
            description="Graph- and metadata-aware retrieval using explicit issue links plus metadata propagation.",
            strengths=("uses issue graph structure", "leverages Jira metadata strongly", "captures subsystem context"),
            limitations=("graph quality depends on available links and metadata", "heavier than non-graph baselines"),
        ),
        "llm-e5-large": ModelSpec(
            name="llm-e5-large",
            family="llm_embedding_retrieval",
            stage="retrieval",
            runnable=True,
            description="Sentence-transformer retrieval using E5-large embeddings with cosine candidate ranking.",
            strengths=("strong semantic retrieval", "robust on paraphrases", "GPU-accelerated embedding scoring"),
            limitations=("requires optional sentence-transformers dependency", "higher compute cost than sparse baselines"),
        ),
        "llm-e5-cross-reranker": ModelSpec(
            name="llm-e5-cross-reranker",
            family="llm_cross_encoder_reranking",
            stage="classification",
            runnable=True,
            description="E5 retrieval with BGE cross-encoder reranking for high-precision duplicate scoring.",
            strengths=("strong pairwise semantic judgments", "better duplicate precision than retrieval-only models"),
            limitations=("slower than bi-encoder retrieval", "requires optional sentence-transformers dependency"),
        ),
    }


def build_runnable_pipeline_registry(
    index,
    *,
    holdout_issue_ids: frozenset[int] = frozenset(),
    compute_device: str = "auto",
    requested_models: frozenset[str] | None = None,
) -> dict[str, RetrievalPipeline]:
    logger.info(
        "Building runnable pipeline registry: holdout_count=%s compute_device=%s requested=%s",
        len(holdout_issue_ids),
        compute_device,
        sorted(requested_models) if requested_models is not None else "all",
    )
    pipelines = build_pipeline_registry(
        index,
        holdout_issue_ids=holdout_issue_ids,
        compute_device=compute_device,
        requested_models=requested_models,
    )
    catalog = build_model_catalog()
    runnable = {name: pipeline for name, pipeline in pipelines.items() if name in catalog}
    logger.info(
        "Runnable pipeline registry complete: total_built=%s runnable=%s",
        len(pipelines),
        len(runnable),
    )
    return runnable


def resolve_model_names(requested: list[str] | tuple[str, ...] | None) -> list[str]:
    catalog = build_model_catalog()
    if not requested:
        logger.debug("resolve_model_names: no names requested, returning all %s models", len(catalog))
        return [name for name in catalog]

    resolved: list[str] = []
    for name in requested:
        lowered = name.strip().lower()
        if lowered in {"all", "all-runnable", "*"}:
            logger.debug("resolve_model_names: expanding wildcard '%s' to all %s models", name, len(catalog))
            resolved.extend(catalog)
            continue
        if lowered not in catalog:
            available = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        resolved.append(lowered)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in resolved:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    logger.debug("resolve_model_names: requested=%s resolved=%s", list(requested), deduped)
    return deduped


def model_supports_gpu(model_name: str) -> bool:
    return model_name in GPU_ACCELERATED_MODELS
