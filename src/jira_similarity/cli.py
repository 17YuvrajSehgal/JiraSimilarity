from __future__ import annotations

import argparse
import json
from typing import Iterable

from .benchmarking import BenchmarkRunner, build_benchmark_suites
from .bootstrap import ApplicationBuilder
from .config import DatabaseConfig, RuntimeConfig, SourceConfig
from .domain import IssueQuery, SearchResult
from .logging_utils import configure_logging
from .model_registry import build_model_catalog, resolve_model_names


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.command == "models":
        catalog = build_model_catalog()
        model_names = resolve_model_names(args.models)
        payload = []
        for model_name in model_names:
            model_spec = catalog[model_name].to_dict()
            payload.append(model_spec)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "suites":
        suites = {name: suite.to_dict() for name, suite in build_benchmark_suites().items()}
        print(json.dumps(suites, indent=2))
        return 0

    runtime = RuntimeConfig.from_env().with_overrides(compute_device=args.compute_device)
    source_config = SourceConfig.from_env().with_overrides(
        kind=args.source,
        json_path=args.json_path,
    )
    application = ApplicationBuilder(
        source_config=source_config,
        runtime_config=runtime,
        database_config=DatabaseConfig.from_env(),
    ).build()
    engine = application.build_engine()
    benchmark_runner = BenchmarkRunner(engine)

    if args.command == "benchmark":
        if args.suite:
            result = benchmark_runner.run_suite(
                args.suite,
                sample_size=args.sample_size,
                top_k_values=tuple(args.top_k_values),
            )
        else:
            if not args.task:
                parser.error("benchmark requires --task when --suite is not provided")
            result = benchmark_runner.run(
                task=args.task,
                model_names=args.models,
                sample_size=args.sample_size,
                top_k_values=tuple(args.top_k_values),
            )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "similar":
        query = build_query(args)
        results = engine.search(query, model_name=args.model, top_k=args.top_k)
        print(_render_results(results))
        return 0

    if args.command == "duplicates":
        query = build_query(args)
        results = engine.find_duplicates(
            query,
            model_name=args.model,
            threshold=args.threshold,
            top_k=args.top_k,
        )
        print(_render_results(results))
        return 0

    if args.command == "evaluate":
        reports = engine.evaluate(
            task=args.task,
            model_names=args.models,
            sample_size=args.sample_size,
            top_k_values=tuple(args.top_k_values),
        )
        print(json.dumps([_report_to_dict(report) for report in reports], indent=2))
        return 0

    parser.error("Unknown command")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jira similarity and duplicate-detection toolkit.")
    parser.add_argument(
        "--source",
        choices=("mysql", "json", "jira_api"),
        help="Select the issue source adapter. Defaults to JIRA_SOURCE_KIND or mysql.",
    )
    parser.add_argument(
        "--json-path",
        help="Path to a JSON file when using the json source adapter.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Set application logging verbosity.",
    )
    parser.add_argument(
        "--compute-device",
        choices=("auto", "cpu", "cuda"),
        help="Execution device for optional torch acceleration. Defaults to JIRA_COMPUTE_DEVICE or auto.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="List the implemented model families and runnable models.")
    models.add_argument("--models", nargs="+", default=["all"])

    subparsers.add_parser("suites", help="List the available benchmark suites.")

    benchmark = subparsers.add_parser("benchmark", help="Run a benchmark suite or ad-hoc evaluation.")
    benchmark.add_argument("--suite")
    benchmark.add_argument("--task", choices=("similarity", "duplicates"))
    benchmark.add_argument("--models", nargs="+", default=["all"])
    benchmark.add_argument("--sample-size", type=int)
    benchmark.add_argument("--top-k-values", nargs="+", type=int, default=[1, 3, 5, 10])

    similar = subparsers.add_parser("similar", help="Find relevant historical issues with the selected model.")
    _add_query_arguments(similar)
    similar.add_argument("--model", default="bm25")
    similar.add_argument("--top-k", type=int, default=10)

    duplicates = subparsers.add_parser("duplicates", help="Find likely duplicates with the selected model.")
    _add_query_arguments(duplicates)
    duplicates.add_argument("--model", default="bm25")
    duplicates.add_argument("--top-k", type=int, default=10)
    duplicates.add_argument("--threshold", type=float, default=0.55)

    evaluate = subparsers.add_parser("evaluate", help="Compare the implemented models on TAWOS links.")
    evaluate.add_argument("--task", choices=("similarity", "duplicates"), required=True)
    evaluate.add_argument(
        "--models",
        nargs="+",
        default=["all"],
    )
    evaluate.add_argument("--sample-size", type=int)
    evaluate.add_argument("--top-k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    return parser


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--project")
    parser.add_argument("--issue-type")
    parser.add_argument("--priority")
    parser.add_argument("--status")
    parser.add_argument("--components", nargs="*", default=[])
    parser.add_argument("--affected-versions", nargs="*", default=[])
    parser.add_argument("--fix-versions", nargs="*", default=[])


def build_query(args: argparse.Namespace) -> IssueQuery:
    return IssueQuery(
        title=args.title,
        description_text=args.description,
        project_key=args.project,
        issue_type=args.issue_type,
        priority=args.priority,
        status=args.status,
        components=tuple(args.components),
        affected_versions=tuple(args.affected_versions),
        fix_versions=tuple(args.fix_versions),
    )


def _render_results(results: Iterable[SearchResult]) -> str:
    lines: list[str] = []
    for result in results:
        reasons = "; ".join(result.reasons) if result.reasons else "no explanation available"
        lines.append(
            f"{result.rank:>2}. {result.issue.issue_key:<20} score={result.score:.3f} "
            f"project={result.issue.project_key or '-'} title={result.issue.title}"
        )
        lines.append(f"    reasons: {reasons}")
    return "\n".join(lines) if lines else "No results matched the current query."


def _report_to_dict(report) -> dict[str, object]:
    return {
        "model_name": report.model_name,
        "task": report.task,
        "queries_evaluated": report.queries_evaluated,
        "mrr": report.mrr,
        "map_at_k": report.map_at_k,
        "recall_at_k": report.recall_at_k,
        "threshold_metrics": report.threshold_metrics,
    }
