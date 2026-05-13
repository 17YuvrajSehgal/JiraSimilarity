from __future__ import annotations

from dataclasses import dataclass
import math
import logging

from ..compute import resolve_torch_runtime
from ..pipeline import (
    BM25CandidateGenerator,
    CandidateMatch,
    FeatureExtractor,
    RetrievalPipeline,
    RerankResult,
    Reranker,
    SearchIndex,
    StandardFeatureExtractor,
)
from ..text import jaccard_similarity, overlap_ratio
from .training_diagnostics import (
    TrainingEpochMetrics,
    TrainingRunDiagnostics,
    binary_cross_entropy,
    binary_metrics_from_probabilities,
    split_binary_examples,
    training_diagnostics_enabled,
    write_training_run_diagnostics,
)

logger = logging.getLogger(__name__)

ENGINEERED_FEATURE_NAMES = (
    "bm25",
    "bm25_plus",
    "tfidf_cosine",
    "title_overlap",
    "title_ngram",
    "description_overlap",
    "term_overlap",
    "component_overlap",
    "affected_version_overlap",
    "fix_version_overlap",
    "project_match",
    "issue_type_match",
    "priority_match",
    "status_match",
    "length_ratio",
    "candidate_seed",
)
CLASSICAL_SUPERVISED_RERANK_POOL_SIZE = 80


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _binary_match(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    return 1.0 if left == right else 0.0


def _length_ratio(left: int, right: int) -> float:
    larger = max(left, right)
    if larger <= 0:
        return 0.0
    return min(left, right) / larger


@dataclass(frozen=True, slots=True)
class PairTrainingExample:
    features: dict[str, float]
    label: int


@dataclass(frozen=True, slots=True)
class LogisticRegressionModel:
    feature_names: tuple[str, ...]
    weights: dict[str, float]
    bias: float

    def predict_proba(self, features: dict[str, float]) -> float:
        score = self.bias
        for name in self.feature_names:
            score += self.weights.get(name, 0.0) * features.get(name, 0.0)
        return _sigmoid(score)


class EngineeredFeatureExtractor(StandardFeatureExtractor):
    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = super().extract(query, candidate, index, seed_score=seed_score)
        feature_scores.update(
            {
                "title_overlap": round(jaccard_similarity(query.title_terms, candidate.title_terms), 4),
                "term_overlap": round(
                    overlap_ratio(frozenset(query.term_frequency), frozenset(candidate.term_frequency)),
                    4,
                ),
                "component_overlap": round(jaccard_similarity(query.component_terms, candidate.component_terms), 4),
                "affected_version_overlap": round(
                    jaccard_similarity(query.affected_version_terms, candidate.affected_version_terms),
                    4,
                ),
                "fix_version_overlap": round(
                    jaccard_similarity(query.fix_version_terms, candidate.fix_version_terms),
                    4,
                ),
                "project_match": _binary_match(query.project_key, candidate.project_key),
                "issue_type_match": _binary_match(query.issue_type, candidate.issue_type),
                "priority_match": _binary_match(query.priority, candidate.priority),
                "status_match": _binary_match(query.status, candidate.status),
                "length_ratio": round(_length_ratio(query.document_length, candidate.document_length), 4),
            }
        )
        return feature_scores


class PairTrainingSetBuilder:
    def __init__(
        self,
        *,
        feature_extractor: FeatureExtractor,
        candidate_generator=None,
        negatives_per_positive: int = 4,
        hard_negative_pool_size: int = 25,
    ):
        self._feature_extractor = feature_extractor
        self._negatives_per_positive = negatives_per_positive
        self._hard_negative_pool_size = hard_negative_pool_size
        self._candidate_generator = candidate_generator or BM25CandidateGenerator()

    def build(
        self,
        index: SearchIndex,
        *,
        holdout_issue_ids: frozenset[int] = frozenset(),
    ) -> list[PairTrainingExample]:
        logger.info(
            "Building engineered-feature training set: issues=%s holdout=%s negatives_per_positive=%s",
            len(index.documents),
            len(holdout_issue_ids),
            self._negatives_per_positive,
        )
        examples: list[PairTrainingExample] = []
        fallback_negatives = self._sorted_issue_ids(index)

        for index_count, issue_id in enumerate(fallback_negatives, start=1):
            if issue_id in holdout_issue_ids:
                continue

            query = index.prepared[issue_id]
            positive_targets = [
                candidate_id
                for candidate_id in sorted(query.linked_issue_ids | query.duplicate_issue_ids)
                if candidate_id in index.prepared and candidate_id not in holdout_issue_ids and candidate_id != issue_id
            ]
            if not positive_targets:
                continue

            if index_count % 25000 == 0:
                logger.info("PairTrainingSetBuilder: processed %s/%s issues...", index_count, len(fallback_negatives))

            candidate_matches = self._candidate_generator.generate(
                query,
                index,
                pool_size=max(self._hard_negative_pool_size, len(positive_targets) * 4),
            )
            candidate_seed_by_issue = {match.issue_id: match.seed_score for match in candidate_matches}
            hard_negatives = self._select_negatives(
                query_id=issue_id,
                positive_targets=set(positive_targets),
                candidate_matches=candidate_matches,
                fallback_issue_ids=fallback_negatives,
                holdout_issue_ids=holdout_issue_ids,
            )

            for target_id in positive_targets:
                candidate = index.prepared[target_id]
                features = self._feature_extractor.extract(
                    query,
                    candidate,
                    index,
                    seed_score=candidate_seed_by_issue.get(target_id, 0.0),
                )
                examples.append(PairTrainingExample(features=features, label=1))

            negative_limit = max(len(positive_targets) * self._negatives_per_positive, 1)
            for candidate_id, seed_score in hard_negatives[:negative_limit]:
                candidate = index.prepared[candidate_id]
                features = self._feature_extractor.extract(query, candidate, index, seed_score=seed_score)
                examples.append(PairTrainingExample(features=features, label=0))

        logger.info("Engineered-feature training set ready: examples=%s", len(examples))
        return examples

    @staticmethod
    def _sorted_issue_ids(index: SearchIndex) -> list[int]:
        return sorted(index.documents)

    def _select_negatives(
        self,
        *,
        query_id: int,
        positive_targets: set[int],
        candidate_matches: list[CandidateMatch],
        fallback_issue_ids: list[int],
        holdout_issue_ids: frozenset[int],
    ) -> list[tuple[int, float]]:
        negatives: list[tuple[int, float]] = []
        seen: set[int] = set()

        for match in candidate_matches:
            candidate_id = match.issue_id
            if self._is_invalid_negative(
                query_id,
                candidate_id,
                positive_targets=positive_targets,
                holdout_issue_ids=holdout_issue_ids,
                seen=seen,
            ):
                continue
            negatives.append((candidate_id, match.seed_score))
            seen.add(candidate_id)

        for candidate_id in fallback_issue_ids:
            if self._is_invalid_negative(
                query_id,
                candidate_id,
                positive_targets=positive_targets,
                holdout_issue_ids=holdout_issue_ids,
                seen=seen,
            ):
                continue
            negatives.append((candidate_id, 0.0))
            seen.add(candidate_id)

        return negatives

    @staticmethod
    def _is_invalid_negative(
        query_id: int,
        candidate_id: int,
        *,
        positive_targets: set[int],
        holdout_issue_ids: frozenset[int],
        seen: set[int],
    ) -> bool:
        return (
            candidate_id == query_id
            or candidate_id in positive_targets
            or candidate_id in holdout_issue_ids
            or candidate_id in seen
        )


class LogisticRegressionTrainer:
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...] = ENGINEERED_FEATURE_NAMES,
        learning_rate: float = 0.3,
        epochs: int = 300,
        l2_penalty: float = 0.01,
        seed: int = 17,
        early_stopping_patience: int = 35,
        min_epochs: int = 40,
        compute_device: str = "auto",
    ):
        self._feature_names = feature_names
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._l2_penalty = l2_penalty
        self._seed = seed
        self._early_stopping_patience = early_stopping_patience
        self._min_epochs = min_epochs
        self._compute_device = compute_device

    def fit(self, examples: list[PairTrainingExample]) -> LogisticRegressionModel:
        logger.info(
            "Training logistic regression reranker: examples=%s epochs=%s lr=%.3f l2=%.3f",
            len(examples),
            self._epochs,
            self._learning_rate,
            self._l2_penalty,
        )
        if not examples:
            return self._empty_model(bias=-1.5)

        train_examples, validation_examples, test_examples = split_binary_examples(
            examples,
            seed=self._seed,
        )
        logger.info(
            "Logistic regression data split: train=%s validation=%s test=%s",
            len(train_examples),
            len(validation_examples),
            len(test_examples),
        )

        positive_count = sum(example.label for example in train_examples)
        negative_count = len(train_examples) - positive_count
        if positive_count == 0 or negative_count == 0:
            return self._empty_model(bias=0.0)

        torch_runtime = resolve_torch_runtime(self._compute_device)
        should_use_torch = torch_runtime.enabled and torch_runtime.torch is not None
        if should_use_torch:
            model, curve, best_epoch, stopped_epoch = self._fit_torch(
                train_examples,
                validation_examples,
                positive_count,
                negative_count,
                torch_runtime,
            )
        else:
            model, curve, best_epoch, stopped_epoch = self._fit_python(
                train_examples,
                validation_examples,
                positive_count,
                negative_count,
            )

        test_metrics = self._evaluate_examples(model, test_examples)
        logger.info(
            "Logistic regression test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f loss=%.4f",
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["loss"],
        )
        if training_diagnostics_enabled():
            write_training_run_diagnostics(
                TrainingRunDiagnostics(
                    model_name="logreg-engineered",
                    trainer_name="logistic_regression",
                    compute_device=torch_runtime.device,
                    total_examples=len(examples),
                    train_examples=len(train_examples),
                    validation_examples=len(validation_examples),
                    test_examples=len(test_examples),
                    train_positive_examples=sum(example.label for example in train_examples),
                    validation_positive_examples=sum(example.label for example in validation_examples),
                    test_positive_examples=sum(example.label for example in test_examples),
                    early_stopping_patience=self._early_stopping_patience,
                    min_epochs=self._min_epochs,
                    best_epoch=best_epoch,
                    stopped_epoch=stopped_epoch,
                    test_metrics=test_metrics,
                    curve=tuple(curve),
                )
            )
        return model

    def _empty_model(self, *, bias: float) -> LogisticRegressionModel:
        return LogisticRegressionModel(
            feature_names=self._feature_names,
            weights={name: 0.0 for name in self._feature_names},
            bias=bias,
        )

    def _fit_python(
        self,
        train_examples: list[PairTrainingExample],
        validation_examples: list[PairTrainingExample],
        positive_count: int,
        negative_count: int,
    ) -> tuple[LogisticRegressionModel, list[TrainingEpochMetrics], int, int]:
        weights = {name: 0.0 for name in self._feature_names}
        bias = math.log(positive_count / negative_count)
        example_count = max(len(train_examples), 1)
        positive_weight = negative_count / max(positive_count, 1)
        best_epoch = 1
        best_val_loss = float("inf")
        best_weights = dict(weights)
        best_bias = bias
        epochs_without_improvement = 0
        curve: list[TrainingEpochMetrics] = []
        stopped_epoch = self._epochs

        for epoch_index in range(1, self._epochs + 1):
            gradient_bias = 0.0
            gradients = {name: 0.0 for name in self._feature_names}

            for example in train_examples:
                margin = bias
                for name in self._feature_names:
                    margin += weights[name] * example.features.get(name, 0.0)
                prediction = _sigmoid(margin)
                sample_weight = positive_weight if example.label == 1 else 1.0
                error = (prediction - example.label) * sample_weight
                gradient_bias += error
                for name in self._feature_names:
                    gradients[name] += error * example.features.get(name, 0.0)

            bias -= self._learning_rate * gradient_bias / example_count
            for name in self._feature_names:
                regularized_gradient = (gradients[name] / example_count) + (self._l2_penalty * weights[name])
                weights[name] -= self._learning_rate * regularized_gradient

            train_metrics = self._evaluate_examples_by_params(weights, bias, train_examples)
            validation_reference = validation_examples if validation_examples else train_examples
            validation_metrics = self._evaluate_examples_by_params(weights, bias, validation_reference)
            curve.append(
                TrainingEpochMetrics(
                    epoch=epoch_index,
                    train_loss=train_metrics["loss"],
                    validation_loss=validation_metrics["loss"],
                    train_accuracy=train_metrics["accuracy"],
                    validation_accuracy=validation_metrics["accuracy"],
                    train_precision=train_metrics["precision"],
                    validation_precision=validation_metrics["precision"],
                    train_recall=train_metrics["recall"],
                    validation_recall=validation_metrics["recall"],
                    train_f1=train_metrics["f1"],
                    validation_f1=validation_metrics["f1"],
                )
            )

            if validation_metrics["loss"] < best_val_loss - 1e-5:
                best_val_loss = validation_metrics["loss"]
                best_epoch = epoch_index
                best_weights = dict(weights)
                best_bias = bias
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch_index == 1 or epoch_index % 10 == 0 or epoch_index == self._epochs:
                logger.info("Logistic regression training progress: epoch=%s/%s", epoch_index, self._epochs)
                logger.info(
                    "Logistic regression metrics: train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f",
                    train_metrics["loss"],
                    validation_metrics["loss"],
                    train_metrics["f1"],
                    validation_metrics["f1"],
                )

            if (
                epoch_index >= self._min_epochs
                and epochs_without_improvement >= self._early_stopping_patience
            ):
                stopped_epoch = epoch_index
                logger.info(
                    "Logistic regression early stopping: epoch=%s best_epoch=%s best_val_loss=%.4f",
                    epoch_index,
                    best_epoch,
                    best_val_loss,
                )
                break

        logger.info("Logistic regression reranker training complete")
        return (
            LogisticRegressionModel(feature_names=self._feature_names, weights=best_weights, bias=best_bias),
            curve,
            best_epoch,
            stopped_epoch,
        )

    def _fit_torch(
        self,
        train_examples: list[PairTrainingExample],
        validation_examples: list[PairTrainingExample],
        positive_count: int,
        negative_count: int,
        torch_runtime,
    ) -> tuple[LogisticRegressionModel, list[TrainingEpochMetrics], int, int]:
        torch = torch_runtime.torch
        device = torch_runtime.device
        logger.info("Using torch-accelerated logistic regression on %s", device)
        torch.manual_seed(self._seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(self._seed)

        feature_matrix = [
            [example.features.get(name, 0.0) for name in self._feature_names]
            for example in train_examples
        ]
        labels = [float(example.label) for example in train_examples]

        inputs = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
        targets = torch.tensor(labels, dtype=torch.float32, device=device)
        weights = torch.zeros(len(self._feature_names), dtype=torch.float32, device=device, requires_grad=True)
        bias = torch.tensor(
            math.log(positive_count / negative_count),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        optimizer = torch.optim.AdamW(
            [weights, bias],
            lr=min(self._learning_rate, 0.05),
            weight_decay=self._l2_penalty,
        )
        pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        batch_size = min(512, len(train_examples))
        sample_count = int(inputs.shape[0])
        validation_inputs = None
        validation_targets = None
        if validation_examples:
            validation_inputs = torch.tensor(
                [[example.features.get(name, 0.0) for name in self._feature_names] for example in validation_examples],
                dtype=torch.float32,
                device=device,
            )
            validation_targets = torch.tensor(
                [float(example.label) for example in validation_examples],
                dtype=torch.float32,
                device=device,
            )

        curve: list[TrainingEpochMetrics] = []
        best_epoch = 1
        best_val_loss = float("inf")
        best_weights = weights.detach().clone()
        best_bias = bias.detach().clone()
        epochs_without_improvement = 0
        stopped_epoch = self._epochs

        for epoch_index in range(1, self._epochs + 1):
            permutation = torch.randperm(sample_count, device=device)

            for start in range(0, sample_count, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]

                optimizer.zero_grad()
                logits = batch_inputs @ weights + bias
                loss = loss_fn(logits, batch_targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                train_logits = inputs @ weights + bias
                train_probabilities = torch.sigmoid(train_logits)
                train_loss = float(loss_fn(train_logits, targets).item())
                train_metrics = binary_metrics_from_probabilities(
                    [int(value) for value in targets.tolist()],
                    [float(value) for value in train_probabilities.tolist()],
                )

                if validation_inputs is not None and validation_targets is not None:
                    validation_logits = validation_inputs @ weights + bias
                    validation_probabilities = torch.sigmoid(validation_logits)
                    validation_loss = float(loss_fn(validation_logits, validation_targets).item())
                    validation_metrics = binary_metrics_from_probabilities(
                        [int(value) for value in validation_targets.tolist()],
                        [float(value) for value in validation_probabilities.tolist()],
                    )
                else:
                    validation_loss = train_loss
                    validation_metrics = train_metrics

            curve.append(
                TrainingEpochMetrics(
                    epoch=epoch_index,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    train_accuracy=train_metrics["accuracy"],
                    validation_accuracy=validation_metrics["accuracy"],
                    train_precision=train_metrics["precision"],
                    validation_precision=validation_metrics["precision"],
                    train_recall=train_metrics["recall"],
                    validation_recall=validation_metrics["recall"],
                    train_f1=train_metrics["f1"],
                    validation_f1=validation_metrics["f1"],
                )
            )

            if validation_loss < best_val_loss - 1e-5:
                best_val_loss = validation_loss
                best_epoch = epoch_index
                best_weights = weights.detach().clone()
                best_bias = bias.detach().clone()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch_index == 1 or epoch_index % 10 == 0 or epoch_index == self._epochs:
                logger.info(
                    "Torch logistic regression progress: epoch=%s/%s train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f",
                    epoch_index,
                    self._epochs,
                    train_loss,
                    validation_loss,
                    train_metrics["f1"],
                    validation_metrics["f1"],
                )
            if (
                epoch_index >= self._min_epochs
                and epochs_without_improvement >= self._early_stopping_patience
            ):
                stopped_epoch = epoch_index
                logger.info(
                    "Torch logistic regression early stopping: epoch=%s best_epoch=%s best_val_loss=%.4f",
                    epoch_index,
                    best_epoch,
                    best_val_loss,
                )
                break

        learned_weights = best_weights.detach().cpu().tolist()
        learned_bias = float(best_bias.detach().cpu().item())
        logger.info("Torch logistic regression training complete")
        return (
            LogisticRegressionModel(
                feature_names=self._feature_names,
                weights={name: float(value) for name, value in zip(self._feature_names, learned_weights)},
                bias=learned_bias,
            ),
            curve,
            best_epoch,
            stopped_epoch,
        )

    def _evaluate_examples(
        self,
        model: LogisticRegressionModel,
        examples: list[PairTrainingExample],
    ) -> dict[str, float]:
        if not examples:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        labels = [int(example.label) for example in examples]
        probabilities = [model.predict_proba(example.features) for example in examples]
        metrics = binary_metrics_from_probabilities(labels, probabilities)
        metrics["loss"] = round(binary_cross_entropy(labels, probabilities), 6)
        return metrics

    def _evaluate_examples_by_params(
        self,
        weights: dict[str, float],
        bias: float,
        examples: list[PairTrainingExample],
    ) -> dict[str, float]:
        if not examples:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        labels = [int(example.label) for example in examples]
        probabilities: list[float] = []
        for example in examples:
            score = bias
            for name in self._feature_names:
                score += weights.get(name, 0.0) * example.features.get(name, 0.0)
            probabilities.append(_sigmoid(score))
        metrics = binary_metrics_from_probabilities(labels, probabilities)
        metrics["loss"] = round(binary_cross_entropy(labels, probabilities), 6)
        return metrics


class SupervisedLinearReranker(Reranker):
    def __init__(
        self,
        model: LogisticRegressionModel,
        *,
        explanation_threshold: float = 0.06,
    ):
        self._model = model
        self._explanation_threshold = explanation_threshold

    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        score = self._model.predict_proba(feature_scores)
        reasons = self._build_reasons(feature_scores)
        return RerankResult(
            score=round(score, 6),
            feature_scores=feature_scores,
            reasons=reasons,
        )

    def _build_reasons(self, feature_scores: dict[str, float]) -> tuple[str, ...]:
        templates = {
            "bm25": "strong lexical overlap supports the match",
            "bm25_plus": "bm25-plus indicates a close sparse-text match",
            "tfidf_cosine": "tf-idf cosine similarity is strong",
            "title_overlap": "title tokens overlap closely",
            "title_ngram": "title wording is very similar",
            "description_overlap": "description text overlaps strongly",
            "component_overlap": "component metadata is aligned",
            "project_match": "the issues belong to the same project",
            "issue_type_match": "issue types match",
            "priority_match": "priority metadata matches",
            "length_ratio": "issue lengths are similar",
        }
        contributions: list[tuple[str, float]] = []
        for name in self._model.feature_names:
            contribution = self._model.weights.get(name, 0.0) * feature_scores.get(name, 0.0)
            if contribution > self._explanation_threshold and name in templates:
                contributions.append((name, contribution))

        contributions.sort(key=lambda item: item[1], reverse=True)
        reasons = [templates[name] for name, _ in contributions[:4]]
        if not reasons:
            return ("the engineered feature model found a moderate overall match",)
        return tuple(reasons)


def build_classical_supervised_pipelines(
    index: SearchIndex,
    *,
    holdout_issue_ids: frozenset[int] = frozenset(),
    compute_device: str = "auto",
) -> dict[str, RetrievalPipeline]:
    feature_extractor = EngineeredFeatureExtractor()
    trainer = LogisticRegressionTrainer(compute_device=compute_device)
    training_examples = PairTrainingSetBuilder(
        feature_extractor=feature_extractor,
        hard_negative_pool_size=CLASSICAL_SUPERVISED_RERANK_POOL_SIZE,
    ).build(
        index,
        holdout_issue_ids=holdout_issue_ids,
    )
    trained_model = trainer.fit(training_examples)

    return {
        "logreg-engineered": RetrievalPipeline(
            name="logreg-engineered",
            candidate_generator=BM25CandidateGenerator(),
            feature_extractor=feature_extractor,
            reranker=SupervisedLinearReranker(trained_model),
            max_candidate_pool_size=CLASSICAL_SUPERVISED_RERANK_POOL_SIZE,
        )
    }
