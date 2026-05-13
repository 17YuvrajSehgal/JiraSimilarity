from __future__ import annotations

from dataclasses import dataclass
import logging
import random

from ..compute import resolve_torch_runtime
from ..pipeline import (
    FeatureExtractor,
    RetrievalPipeline,
    RerankResult,
    Reranker,
    SearchIndex,
)
from .classical_supervised import EngineeredFeatureExtractor, PairTrainingExample, PairTrainingSetBuilder, _sigmoid
from .dense_semantic import DenseEmbeddingSpace, DenseSemanticFeatureExtractor
from .hybrid_sparse_dense import ReciprocalRankFusionCandidateGenerator
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

NEURAL_FEATURE_NAMES = (
    "bm25",
    "bm25_plus",
    "tfidf_cosine",
    "dense_cosine",
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
PAIRWISE_RERANK_POOL_SIZE = 80


def _relu(value: float) -> float:
    return value if value > 0.0 else 0.0


def _relu_derivative(value: float) -> float:
    return 1.0 if value > 0.0 else 0.0


class NeuralPairFeatureExtractor(FeatureExtractor):
    def __init__(self, dense_space: DenseEmbeddingSpace):
        self._engineered = EngineeredFeatureExtractor()
        self._dense = DenseSemanticFeatureExtractor(dense_space)

    def extract(self, query, candidate, index: SearchIndex, *, seed_score: float) -> dict[str, float]:
        feature_scores = self._engineered.extract(query, candidate, index, seed_score=seed_score)
        dense_scores = self._dense.extract(query, candidate, index, seed_score=seed_score)
        feature_scores["dense_cosine"] = dense_scores.get("dense_cosine", 0.0)
        return feature_scores


@dataclass(frozen=True, slots=True)
class MLPClassifierModel:
    feature_names: tuple[str, ...]
    hidden_size: int
    input_hidden: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    hidden_output: tuple[float, ...]
    output_bias: float

    def predict_proba(self, features: dict[str, float]) -> float:
        inputs = [features.get(name, 0.0) for name in self.feature_names]
        hidden_values: list[float] = []
        for hidden_index in range(self.hidden_size):
            activation = self.hidden_bias[hidden_index]
            for feature_index, feature_value in enumerate(inputs):
                activation += self.input_hidden[hidden_index][feature_index] * feature_value
            hidden_values.append(_relu(activation))

        output_activation = self.output_bias
        for hidden_index, hidden_value in enumerate(hidden_values):
            output_activation += self.hidden_output[hidden_index] * hidden_value
        return _sigmoid(output_activation)


class TorchMLPClassifierModel:
    def __init__(self, *, feature_names: tuple[str, ...], model, torch, device: str):
        self.feature_names = feature_names
        self._model = model
        self._torch = torch
        self._device = device

    def predict_proba(self, features: dict[str, float]) -> float:
        values = [features.get(name, 0.0) for name in self.feature_names]
        with self._torch.no_grad():
            inputs = self._torch.tensor(values, dtype=self._torch.float32, device=self._device).unsqueeze(0)
            logits = self._model(inputs).squeeze(0).squeeze(0)
            return float(self._torch.sigmoid(logits).item())


class SimpleMLPTrainer:
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...] = NEURAL_FEATURE_NAMES,
        hidden_size: int = 32,
        learning_rate: float = 0.03,
        epochs: int = 300,
        l2_penalty: float = 0.001,
        seed: int = 13,
        early_stopping_patience: int = 35,
        min_epochs: int = 40,
        compute_device: str = "auto",
    ):
        self._feature_names = feature_names
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._l2_penalty = l2_penalty
        self._seed = seed
        self._early_stopping_patience = early_stopping_patience
        self._min_epochs = min_epochs
        self._compute_device = compute_device

    def fit(self, examples: list[PairTrainingExample]):
        logger.info(
            "Training neural pairwise classifier: examples=%s hidden=%s epochs=%s lr=%.3f",
            len(examples),
            self._hidden_size,
            self._epochs,
            self._learning_rate,
        )
        if not examples:
            return self._empty_model()

        train_examples, validation_examples, test_examples = split_binary_examples(
            examples,
            seed=self._seed,
        )
        logger.info(
            "Neural pairwise data split: train=%s validation=%s test=%s",
            len(train_examples),
            len(validation_examples),
            len(test_examples),
        )

        torch_runtime = resolve_torch_runtime(self._compute_device)
        should_use_torch = torch_runtime.enabled and torch_runtime.torch is not None
        if should_use_torch:
            model, curve, best_epoch, stopped_epoch = self._fit_torch(
                train_examples,
                validation_examples,
                torch_runtime,
            )
        else:
            model, curve, best_epoch, stopped_epoch = self._fit_python(
                train_examples,
                validation_examples,
            )

        test_metrics = self._evaluate_examples(model, test_examples)
        logger.info(
            "Neural pairwise test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f loss=%.4f",
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["loss"],
        )
        if training_diagnostics_enabled():
            write_training_run_diagnostics(
                TrainingRunDiagnostics(
                    model_name="pairwise-neural-mlp",
                    trainer_name="pairwise_mlp",
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

    def _fit_python(
        self,
        train_examples: list[PairTrainingExample],
        validation_examples: list[PairTrainingExample],
    ) -> tuple[MLPClassifierModel, list[TrainingEpochMetrics], int, int]:
        rng = random.Random(self._seed)
        input_size = len(self._feature_names)
        input_hidden = [
            [rng.uniform(-0.08, 0.08) for _ in range(input_size)]
            for _ in range(self._hidden_size)
        ]
        hidden_bias = [0.0 for _ in range(self._hidden_size)]
        hidden_output = [rng.uniform(-0.08, 0.08) for _ in range(self._hidden_size)]
        output_bias = 0.0

        training_rows = [
            ([example.features.get(name, 0.0) for name in self._feature_names], float(example.label))
            for example in train_examples
        ]
        positive_count = sum(example.label for example in train_examples)
        negative_count = len(train_examples) - positive_count
        positive_weight = negative_count / max(positive_count, 1)

        best_epoch = 1
        best_val_loss = float("inf")
        best_state = (
            [list(row) for row in input_hidden],
            list(hidden_bias),
            list(hidden_output),
            output_bias,
        )
        epochs_without_improvement = 0
        curve: list[TrainingEpochMetrics] = []
        stopped_epoch = self._epochs

        for epoch_index in range(1, self._epochs + 1):
            for inputs, label in training_rows:
                hidden_linear = []
                hidden_activations = []
                for hidden_index in range(self._hidden_size):
                    activation = hidden_bias[hidden_index]
                    for feature_index, feature_value in enumerate(inputs):
                        activation += input_hidden[hidden_index][feature_index] * feature_value
                    hidden_linear.append(activation)
                    hidden_activations.append(_relu(activation))

                output_linear = output_bias
                for hidden_index, hidden_value in enumerate(hidden_activations):
                    output_linear += hidden_output[hidden_index] * hidden_value
                prediction = _sigmoid(output_linear)
                sample_weight = positive_weight if label >= 0.5 else 1.0
                error = (prediction - label) * sample_weight

                output_bias -= self._learning_rate * error
                hidden_output_snapshot = list(hidden_output)
                for hidden_index, hidden_value in enumerate(hidden_activations):
                    gradient = error * hidden_value + (self._l2_penalty * hidden_output[hidden_index])
                    hidden_output[hidden_index] -= self._learning_rate * gradient

                for hidden_index in range(self._hidden_size):
                    hidden_error = error * hidden_output_snapshot[hidden_index] * _relu_derivative(hidden_linear[hidden_index])
                    hidden_bias[hidden_index] -= self._learning_rate * hidden_error
                    for feature_index, feature_value in enumerate(inputs):
                        gradient = hidden_error * feature_value + (self._l2_penalty * input_hidden[hidden_index][feature_index])
                        input_hidden[hidden_index][feature_index] -= self._learning_rate * gradient

            model_snapshot = self._build_python_model(input_hidden, hidden_bias, hidden_output, output_bias)
            train_metrics = self._evaluate_examples(model_snapshot, train_examples)
            validation_reference = validation_examples if validation_examples else train_examples
            validation_metrics = self._evaluate_examples(model_snapshot, validation_reference)
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
                best_state = (
                    [list(row) for row in input_hidden],
                    list(hidden_bias),
                    list(hidden_output),
                    output_bias,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch_index == 1 or epoch_index % 10 == 0 or epoch_index == self._epochs:
                logger.info(
                    "Neural pairwise classifier progress: epoch=%s/%s train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f",
                    epoch_index,
                    self._epochs,
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
                    "Neural pairwise early stopping: epoch=%s best_epoch=%s best_val_loss=%.4f",
                    epoch_index,
                    best_epoch,
                    best_val_loss,
                )
                break

        logger.info("Neural pairwise classifier training complete")
        best_model = self._build_python_model(*best_state)
        return (
            best_model,
            curve,
            best_epoch,
            stopped_epoch,
        )

    def _fit_torch(
        self,
        train_examples: list[PairTrainingExample],
        validation_examples: list[PairTrainingExample],
        torch_runtime,
    ):
        torch = torch_runtime.torch
        device = torch_runtime.device
        logger.info("Using torch-accelerated neural pairwise training on %s", device)

        torch.manual_seed(self._seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(self._seed)

        feature_matrix = [
            [example.features.get(name, 0.0) for name in self._feature_names]
            for example in train_examples
        ]
        labels = [float(example.label) for example in train_examples]

        inputs = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
        targets = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
        positive_count = sum(example.label for example in train_examples)
        negative_count = len(train_examples) - positive_count

        model = torch.nn.Sequential(
            torch.nn.Linear(len(self._feature_names), self._hidden_size),
            torch.nn.LayerNorm(self._hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(self._hidden_size, max(8, self._hidden_size // 2)),
            torch.nn.GELU(),
            torch.nn.Linear(max(8, self._hidden_size // 2), 1),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=min(self._learning_rate, 0.03),
            weight_decay=self._l2_penalty,
        )
        pos_weight = torch.tensor(
            [negative_count / max(positive_count, 1)],
            dtype=torch.float32,
            device=device,
        )
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
            ).unsqueeze(1)

        curve: list[TrainingEpochMetrics] = []
        best_epoch = 1
        best_val_loss = float("inf")
        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        epochs_without_improvement = 0
        stopped_epoch = self._epochs

        model.train()
        for epoch_index in range(1, self._epochs + 1):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]
                optimizer.zero_grad()
                logits = model(batch_inputs)
                loss = loss_fn(logits, batch_targets)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                train_logits = model(inputs)
                train_probabilities = torch.sigmoid(train_logits).squeeze(1)
                train_loss = float(loss_fn(train_logits, targets).item())
                train_metrics = binary_metrics_from_probabilities(
                    [int(value) for value in targets.squeeze(1).tolist()],
                    [float(value) for value in train_probabilities.tolist()],
                )

                if validation_inputs is not None and validation_targets is not None:
                    validation_logits = model(validation_inputs)
                    validation_probabilities = torch.sigmoid(validation_logits).squeeze(1)
                    validation_loss = float(loss_fn(validation_logits, validation_targets).item())
                    validation_metrics = binary_metrics_from_probabilities(
                        [int(value) for value in validation_targets.squeeze(1).tolist()],
                        [float(value) for value in validation_probabilities.tolist()],
                    )
                else:
                    validation_loss = train_loss
                    validation_metrics = train_metrics
            model.train()

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
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch_index == 1 or epoch_index % 10 == 0 or epoch_index == self._epochs:
                logger.info(
                    "Torch neural pairwise progress: epoch=%s/%s train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f",
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
                    "Torch neural pairwise early stopping: epoch=%s best_epoch=%s best_val_loss=%.4f",
                    epoch_index,
                    best_epoch,
                    best_val_loss,
                )
                break

        model.load_state_dict(best_state)
        model.eval()
        logger.info("Torch neural pairwise training complete")
        return (
            TorchMLPClassifierModel(
                feature_names=self._feature_names,
                model=model,
                torch=torch,
                device=device,
            ),
            curve,
            best_epoch,
            stopped_epoch,
        )

    def _empty_model(self) -> MLPClassifierModel:
        input_size = len(self._feature_names)
        return MLPClassifierModel(
            feature_names=self._feature_names,
            hidden_size=self._hidden_size,
            input_hidden=tuple(tuple(0.0 for _ in range(input_size)) for _ in range(self._hidden_size)),
            hidden_bias=tuple(0.0 for _ in range(self._hidden_size)),
            hidden_output=tuple(0.0 for _ in range(self._hidden_size)),
            output_bias=-1.0,
        )

    def _build_python_model(
        self,
        input_hidden: list[list[float]],
        hidden_bias: list[float],
        hidden_output: list[float],
        output_bias: float,
    ) -> MLPClassifierModel:
        return MLPClassifierModel(
            feature_names=self._feature_names,
            hidden_size=self._hidden_size,
            input_hidden=tuple(tuple(row) for row in input_hidden),
            hidden_bias=tuple(hidden_bias),
            hidden_output=tuple(hidden_output),
            output_bias=output_bias,
        )

    def _evaluate_examples(self, model, examples: list[PairTrainingExample]) -> dict[str, float]:
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


class NeuralPairwiseReranker(Reranker):
    def __init__(
        self,
        model,
        *,
        explanation_threshold: float = 0.08,
    ):
        self._model = model
        self._explanation_threshold = explanation_threshold

    def rerank(self, feature_scores: dict[str, float]) -> RerankResult:
        score = self._model.predict_proba(feature_scores)
        return RerankResult(
            score=round(score, 6),
            feature_scores=feature_scores,
            reasons=self._build_reasons(feature_scores, score),
        )

    def _build_reasons(self, feature_scores: dict[str, float], score: float) -> tuple[str, ...]:
        templates = {
            "dense_cosine": "neural duplicate model found strong semantic alignment",
            "bm25_plus": "sparse overlap still supports the pair strongly",
            "title_ngram": "title phrasing is closely aligned",
            "description_overlap": "descriptions share important duplicate signals",
            "component_overlap": "component metadata points to the same subsystem",
            "project_match": "both issues belong to the same project",
        }
        ranked = sorted(
            (
                (name, value)
                for name, value in feature_scores.items()
                if value >= self._explanation_threshold and name in templates
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        reasons = [templates[name] for name, _ in ranked[:4]]
        if not reasons:
            if score >= 0.5:
                return ("the neural duplicate classifier found a strong overall pairwise match",)
            return ("the neural duplicate classifier found only weak duplicate evidence",)
        return tuple(reasons)


def build_deep_pairwise_pipelines(
    index: SearchIndex,
    *,
    dense_space: DenseEmbeddingSpace,
    holdout_issue_ids: frozenset[int] = frozenset(),
    compute_device: str = "auto",
) -> dict[str, RetrievalPipeline]:
    logger.info("Building deep pairwise duplicate classification pipelines")
    feature_extractor = NeuralPairFeatureExtractor(dense_space)
    candidate_generator = ReciprocalRankFusionCandidateGenerator(
        dense_space=dense_space,
        compute_device=compute_device,
    )
    trainer = SimpleMLPTrainer(compute_device=compute_device)
    training_examples = PairTrainingSetBuilder(
        feature_extractor=feature_extractor,
        candidate_generator=candidate_generator,
        negatives_per_positive=4,
        hard_negative_pool_size=PAIRWISE_RERANK_POOL_SIZE,
    ).build(
        index,
        holdout_issue_ids=holdout_issue_ids,
    )
    trained_model = trainer.fit(training_examples)
    return {
        "pairwise-neural-mlp": RetrievalPipeline(
            name="pairwise-neural-mlp",
            candidate_generator=candidate_generator,
            feature_extractor=feature_extractor,
            reranker=NeuralPairwiseReranker(trained_model),
            max_candidate_pool_size=PAIRWISE_RERANK_POOL_SIZE,
        )
    }
