from __future__ import annotations

from dataclasses import dataclass
import logging
import math
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
        hidden_size: int = 12,
        learning_rate: float = 0.08,
        epochs: int = 220,
        l2_penalty: float = 0.0005,
        seed: int = 13,
        compute_device: str = "auto",
    ):
        self._feature_names = feature_names
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._l2_penalty = l2_penalty
        self._seed = seed
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

        torch_runtime = resolve_torch_runtime(self._compute_device)
        should_use_torch = (
            torch_runtime.enabled
            and torch_runtime.torch is not None
            and (torch_runtime.device == "cuda" or self._compute_device == "cpu")
        )
        if should_use_torch:
            return self._fit_torch(examples, torch_runtime)
        return self._fit_python(examples)

    def _fit_python(self, examples: list[PairTrainingExample]) -> MLPClassifierModel:
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
            for example in examples
        ]

        for epoch_index in range(1, self._epochs + 1):
            total_loss = 0.0
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
                error = prediction - label
                total_loss += -(
                    label * math.log(max(prediction, 1e-9))
                    + (1.0 - label) * math.log(max(1.0 - prediction, 1e-9))
                )

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

            if epoch_index == 1 or epoch_index % 40 == 0 or epoch_index == self._epochs:
                mean_loss = total_loss / max(len(training_rows), 1)
                logger.info(
                    "Neural pairwise classifier progress: epoch=%s/%s loss=%.4f",
                    epoch_index,
                    self._epochs,
                    mean_loss,
                )

        logger.info("Neural pairwise classifier training complete")
        return MLPClassifierModel(
            feature_names=self._feature_names,
            hidden_size=self._hidden_size,
            input_hidden=tuple(tuple(row) for row in input_hidden),
            hidden_bias=tuple(hidden_bias),
            hidden_output=tuple(hidden_output),
            output_bias=output_bias,
        )

    def _fit_torch(self, examples: list[PairTrainingExample], torch_runtime):
        torch = torch_runtime.torch
        device = torch_runtime.device
        logger.info("Using torch-accelerated neural pairwise training on %s", device)

        torch.manual_seed(self._seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(self._seed)

        feature_matrix = [
            [example.features.get(name, 0.0) for name in self._feature_names]
            for example in examples
        ]
        labels = [float(example.label) for example in examples]

        inputs = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
        targets = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)

        model = torch.nn.Sequential(
            torch.nn.Linear(len(self._feature_names), self._hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden_size, 1),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._l2_penalty,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        model.train()
        for epoch_index in range(1, self._epochs + 1):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            if epoch_index == 1 or epoch_index % 40 == 0 or epoch_index == self._epochs:
                logger.info(
                    "Torch neural pairwise progress: epoch=%s/%s loss=%.4f",
                    epoch_index,
                    self._epochs,
                    float(loss.item()),
                )

        model.eval()
        logger.info("Torch neural pairwise training complete")
        return TorchMLPClassifierModel(
            feature_names=self._feature_names,
            model=model,
            torch=torch,
            device=device,
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
        negatives_per_positive=3,
        hard_negative_pool_size=30,
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
        )
    }
