from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_COMPUTE_DEVICES = frozenset({"auto", "cpu", "cuda"})


def normalize_compute_device(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in VALID_COMPUTE_DEVICES:
        raise ValueError(
            f"Unsupported compute device '{value}'. Supported values: auto, cpu, cuda."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class TorchRuntime:
    torch: Any | None
    device: str
    enabled: bool
    reason: str


def resolve_torch_runtime(preferred_device: str | None = "auto") -> TorchRuntime:
    normalized_device = normalize_compute_device(preferred_device)
    return _resolve_torch_runtime_cached(normalized_device)


@lru_cache(maxsize=3)
def _resolve_torch_runtime_cached(normalized_device: str) -> TorchRuntime:
    try:
        import torch  # type: ignore
    except ImportError:
        return TorchRuntime(
            torch=None,
            device="cpu",
            enabled=False,
            reason="PyTorch is not installed",
        )

    if normalized_device == "cpu":
        return TorchRuntime(
            torch=torch,
            device="cpu",
            enabled=True,
            reason="PyTorch enabled on CPU",
        )

    if torch.cuda.is_available():
        return TorchRuntime(
            torch=torch,
            device="cuda",
            enabled=True,
            reason="PyTorch enabled on CUDA",
        )

    if normalized_device == "cuda":
        logger.warning("CUDA requested but not available. Falling back to CPU.")
    return TorchRuntime(
        torch=torch,
        device="cpu",
        enabled=True,
        reason="PyTorch enabled on CPU (CUDA unavailable)",
    )
