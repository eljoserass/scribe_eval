"""Shared pieces for metrics that run a neural model, and settings for any metric.

Model metrics need the `semantic` extra. Weights download from Hugging Face on first
use and stay in its cache, so `HF_HOME` and `HF_HUB_OFFLINE` apply as usual.
"""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import replace
from typing import TypeVar

_Metric = TypeVar("_Metric")
_Model = TypeVar("_Model")


class LoadedModels:
    """Models one configured metric has loaded, kept until scoring ends.

    A metric is configured once per experiment, so datasets share what it loads.
    """

    def __init__(self) -> None:
        """Start empty; nothing loads until a metric asks."""
        self._models: dict[Hashable, object] = {}

    def get(self, key: Hashable, load: Callable[[], _Model]) -> _Model:
        """Return the model for `key`, loading and maybe downloading it once."""
        if key not in self._models:
            self._models[key] = load()
        return self._models[key]


def apply_settings(
    metric: _Metric, settings: Mapping[str, object], allowed: tuple[str, ...]
) -> _Metric:
    """Copy a metric with new settings, rejecting names it doesn't take.

    The copy starts with no loaded models.

    Raises:
        ValueError: If a setting isn't in `allowed`.
    """
    unknown = sorted(set(settings) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown settings for metric {metric.name!r}: {', '.join(unknown)}. "
            f"It takes: {', '.join(allowed)}."
        )
    return replace(metric, **settings)


def require_semantic(module: str, metric: str) -> None:
    """Fail early, with the fix, when the `semantic` extra is missing.

    Raises:
        ValueError: If `module` can't be found.
    """
    from importlib.util import find_spec

    if find_spec(module) is None:
        raise ValueError(
            f"Metric {metric!r} needs the semantic extra: "
            "uv sync --extra semantic, or pip install 'scribe-eval[semantic]'."
        )


def resolve_device(device: str) -> str:
    """Turn `auto` into the best device present: CUDA, then Apple MPS, then CPU."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
