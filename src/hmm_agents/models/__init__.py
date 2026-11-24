from __future__ import annotations

from .base import FailurePredictor, group_holdout, prefix_view
from .feature_lr import FeatureLR
from .neural import NeuralSeq

REGISTRY: dict[str, type[FailurePredictor]] = {
    "feature_lr": FeatureLR,
    "neural": NeuralSeq,
}


def build_model(spec: dict, seed: int = 0) -> FailurePredictor:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in REGISTRY:
        raise ValueError(f"unknown model type {kind!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[kind](**spec, seed=seed)


__all__ = ["FailurePredictor", "group_holdout", "prefix_view", "build_model", "REGISTRY", "FeatureLR", "NeuralSeq"]
