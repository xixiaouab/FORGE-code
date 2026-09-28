from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from .objectives import DatasetCostNormalizer, StructuredStandardizer
from .policy import ACTION_LAYOUT, FactorizedRouter, FlatRouter, Router, RouterConfig


SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: Router,
    *,
    feature_names: Sequence[str],
    standardizer: StructuredStandardizer | None = None,
    cost_normalizer: DatasetCostNormalizer | None = None,
    metadata: dict | None = None,
) -> None:
    if len(feature_names) != model.config.input_dim or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must uniquely identify every input dimension in order")
    if standardizer is not None:
        state = standardizer.to_dict()
        if state["embedding_dim"] + len(state["mean"]) != model.config.input_dim:
            raise ValueError("Standardizer does not match router input dimension")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": model.architecture,
        "router_config": model.config.to_dict(),
        "action_layout": ACTION_LAYOUT,
        "feature_names": list(feature_names),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "standardizer": None if standardizer is None else standardizer.to_dict(),
        "cost_normalizer": None if cost_normalizer is None else cost_normalizer.to_dict(),
        "training_config": getattr(model, "training_config", None),
        "metadata": metadata or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_variant: str | None = None,
    expected_feature_names: Sequence[str] | None = None,
    expected_n_thinking: int | None = None,
) -> tuple[Router, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported FORGE checkpoint schema")
    if payload.get("action_layout") != ACTION_LAYOUT:
        raise ValueError("Checkpoint action layout is incompatible")
    config = RouterConfig(**payload["router_config"])
    if expected_variant is not None and config.variant != expected_variant:
        raise ValueError(f"Expected variant {expected_variant!r}, found {config.variant!r}")
    if expected_n_thinking is not None and config.n_thinking != expected_n_thinking:
        raise ValueError("Checkpoint thinking alphabet is incompatible")
    names = payload.get("feature_names", [])
    if len(names) != config.input_dim or len(set(names)) != len(names):
        raise ValueError("Checkpoint feature names do not match its input dimension")
    if expected_feature_names is not None and list(expected_feature_names) != names:
        raise ValueError("Checkpoint feature order does not match requested feature order")
    factories = {"factorized": FactorizedRouter, "flat": FlatRouter}
    if payload.get("architecture") not in factories:
        raise ValueError("Unsupported router architecture")
    model = factories[payload["architecture"]](config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    model.training_config = payload.get("training_config")
    if payload.get("standardizer") is not None:
        standardizer = StructuredStandardizer.from_dict(payload["standardizer"])
        if standardizer.embedding_dim + len(standardizer.mean) != config.input_dim:
            raise ValueError("Checkpoint standardizer dimension is incompatible")
    if payload.get("cost_normalizer") is not None:
        DatasetCostNormalizer.from_dict(payload["cost_normalizer"])
    return model, payload
