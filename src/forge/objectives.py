from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def boltzmann_targets(utilities: Tensor, temperature: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(utilities).all():
        raise ValueError("utilities must be finite")
    return F.softmax(utilities / temperature, dim=-1)


def forward_kl(log_probs: Tensor, target_probs: Tensor) -> Tensor:
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def group_advantages(rewards: Tensor, epsilon: float = 1e-4, method: str = "grpo") -> Tensor:
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (batch, group>=2)")
    if epsilon <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite and epsilon positive")
    centered = rewards - rewards.mean(-1, keepdim=True)
    if method == "grpo":
        sd = rewards.std(-1, correction=0, keepdim=True)
        return torch.where(sd > 0, centered / (sd + epsilon), torch.zeros_like(centered))
    if method == "dr_grpo":
        return centered
    if method == "rloo":
        return centered * rewards.shape[1] / (rewards.shape[1] - 1)
    raise ValueError(f"Unknown group advantage method: {method}")


def clipped_policy_loss(
    selected_log_probs: Tensor,
    old_selected_log_probs: Tensor,
    advantages: Tensor,
    clip_range: float = 0.2,
) -> Tensor:
    if not 0 < clip_range < 1:
        raise ValueError("clip_range must be in (0, 1)")
    ratios = (selected_log_probs - old_selected_log_probs.detach()).exp()
    advantage = advantages.detach()
    return -torch.minimum(ratios * advantage, ratios.clamp(1 - clip_range, 1 + clip_range) * advantage).mean()


def categorical_kl(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    p = log_probs.exp()
    return (p * (log_probs - reference_log_probs.detach())).sum(-1).mean()


def categorical_entropy(log_probs: Tensor) -> Tensor:
    return -(log_probs.exp() * log_probs).sum(-1).mean()


def dpo_loss(
    log_probs: Tensor,
    reference_log_probs: Tensor,
    chosen: Tensor,
    rejected: Tensor,
    beta: float = 0.1,
) -> Tensor:
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    delta = log_probs - reference_log_probs.detach()
    margin = delta.gather(1, chosen[:, None]) - delta.gather(1, rejected[:, None])
    return -F.logsigmoid(beta * margin).mean()


@dataclass
class DatasetCostNormalizer:
    maxima: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(
        self,
        input_tokens: np.ndarray,
        output_tokens: np.ndarray,
        datasets: Sequence[str],
        *,
        split: str,
    ) -> "DatasetCostNormalizer":
        if split != "train":
            raise ValueError("Cost maxima may only be fitted on the training split")
        cin = np.asarray(input_tokens, dtype=np.float64)
        cout = np.asarray(output_tokens, dtype=np.float64)
        labels = np.asarray(datasets, dtype=str)
        self._validate(cin, cout, labels)
        self.maxima = {
            str(name): (float(cin[labels == name].max()), float(cout[labels == name].max()))
            for name in np.unique(labels)
        }
        return self

    @staticmethod
    def _validate(cin: np.ndarray, cout: np.ndarray, labels: np.ndarray) -> None:
        if cin.shape != cout.shape or cin.ndim not in (1, 2) or len(cin) != len(labels) or len(cin) == 0:
            raise ValueError("Token arrays must agree and have one row per dataset label")
        if not np.isfinite(cin).all() or not np.isfinite(cout).all() or (cin < 0).any() or (cout < 0).any():
            raise ValueError("Token counts must be finite and nonnegative")

    def transform(
        self, input_tokens: np.ndarray, output_tokens: np.ndarray, datasets: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        cin, cout = np.asarray(input_tokens, dtype=np.float64), np.asarray(output_tokens, dtype=np.float64)
        labels = np.asarray(datasets, dtype=str)
        self._validate(cin, cout, labels)
        missing = set(labels) - self.maxima.keys()
        if missing:
            raise ValueError(f"No training cost maxima for datasets: {sorted(missing)}")
        denominators = np.asarray([self.maxima[str(name)] for name in labels], dtype=np.float64)
        denominators = np.where(denominators > 0, denominators, 1.0)
        shape = (len(labels),) + (1,) * (cin.ndim - 1)
        return cin / denominators[:, 0].reshape(shape), cout / denominators[:, 1].reshape(shape)

    def to_dict(self) -> dict:
        return {"fit_split": "train", "maxima": {k: list(v) for k, v in self.maxima.items()}}

    @classmethod
    def from_dict(cls, state: dict) -> "DatasetCostNormalizer":
        if state.get("fit_split") != "train":
            raise ValueError("Refusing a cost normalizer not fitted on train")
        maxima = {str(k): tuple(float(x) for x in v) for k, v in state["maxima"].items()}
        if any(len(v) != 2 or not all(np.isfinite(x) and x >= 0 for x in v) for v in maxima.values()):
            raise ValueError("Invalid normalizer maxima")
        return cls(maxima)


def pareto_utility(
    f1: np.ndarray,
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    datasets: Sequence[str],
    normalizer: DatasetCostNormalizer,
    lambda_in: float = 0.1,
    lambda_out: float = 0.2,
) -> np.ndarray:
    scores = np.asarray(f1, dtype=np.float64)
    if lambda_in < 0 or lambda_out < 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and cost weights nonnegative")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("F1 rewards must be fractions in [0, 1], not percentages")
    cin, cout = normalizer.transform(input_tokens, output_tokens, datasets)
    if scores.shape != cin.shape:
        raise ValueError("F1 and token arrays must have the same shape")
    return scores - lambda_in * cin - lambda_out * cout


@dataclass
class StructuredStandardizer:
    embedding_dim: int = 768
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, *, split: str) -> "StructuredStandardizer":
        if split != "train":
            raise ValueError("Feature standardization may only be fitted on train")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] < self.embedding_dim or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        values = x[:, self.embedding_dim :]
        self.mean = values.mean(0)
        sd = values.std(0, ddof=0)
        self.scale = np.where(sd > 1e-8, sd, 1.0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError("Feature standardizer has not been fitted")
        x = np.asarray(features, dtype=np.float32).copy()
        if x.ndim != 2 or x.shape[1] != self.embedding_dim + len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Feature matrix does not match the fitted scaler")
        x[:, self.embedding_dim :] = (x[:, self.embedding_dim :] - self.mean) / self.scale
        return x

    def to_dict(self) -> dict:
        if self.mean is None or self.scale is None:
            raise ValueError("Feature standardizer has not been fitted")
        return {"fit_split": "train", "embedding_dim": self.embedding_dim, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, state: dict) -> "StructuredStandardizer":
        if state.get("fit_split") != "train":
            raise ValueError("Refusing a feature standardizer not fitted on train")
        mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
        if mean.ndim != 1 or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid feature standardizer state")
        return cls(int(state["embedding_dim"]), mean, scale)
