from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ACTION_LAYOUT = "support-major:direct,summary,raw;thinking:no_think,cot,low,high"


@dataclass(frozen=True)
class RouterConfig:
    input_dim: int = 789
    n_thinking: int = 2
    hidden_dim: int = 256
    embedding_dim: int = 16
    dropout: float = 0.1
    variant: str = "full"

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1 or self.embedding_dim < 1:
            raise ValueError("Router dimensions must be positive")
        if self.n_thinking not in (1, 2, 3, 4):
            raise ValueError("n_thinking must be 1, 2, 3, or 4")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def n_actions(self) -> int:
        return 3 * self.n_thinking

    def to_dict(self) -> dict:
        return asdict(self)


def _encoder(config: RouterConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
    )


class FactorizedRouter(nn.Module):
    architecture = "factorized"

    def __init__(self, config: RouterConfig | None = None) -> None:
        super().__init__()
        self.config = config or RouterConfig()
        c = self.config
        self.encoder = _encoder(c)
        self.support_head = nn.Linear(c.hidden_dim, 3)
        self.support_embedding = nn.Embedding(3, c.embedding_dim)
        self.thinking_head = nn.Linear(c.hidden_dim + c.embedding_dim, c.n_thinking)
        self.reset_thinking_uniform()

    def _hidden(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected (batch, {self.config.input_dim}) features")
        return self.encoder(x)

    def support_logits(self, x: Tensor) -> Tensor:
        return self.support_head(self._hidden(x))

    def support_log_probs(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.support_logits(x), dim=-1)

    def joint_log_probs(self, x: Tensor) -> Tensor:
        h = self._hidden(x)
        support = F.log_softmax(self.support_head(h), dim=-1)
        embeddings = self.support_embedding.weight.unsqueeze(0).expand(h.shape[0], -1, -1)
        conditional_input = torch.cat((h.unsqueeze(1).expand(-1, 3, -1), embeddings), dim=-1)
        thinking = F.log_softmax(self.thinking_head(conditional_input), dim=-1)
        return support.unsqueeze(-1) + thinking

    def forward(self, x: Tensor) -> Tensor:
        return self.joint_log_probs(x)

    @torch.no_grad()
    def reset_thinking_uniform(self) -> None:
        self.thinking_head.weight.zero_()
        self.thinking_head.bias.zero_()

    @torch.no_grad()
    def select(self, x: Tensor) -> Tensor:
        return self.joint_log_probs(x).flatten(1).argmax(-1)


class FlatRouter(nn.Module):
    architecture = "flat"

    def __init__(self, config: RouterConfig | None = None) -> None:
        super().__init__()
        self.config = config or RouterConfig()
        self.encoder = _encoder(self.config)
        self.head = nn.Linear(self.config.hidden_dim, self.config.n_actions)
        self.reset_thinking_uniform()

    def joint_log_probs(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected (batch, {self.config.input_dim}) features")
        logits = self.head(self.encoder(x))
        return F.log_softmax(logits, dim=-1).reshape(-1, 3, self.config.n_thinking)

    def support_log_probs(self, x: Tensor) -> Tensor:
        return self.joint_log_probs(x).logsumexp(-1)

    def support_logits(self, x: Tensor) -> Tensor:
        return self.support_log_probs(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.joint_log_probs(x)

    @torch.no_grad()
    def reset_thinking_uniform(self) -> None:
        t = self.config.n_thinking
        weights = self.head.weight.reshape(3, t, -1)
        biases = self.head.bias.reshape(3, t)
        weights.copy_(weights[:, :1].clone().expand_as(weights))
        biases.copy_(biases[:, :1].clone().expand_as(biases))

    @torch.no_grad()
    def select(self, x: Tensor) -> Tensor:
        return self.joint_log_probs(x).flatten(1).argmax(-1)


Router = FactorizedRouter | FlatRouter


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
