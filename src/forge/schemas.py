from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answers: tuple[str, ...]
    benchmark: str = ""
    documents: tuple[Document, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    support: int
    thinking: int = 0

    def __post_init__(self):
        if self.support not in range(3) or self.thinking not in range(4):
            raise ValueError("Action requires support in [0, 2], thinking in [0, 3]")

    @property
    def name(self) -> str:
        return f"{('direct', 'summary', 'raw')[self.support]}:{('no_think', 'cot_prompt', 'think_low', 'think_high')[self.thinking]}"

    def index(self, n_thinking: int = 2) -> int:
        if n_thinking not in (1, 2, 3, 4) or self.thinking >= n_thinking:
            raise ValueError("Action is not in this alphabet")
        return self.support * n_thinking + self.thinking

    @classmethod
    def from_index(cls, index: int, n_thinking: int = 2) -> Action:
        if n_thinking not in (1, 2, 3, 4) or not 0 <= index < 3 * n_thinking:
            raise ValueError("Invalid action index or alphabet")
        return cls(*divmod(index, n_thinking))


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_output_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    reasoning_budget: int | None = None
    seed: int | None = None
    tag: str = "answer"

    def __post_init__(self):
        if self.max_output_tokens <= 0 or not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid generation limits or sampling parameters")
        if self.reasoning_budget is not None and self.reasoning_budget <= 0:
            raise ValueError("reasoning_budget must be positive")


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    reasoning_tokens: int = 0
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if min(self.input_tokens, self.output_tokens, self.reasoning_tokens, self.latency_s) < 0:
            raise ValueError("Token usage and latency must be nonnegative")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0
    reasoning_tokens: int = 0

    @classmethod
    def from_generation(cls, generation: Generation) -> Usage:
        return cls(generation.input_tokens, generation.output_tokens, 1,
                   generation.latency_s, generation.reasoning_tokens)

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens,
                     self.calls + other.calls, self.latency_s + other.latency_s,
                     self.reasoning_tokens + other.reasoning_tokens)


@dataclass(frozen=True)
class AnswerResult:
    example_id: str
    benchmark: str
    action: Action
    answer: str
    f1: float
    em: float
    selected_usage: Usage
    probe_usage: Usage
    total_usage: Usage
    generation: Generation
    protocol: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
