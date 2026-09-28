from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from .data import import_example
from .features import FEATURE_DIMENSIONS, FeatureExtractor, canonical_variant, full_from_lite
from .hosts import Host
from .metrics import score_answer
from .retrieval import BM25Retriever, RetrievedDocument, raw_support, summary_support
from .schemas import Action, AnswerResult, Document, Example, Generation, GenerationRequest, Usage


FEATURE_SCHEMA = "forge-reconstruction-v1"
FIXED_SCHEMA = "forge-fixed-no-features-v1"


@dataclass
class PreparedQuery:
    example: Example
    retrieved: tuple[RetrievedDocument, ...]
    features: np.ndarray
    probe_usage: Usage
    variant: str = "lite"
    probe: Generation | None = None
    samples: tuple[Generation, ...] = ()
    feature_schema: str = FEATURE_SCHEMA
    preparation_latency_s: float = 0.0
    _started_at: float | None = field(default=None, repr=False)
    _fresh_consumed: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        return {"example": asdict(self.example), "retrieved": [asdict(item) for item in self.retrieved],
                "features": self.features.tolist(), "probe_usage": asdict(self.probe_usage),
                "variant": self.variant, "feature_schema": self.feature_schema,
                "probe": asdict(self.probe) if self.probe else None,
                "samples": [asdict(sample) for sample in self.samples],
                "preparation_latency_s": self.preparation_latency_s}

    @classmethod
    def from_dict(cls, record: dict) -> PreparedQuery:
        if record.get("feature_schema") not in (FEATURE_SCHEMA, FIXED_SCHEMA):
            raise ValueError("Unsupported prepared-query feature schema")
        variant = canonical_variant(record["variant"])
        features = np.asarray(record["features"], dtype=np.float32)
        expected = 0 if record["feature_schema"] == FIXED_SCHEMA else FEATURE_DIMENSIONS[variant]
        if features.shape != (expected,) or not np.isfinite(features).all():
            raise ValueError("Prepared query has invalid feature dimensions or values")
        return cls(import_example(record["example"]),
                   tuple(RetrievedDocument(Document(**item["document"]), float(item["score"])) for item in record["retrieved"]),
                   features, Usage(**record["probe_usage"]), variant,
                   Generation(**record["probe"]) if record.get("probe") else None,
                   tuple(Generation(**sample) for sample in record.get("samples", [])),
                   record["feature_schema"], float(record.get("preparation_latency_s", 0)))


def save_prepared(path: str | Path, prepared: Iterable[PreparedQuery]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def load_prepared(path: str | Path) -> list[PreparedQuery]:
    with Path(path).open(encoding="utf-8") as handle:
        return [PreparedQuery.from_dict(json.loads(line)) for line in handle if line.strip()]


def build_prompt(example: Example, retrieved: Sequence[RetrievedDocument], action: Action) -> str:
    support = ""
    if action.support == 1:
        support = summary_support(example.question, retrieved)
    elif action.support == 2:
        support = raw_support(retrieved)
    instruction = "Answer the question concisely."
    if example.benchmark.lower() == "fever":
        instruction = "Classify the claim. Return exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    prefix = f"{instruction}\n\n"
    if support:
        prefix += f"Evidence:\n{support}\n\n"
    prefix += f"Question: {example.question}"
    if action.thinking == 1:
        prefix += "\nLet us think step by step"
    return prefix + "\nAnswer:"


class ForgePipeline:
    def __init__(self, retriever: BM25Retriever | None, feature_extractor: FeatureExtractor,
                 host: Host, variant: str = "lite"):
        self.retriever, self.feature_extractor, self.host = retriever, feature_extractor, host
        self.variant = canonical_variant(variant)

    def prepare(self, example: Example, protocol: str = "fresh_online", *,
                probe: Generation | None = None, samples: Sequence[Generation] = ()) -> PreparedQuery:
        if protocol not in ("cached", "fresh_online"):
            raise ValueError("Protocol must be cached or fresh_online")
        started = time.perf_counter()
        retriever = self.retriever
        if retriever is None:
            if not example.documents:
                raise ValueError("Provide a corpus retriever or per-example candidate documents")
            retriever = BM25Retriever(example.documents)
        retrieved = tuple(retriever.retrieve(example.question, 5))
        features = self.feature_extractor.extract(example, retrieved, "lite" if self.variant == "full" else self.variant)
        usage = Usage()
        if self.variant == "full":
            if protocol == "fresh_online":
                if probe is not None or samples:
                    raise ValueError("Fresh Online obtains new probes; use cached for supplied generations")
                prompt = build_prompt(example, retrieved, Action(0, 0))
                requests = [GenerationRequest(prompt, tag="probe_greedy")]
                requests.extend(GenerationRequest(prompt, max_output_tokens=24, temperature=0.7,
                                                   top_p=0.9, tag=f"probe_sc_{i}") for i in range(3))
                with ThreadPoolExecutor(max_workers=4) as executor:
                    generations = list(executor.map(self.host.generate, requests))
                probe, samples = generations[0], tuple(generations[1:])
            elif probe is None or len(samples) != 3:
                raise ValueError("Cached Full features require one recorded probe and three SC samples, or load_prepared()")
            for generation in (probe, *samples):
                usage += Usage.from_generation(generation)
        elif probe is not None or samples:
            raise ValueError("Host-independent variants do not consume probe data")
        if self.variant == "full":
            features = full_from_lite(features, example.question, probe, samples)
        return PreparedQuery(example, retrieved, features, usage, self.variant, probe, tuple(samples),
                             preparation_latency_s=time.perf_counter() - started,
                             _started_at=started if protocol == "fresh_online" else None)

    def prepare_fixed(self, example: Example, action: Action | None = None) -> PreparedQuery:
        started = time.perf_counter()
        if action is not None and action.support == 0:
            retrieved = ()
        elif self.retriever is None and not example.documents:
            raise ValueError("Fixed support actions require a corpus or per-example passages")
        else:
            retriever = self.retriever or BM25Retriever(example.documents)
            retrieved = tuple(retriever.retrieve(example.question, 5))
        return PreparedQuery(example, retrieved, np.empty(0, dtype=np.float32), Usage(), self.variant,
                             feature_schema=FIXED_SCHEMA, preparation_latency_s=time.perf_counter() - started,
                             _started_at=started)

    def answer(self, prepared: PreparedQuery, action: Action, protocol: str = "fresh_online") -> AnswerResult:
        if protocol not in ("cached", "fresh_online"):
            raise ValueError("Protocol must be cached or fresh_online")
        if prepared.variant != self.variant or prepared.feature_schema not in (FEATURE_SCHEMA, FIXED_SCHEMA):
            raise ValueError("Prepared query does not match pipeline feature configuration")
        if action.support != 0 and not prepared.retrieved:
            raise ValueError("Summary/Raw requires retrieved passages")
        if protocol == "fresh_online" and (prepared._started_at is None or prepared._fresh_consumed):
            raise ValueError("Fresh Online needs a newly prepared query per routed answer; cached supports reusable features")
        prompt = build_prompt(prepared.example, prepared.retrieved, action)
        budget = {2: 1024, 3: 4096}.get(action.thinking)
        generation = self.host.generate(GenerationRequest(prompt, reasoning_budget=budget, tag=f"answer:{action.name}"))
        selected_usage = Usage.from_generation(generation)
        probes = prepared.probe_usage
        if protocol == "fresh_online":
            prepared._fresh_consumed = True
            live = not generation.metadata.get("replayed") and all(not item.metadata.get("replayed") for item in ((prepared.probe,) if prepared.probe else ()) + prepared.samples)
            if live:
                generation = Generation(generation.text, generation.input_tokens, generation.output_tokens,
                                        generation.latency_s, generation.reasoning_tokens, generation.confidence,
                                        {**generation.metadata, "end_to_end_latency_s": time.perf_counter() - prepared._started_at,
                                         "latency_source": "measured_wall_clock"})
        f1, em = score_answer(generation.text, prepared.example.answers)
        return AnswerResult(prepared.example.id, prepared.example.benchmark, action, generation.text,
                            f1, em, selected_usage, probes, selected_usage + probes, generation, protocol)

    def enumerate(self, prepared: PreparedQuery, actions: Iterable[Action], protocol: str = "cached") -> list[AnswerResult]:
        if protocol != "cached":
            raise ValueError("Arm enumeration reuses features and must use protocol='cached'")
        return [self.answer(prepared, action, protocol=protocol) for action in actions]
