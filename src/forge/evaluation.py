from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schemas import Action, AnswerResult, Generation, Usage


PROTOCOL_LABELS = {"cached": "Cached", "fresh_online": "Fresh Online"}
_USAGE_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "calls")
_SUPPORTS = ("direct", "summary", "raw")
_THINKING = ("no_think", "cot_prompt", "think_low", "think_high")


def _validate_result(result: AnswerResult) -> None:
    if result.protocol not in PROTOCOL_LABELS:
        raise ValueError(f"Unknown evaluation protocol: {result.protocol!r}")
    if not isinstance(result.example_id, str) or not result.example_id.strip():
        raise ValueError("Results require a nonempty example_id")
    if not isinstance(result.benchmark, str) or not result.benchmark.strip():
        raise ValueError("Results require a nonempty benchmark")
    if not isinstance(result.answer, str):
        raise ValueError("The answer must be a string")
    for name in ("f1", "em"):
        value = getattr(result, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    if result.em not in (0, 1):
        raise ValueError("Per-query exact match must be 0 or 1")
    for scope in ("selected_usage", "probe_usage", "total_usage"):
        usage = getattr(result, scope)
        for name in _USAGE_FIELDS:
            value = getattr(usage, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{scope}.{name} must be a nonnegative integer")
        if not math.isfinite(usage.latency_s) or usage.latency_s < 0:
            raise ValueError(f"{scope}.latency_s must be finite and nonnegative")
        if usage.reasoning_tokens > usage.output_tokens:
            raise ValueError("Reasoning tokens must be included in output_tokens")
    for name in _USAGE_FIELDS:
        if getattr(result.total_usage, name) != (
            getattr(result.selected_usage, name) + getattr(result.probe_usage, name)
        ):
            raise ValueError(f"total_usage.{name} must equal selected plus probe usage")
    metadata = result.generation.metadata
    if not isinstance(metadata, dict):
        raise ValueError("Generation metadata must be an object")
    if result.protocol == "fresh_online" and any(
        metadata.get(name) is True for name in ("cache_hit", "replayed", "simulated")
    ):
        raise ValueError("Cached, replayed, or simulated outputs are not Fresh Online measurements")


def result_from_dict(value: Mapping[str, Any]) -> AnswerResult:
    data = dict(value)
    data["action"] = Action(**data["action"])
    data["generation"] = Generation(**data["generation"])
    for name in ("selected_usage", "probe_usage", "total_usage"):
        data[name] = Usage(**data[name])
    result = AnswerResult(**data)
    _validate_result(result)
    return result


def read_results(path: str | Path) -> list[AnswerResult]:
    results = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Each JSONL record must be an object")
                results.append(result_from_dict(value))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return results


def write_results(path: str | Path, results: Iterable[AnswerResult]) -> None:
    rows = list(results)
    for row in rows:
        _validate_result(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for result in rows:
                stream.write(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _index_results(results: Iterable[AnswerResult]) -> tuple[str, dict[tuple[str, str], AnswerResult]]:
    indexed = {}
    protocols = set()
    for result in results:
        _validate_result(result)
        key = (result.benchmark, result.example_id)
        if key in indexed:
            raise ValueError(f"Duplicate benchmark/query ID: {key!r}")
        indexed[key] = result
        protocols.add(result.protocol)
    if not indexed:
        raise ValueError("Evaluation requires at least one result")
    if len(protocols) != 1:
        raise ValueError("Cannot combine Cached and Fresh Online results")
    return protocols.pop(), indexed


def _usage_summary(usages: Sequence[Usage]) -> dict[str, Any]:
    total = {name: sum(getattr(usage, name) for usage in usages) for name in _USAGE_FIELDS}
    total["tokens"] = total["input_tokens"] + total["output_tokens"]
    total["request_latency_sum_s"] = sum(usage.latency_s for usage in usages)
    return {"total": total, "per_query": {key: value / len(usages) for key, value in total.items()}}


def _latency_summary(results: Sequence[AnswerResult], protocol: str) -> dict[str, Any]:
    if protocol != "fresh_online":
        return {"available": False, "reason": "Cached evaluation is not Fresh Online latency"}
    values = []
    for result in results:
        metadata = result.generation.metadata
        value = metadata.get("end_to_end_latency_s")
        if metadata.get("latency_source") != "measured_wall_clock" or value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("end_to_end_latency_s must be finite and nonnegative")
        values.append(value)
    if len(values) != len(results):
        return {"available": False, "measured_queries": len(values),
                "reason": "Complete measured end-to-end wall-clock timings were not supplied"}
    return {"available": True, "source": "measured_wall_clock", "n_queries": len(values),
            "mean_s": float(np.mean(values)), "p50_s": float(np.quantile(values, 0.5)),
            "p95_s": float(np.quantile(values, 0.95))}


def _summarize_group(results: Sequence[AnswerResult], protocol: str) -> dict[str, Any]:
    size = len(results)
    counts = Counter(result.action.name for result in results)
    support_counts = Counter(_SUPPORTS[result.action.support] for result in results)
    thinking_counts = Counter(_THINKING[result.action.thinking] for result in results)
    usage = {name: _usage_summary([getattr(result, f"{name}_usage") for result in results])
             for name in ("selected", "probe", "total")}
    scope = "selected" if protocol == "cached" else "total"
    return {
        "n_queries": size,
        "f1": float(np.mean([result.f1 for result in results])),
        "em": float(np.mean([result.em for result in results])),
        "usage": usage,
        "reported_cost_tokens_per_query": usage[scope]["per_query"]["tokens"],
        "reported_cost_k_tokens_per_query": usage[scope]["per_query"]["tokens"] / 1000,
        "action_counts": dict(sorted(counts.items())),
        "action_fractions": {name: count / size for name, count in sorted(counts.items())},
        "support_fractions": {name: support_counts[name] / size for name in _SUPPORTS},
        "thinking_fractions": {name: thinking_counts[name] / size for name in _THINKING},
        "online_latency": _latency_summary(results, protocol),
    }


def summarize_results(results: Iterable[AnswerResult]) -> dict[str, Any]:
    protocol, indexed = _index_results(results)
    groups = defaultdict(list)
    for result in indexed.values():
        groups[result.benchmark].append(result)
    benchmarks = {name: _summarize_group(rows, protocol) for name, rows in sorted(groups.items())}
    macro = {name: float(np.mean([summary[name] for summary in benchmarks.values()])) for name in
             ("f1", "em", "reported_cost_tokens_per_query", "reported_cost_k_tokens_per_query")}
    macro["usage_per_query"] = {
        scope: {field: float(np.mean([summary["usage"][scope]["per_query"][field]
                                    for summary in benchmarks.values()]))
                for field in next(iter(benchmarks.values()))["usage"][scope]["per_query"]}
        for scope in ("selected", "probe", "total")
    }
    action_names = sorted({result.action.name for result in indexed.values()})
    macro["action_fractions"] = {
        name: float(np.mean([summary["action_fractions"].get(name, 0.0) for summary in benchmarks.values()]))
        for name in action_names
    }
    return {
        "protocol": protocol, "protocol_label": PROTOCOL_LABELS[protocol],
        "n_queries": len(indexed), "n_benchmarks": len(benchmarks), "score_unit": "fraction",
        "macro_weighting": "equal_benchmark",
        "cost_usage": "selected" if protocol == "cached" else "total",
        "reasoning_tokens_included_in_output_tokens": True,
        "benchmarks": benchmarks, "macro": macro,
        "pooled": _summarize_group(list(indexed.values()), protocol),
    }


def paired_bootstrap(
    candidate: Iterable[AnswerResult], baseline: Iterable[AnswerResult], *, metric: str = "f1",
    n_resamples: int = 10000, confidence: float = 0.95, seed: int = 42,
) -> dict[str, Any]:
    protocol, left = _index_results(candidate)
    other_protocol, right = _index_results(baseline)
    if protocol != other_protocol:
        raise ValueError("Paired comparisons require the same evaluation protocol")
    if left.keys() != right.keys():
        raise ValueError("Paired comparisons require identical benchmark/query ID sets")
    if metric not in ("f1", "em", "reported_cost_k_tokens"):
        raise ValueError("metric must be f1, em, or reported_cost_k_tokens")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    def value(result: AnswerResult) -> float:
        if metric in ("f1", "em"):
            return getattr(result, metric)
        usage = result.selected_usage if protocol == "cached" else result.total_usage
        return (usage.input_tokens + usage.output_tokens) / 1000

    differences = defaultdict(list)
    for key in sorted(left):
        differences[key[0]].append(value(left[key]) - value(right[key]))
    rng = np.random.default_rng(seed)
    tails = ((1 - confidence) / 2, (1 + confidence) / 2)
    macro_samples = np.zeros(n_resamples, dtype=np.float64)
    benchmarks = {}
    estimates = []
    for name, group in sorted(differences.items()):
        delta = np.asarray(group, dtype=np.float64)
        samples = np.empty(n_resamples, dtype=np.float64)
        for start in range(0, n_resamples, 256):
            stop = min(start + 256, n_resamples)
            indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            samples[start:stop] = delta[indices].mean(axis=1)
        estimate = float(delta.mean())
        benchmarks[name] = {"n_pairs": len(delta), "estimate": estimate,
                            "ci": np.quantile(samples, tails).tolist()}
        estimates.append(estimate)
        macro_samples += samples / len(differences)
    return {
        "protocol": protocol, "metric": metric,
        "unit": "fraction" if metric in ("f1", "em") else "k_tokens_per_query",
        "difference": "candidate_minus_baseline", "method": "stratified_paired_query_percentile",
        "confidence": confidence, "n_resamples": n_resamples, "seed": seed,
        "n_pairs": len(left), "benchmarks": benchmarks,
        "macro": {"weighting": "equal_benchmark", "estimate": float(np.mean(estimates)),
                  "ci": np.quantile(macro_samples, tails).tolist()},
    }
