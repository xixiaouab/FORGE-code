from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from threading import Lock
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .schemas import Generation, GenerationRequest


class Host(Protocol):
    def generate(self, request: GenerationRequest) -> Generation: ...


class HostError(RuntimeError):
    pass


class UnsupportedThinkingBudget(HostError):
    pass


def _set_path(payload: dict, path: str, value: Any) -> None:
    components = path.split(".")
    cursor = payload
    for component in components[:-1]:
        cursor = cursor.setdefault(component, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Conflicting request field: {path}")
    cursor[components[-1]] = value


def request_key(request: GenerationRequest) -> str:
    return hashlib.sha256(json.dumps(asdict(request), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _HTTPHost:
    def __init__(self, model: str, base_url: str, api_key_env: str | None = None,
                 timeout: float = 120, requests_per_minute: float = 25,
                 max_calls: int | None = None, allow_requests: bool = False):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Use HTTPS for remote hosts")
        if timeout <= 0 or requests_per_minute <= 0 or (max_calls is not None and max_calls <= 0):
            raise ValueError("Invalid request timeout, rate limit or call limit")
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.timeout, self.requests_per_minute = timeout, requests_per_minute
        self.max_calls, self.allow_requests = max_calls, allow_requests
        self.calls = 0
        self._last_request = 0.0
        self._request_lock = Lock()

    def _post(self, suffix: str, payload: dict, headers: dict) -> tuple[dict, float]:
        if not self.allow_requests:
            raise HostError("Live host calls are disabled; explicitly set allow_requests=True to enable billable requests")
        with self._request_lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise HostError(f"Host call limit reached ({self.max_calls})")
            pause = 60 / self.requests_per_minute - (time.monotonic() - self._last_request)
            if pause > 0:
                time.sleep(pause)
            self._last_request = time.monotonic()
            self.calls += 1
        encoded = json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + suffix, data=encoded,
                          headers={"Content-Type": "application/json", **headers}, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise HostError(f"Host HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise HostError(f"Host request failed: {error}") from error
        return result, time.perf_counter() - started

    def _key(self) -> str | None:
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise HostError(f"Missing API key environment variable: {self.api_key_env}")
        return key


class OpenAICompatibleHost(_HTTPHost):
    def __init__(self, model: str, base_url: str, api_key_env: str | None = "OPENAI_API_KEY",
                 output_token_field: str = "max_tokens", extra_fields: dict | None = None,
                 thinking_budget_field: str | None = None, thinking_fields: dict | None = None,
                 no_thinking_fields: dict | None = None, separate_answer_budget: bool = False,
                 **kwargs):
        super().__init__(model, base_url, api_key_env, **kwargs)
        if thinking_budget_field in {"max_tokens", "max_completion_tokens", "max_output_tokens"}:
            raise ValueError("An output token ceiling is not a reasoning budget")
        self.output_token_field = output_token_field
        self.extra_fields = extra_fields or {}
        self.thinking_budget_field = thinking_budget_field
        self.thinking_fields, self.no_thinking_fields = thinking_fields or {}, no_thinking_fields or {}
        self.separate_answer_budget = separate_answer_budget

    def build_payload(self, request: GenerationRequest) -> dict:
        payload = deepcopy(self.extra_fields)
        payload.update({"model": self.model, "messages": [{"role": "user", "content": request.prompt}],
                        "temperature": request.temperature, "top_p": request.top_p, "stream": False})
        if request.reasoning_budget is not None:
            if not self.thinking_budget_field or not self.separate_answer_budget:
                raise UnsupportedThinkingBudget("This endpoint needs a verified reasoning-budget field and independent answer-token cap; max_tokens/reasoning_effort do not implement the paper's exact budgets")
            for field, value in self.thinking_fields.items():
                _set_path(payload, field, value)
            _set_path(payload, self.thinking_budget_field, request.reasoning_budget)
        else:
            for field, value in self.no_thinking_fields.items():
                _set_path(payload, field, value)
        _set_path(payload, self.output_token_field, request.max_output_tokens)
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    def generate(self, request: GenerationRequest) -> Generation:
        payload = self.build_payload(request)
        key = self._key()
        result, latency = self._post("/chat/completions", payload, {"Authorization": f"Bearer {key}"} if key else {})
        try:
            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            if isinstance(content, list):
                content = "".join(block.get("text", "") for block in content if block.get("type") == "text")
            usage = result["usage"]
            inputs, outputs = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
            reasoning = int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0))
        except (KeyError, TypeError, IndexError, ValueError) as error:
            raise HostError("Host response lacks completion text or actual token usage") from error
        logprobs = (choice.get("logprobs") or {}).get("content") or []
        values = [item["logprob"] for item in logprobs if isinstance(item.get("logprob"), (int, float)) and math.isfinite(item["logprob"])]
        confidence = math.exp(sum(values) / len(values)) if values else None
        metadata = {"model": result.get("model", self.model), "request_id": result.get("id"),
                    "finish_reason": choice.get("finish_reason"), "reasoning_budget": request.reasoning_budget,
                    "usage": usage, "reasoning_usage_reported": "reasoning_tokens" in usage.get("completion_tokens_details", {})}
        return Generation(content, inputs, outputs, latency, reasoning, confidence, metadata)


class AnthropicHost(_HTTPHost):
    def __init__(self, model: str, base_url: str = "https://api.anthropic.com/v1",
                 api_key_env: str | None = "ANTHROPIC_API_KEY", allow_native_thinking_defaults: bool = False,
                 extra_fields: dict | None = None, **kwargs):
        super().__init__(model, base_url, api_key_env, **kwargs)
        self.allow_native_thinking_defaults = allow_native_thinking_defaults
        self.extra_fields = extra_fields or {}

    def build_payload(self, request: GenerationRequest) -> dict:
        payload = deepcopy(self.extra_fields)
        payload.update({"model": self.model, "messages": [{"role": "user", "content": request.prompt}],
                        "max_tokens": request.max_output_tokens, "stream": False})
        if request.reasoning_budget is not None:
            if not self.allow_native_thinking_defaults:
                raise UnsupportedThinkingBudget("Anthropic thinking requires native sampling and a combined output cap; enable allow_native_thinking_defaults to acknowledge these documented differences")
            if request.reasoning_budget < 1024:
                raise UnsupportedThinkingBudget("Manual Anthropic thinking budgets must be at least 1024 tokens")
            payload["thinking"] = {"type": "enabled", "budget_tokens": request.reasoning_budget}
            payload["max_tokens"] = request.reasoning_budget + request.max_output_tokens
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        else:
            payload["thinking"] = {"type": "disabled"}
            payload["temperature"] = request.temperature
            if request.top_p != 1.0:
                payload["top_p"] = request.top_p
        if request.seed is not None:
            raise HostError("Anthropic Messages does not support a request seed")
        return payload

    def generate(self, request: GenerationRequest) -> Generation:
        payload = self.build_payload(request)
        key = self._key()
        if not key:
            raise HostError("Anthropic requires an API key")
        result, latency = self._post("/messages", payload, {"x-api-key": key, "anthropic-version": "2023-06-01"})
        try:
            text = "".join(block["text"] for block in result["content"] if block.get("type") == "text")
            usage = result["usage"]
            inputs = int(usage["input_tokens"]) + int(usage.get("cache_creation_input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0))
            outputs = int(usage["output_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise HostError("Anthropic response lacks actual token usage or content") from error
        metadata = {"model": result.get("model", self.model), "request_id": result.get("id"),
                    "stop_reason": result.get("stop_reason"), "usage": usage,
                    "reasoning_budget": request.reasoning_budget, "reasoning_usage_reported": False,
                    "native_thinking_defaults": request.reasoning_budget is not None,
                    "independent_answer_cap": request.reasoning_budget is None}
        return Generation(text, inputs, outputs, latency, metadata=metadata)


class RecordingHost:
    def __init__(self, host: Host, path: str | Path, namespace: str):
        self.host, self.path, self.namespace = host, Path(path), namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8"):
            pass
        self._lock = Lock()

    def generate(self, request: GenerationRequest) -> Generation:
        generation = self.host.generate(request)
        record = {"namespace": self.namespace, "key": request_key(request),
                  "request": asdict(request), "generation": asdict(generation)}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return generation


class ReplayHost:
    def __init__(self, path: str | Path, namespace: str | None = None):
        self.responses = defaultdict(deque)
        self._lock = Lock()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if namespace is not None and record.get("namespace") != namespace:
                    continue
                request = GenerationRequest(**record["request"])
                key = request_key(request)
                if record.get("key", key) != key:
                    raise ValueError("Replay record request hash mismatch")
                self.responses[key].append(Generation(**record["generation"]))
        if not self.responses:
            raise ValueError("No replay records match this namespace")

    def generate(self, request: GenerationRequest) -> Generation:
        key = request_key(request)
        with self._lock:
            if not self.responses.get(key):
                raise HostError(f"No remaining recorded completion for request {key}")
            generation = self.responses[key].popleft()
        return Generation(generation.text, generation.input_tokens, generation.output_tokens,
                          generation.latency_s, generation.reasoning_tokens, generation.confidence,
                          {**generation.metadata, "replayed": True})


def create_host(config: dict, allow_requests: bool = False) -> Host:
    config = dict(config)
    provider = config.pop("provider")
    recording = config.pop("recording", None)
    namespace = config.pop("namespace", config.get("model", "default"))
    if provider == "replay":
        host = ReplayHost(config.pop("path"), namespace=config.pop("replay_namespace", None))
        if config:
            raise ValueError(f"Unsupported replay options: {sorted(config)}")
        return host
    config["allow_requests"] = allow_requests
    if provider in ("openai", "openai_compatible"):
        host = OpenAICompatibleHost(**config)
    elif provider == "anthropic":
        host = AnthropicHost(**config)
    else:
        raise ValueError(f"Unknown host provider: {provider}")
    return RecordingHost(host, recording, namespace) if recording else host
