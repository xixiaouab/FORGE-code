from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from threading import Barrier, Thread

import numpy as np
import pytest

from forge.data import import_example, load_examples, write_examples
from forge.features import FeatureExtractor, feature_names, full_from_lite
from forge.hosts import (AnthropicHost, HostError, OpenAICompatibleHost, RecordingHost,
                         ReplayHost, UnsupportedThinkingBudget)
from forge.metrics import score_answer
from forge.pipeline import ForgePipeline, build_prompt, load_prepared, save_prepared
from forge.retrieval import BM25Retriever, STOPWORDS, raw_support, summary_support, tokenize
from forge.schemas import Action, Document, Example, Generation, GenerationRequest


class TestEmbedder:
    __test__ = False

    def encode(self, texts):
        array = np.zeros((len(texts), 768), dtype=np.float32)
        array[:, 0] = 1
        return array


class TestHost:
    __test__ = False

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return Generation("Paris", 20, 2, 0.01, metadata={"test_fixture": True})


@pytest.fixture
def example():
    return Example("q1", "Which city is France's capital?", ("Paris",), "hotpotqa",
                   (Document("p1", "Paris is the capital of France. It is on the Seine."),
                    Document("p2", "Rome is the capital of Italy."),
                    Document("p3", "Berlin is Germany's capital.")))


def test_bm25_matches_declared_backend():
    from rank_bm25 import BM25Okapi
    documents = [Document(str(i), text) for i, text in enumerate([
        "Paris France river", "Rome Italy river", "Berlin Germany river",
        "Tokyo Japan island", "Paris France city city", "nothing else"
    ])]
    reference = BM25Okapi([tokenize(document.text) for document in documents])
    actual = BM25Retriever(documents)
    for question in ["Paris France", "river", "unknown", "city city"]:
        np.testing.assert_allclose(actual.scores(question), reference.get_scores(tokenize(question)))
    assert len(STOPWORDS) == 57


def test_support_budgets_and_deterministic_ties():
    docs = [Document(str(i), " ".join([f"word{i}"] * 210) + ". Short useful sentence.") for i in range(5)]
    retriever = BM25Retriever(docs)
    result = retriever.retrieve("unseen", 5)
    assert [item.document.id for item in result] == [str(i) for i in range(5)]
    assert len(raw_support(result).split()) == 600
    summary = summary_support("useful", result)
    assert summary == summary_support("useful", result)
    assert len(summary.split()) <= 140
    assert "Short useful sentence." in summary
    assert "word3" not in raw_support(result)


@pytest.mark.parametrize("benchmark,record,expected", [
    ("hotpotqa", {"_id": "a", "question": "Capital?", "answer": "Paris", "context": [["France", ["Paris."]]]}, "Paris"),
    ("2wikimultihopqa", {"_id": "b", "question": "Capital?", "answer": "Paris", "context": {"title": ["France"], "sentences": [["Paris."]]}}, "Paris"),
    ("musique", {"id": "c", "question": "Capital?", "answer": "Paris", "answer_aliases": ["City of Paris"], "paragraphs": [{"idx": 0, "title": "France", "paragraph_text": "Paris."}]}, "Paris"),
    ("popqa", {"id": "d", "question": "Capital?", "possible_answers": '["Paris", "City of Paris"]'}, "Paris"),
    ("fever", {"id": 7, "claim": "Paris is French.", "label": "SUPPORTS", "evidence": [[[1, 2, "Paris", 0]]]}, "SUPPORTS"),
])
def test_benchmark_imports(benchmark, record, expected, tmp_path):
    example = import_example(record, benchmark)
    assert example.answers[0] == expected
    path = tmp_path / "examples.jsonl"
    write_examples(path, [example])
    assert load_examples(path) == [example]


def test_unlabeled_and_duplicate_data_rejected(tmp_path):
    with pytest.raises(ValueError, match="no reference"):
        import_example({"id": "test", "question": "Unknown?"}, "popqa")
    path = tmp_path / "duplicates.jsonl"
    row = json.dumps({"id": "x", "question": "Q?", "answers": ["A"]})
    path.write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="unique"):
        load_examples(path)


def test_dimensions_and_full_group_order(example):
    retrieved = BM25Retriever(example.documents).retrieve(example.question, 5)
    extractor = FeatureExtractor(TestEmbedder())
    probe = Generation("Paris", 20, 2, 0.01)
    samples = [probe, probe, Generation("Rome", 20, 2, 0.01)]
    lite = extractor.extract(example, retrieved, "lite")
    full = extractor.extract(example, retrieved, "full", probe, samples)
    assert extractor.extract(example, retrieved, "bge").shape == (773,)
    assert lite.shape == (778,) and full.shape == (789,)
    np.testing.assert_array_equal(full, full_from_lite(lite, example.question, probe, samples))
    np.testing.assert_allclose(full[-5:-1], [2 / 3, 2 / 3, 1, 2 / 3])
    assert full[-1] == pytest.approx(1)
    for variant, dimensions in (("bge", 773), ("lite", 778), ("full", 789)):
        names = feature_names(variant)
        assert len(names) == len(set(names)) == dimensions
        assert names[-1] == "query_passage_cosine"


def test_full_four_probes_and_fresh_accounting(example, tmp_path):
    host = TestHost()
    pipeline = ForgePipeline(None, FeatureExtractor(TestEmbedder()), host, "full")
    prepared = pipeline.prepare(example)
    assert len(host.requests) == 4
    assert host.requests[0].temperature == 0 and host.requests[0].max_output_tokens == 64
    assert all(request.temperature == 0.7 and request.top_p == 0.9 and request.max_output_tokens == 24
               for request in host.requests[1:])
    result = pipeline.answer(prepared, Action(1, 0))
    assert result.f1 == result.em == 1
    assert result.selected_usage.calls == 1 and result.probe_usage.calls == 4 and result.total_usage.calls == 5
    assert result.total_usage.input_tokens == 100
    assert result.generation.metadata["latency_source"] == "measured_wall_clock"
    with pytest.raises(ValueError, match="newly prepared"):
        pipeline.answer(prepared, Action(2, 0))
    path = tmp_path / "prepared.jsonl"
    save_prepared(path, [prepared])
    cached = load_prepared(path)[0]
    assert cached.to_dict() == prepared.to_dict()
    previous = len(host.requests)
    cached_result = pipeline.answer(cached, Action(2, 0), "cached")
    assert len(host.requests) == previous + 1
    assert cached_result.selected_usage.calls == 1 and cached_result.total_usage.calls == 5
    assert "end_to_end_latency_s" not in cached_result.generation.metadata


def test_lite_never_calls_probe(example):
    host = TestHost()
    pipeline = ForgePipeline(None, FeatureExtractor(TestEmbedder()), host, "lite")
    prepared = pipeline.prepare(example)
    assert host.requests == []
    assert prepared.probe_usage.calls == 0
    result = pipeline.answer(prepared, Action(0, 0))
    assert result.total_usage.calls == 1


def test_full_probes_run_concurrently(example):
    barrier = Barrier(4, timeout=3)

    class ConcurrentHost(TestHost):
        def generate(self, request):
            barrier.wait()
            return super().generate(request)

    pipeline = ForgePipeline(None, FeatureExtractor(TestEmbedder()), ConcurrentHost(), "full")
    prepared = pipeline.prepare(example)
    assert prepared.probe_usage.calls == 4


def test_fixed_baselines_skip_embeddings_and_probes(example):
    class NoEmbedding:
        def encode(self, texts):
            raise AssertionError("Fixed actions must not construct routing features")

    host = TestHost()
    pipeline = ForgePipeline(None, FeatureExtractor(NoEmbedding()), host)
    prepared = pipeline.prepare_fixed(example)
    assert prepared.features.size == 0 and not host.requests
    result = pipeline.answer(prepared, Action(2, 0))
    assert result.total_usage.calls == 1
    direct_example = Example("direct", "Capital?", ("Paris",), "popqa")
    prepared = pipeline.prepare_fixed(direct_example, Action(0, 0))
    assert not prepared.retrieved
    with pytest.raises(ValueError, match="requires retrieved"):
        pipeline.answer(prepared, Action(2, 0))
    assert pipeline.answer(prepared, Action(0, 0)).total_usage.calls == 1


def test_cached_full_requires_probe_assets(example):
    host = TestHost()
    pipeline = ForgePipeline(None, FeatureExtractor(TestEmbedder()), host, "full")
    with pytest.raises(ValueError, match="recorded probe"):
        pipeline.prepare(example, "cached")
    assert not host.requests


def test_action_axis_and_cot_prompt(example):
    for count in range(1, 5):
        for index in range(3 * count):
            assert Action.from_index(index, count).index(count) == index
    retrieved = BM25Retriever(example.documents).retrieve(example.question)
    direct = build_prompt(example, retrieved, Action(0, 0))
    cot = build_prompt(example, retrieved, Action(0, 1))
    assert "Evidence:" not in direct
    assert "Let us think step by step" in cot
    assert "Evidence:" in build_prompt(example, retrieved, Action(2, 0))


def test_strict_thinking_budget_payload():
    request = GenerationRequest("Question?", reasoning_budget=1024)
    host = OpenAICompatibleHost("served-model", "http://localhost:8080/v1", api_key_env=None)
    with pytest.raises(UnsupportedThinkingBudget):
        host.build_payload(request)
    with pytest.raises(ValueError, match="not a reasoning"):
        OpenAICompatibleHost("model", "https://example.org/v1", thinking_budget_field="max_tokens")
    host = OpenAICompatibleHost("served-model", "http://localhost:8080/v1", api_key_env=None,
                                thinking_budget_field="thinking.budget_tokens",
                                thinking_fields={"thinking.type": "enabled"},
                                no_thinking_fields={"thinking.type": "disabled"}, separate_answer_budget=True)
    payload = host.build_payload(request)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert payload["max_tokens"] == 64
    assert host.build_payload(GenerationRequest("Question?"))["thinking"]["type"] == "disabled"
    anthropic = AnthropicHost("claude-sonnet-4-20250514")
    with pytest.raises(UnsupportedThinkingBudget):
        anthropic.build_payload(request)
    anthropic = AnthropicHost("claude-sonnet-4-20250514", allow_native_thinking_defaults=True)
    payload = anthropic.build_payload(request)
    assert payload["thinking"]["budget_tokens"] == 1024
    assert payload["max_tokens"] == 1088
    assert "temperature" not in payload and "top_p" not in payload


def test_recording_replay_consumes_requests(example, tmp_path):
    path = tmp_path / "new" / "nested" / "calls.jsonl"
    recording = RecordingHost(TestHost(), path, "test-fixture")
    request = GenerationRequest("Question?", temperature=0.7, max_output_tokens=24)
    recording.generate(request)
    recording.generate(request)
    replay = ReplayHost(path, "test-fixture")
    assert replay.generate(request).metadata["replayed"]
    assert replay.generate(request).text == "Paris"
    with pytest.raises(HostError, match="No remaining"):
        replay.generate(request)


def test_live_requests_disabled_by_default():
    host = OpenAICompatibleHost("model", "http://localhost:1/v1", api_key_env=None)
    with pytest.raises(HostError, match="disabled"):
        host.generate(GenerationRequest("Question?"))


def test_http_adapters_parse_real_response_contract(monkeypatch):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            if self.path.endswith("/chat/completions"):
                result = {"id": "test-openai", "model": "test", "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}}}
            else:
                result = {"id": "test-anthropic", "model": "test", "content": [{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "Paris"}],
                          "usage": {"input_tokens": 7, "cache_read_input_tokens": 3, "output_tokens": 4}}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("FORGE_TEST_API_KEY", "local-test-only")
    try:
        openai = OpenAICompatibleHost("test", url, api_key_env=None, allow_requests=True)
        output = openai.generate(GenerationRequest("Q?"))
        assert (output.text, output.input_tokens, output.output_tokens, output.reasoning_tokens) == ("Paris", 10, 3, 1)
        anthropic = AnthropicHost("test", url, api_key_env="FORGE_TEST_API_KEY", allow_requests=True)
        output = anthropic.generate(GenerationRequest("Q?"))
        assert (output.text, output.input_tokens, output.output_tokens) == ("Paris", 10, 4)
        assert [path for path, _ in captured] == ["/v1/chat/completions", "/v1/messages"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_answer_metrics():
    assert score_answer("The Paris.", ["Paris"]) == (1, 1)
    assert score_answer("new york", ["York", "New York City"])[0] == pytest.approx(0.8)
