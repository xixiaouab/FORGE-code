from dataclasses import replace
import json

import pytest

from forge.evaluation import paired_bootstrap, read_results, summarize_results, write_results
from forge.schemas import Action, AnswerResult, Generation, Usage


def result(index="1", benchmark="hotpotqa", f1=1.0, em=1.0, protocol="cached", action=None,
           selected=None, probe=None, metadata=None):
    selected = selected or Usage(100, 20, 1, 0.2, 5)
    probe = probe or Usage(240, 40, 4, 0.8, 0)
    generation = Generation("answer", selected.input_tokens, selected.output_tokens,
                            selected.latency_s, selected.reasoning_tokens, metadata=metadata or {})
    return AnswerResult(index, benchmark, action or Action(1, 0), "answer", f1, em,
                        selected, probe, selected + probe, generation, protocol)


def test_jsonl_roundtrip_preserves_nested_dataclasses(tmp_path):
    rows = [result(metadata={"model": "fixture", "sampling": {"seed": 42}}),
            result("2", action=Action(2, 1), f1=0.25, em=0)]
    path = tmp_path / "nested" / "results.jsonl"
    write_results(path, iter(rows))
    assert read_results(path) == rows
    assert isinstance(read_results(path)[0].action, Action)
    assert json.loads(path.read_text().splitlines()[0])["selected_usage"]["calls"] == 1


def test_invalid_jsonl_reports_line_and_does_not_silently_skip(tmp_path):
    path = tmp_path / "broken.jsonl"
    write_results(path, [result()])
    with path.open("a") as stream:
        stream.write('{"example_id": "missing fields"}\n')
    with pytest.raises(ValueError, match=r"broken.jsonl:2"):
        read_results(path)


def test_invalid_write_preserves_existing_file(tmp_path):
    path = tmp_path / "results.jsonl"
    write_results(path, [result()])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="f1"):
        write_results(path, [result("2", f1=float("nan"))])
    assert path.read_bytes() == original


def test_macro_equally_weights_benchmarks_not_queries():
    rows = [result(str(i), f1=1, em=1, action=Action(2, 0)) for i in range(3)]
    rows.append(result("1", benchmark="musique", f1=0, em=0, action=Action(0, 1)))
    summary = summarize_results(rows)
    assert summary["macro"]["f1"] == 0.5
    assert summary["macro"]["em"] == 0.5
    assert summary["pooled"]["f1"] == 0.75
    assert summary["macro"]["action_fractions"] == {"direct:cot_prompt": 0.5, "raw:no_think": 0.5}
    assert summary["pooled"]["support_fractions"] == {"direct": 0.25, "summary": 0, "raw": 0.75}
    assert summary["benchmarks"]["musique"]["n_queries"] == 1


def test_costs_keep_selected_probe_and_total_separate():
    cached = summarize_results([result()])
    fresh = summarize_results([result(protocol="fresh_online")])
    assert cached["protocol_label"] == "Cached"
    assert fresh["protocol_label"] == "Fresh Online"
    assert cached["cost_usage"] == "selected"
    assert fresh["cost_usage"] == "total"
    assert cached["macro"]["reported_cost_k_tokens_per_query"] == pytest.approx(0.120)
    assert fresh["macro"]["reported_cost_k_tokens_per_query"] == pytest.approx(0.400)
    usage = cached["pooled"]["usage"]
    assert usage["selected"]["total"]["tokens"] == 120
    assert usage["probe"]["total"]["tokens"] == 280
    assert usage["total"]["total"]["calls"] == 5
    assert usage["selected"]["total"]["reasoning_tokens"] == 5
    assert cached["reasoning_tokens_included_in_output_tokens"] is True


def test_different_benchmark_costs_are_macro_averaged():
    rows = [result(str(i), selected=Usage(100, 0, 1)) for i in range(3)]
    rows.append(result("1", "musique", selected=Usage(500, 0, 1)))
    summary = summarize_results(rows)
    assert summary["macro"]["reported_cost_tokens_per_query"] == 300
    assert summary["pooled"]["reported_cost_tokens_per_query"] == 200


def test_cached_timings_are_never_reported_as_fresh_online_latency():
    summary = summarize_results([result(metadata={"end_to_end_latency_s": 0.001,
                                                 "latency_source": "measured_wall_clock"})])
    assert summary["pooled"]["online_latency"]["available"] is False
    assert "p50_s" not in summary["pooled"]["online_latency"]


def test_online_latency_uses_measured_wall_clock_not_sum_of_parallel_requests():
    rows = [result(str(i), protocol="fresh_online", metadata={
        "end_to_end_latency_s": value, "latency_source": "measured_wall_clock"
    }) for i, value in enumerate((0.30, 0.50, 0.70))]
    summary = summarize_results(rows)
    latency = summary["pooled"]["online_latency"]
    assert latency["available"] is True
    assert latency["p50_s"] == pytest.approx(0.5)
    assert latency["p95_s"] == pytest.approx(0.68)
    assert summary["pooled"]["usage"]["total"]["per_query"]["request_latency_sum_s"] == 1


def test_partial_online_timings_do_not_produce_biased_quantiles():
    rows = [result("1", protocol="fresh_online", metadata={
        "end_to_end_latency_s": 0.2, "latency_source": "measured_wall_clock"
    }), result("2", protocol="fresh_online")]
    latency = summarize_results(rows)["pooled"]["online_latency"]
    assert latency["available"] is False
    assert latency["measured_queries"] == 1
    assert "p95_s" not in latency


@pytest.mark.parametrize("metadata", [{"cache_hit": True}, {"simulated": True}, {"replayed": True}])
def test_replays_cannot_be_labeled_fresh_online(metadata):
    with pytest.raises(ValueError, match="not Fresh Online"):
        summarize_results([result(protocol="fresh_online", metadata=metadata)])


def test_mixed_protocol_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="Cannot combine"):
        summarize_results([result("1"), result("2", protocol="fresh_online")])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_results([result(), result()])
    with pytest.raises(ValueError, match="at least one"):
        summarize_results([])


def test_inconsistent_total_usage_is_rejected():
    row = replace(result(), total_usage=Usage(340, 60, 4, 1.0, 5))
    with pytest.raises(ValueError, match="total_usage.calls"):
        summarize_results([row])


def test_bootstrap_matches_ids_not_input_order_and_keeps_pair_correlation():
    candidate = [result(str(i), f1=float(i % 2), em=i % 2) for i in range(20)]
    baseline = list(reversed(candidate))
    outcome = paired_bootstrap(candidate, baseline, n_resamples=200)
    assert outcome["macro"]["estimate"] == 0
    assert outcome["macro"]["ci"] == [0.0, 0.0]
    assert outcome["n_pairs"] == 20


def test_bootstrap_macro_preserves_equal_benchmark_weighting():
    candidate = [result(str(i), f1=1, em=1) for i in range(10)]
    baseline = [result(str(i), f1=0, em=0) for i in range(10)]
    candidate.append(result("1", "musique", f1=0, em=0))
    baseline.append(result("1", "musique", f1=1, em=1))
    outcome = paired_bootstrap(candidate, baseline, n_resamples=300)
    assert outcome["macro"]["estimate"] == 0
    assert outcome["macro"]["ci"] == [0.0, 0.0]
    assert outcome["benchmarks"]["hotpotqa"]["ci"] == [1.0, 1.0]
    assert outcome["benchmarks"]["musique"]["ci"] == [-1.0, -1.0]


def test_bootstrap_interval_matches_a_small_exact_resampling_distribution():
    candidate = [result("1", f1=0, em=0), result("2", f1=1, em=1)]
    baseline = [result("1", f1=0, em=0), result("2", f1=0, em=0)]
    outcome = paired_bootstrap(candidate, baseline, n_resamples=10000)
    assert outcome["macro"]["estimate"] == 0.5
    assert outcome["macro"]["ci"] == [0.0, 1.0]
    assert outcome == paired_bootstrap(list(reversed(candidate)), baseline, n_resamples=10000)


def test_bootstrap_cost_uses_protocol_accounting():
    selected = Usage(80, 20, 1)
    candidate = result(selected=selected, probe=Usage(800, 100, 4))
    baseline = result(selected=Usage(280, 20, 1), probe=Usage())
    cached = paired_bootstrap([candidate], [baseline], metric="reported_cost_k_tokens", n_resamples=10)
    fresh = paired_bootstrap([replace(candidate, protocol="fresh_online")],
                             [replace(baseline, protocol="fresh_online")],
                             metric="reported_cost_k_tokens", n_resamples=10)
    assert cached["macro"]["estimate"] == pytest.approx(-0.2)
    assert fresh["macro"]["estimate"] == pytest.approx(0.7)


@pytest.mark.parametrize("candidate,baseline", [
    ([result("1")], [result("2")]),
    ([result("1")], [result("1", benchmark="musique")]),
    ([result("1")], [result("1", protocol="fresh_online")]),
])
def test_bootstrap_requires_exact_pairs_and_same_protocol(candidate, baseline):
    with pytest.raises(ValueError):
        paired_bootstrap(candidate, baseline, n_resamples=10)


@pytest.mark.parametrize("options", [{"n_resamples": 0}, {"confidence": 1}, {"metric": "latency"}])
def test_bootstrap_rejects_invalid_settings(options):
    with pytest.raises(ValueError):
        paired_bootstrap([result()], [result()], **options)
