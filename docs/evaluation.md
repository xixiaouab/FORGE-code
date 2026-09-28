# Evaluation

`write_results(path, results)` and `read_results(path)` use one serialized
`AnswerResult` per JSONL line. Query identity is `(benchmark, example_id)`.
Repeated IDs within a benchmark and mixed evaluation protocols are errors.
F1 and EM are fractions; multiply by 100 when formatting paper-style percentages.

```python
from forge.evaluation import read_results, summarize_results, paired_bootstrap

candidate = read_results("runs/forge/results.jsonl")
baseline = read_results("runs/raw/results.jsonl")
summary = summarize_results(candidate)
interval = paired_bootstrap(candidate, baseline, metric="f1")
```

## Cost and latency

Every report retains `selected`, `probe`, and `total` usage separately. Total
token usage and calls must equal selected plus probe usage. Reasoning tokens are
a subset of output tokens, so they are not added again to the token total.

- `cached` reports selected-answer tokens per query. Probe acquisition remains
  visible in the separate usage record but is excluded from the reported cost.
- `fresh_online` reports all selected-answer and probe tokens per query.
- `macro` weights benchmarks equally. `pooled` weights individual queries equally.
  Per-benchmark values and both aggregations are retained.

`Usage.latency_s` is accumulated request time. Concurrent probe calls make its
sum different from end-to-end elapsed time. It is exposed only as
`request_latency_sum_s` and is never substituted for online latency.

Fresh Online p50/p95 require actual timings for every reported query in
`generation.metadata`:

```json
{"latency_source": "measured_wall_clock", "end_to_end_latency_s": 0.2714}
```

The timer must span retrieval, feature acquisition, routing, and the final host
answer for a new query, under the stated batching/concurrency setup. Cached
reports never emit online latency quantiles. Replay, simulated, or cache-hit
records cannot be labeled Fresh Online. Missing timings remain unavailable;
the evaluator does not estimate them from cache access or token counts.

## Paired uncertainty

`paired_bootstrap` requires identical protocol and benchmark/query ID sets. It
resamples matched query pairs within each benchmark, then averages benchmark
means equally in every replicate. The reported difference is candidate minus
baseline. Supported metrics are `f1`, `em`, and `reported_cost_k_tokens`.

The returned percentile interval measures uncertainty over matched queries on
the evaluated split. Defaults are 10,000 replicates, 95% confidence, and RNG
seed 42. Compute variation across data splits and training seeds separately.
