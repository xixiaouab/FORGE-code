# End-to-End Workflow

This guide runs FORGE from local benchmark data through training and evaluation. See the [README](../README.md) for installation and a method overview.

## 1. Import Local Data and Save a Split Manifest

Importers accept original local HotpotQA, 2WikiMultiHopQA, MuSiQue, PopQA, and FEVER JSON/JSONL files; PopQA TSV is also supported. Supply a labeled source pool, then create disjoint train/dev/test splits. This example runs HotpotQA:

```sh
mkdir -p data/canonical data/splits runs
forge import \
  --benchmark hotpotqa \
  --data data/raw/hotpot_train_v1.1.json \
  --output data/canonical/hotpotqa.jsonl

forge split \
  --data data/canonical/hotpotqa.jsonl \
  --output data/splits/hotpotqa-seed42 \
  --seed 42 \
  --train-size 1500 --dev-size 500 --test-size 300 --transfer-size 75
```

Replace the source path with the dataset you have. Accepted benchmark names include `hotpotqa`, `2wikimultihopqa`, `musique`, `popqa`, and `fever`. The source must contain at least 2,300 distinct labeled questions for the sizes above. Split output contains `train.jsonl`, `dev.jsonl`, `test.jsonl`, `transfer.jsonl`, and `manifest.json`. Transfer is a matched subset of test. Existing split files are not overwritten.

The example uses seed 42 and records the selected query IDs in the split manifest. Adjust the split sizes for your dataset.

HotpotQA/2Wiki/MuSiQue candidate paragraphs are imported without consulting gold supporting-fact annotations. PopQA and FEVER need a separate textual corpus. Use `--corpus data/corpus.jsonl` consistently in all host-processing commands when using one. FEVER evidence page identifiers alone are not passage text.

Canonical formats are:

```json
{"id":"original-query-id","benchmark":"hotpotqa","question":"...","answers":["..."],"documents":[{"id":"passage-id","title":"...","text":"..."}]}
```

```json
{"id":"passage-id","title":"...","text":"..."}
```

## 2. Configure the Frozen Host

Copy and edit the example configuration before running inference:

```sh
cp configs/host.example.yaml configs/host.local.yaml
```

Set `model` and `base_url` to your served model and OpenAI-compatible `/v1` endpoint. `api_key_env` names the environment variable containing the key; use `null` only for an unauthenticated local endpoint. Configure the environment variable outside version control. Set model-specific `no_thinking_fields` when the server needs an explicit switch to disable native thinking.

`timeout` is measured in seconds. The default rate is 25 requests/minute. Increase it only for an endpoint that permits the corresponding throughput. The `recording` file stores exact requests, outputs, usage, and response metadata; its namespace should identify the frozen host revision. Preserve the same host configuration file throughout each training run, because its SHA-256 is checked across stages. Optional `max_calls` limits requests in one process.

**Network inference requires `--allow-host-requests`.** The commands below explicitly enable it and can incur inference charges. Stage 0 enumerates three selected answers per query for the warm alphabet, plus four feature probes per query for Full. Default Stage 2 requests 5,000 × 32 × 8 = **1,280,000 completions** per operating point. The example 25/minute limit is not the paper's cluster throughput.

For twelve-action experiments, consult [host restrictions](method-decisions.md#prompts-adapters-and-scoring). An output-token ceiling or `reasoning_effort` is not substituted for an exact reasoning budget. Unsupported settings raise an error. Anthropic's native thinking mode requires an explicit opt-in because its sampling and combined token cap differ from the manuscript's independent answer budget.

## 3. Stage 0: Enumerate the Warm and Test Actions

The default warm alphabet is Direct/Summary/Raw × NoThink, while the router is initialized for six actions. Prepare separate train and dev artifacts; enumerate all six actions on test for cached evaluation:

```sh
forge stage0 \
  --data data/splits/hotpotqa-seed42/train.jsonl \
  --split train --variant full --alphabet warm --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/train-warm

forge stage0 \
  --data data/splits/hotpotqa-seed42/dev.jsonl \
  --split dev --variant full --alphabet warm --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/dev-warm

forge stage0 \
  --data data/splits/hotpotqa-seed42/test.jsonl \
  --split test --variant full --alphabet full --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/test-full
```

Each directory contains a hash-bearing `metadata.json`, `prepared.jsonl` with features/probes, and `outcomes.jsonl` with actual selected answers and usage. Rerunning an identical Stage-0 command resumes missing query/action records. Changing input/configuration requires a new output directory.

## 4. Stage 1: Supervised KL Warm-Start

```sh
forge train \
  --train runs/hotpotqa/full/train-warm \
  --dev runs/hotpotqa/full/dev-warm \
  --config configs/training.yaml \
  --seed 42 --device cpu \
  --output runs/hotpotqa/full/stage1.pt
```

The checkpoint includes architecture, weights, feature order, training-fitted standardization and cost normalization, and data/host lineage. A companion history JSON records optimization. Train/dev overlap is rejected. The thinking head starts uniform for the three-action warm-start.

## 5. Stage 2: Live Policy-Guided Refinement

```sh
forge refine \
  --train runs/hotpotqa/full/train-warm \
  --checkpoint runs/hotpotqa/full/stage1.pt \
  --config configs/training.yaml \
  --host configs/host.local.yaml --allow-host-requests \
  --seed 42 --device cpu \
  --output runs/hotpotqa/full/stage2.pt
```

The router can train on CPU while the frozen host runs on the configured server. Each sampled action obtains a new host completion; repeated actions are not silently memoized. The three-action table supplies training queries/features, while the live callback evaluates the expanded six-action alphabet. Rollout rows and training history are saved next to the checkpoint.

`--steps 2` can be used for a deliberately short integration run; it is not the paper's 5,000-step setting. `--replay-table` is a separate offline alternative requiring a full-action training table and is labeled as replay in checkpoint metadata. It does not recreate live Stage-2 calls or reported compute measurements. See [training details](training.md).

## 6. Evaluate Cached and Fresh Online Separately

### Cached: Select Already Recorded Test Answers

For six actions, always-Raw/NoThink is action **4**, not action 2:

```sh
forge cached-eval \
  --data runs/hotpotqa/full/test-full \
  --checkpoint runs/hotpotqa/full/stage2.pt \
  --output runs/hotpotqa/full/cached-forge.jsonl

forge cached-eval \
  --data runs/hotpotqa/full/test-full --fixed 4 \
  --output runs/hotpotqa/full/cached-raw.jsonl

forge evaluate \
  --data runs/hotpotqa/full/cached-forge.jsonl \
  --output runs/hotpotqa/full/cached-summary.json

forge compare \
  --candidate runs/hotpotqa/full/cached-forge.jsonl \
  --baseline runs/hotpotqa/full/cached-raw.jsonl \
  --metric f1 --resamples 10000 --seed 42 \
  --output runs/hotpotqa/full/cached-paired-f1.json
```

Cached evaluation reports selected-answer token cost. Historical probe acquisition remains visible in result records and is excluded from its reported cost. It does not report online latency.

### Fresh Online: Acquire Features and Obtain a New Answer

```sh
forge infer \
  --data data/splits/hotpotqa-seed42/test.jsonl \
  --checkpoint runs/hotpotqa/full/stage2.pt \
  --protocol fresh_online \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/fresh-forge.jsonl

forge infer \
  --data data/splits/hotpotqa-seed42/test.jsonl \
  --fixed 4 --n-thinking 2 --protocol fresh_online \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/fresh-raw.jsonl

forge compare \
  --candidate runs/hotpotqa/full/fresh-forge.jsonl \
  --baseline runs/hotpotqa/full/fresh-raw.jsonl \
  --metric reported_cost_k_tokens --resamples 10000 --seed 42 \
  --output runs/hotpotqa/full/fresh-paired-cost.json
```

Full Fresh Online includes all four probes plus the routed answer. Fixed baselines bypass BGE and routing probes. End-to-end timing includes retrieval, feature acquisition, routing, and the final request; accumulated request durations are kept separately. Inference creates a companion summary JSON automatically and refuses to overwrite an existing output.

`forge infer --protocol cached` instead accepts a `prepared.jsonl` and obtains **new answers with cached features**. `forge cached-eval` selects answers already in an outcome table. Both use Cached accounting; neither is a Fresh Online latency measurement.

Repeat the workflow on the five benchmarks and preserve benchmark/query IDs when combining result JSONL files. Evaluation provides per-benchmark, equally weighted macro, and query-weighted pooled summaries. F1/EM are fractions in saved files; multiply by 100 for percentages. Paired comparison requires identical benchmark/query IDs and the same protocol.

## Experiments

- [Experiments](experiments.md): Full/Lite/BGE variants, 3/6/9/12 actions, full-alphabet Stage-1 control, cost weights, optional optimizer ablations and baseline APIs.
- [Method details](method-decisions.md): exact feature formulas, prompts, retrieval choices, provider restrictions and accounting.
- [Training](training.md): objectives, sampling proposal, normalization and checkpoint contract.
- [Evaluation](evaluation.md): metrics, macro averaging and paired uncertainty.

For the nine-run analysis, use three independent data splits × three training seeds **per method and host**. Record the seeds, regenerate the artifacts, and retune baseline controls on each dev split with a matched budget. Paired-query bootstrap intervals and nine-run standard deviations describe different uncertainty sources.

