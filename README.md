# FORGE

**Form-Optimal Routing of Grounded Evidence for Frozen LLM Agents**

[Project website](https://xixiaouab.github.io/projects/FORGE/) · [Paper](https://xixiaouab.github.io/projects/FORGE/assets/FORGE.pdf) · [Method details](docs/method-decisions.md) · [Experiment recipes](docs/experiments.md)

FORGE learns which evidence form and thinking setting a frozen language model should use for each question. It combines offline action enumeration, supervised KL distillation, and policy-guided group-relative refinement.

## Features

| Component | Implementation |
|---|---|
| Support | Direct; deterministic extractive Summary; top-3 BM25 Raw |
| Thinking | NoThink; CoT-Prompt; supported host reasoning budgets of 1,024/4,096 tokens |
| Router | Two-layer MLP with factorized support and conditional thinking heads |
| Features | BGE+BM25: 773; Lite: 778; Full: 789 |
| Stage 1 | Boltzmann targets and forward-KL warm-start; hard-label control |
| Stage 2 | Group-relative clipped surrogate, Stage-1 KL anchor, entropy bonus |
| Inference | OpenAI-compatible HTTP; Anthropic HTTP; explicit recorded-response replay |
| Evaluation | Token F1, exact match, separate Cached/Fresh Online costs, paired bootstrap |

Full uses one greedy and three self-consistency probe calls before its routed answer. Lite and BGE+BM25 use no host probes. Four Full probes are submitted concurrently; provider rate limits still apply.

The main non-thinking setup has six actions. Thinking-capable hosts have twelve. Action indices are support-major: `support * n_thinking + thinking`, with support `[Direct, Summary, Raw]` and thinking `[NoThink, CoT-Prompt, Think-Low, Think-High]`.

## Install

Use Python 3.11 or later from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embeddings,test]'
pytest -q
forge --help
```

The embedding model loads lazily on first feature extraction. Model weights and benchmark data are obtained separately under their original licenses. Tests use small, explicitly synthetic fixtures and a localhost HTTP server; they make no external model calls.

## 1. Import local data and save a split manifest

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

## 2. Configure the frozen host

Copy and edit the example configuration before running inference:

```sh
cp configs/host.example.yaml configs/host.local.yaml
```

Set `model` and `base_url` to your served model and OpenAI-compatible `/v1` endpoint. `api_key_env` names the environment variable containing the key; use `null` only for an unauthenticated local endpoint. Configure the environment variable outside version control. Set model-specific `no_thinking_fields` when the server needs an explicit switch to disable native thinking.

`timeout` is measured in seconds. The default rate is 25 requests/minute. Increase it only for an endpoint that permits the corresponding throughput. The `recording` file stores exact requests, outputs, usage, and response metadata; its namespace should identify the frozen host revision. Preserve the same host configuration file throughout each training run, because its SHA-256 is checked across stages. Optional `max_calls` limits requests in one process.

**Network inference requires `--allow-host-requests`.** The commands below explicitly enable it and can incur inference charges. Stage 0 enumerates three selected answers per query for the warm alphabet, plus four feature probes per query for Full. Default Stage 2 requests 5,000 × 32 × 8 = **1,280,000 completions** per operating point. The example 25/minute limit is not the paper's cluster throughput.

For twelve-action experiments, consult [host restrictions](docs/method-decisions.md#prompts-adapters-and-scoring). An output-token ceiling or `reasoning_effort` is not substituted for an exact reasoning budget. Unsupported settings raise an error. Anthropic's native thinking mode requires an explicit opt-in because its sampling and combined token cap differ from the manuscript's independent answer budget.

## 3. Stage 0: enumerate the warm and test actions

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

## 4. Stage 1: supervised KL warm-start

```sh
forge train \
  --train runs/hotpotqa/full/train-warm \
  --dev runs/hotpotqa/full/dev-warm \
  --config configs/training.yaml \
  --seed 42 --device cpu \
  --output runs/hotpotqa/full/stage1.pt
```

The checkpoint includes architecture, weights, feature order, training-fitted standardization and cost normalization, and data/host lineage. A companion history JSON records optimization. Train/dev overlap is rejected. The thinking head starts uniform for the three-action warm-start.

## 5. Stage 2: live policy-guided refinement

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

`--steps 2` can be used for a deliberately short integration run; it is not the paper's 5,000-step setting. `--replay-table` is a separate offline alternative requiring a full-action training table and is labeled as replay in checkpoint metadata. It does not recreate live Stage-2 calls or reported compute measurements. See [training details](docs/training.md).

## 6. Evaluate Cached and Fresh Online separately

### Cached: select already recorded test answers

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

### Fresh Online: acquire features and obtain a new answer

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

- [Experiments](docs/experiments.md): Full/Lite/BGE variants, 3/6/9/12 actions, full-alphabet Stage-1 control, cost weights, optional optimizer ablations and baseline APIs.
- [Method details](docs/method-decisions.md): exact feature formulas, prompts, retrieval choices, provider restrictions and accounting.
- [Training](docs/training.md): objectives, sampling proposal, normalization and checkpoint contract.
- [Evaluation](docs/evaluation.md): metrics, macro averaging and paired uncertainty.

For the nine-run analysis, use three independent data splits × three training seeds **per method and host**. Record the seeds, regenerate the artifacts, and retune baseline controls on each dev split with a matched budget. Paired-query bootstrap intervals and nine-run standard deviations describe different uncertainty sources.

## Citation

```bibtex
@article{xiao2026forge,
  title={FORGE: Form-Optimal Routing of Grounded Evidence for Frozen LLM Agents},
  author={Xiao, Xi and Zhang, Yunbei and Liu, Chen and Zhao, Lin and Chen, Jialin and Zhao, Tianchen and Xu, Xiang and Kim, Youngeun and Wang, Tianyang and Xu, Min},
  year={2026},
  note={Preprint},
  url={https://xixiaouab.github.io/projects/FORGE/}
}
```
