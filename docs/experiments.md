# Experiment recipes

These recipes extend the [README workflow](../README.md). Save each run's configuration, source/corpus hashes, split manifest, outputs, and checkpoint together.

## Features and action alphabets

| `--variant` | Input dimensions | Fresh feature calls | Contents |
|---|---:|---:|---|
| `bge` | 773 | 0 | BGE + four BM25 statistics + query/passage cosine |
| `lite` | 778 | 0 | BGE variant + five query-structure features |
| `full` | 789 | 4 | Lite + seven greedy-probe and four self-consistency features |

Prepare separate Stage-0 artifacts and train separate checkpoints for each feature variant. Removing coordinates from a trained Full checkpoint does not produce a Lite model.

| `--n-thinking` | Total actions | Within-support action order |
|---|---:|---|
| 1 | 3 | NoThink |
| 2 | 6 | NoThink, CoT-Prompt |
| 3 | 9 | NoThink, CoT-Prompt, Think-Low |
| 4 | 12 | NoThink, CoT-Prompt, Think-Low, Think-High |

The four thinking settings are ordered consistently in training, checkpoints and results. The nine-action recipe uses NoThink/CoT-Prompt/Think-Low. Six- and twelve-action alphabets are the primary non-thinking and thinking-host settings. `--alphabet warm` always enumerates three NoThink actions while retaining the configured full policy size. `--alphabet full` enumerates all `3 * n_thinking` actions.

Generate variant configurations from the provided defaults:

```sh
python - <<'PY'
from pathlib import Path
import yaml

for variant, dimensions in [('bge', 773), ('lite', 778), ('full', 789)]:
    for n_thinking in (1, 2, 3, 4):
        config = yaml.safe_load(Path('configs/training.yaml').read_text())
        config['router'].update(variant=variant, input_dim=dimensions, n_thinking=n_thinking)
        Path(f'configs/{variant}-{3 * n_thinking}arm.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

For example, independently prepare Lite warm data and train its six-action router:

```sh
forge stage0 --data data/splits/hotpotqa-seed42/train.jsonl \
  --split train --variant lite --alphabet warm --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/lite/train-warm

forge stage0 --data data/splits/hotpotqa-seed42/dev.jsonl \
  --split dev --variant lite --alphabet warm --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/lite/dev-warm

forge train --train runs/hotpotqa/lite/train-warm \
  --dev runs/hotpotqa/lite/dev-warm --config configs/lite-6arm.yaml \
  --seed 42 --output runs/hotpotqa/lite/stage1.pt
```

Repeat test enumeration, refinement and evaluation using this variant's own paths. For a twelve-action experiment, use a verified thinking-capable host, `--n-thinking 4` during all Stage-0 calls, and the corresponding `*-12arm.yaml`. An unsupported budget adapter fails rather than substituting a token ceiling. Anthropic's explicitly enabled native alternative is recorded as a protocol difference; see [adapter limitations](method-decisions.md#prompts-adapters-and-scoring).

## Match the alphabet when isolating Stage 2

The default pipeline expands three warm actions to six refined actions. Its change in quality includes both alphabet expansion and optimization. To isolate refinement, train a separate six-action Stage-1 control with all six actions available:

```sh
forge stage0 --data data/splits/hotpotqa-seed42/train.jsonl \
  --split train --variant full --alphabet full --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/train-six

forge stage0 --data data/splits/hotpotqa-seed42/dev.jsonl \
  --split dev --variant full --alphabet full --n-thinking 2 \
  --host configs/host.local.yaml --allow-host-requests \
  --output runs/hotpotqa/full/dev-six

forge train --train runs/hotpotqa/full/train-six \
  --dev runs/hotpotqa/full/dev-six --config configs/training.yaml \
  --seed 42 --output runs/hotpotqa/full/stage1-six.pt

forge refine --train runs/hotpotqa/full/train-six \
  --checkpoint runs/hotpotqa/full/stage1-six.pt --config configs/training.yaml \
  --host configs/host.local.yaml --allow-host-requests \
  --seed 42 --output runs/hotpotqa/full/stage2-six.pt

forge cached-eval --data runs/hotpotqa/full/test-full \
  --checkpoint runs/hotpotqa/full/stage1-six.pt \
  --output runs/hotpotqa/full/control-stage1.jsonl

forge cached-eval --data runs/hotpotqa/full/test-full \
  --checkpoint runs/hotpotqa/full/stage2-six.pt \
  --output runs/hotpotqa/full/control-stage2.jsonl

forge compare --candidate runs/hotpotqa/full/control-stage2.jsonl \
  --baseline runs/hotpotqa/full/control-stage1.jsonl \
  --metric f1 --output runs/hotpotqa/full/control-paired.json
```

Use the same feature schema, selected-answer decoding, costs, test IDs and host. The train-only cost normalizer is saved in each Stage-1 checkpoint and carried into its refinement. If comparing the three-action and full-alphabet controls directly, note that separately fitted action tables can yield different cost maxima; use the Python API with a shared, explicitly frozen training normalizer for a strict normalization-matched ablation.

## Cost-weight sensitivity

The manuscript's default is `(lambda_in, lambda_out) = (0.10, 0.20)`. Its canonical sensitivity table contains these six pairs:

| Input weight | Output weight |
|---:|---:|
| 0.05 | 0.00 |
| 0.05 | 0.20 |
| 0.10 | 0.00 |
| 0.10 | 0.20 |
| 0.10 | 0.50 |
| 0.20 | 0.20 |

Create one Stage-1 checkpoint per pair using the same observed action table:

```sh
python - <<'PY'
import subprocess

pairs = [(0.05, 0.0), (0.05, 0.2), (0.1, 0.0), (0.1, 0.2), (0.1, 0.5), (0.2, 0.2)]
for lambda_in, lambda_out in pairs:
    tag = f'{lambda_in:.2f}-{lambda_out:.2f}'
    subprocess.run([
        'forge', 'train', '--train', 'runs/hotpotqa/full/train-warm',
        '--dev', 'runs/hotpotqa/full/dev-warm', '--config', 'configs/training.yaml',
        '--lambda-in', str(lambda_in), '--lambda-out', str(lambda_out), '--seed', '42',
        '--output', f'runs/hotpotqa/lambda-{tag}/stage1.pt',
    ], check=True)
PY
```

These commands train Stage 1 only and incur no additional host calls. For each full-pipeline operating point, run `forge refine` from that checkpoint using its own output path, then evaluate on the same held-out data. Refinement takes the cost weights from the checkpoint; it does not silently reset them to the YAML defaults. The six trained outputs are not the published table values until measured.

## Architecture, label and optimization controls

| Control | Configuration |
|---|---|
| Hard labels | Add `--hard-labels` to `forge train` |
| Flat joint head | Add `--flat` to `forge train` |
| Stage-1-only | Evaluate the Stage-1 checkpoint and skip `forge refine` |
| GRPO / RLOO / Dr.GRPO / DPO | `forge refine --algorithm grpo/rloo/dr_grpo/dpo` |
| Remove KL anchor | Set `stage2.beta_initial: 0` and `stage2.adaptive_beta: false` |
| Fixed KL coefficient | Set the desired `stage2.beta_initial` and disable adaptation |
| Rollout length | `forge refine --steps 2500` or the default 5000 |
| Training size | Construct recorded nested train subsets of 200/500/1000/1500/2000 queries, keeping dev/test fixed |

For example:

```sh
forge train --train runs/hotpotqa/full/train-warm \
  --dev runs/hotpotqa/full/dev-warm --config configs/training.yaml \
  --hard-labels --seed 42 --output runs/hotpotqa/hard/stage1.pt

forge train --train runs/hotpotqa/full/train-warm \
  --dev runs/hotpotqa/full/dev-warm --config configs/training.yaml \
  --flat --seed 42 --output runs/hotpotqa/flat/stage1.pt

forge refine --train runs/hotpotqa/full/train-warm \
  --checkpoint runs/hotpotqa/full/stage1.pt --config configs/training.yaml \
  --algorithm rloo --host configs/host.local.yaml --allow-host-requests \
  --seed 42 --output runs/hotpotqa/rloo/stage2.pt
```

See [training.md](training.md#optional-optimizer-ablations) for the DPO, RLOO, and Dr.GRPO objectives.

Offline refinement is useful for code and objective checks:

```sh
forge refine --train runs/hotpotqa/full/train-six \
  --checkpoint runs/hotpotqa/full/stage1-six.pt --config configs/training.yaml \
  --replay-table --steps 20 --seed 42 \
  --output runs/hotpotqa/replay/stage2.pt
```

This is labeled `offline_table_replay`; it is not a live GRPO experiment and does not count cached reads as atomic model completions.

## Baseline APIs

Fixed Direct/Summary/Raw are available through `forge infer --fixed` and `forge cached-eval --fixed`. Their NoThink action IDs are `0`, `n_thinking`, and `2 * n_thinking` respectively. Use identical query sets and cost protocol for comparisons.

`forge.baselines` provides three utilities:

- `BM25ThresholdPolicy.fit(scores, utilities, split="dev", grid_size=21)`: calibrates two thresholds on development utility; lower/middle/higher score regions select Direct/Summary/Raw.
- `dev_fallback(router_utilities, fixed_utilities, split="dev", seed=..., n_resamples=...)`: uses a dev-only paired interval to choose routing or a fixed alternative.
- `utility_oracle(utilities)`: per-query observed-utility argmax, a descriptive enumerated reference. It is not a deployable router or a universal F1/EM upper bound.

This complete example fits and evaluates the threshold baseline from the six-action dev and test artifacts above:

```sh
python - <<'PY'
import numpy as np

from forge.artifacts import OutcomeTable, read_json, read_jsonl
from forge.baselines import BM25ThresholdPolicy
from forge.checkpoint import load_checkpoint
from forge.evaluation import result_from_dict, write_results
from forge.objectives import DatasetCostNormalizer
from forge.pipeline import load_prepared
from forge.schemas import Action

def table(path):
    metadata = read_json(f'{path}/metadata.json')
    return OutcomeTable(
        load_prepared(f'{path}/prepared.jsonl'), read_jsonl(f'{path}/outcomes.jsonl'),
        [Action(**action) for action in metadata['actions']], split=metadata['split'],
    )

dev = table('runs/hotpotqa/full/dev-six')
test = table('runs/hotpotqa/full/test-full')
_, checkpoint = load_checkpoint('runs/hotpotqa/full/stage1-six.pt')
normalizer = DatasetCostNormalizer.from_dict(checkpoint['cost_normalizer'])
dev_columns = [dev.actions.index(Action(support, 0)) for support in range(3)]
test_columns = [test.actions.index(Action(support, 0)) for support in range(3)]
policy = BM25ThresholdPolicy.fit(
    [item.retrieved[0].score for item in dev.prepared],
    dev.utilities(normalizer)[:, dev_columns], split='dev',
)
supports = policy.select([item.retrieved[0].score for item in test.prepared])
rows = test.select([test_columns[int(support)] for support in supports])
write_results('runs/hotpotqa/full/cached-bm25.jsonl', [result_from_dict(row) for row in rows])
print({'low': policy.low, 'high': policy.high})
PY

forge evaluate --data runs/hotpotqa/full/cached-bm25.jsonl \
  --output runs/hotpotqa/full/cached-bm25-summary.json
```

Use the checkpoint's saved cost weights if they differ from the `(0.1, 0.2)` defaults shown. Tune any heuristic on dev only and keep its search budget fixed across runs.

Adaptive-RAG, TierMem, s3, Sysformer, AdaReasoner, and provider-specific judged-answer protocols require separate integrations. Use the same data splits, hosts, and evaluation budget for comparisons.

## Split seeds, training seeds, and nine-run summaries

The example uses data seed 42. For nine-run summaries, choose **three data-sampling seeds** and **three router-training seeds**, and save those values before running the experiment.

For every data seed, run `forge split` into its own directory, enumerate that split with each frozen host, and fit scalers/normalizers using that split's training rows only. Train and refine three policies using the three recorded training seeds, preserving distinct outputs. Retune tunable baselines on that split's dev rows with the same budget. A shorter Stage-2 run must be labeled separately.

For each of the resulting nine runs, compute the five-benchmark macro F1 with equal benchmark weights. Then summarize those nine run-level values using a stated standard-deviation convention. The evaluator's 95% paired bootstrap instead resamples matched queries within benchmark on one evaluated split; it must not be presented as the standard deviation across nine training runs.

## Frozen cross-host transfer

Use the exact saved source-host checkpoint, feature order, scaler and cost weights on the recorded matched transfer subset. Switch only the host configuration and generate target-host features/answers under the stated protocol. Do not retrain Stage 2 or choose target-specific cost weights and still call it zero-shot transfer. A Full Fresh Online target query still requires four target-host probes.

Report Cached transfer results and measured Fresh Online latency separately.
