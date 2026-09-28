# Training and implementation choices

FORGE selects a support form and thinking setting before a frozen host answers. The primary pipeline is three-action offline enumeration, supervised KL warm-start, and six- or twelve-action policy-guided refinement. Full and Lite use separately trained policies.

This repository implements the manuscript's stated method. The original experimental checkpoints, responses and unpublished configuration are not bundled. Where the manuscript does not define a unique algorithm, the concrete choices below make this implementation reproducible; they should not be mistaken for recovered settings from the authors' original runs.

## Architecture and action order

`RouterConfig` defines `input_dim`, `n_thinking`, `hidden_dim`, `embedding_dim`, `dropout`, and `variant`. Defaults are 789,2,256,16,0.1,`full`. Full has 789 inputs, Lite 778, and BGE+BM25 (773).

`FactorizedRouter` contains:

1. Two biased linear layers, input→256→256, each followed by ReLU and dropout 0.1.
2. A linear 256→3 support head.
3. Three learned 16-dimensional support embeddings.
4. A linear 272→T thinking head that conditions on the encoder output and support embedding.

The distribution is `p(support|x) * p(thinking|x,support)`. `joint_log_probs(x)` returns shape`(B,3,T)`. Flattening dimensions 1–2 gives support-major order: Direct, Summary, Raw; within each support, NoThink, CoT-Prompt, Think-Low, Think-High. T=1/2/3/4 gives 3/6/9/12 actions. Argmax selects the first action on an exact tie.

`FlatRouter` uses the same encoder and one linear head with 3T logits. It exposes the same joint-probability and support-marginal interfaces.

Biased parameter counts:

| Feature variant | Factorized 6-arm | Factorized 12-arm |
|---|---:|---:|
| Full 789 | 269,397 | 269,943 |
| Lite 778 | 266,581 | 267,127 |
| BGE+BM25 (773) | 265,301 | 265,847 |

The stated Full six-arm architecture agrees with the manuscript's approximate 269K. A flat Full head has 269,574 parameters atK=6 and 271,116 atK=12; the manuscript's single flat 273K entry cannot be recovered from the described encoder/head alone. The implementation retains the stated layers rather than adding unexplained parameters.

Bias flags, exact dropout placement and initialization were not fully specified in the manuscript. The implementation uses the choices above and PyTorch initialization, then explicitly initializes the thinking head to uniform for Stage 1. Router training uses float 32.

## Preprocessing and utility

`StructuredStandardizer` fits population mean/standard deviation on training features only. It preserves the first 768 normalized embedding dimensions and standardizes the remaining structured features. Constant features use scale 1. Calling `fit(..., split="dev")` or`"test"` fails. Reuse the same saved scaler for dev, test and deployment.

`DatasetCostNormalizer` fits separate input/output token maxima for each dataset using training records only. A zero maximum becomes divisor 1. Costs above the fitted maximum are allowed to exceed 1; evaluation never refits or clips them.

```python
from forge.objectives import DatasetCostNormalizer, StructuredStandardizer, pareto_utility

scaler = StructuredStandardizer().fit(train_features, split="train")
x_train = scaler.transform(train_features)
x_dev = scaler.transform(dev_features)

costs = DatasetCostNormalizer().fit(
    train_input_tokens, train_output_tokens, train_dataset_names, split="train"
)
u_train = pareto_utility(
    train_f1, train_input_tokens, train_output_tokens, train_dataset_names, costs
)
u_dev = pareto_utility(
    dev_f1, dev_input_tokens, dev_output_tokens, dev_dataset_names, costs
)
```

Array shapes are`(N,actions)` for F1/token records and`(N,features)` for features. F1 rewards must be fractions in[0,1], not displayed percentages. Utility is F1−0.1×normalized input tokens−0.2×normalized output tokens. The same training-fitted denominators must be used throughout each run. The paper says dataset-level maxima but does not identify their fitting population; training-only fitting is an explicit implementation choice that prevents held-out leakage. Caller-provided records determine which training actions enter the maxima.

The training functions expect already transformed features and computed utilities. They do not fit scalers implicitly.

## Stage 1

`warm_start(features, utilities, dev_features=None, dev_utilities=None, config=None, router_config=None, model=None)` returns`(model, history)`.

- Utilities with three columns mean the default support-only warm start. Targets are`softmax(utility / 1.0)`, and loss is mean`KL(target || support_policy)`.
- Utilities with 3T columns mean the full-alphabet control. Loss is forward KL against the full joint policy.
- Three-arm warm-start resets thinking to a uniform conditional distribution. On a factorized head, thinking weights/bias start at zero and receive no warm-start gradient. On a flat head, each support's initial thinking logits are equal.
- `hard_labels=True` gives the hard-label baseline, using the first utility argmax.
- Default AdamW: learning rate 2 e-4, weight decay 0.01, batch 64; maximum 50 epochs; patience 7.
- Selection accuracy is agreement with the first argmax of development utility. The earliest best-accuracy checkpoint is restored. These exact tie/accuracy definitions are implementation choices.
- If dev arrays are omitted, training-data selection is explicitly marked`selection_split="train"` in history. Use a disjoint dev split for reported experiments.

```python
from forge.policy import RouterConfig
from forge.training import WarmStartConfig, warm_start

model, history = warm_start(
    x_train, u_train, x_dev, u_dev,
    config=WarmStartConfig(),
    router_config=RouterConfig(input_dim=789, n_thinking=2, variant="full"),
)
```

Provide a`FlatRouter` through`model=` for the flat ablation. A default three-action warm-start may initialize a six- or twelve-action policy, whereas the extra full-alphabet Stage 1 control must receive observations of every corresponding action.

## Stage 2

`refine(model, features, reward_fn, config=None, reference_model=None, progress=None)` returns`(model, history)`. The reward callback receives NumPy query indices of shape`(B,)` and support-major action indices of shape`(B,G)` and returns finite utility rewards of shape`(B,G)`.

```python
from forge.training import RefineConfig, refine

def rewards(query_indices, actions):
    return training_utility_table[query_indices[:, None], actions]

model, history = refine(model, x_train, rewards, RefineConfig())
```

That example is explicitly **offline replay**. A live callback must execute the selected host actions and calculate rewards with the same frozen training normalizers. Replaying records does not recreate 1.28M actual host completions or the paper's GPU-hour measurement.

### Primary GRPO configuration

| Setting | Default |
|---|---:|
| Rollout steps | 5,000 |
| Queries per rollout B | 32 |
| Sampled actions per query G | 8 |
| Inner optimizer steps per rollout | 4 |
| AdamW learning rate | 1 e-5 |
| PPO clip | 0.2 |
| Advantage epsilon | 1 e-4 |
| Initial KL beta | 0.05 |
| Entropy coefficient | 0.01 |
| Adaptive beta interval | 100 rollouts |
| Adaptive beta lower/upper KL | 0.005/0.05 |
| Adaptive beta factor | 1.5 |

Five thousand rollout steps imply 20,000 optimizer steps and 5,000×32×8=1,280,000 sampled action requests. The training history records these separately.

For each group, subtract its reward mean and divide by its population standard deviation plus 1 e-4. A constant-reward group gets zero advantage. The clipped policy term uses the ratio of current to old-policy action probability. Add mean joint`KL(current || Stage1_reference)` weighted by beta and subtract joint entropy weighted by 0.01. The Stage 1 reference is copied and frozen. If supplied explicitly, it must match the current router configuration.

Every 100 rollouts, use the post-update mean per-query joint KL of the last rollout: multiply beta by 1.5 above 0.05, divide by 1.5 below 0.005, otherwise keep it. The paper's nominal target 0.02 lies inside this band; no extra target penalty is added.

### Exact sampling and optimization choices

The manuscript specifies support coverage but not the complete proposal. This implementation reserves one conditional thinking draw for each of Direct/Summary/Raw, samples the remainingG−3 actions with replacement from the old joint distribution, and shuffles their positions. WhenG<3 it samples jointly without coverage guarantees. Conditional draws remain numerically stable even when a support has vanishing joint mass.

Stratification changes the proposal. The manuscript explicitly uses the uncorrected old-policy ratio and calls it a policy-guided clipped surrogate; this implementation preserves that choice and does not claim an unbiased policy gradient or add a hidden importance correction.

Further choices absent from the manuscript:

- Query sampling is uniform with replacement; default seed 42 controls query/action sampling and initialization.
- Each inner step uses the entire fixed rollout. Old selected log probabilities and the Stage 1 reference remain fixed for all four steps.
- Dropout is disabled during Stage 2 rollout and gradient updates. This keeps likelihood ratios independent of resampled dropout masks; gradients are still enabled.
- Stage 2 AdamW weight decay is 0.01, matching Stage 1; betas/epsilon use PyTorch defaults. No scheduler, gradient clipping or beta bounds are added.
- Repeated sampled actions are passed to the reward callback individually. Whether they trigger new host calls or read cached outcomes is an explicit backend decision.

## Optional optimizer ablations

`RefineConfig.algorithm` accepts`grpo`,`dr_grpo`,`rloo`,`dpo`. The paper names the latter three without publishing full settings, so these are transparent reference implementations rather than reconstructed original ablation runs.

| Algorithm | Implemented difference |
|---|---|
| GRPO | Mean-centered, population-SD-normalized rewards; PPO-clipped surrogate. |
| Dr.GRPO | Mean-centered rewards without SD normalization; same clipped surrogate. |
| RLOO | Leave-one-out advantage`G/(G-1)*(r-mean(r))`; unclipped log-policy loss. |
| DPO | Per query, pair highest/lowest sampled utility actions; ignore all-tie groups; use frozen-reference log-ratio DPO loss with inverse temperature 0.1. |

All four use the stated loop's explicit joint KL and entropy terms. For this single-step categorical policy there is no generated-token-length loss normalization. `dpo_beta` controls DPO temperature separately from the KL-anchor beta.

Other ablations are exposed through existing configuration: beta 0 removes the anchor, `adaptive_beta=False` uses a fixed coefficient, `n_thinking` controls 3/6/9/12 actions, `FlatRouter` selects the flat head, stage 1-only skips refinement, and the warm input table controls training size. The 773/778/789 variants require independently prepared feature matrices and independently trained checkpoints. Optimizer ablations do not substitute for the separately adapted Adaptive-RAG/TierMem/s 3/Sysformer/AdaReasoner baselines.

## Checkpoints and deployment contract

```python
from forge.checkpoint import load_checkpoint, save_checkpoint

save_checkpoint(
    "runs/full/policy.pt", model, feature_names=ordered_feature_names,
    standardizer=scaler, cost_normalizer=costs,
    metadata={"protocol": "cached", "reward_source": "gold_f1"},
)
model, checkpoint = load_checkpoint(
    "runs/full/policy.pt", expected_variant="full",
    expected_feature_names=ordered_feature_names, expected_n_thinking=2,
)
```

The payload stores schema version, exact architecture/configuration, action layout, ordered feature names, weights, optional train-fitted normalization statistics, training configuration and metadata. Loading rejects mismatched schemas, state dict shapes, variants, alphabets and feature orders. It uses PyTorch`weights_only=True`. The serializer writes atomically.

The supplied checkpoint API stores a trained policy and preprocessing state, not optimizer/RNG state for exact mid-run resume. Save run histories and source/corpus/split/model manifests alongside it. A checkpoint round-trip test confirms its predictions survive save/load; it does not validate historical paper scores.

## Defaults and tests

[`configs/training.yaml`](../configs/training.yaml) contains defaults compatible with the configuration dataclasses. Router architecture is in`router`, cost weights in`utility`, and loop settings in`stage1`/`stage2`. The extra defaults absent from the paper are documented above.

```sh
pytest tests/test_training.py -q
```

Tests check probability normalization/factorization, initial uniform thinking, parameter counts, Boltzmann labels, clipping values/gradients, zero-variance groups, sampler coverage/stability, train-only normalizers, Stage 1 and Stage 2 learning, deterministic reruns, optional objectives and strict checkpoint round-trips. They use explicitly synthetic inputs and do not claim to reproduce benchmark results.
