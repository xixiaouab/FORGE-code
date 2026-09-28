from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .artifacts import OutcomeTable, append_jsonl, read_json, read_jsonl, write_json
from .checkpoint import load_checkpoint, save_checkpoint
from .data import load_documents, load_examples, write_examples
from .evaluation import paired_bootstrap, read_results, result_from_dict, summarize_results, write_results
from .features import BGEEmbedder, FeatureExtractor, feature_names
from .hosts import create_host
from .objectives import DatasetCostNormalizer, StructuredStandardizer, pareto_utility
from .pipeline import FEATURE_SCHEMA, ForgePipeline, load_prepared
from .policy import FactorizedRouter, FlatRouter, RouterConfig
from .schemas import Action
from .splits import assert_disjoint, make_splits
from .training import RefineConfig, WarmStartConfig, refine, warm_start


def _config(path):
    value = yaml.safe_load(Path(path).read_text()) if path else {}
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def _pipeline(args, variant):
    if not args.host:
        raise ValueError("Provide --host with a frozen-host configuration YAML")
    host = create_host(_config(args.host), allow_requests=args.allow_host_requests)
    retriever = None
    if args.corpus:
        from .retrieval import BM25Retriever
        retriever = BM25Retriever(load_documents(args.corpus))
    embedder = BGEEmbedder(device=args.embedding_device)
    return ForgePipeline(retriever, FeatureExtractor(embedder), host, variant)


def _load_artifact(path):
    path = Path(path)
    metadata = read_json(path / "metadata.json")
    if metadata.get("feature_schema") != FEATURE_SCHEMA or not metadata.get("complete"):
        raise ValueError("Stage-0 artifact is incomplete or uses a different feature schema")
    prepared = load_prepared(path / "prepared.jsonl")
    if any(p.variant != metadata["variant"] for p in prepared):
        raise ValueError("Prepared feature variant does not match artifact metadata")
    if metadata["n_thinking"] not in (1, 2, 3, 4) or metadata["alphabet"] not in ("warm", "full"):
        raise ValueError("Invalid action alphabet in artifact metadata")
    expected_actions = [asdict(Action(s, t)) for s in range(3)
                        for t in (range(metadata["n_thinking"]) if metadata["alphabet"] == "full" else [0])]
    if metadata["actions"] != expected_actions:
        raise ValueError("Artifact actions must match the declared support-major alphabet")
    rows = read_jsonl(path / "outcomes.jsonl")
    if any(row.get("protocol") != "cached" for row in rows):
        raise ValueError("Stage-0 outcomes must use cached selected-answer accounting")
    table = OutcomeTable(prepared, rows,
                         [Action(**a) for a in metadata["actions"]], split=metadata["split"])
    if len(prepared) != metadata["n_queries"]:
        raise ValueError("Stage-0 query count does not match manifest")
    return metadata, table


def stage0(args):
    examples = load_examples(args.data, args.benchmark)
    if not examples:
        raise ValueError("Stage 0 needs at least one query")
    actions = [Action(s, t) for s in range(3) for t in (range(args.n_thinking) if args.alphabet == "full" else [0])]
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"feature_schema": FEATURE_SCHEMA, "variant": args.variant, "n_thinking": args.n_thinking,
                "alphabet": args.alphabet, "actions": [asdict(a) for a in actions], "split": args.split,
                "source_sha256": _sha(args.data), "corpus_sha256": _sha(args.corpus),
                "host_config_sha256": _sha(args.host), "n_queries": len(examples), "complete": False}
    manifest_path = destination / "metadata.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if {**previous, "complete": False} != metadata:
            raise ValueError("Existing Stage-0 directory has a different manifest; use a new output path")
        if previous["complete"]:
            _load_artifact(destination)
            print(f"Already complete: {destination}")
            return
    else:
        if any(destination.iterdir()):
            raise ValueError("Output directory is nonempty and has no Stage-0 manifest")
        write_json(manifest_path, metadata)
    pipeline = _pipeline(args, args.variant)
    prepared_path, outcomes_path = destination / "prepared.jsonl", destination / "outcomes.jsonl"
    prior = load_prepared(prepared_path) if prepared_path.exists() else []
    prepared_map = {(p.example.benchmark, p.example.id): p for p in prior}
    rows = read_jsonl(outcomes_path) if outcomes_path.exists() else []
    done = {(r["benchmark"], r["example_id"], r["action"]["support"], r["action"]["thinking"]) for r in rows}
    for i, example in enumerate(examples):
        key = (example.benchmark, example.id)
        prepared = prepared_map.get(key)
        if prepared is None:
            prepared = pipeline.prepare(example, protocol="fresh_online")
            append_jsonl(prepared_path, prepared.to_dict())
        for action in actions:
            identity = (*key, action.support, action.thinking)
            if identity not in done:
                result = pipeline.answer(prepared, action, protocol="cached")
                append_jsonl(outcomes_path, result.to_dict())
                done.add(identity)
        print(f"Stage 0: {i + 1}/{len(examples)}", flush=True)
    OutcomeTable(load_prepared(prepared_path), read_jsonl(outcomes_path), actions, split=args.split)
    write_json(manifest_path, {**metadata, "complete": True})


def train(args):
    metadata, table = _load_artifact(args.train)
    devmeta, dev = _load_artifact(args.dev)
    if metadata["split"] != "train" or devmeta["split"] != "dev":
        raise ValueError("Stage 1 requires train and dev artifacts with those split labels")
    for key in ("variant", "n_thinking", "actions", "host_config_sha256", "corpus_sha256", "feature_schema"):
        if metadata[key] != devmeta[key]:
            raise ValueError(f"Train/dev mismatch: {key}")
    assert_disjoint([p.example for p in table.prepared], [p.example for p in dev.prepared])
    scaler = StructuredStandardizer().fit(table.features, split="train")
    normalizer = table.fit_normalizer()
    settings = _config(args.config)
    utility_config = settings.get("utility", {})
    if utility_config.get("f1_scale", "fraction") != "fraction" or utility_config.get("normalization_fit_split", "train") != "train":
        raise ValueError("Utility configuration requires F1 fractions and train-only normalization")
    lambda_in = args.lambda_in if args.lambda_in is not None else utility_config.get("lambda_in", 0.1)
    lambda_out = args.lambda_out if args.lambda_out is not None else utility_config.get("lambda_out", 0.2)
    train_u = table.utilities(normalizer, lambda_in, lambda_out)
    dev_u = dev.utilities(normalizer, lambda_in, lambda_out)
    options = dict(settings.get("stage1", {}))
    options.update({key: value for key, value in {"seed": args.seed, "device": args.device, "hard_labels": args.hard_labels}.items()
                    if value is not None})
    if args.epochs is not None:
        options["max_epochs"] = args.epochs
    config = WarmStartConfig(**options)
    router_options = dict(settings.get("router", {}))
    for key, actual in {"input_dim": table.features.shape[1], "n_thinking": metadata["n_thinking"], "variant": metadata["variant"]}.items():
        if key in router_options and router_options[key] != actual:
            raise ValueError(f"Router configuration does not match training artifact: {key}")
        router_options[key] = actual
    router_config = RouterConfig(**router_options)
    torch.manual_seed(config.seed)
    model = (FlatRouter if args.flat else FactorizedRouter)(router_config)
    model, history = warm_start(scaler.transform(table.features), train_u, scaler.transform(dev.features), dev_u,
                                config=config, model=model)
    lineage = {"stage": 1, "feature_schema": FEATURE_SCHEMA, "stage0": metadata,
               "lambda_in": lambda_in, "lambda_out": lambda_out,
               "warm_alphabet": metadata["alphabet"], "host_config_sha256": metadata["host_config_sha256"]}
    save_checkpoint(args.output, model, feature_names=feature_names(metadata["variant"]),
                    standardizer=scaler, cost_normalizer=normalizer, metadata=lineage)
    write_json(str(args.output) + ".history.json", history)
    print(json.dumps({"checkpoint": args.output, "epochs": len(history), "last": history[-1]}))


def refine_command(args):
    metadata, table = _load_artifact(args.train)
    if metadata["split"] != "train":
        raise ValueError("Refinement must use the training split")
    model, payload = load_checkpoint(args.checkpoint, expected_variant=metadata["variant"],
                                     expected_feature_names=feature_names(metadata["variant"]),
                                     expected_n_thinking=metadata["n_thinking"])
    if payload["metadata"].get("stage") != 1:
        raise ValueError("Refinement starts from a Stage-1 checkpoint with its fixed KL anchor")
    for key in ("source_sha256", "corpus_sha256", "host_config_sha256", "variant", "feature_schema", "n_thinking", "n_queries"):
        if payload["metadata"]["stage0"][key] != metadata[key]:
            raise ValueError(f"Refinement artifact differs from the Stage-1 training manifest: {key}")
    scaler = StructuredStandardizer.from_dict(payload["standardizer"])
    normalizer = DatasetCostNormalizer.from_dict(payload["cost_normalizer"])
    lin, lout = payload["metadata"]["lambda_in"], payload["metadata"]["lambda_out"]
    options = dict(_config(args.config).get("stage2", {}))
    options.update({key: value for key, value in {"seed": args.seed, "device": args.device, "algorithm": args.algorithm}.items()
                    if value is not None})
    if args.steps is not None:
        options["steps"] = args.steps
    config = RefineConfig(**options)
    if args.replay_table:
        if metadata["alphabet"] != "full":
            raise ValueError("Offline replay refinement requires a complete full-action table")
        utilities = table.utilities(normalizer, lin, lout)
        reward_fn = lambda indices, actions: utilities[indices[:, None], actions]
        mode = "offline_table_replay"
    else:
        if not args.host:
            raise ValueError("Online refinement needs --host and --allow-host-requests")
        if _sha(args.host) != payload["metadata"]["host_config_sha256"]:
            raise ValueError("Stage 2 must use the same frozen host configuration as Stage 1")
        pipeline = _pipeline(args, metadata["variant"])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        def reward_fn(indices, actions):
            scores, cin, cout = [np.zeros(actions.shape, dtype=float) for _ in range(3)]
            datasets = [table.datasets[i] for i in indices]
            for b, index in enumerate(indices):
                for g, action_index in enumerate(actions[b]):
                    result = pipeline.answer(table.prepared[index], Action.from_index(int(action_index), model.config.n_thinking), protocol="cached")
                    scores[b, g] = result.f1
                    cin[b, g] = result.selected_usage.input_tokens
                    cout[b, g] = result.selected_usage.output_tokens
                    append_jsonl(str(args.output) + ".rollouts.jsonl", result.to_dict())
            return pareto_utility(scores, cin, cout, datasets, normalizer, lin, lout)
        mode = "online_host_feedback"
    model, history = refine(model, scaler.transform(table.features), reward_fn, config=config)
    lineage = {**payload["metadata"], "stage": 2, "refinement_mode": mode, "stage1_sha256": _sha(args.checkpoint)}
    save_checkpoint(args.output, model, feature_names=payload["feature_names"], standardizer=scaler,
                    cost_normalizer=normalizer, metadata=lineage)
    write_json(str(args.output) + ".history.json", history)
    print(json.dumps({"checkpoint": args.output, "mode": mode, "rollout_steps": len(history)}))


def _select(model, payload, prepared):
    scaler = StructuredStandardizer.from_dict(payload["standardizer"])
    x = torch.from_numpy(scaler.transform(np.stack([p.features for p in prepared])))
    model.eval()
    with torch.no_grad():
        return model.select(x).cpu().numpy()


def cached_eval(args):
    metadata, table = _load_artifact(args.data)
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, expected_variant=metadata["variant"],
                                         expected_feature_names=feature_names(metadata["variant"]),
                                         expected_n_thinking=metadata["n_thinking"])
        indices = _select(model, payload, table.prepared)
        lookup = {a.index(metadata["n_thinking"]): i for i, a in enumerate(table.actions)}
        try:
            indices = [lookup[int(i)] for i in indices]
        except KeyError as error:
            raise ValueError("Selected action absent from cache; enumerate the full alphabet for evaluation") from error
    else:
        fixed = Action.from_index(args.fixed, metadata["n_thinking"])
        indices = [table.actions.index(fixed)] * len(table.prepared)
    rows = [result_from_dict(r) for r in table.select(indices)]
    write_results(args.output, rows)
    write_json(str(args.output) + ".summary.json", summarize_results(rows))


def infer(args):
    model = payload = None
    variant = "lite"
    fixed_action = Action.from_index(args.fixed, args.n_thinking) if args.fixed is not None else None
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint)
        variant = model.config.variant
        if payload["feature_names"] != feature_names(variant):
            raise ValueError("Checkpoint feature schema does not match extractor")
    if args.protocol == "cached":
        queries = load_prepared(args.data)
        if model is None and queries:
            variant = queries[0].variant
        if any(p.variant != variant for p in queries):
            raise ValueError("Cached query feature variant differs from checkpoint")
    else:
        queries = load_examples(args.data, args.benchmark)
    if not queries:
        raise ValueError("Inference needs at least one query")
    pipeline = _pipeline(args, variant)
    if Path(args.output).exists():
        raise FileExistsError("Inference output exists; use a new path")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for i, query in enumerate(queries):
        if args.protocol == "cached":
            prepared = query
        else:
            prepared = pipeline.prepare(query) if model else pipeline.prepare_fixed(query, fixed_action)
        action = (Action.from_index(int(_select(model, payload, [prepared])[0]), model.config.n_thinking)
                  if model else fixed_action)
        result = pipeline.answer(prepared, action, protocol=args.protocol)
        append_jsonl(args.output, result.to_dict())
        print(f"Inference: {i + 1}/{len(queries)}", flush=True)
    write_json(str(args.output) + ".summary.json", summarize_results(read_results(args.output)))


def _host_options(parser):
    parser.add_argument("--host", help="Host YAML; credentials are read from an environment variable")
    parser.add_argument("--corpus", help="Corpus JSONL; otherwise use per-example documents")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--allow-host-requests", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="forge", description="FORGE paper-based implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("import", help="Convert a locally supplied benchmark to canonical JSONL")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--benchmark", required=True)
    p.set_defaults(run=lambda a: write_examples(a.output, load_examples(a.data, a.benchmark)))
    p = commands.add_parser("split", help="Create disjoint splits and a matched transfer subset")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--benchmark"); p.add_argument("--seed", type=int, default=42)
    for name, value in (("train", 1500), ("dev", 500), ("test", 300), ("transfer", 75)):
        p.add_argument(f"--{name}-size", type=int, default=value)
    p.set_defaults(run=lambda a: make_splits(a.data, a.output, benchmark=a.benchmark, seed=a.seed,
                   train_size=a.train_size, dev_size=a.dev_size, test_size=a.test_size, transfer_size=a.transfer_size))
    p = commands.add_parser("stage0", help="Enumerate frozen-host actions with resumable row-level outputs")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--benchmark"); p.add_argument("--variant", choices=["full", "lite", "bge"], default="full")
    p.add_argument("--split", choices=["train", "dev", "test", "transfer"], required=True)
    p.add_argument("--alphabet", choices=["warm", "full"], default="warm")
    p.add_argument("--n-thinking", type=int, choices=[1, 2, 3, 4], default=2)
    _host_options(p); p.set_defaults(run=stage0)
    p = commands.add_parser("train", help="Stage 1: KL distillation or matched hard-label control")
    p.add_argument("--train", required=True); p.add_argument("--dev", required=True)
    p.add_argument("--output", required=True); p.add_argument("--config")
    p.add_argument("--lambda-in", type=float); p.add_argument("--lambda-out", type=float)
    p.add_argument("--seed", type=int); p.add_argument("--device")
    p.add_argument("--epochs", type=int); p.add_argument("--hard-labels", action="store_true", default=None)
    p.add_argument("--flat", action="store_true"); p.set_defaults(run=train)
    p = commands.add_parser("refine", help="Stage 2: fresh host feedback, or explicitly labeled offline replay")
    p.add_argument("--train", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True); p.add_argument("--config"); p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int); p.add_argument("--device")
    p.add_argument("--algorithm", choices=["grpo", "rloo", "dr_grpo", "dpo"])
    p.add_argument("--replay-table", action="store_true"); _host_options(p); p.set_defaults(run=refine_command)
    p = commands.add_parser("cached-eval", help="Select recorded answers; never report replay as online latency")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint"); selection.add_argument("--fixed", type=int)
    p.set_defaults(run=cached_eval)
    p = commands.add_parser("infer", help="Route and generate answers, or run a fixed action")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--benchmark")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint"); selection.add_argument("--fixed", type=int)
    p.add_argument("--protocol", choices=["cached", "fresh_online"], default="fresh_online")
    p.add_argument("--n-thinking", choices=[1, 2, 3, 4], type=int, default=2)
    _host_options(p); p.set_defaults(run=infer)
    p = commands.add_parser("evaluate", help="Aggregate results by benchmark and equally weighted macro")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.set_defaults(run=lambda a: write_json(a.output, summarize_results(read_results(a.data))))
    p = commands.add_parser("compare", help="Matched, benchmark-stratified paired bootstrap")
    p.add_argument("--candidate", required=True); p.add_argument("--baseline", required=True); p.add_argument("--output", required=True)
    p.add_argument("--metric", choices=["f1", "em", "reported_cost_k_tokens"], default="f1")
    p.add_argument("--resamples", type=int, default=10000); p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run=lambda a: write_json(a.output, paired_bootstrap(read_results(a.candidate), read_results(a.baseline),
                   metric=a.metric, n_resamples=a.resamples, seed=a.seed)))
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
