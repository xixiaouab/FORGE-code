from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from forge.checkpoint import load_checkpoint
from forge.pipeline import FEATURE_SCHEMA, PreparedQuery, save_prepared
from forge.schemas import Action, AnswerResult, Example, Generation, Usage


PROJECT = Path(__file__).resolve().parents[1]


def run_cli(*args, success=True):
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(PROJECT / "src"))
    process = subprocess.run([sys.executable, "-m", "forge", *map(str, args)], cwd=PROJECT,
                             env=environment, capture_output=True, text=True, timeout=45)
    if success:
        assert process.returncode == 0, process.stdout + process.stderr
    else:
        assert process.returncode != 0, process.stdout + process.stderr
    return process


def make_artifact(path, split, ids, alphabet="warm", *, host="synthetic-host"):
    path.mkdir(parents=True)
    rng = np.random.default_rng(10)
    prepared = [PreparedQuery(Example(identifier, "Synthetic test question?", ("correct",), "test-benchmark"),
                              (), rng.normal(size=778).astype(np.float32), Usage(), "lite") for identifier in ids]
    save_prepared(path / "prepared.jsonl", prepared)
    actions = [Action(s, t) for s in range(3) for t in (range(2) if alphabet == "full" else [0])]
    rows = []
    for item in prepared:
        for action in actions:
            score = float(action.support == 2)
            answer = "correct" if score else "wrong"
            generation = Generation(answer, 5 + 30 * action.support, 5 + 10 * action.thinking,
                                    0.01, metadata={"simulated": True, "purpose": "integration test fixture"})
            selected = Usage.from_generation(generation)
            rows.append(AnswerResult(item.example.id, item.example.benchmark, action, answer, score, score,
                                     selected, Usage(), selected, generation, "cached").to_dict())
    (path / "outcomes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"feature_schema": FEATURE_SCHEMA, "variant": "lite", "n_thinking": 2,
                "alphabet": alphabet, "actions": [asdict(action) for action in actions], "split": split,
                "source_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "corpus_sha256": None, "host_config_sha256": host, "n_queries": len(ids), "complete": True}
    (path / "metadata.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("cli-artifacts")
    train_ids = [f"train-{i}" for i in range(6)]
    train = make_artifact(base / "train-warm", "train", train_ids)
    full_train = make_artifact(base / "train-full", "train", train_ids, "full")
    dev = make_artifact(base / "dev-warm", "dev", [f"dev-{i}" for i in range(4)])
    test = make_artifact(base / "test-full", "test", [f"test-{i}" for i in range(4)], "full")
    config = base / "config.yaml"
    config.write_text(yaml.safe_dump({
        "router": {"hidden_dim": 8, "embedding_dim": 3, "dropout": 0},
        "utility": {"lambda_in": 0.07, "lambda_out": 0.15},
        "stage1": {"max_epochs": 2, "patience": 1, "batch_size": 3, "learning_rate": 0.01, "seed": 17},
        "stage2": {"steps": 2, "batch_size": 2, "group_size": 3, "inner_steps": 2,
                   "learning_rate": 0.001, "seed": 19, "algorithm": "dr_grpo", "adaptive_beta": False},
    }))
    checkpoint = base / "stage1.pt"
    run_cli("train", "--train", train, "--dev", dev, "--output", checkpoint,
            "--config", config, "--epochs", "1")
    return {"base": base, "train": train, "full_train": full_train, "dev": dev, "test": test,
            "config": config, "checkpoint": checkpoint}


def test_real_cli_train_refine_cached_evaluate_and_compare(artifacts):
    a = artifacts
    model, payload = load_checkpoint(a["checkpoint"])
    assert model.config.hidden_dim == 8
    assert payload["training_config"]["seed"] == 17
    assert payload["training_config"]["max_epochs"] == 1
    assert payload["metadata"]["lambda_in"] == 0.07
    assert payload["metadata"]["lambda_out"] == 0.15
    stage2 = a["base"] / "nested" / "stage2.pt"
    run_cli("refine", "--train", a["full_train"], "--checkpoint", a["checkpoint"],
            "--output", stage2, "--replay-table", "--config", a["config"])
    _, refined = load_checkpoint(stage2)
    assert refined["metadata"]["refinement_mode"] == "offline_table_replay"
    assert refined["training_config"]["algorithm"] == "dr_grpo"
    assert refined["training_config"]["seed"] == 19
    history = json.loads(Path(str(stage2) + ".history.json").read_text())
    assert len(history) == 2
    assert history[-1]["sampled_actions"] == 12
    assert history[-1]["optimizer_steps"] == 4
    policy_rows, raw_rows = a["base"] / "policy.jsonl", a["base"] / "raw.jsonl"
    run_cli("cached-eval", "--data", a["test"], "--checkpoint", stage2, "--output", policy_rows)
    run_cli("cached-eval", "--data", a["test"], "--fixed", "4", "--output", raw_rows)
    summary_path, comparison_path = a["base"] / "summary.json", a["base"] / "comparison.json"
    run_cli("evaluate", "--data", policy_rows, "--output", summary_path)
    run_cli("compare", "--candidate", policy_rows, "--baseline", raw_rows,
            "--resamples", "30", "--output", comparison_path)
    summary = json.loads(summary_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    assert summary["protocol"] == "cached"
    assert summary["n_queries"] == 4
    assert summary["pooled"]["online_latency"]["available"] is False
    assert comparison["n_pairs"] == 4
    assert comparison["unit"] == "fraction"
    assert comparison["method"] == "stratified_paired_query_percentile"


def test_cli_refinement_rejects_warm_replay_and_host_mismatch(artifacts):
    a = artifacts
    result = run_cli("refine", "--train", a["train"], "--checkpoint", a["checkpoint"],
                     "--output", a["base"] / "invalid.pt", "--replay-table", "--steps", "1", success=False)
    assert "complete full-action table" in result.stderr
    mismatched = make_artifact(a["base"] / "different-host", "train", [f"train-{i}" for i in range(6)],
                               "full", host="other-synthetic-host")
    result = run_cli("refine", "--train", mismatched, "--checkpoint", a["checkpoint"],
                     "--output", a["base"] / "invalid2.pt", "--replay-table", "--steps", "1", success=False)
    assert "host_config_sha256" in result.stderr


def test_cli_train_rejects_query_leakage(artifacts):
    a = artifacts
    overlap = make_artifact(a["base"] / "overlap-dev", "dev", ["train-0", "new-dev"])
    result = run_cli("train", "--train", a["train"], "--dev", overlap,
                     "--output", a["base"] / "leak.pt", "--epochs", "1", success=False)
    assert "query overlap" in result.stderr
    assert not (a["base"] / "leak.pt").exists()


def test_cli_rejects_misordered_action_manifest(artifacts):
    a = artifacts
    malformed = make_artifact(a["base"] / "reordered", "test", ["reordered-test"], "full")
    manifest_path = malformed / "metadata.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["actions"] = list(reversed(manifest["actions"]))
    manifest_path.write_text(json.dumps(manifest))
    result = run_cli("cached-eval", "--data", malformed, "--fixed", "4",
                     "--output", a["base"] / "invalid-rows.jsonl", success=False)
    assert "support-major alphabet" in result.stderr
