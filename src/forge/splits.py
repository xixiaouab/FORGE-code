from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_examples, write_examples


def make_splits(source, output, *, benchmark=None, train_size=1500, dev_size=500,
                test_size=300, transfer_size=75, seed=42):
    examples = load_examples(source, benchmark)
    if len({example.benchmark for example in examples}) != 1:
        raise ValueError("Split each benchmark separately to preserve per-benchmark sample sizes")
    sizes = (train_size, dev_size, test_size, transfer_size)
    if any(n < 0 for n in sizes) or transfer_size > test_size:
        raise ValueError("Split sizes must be nonnegative; transfer is a subset of test")
    if len(examples) < train_size + dev_size + test_size:
        raise ValueError("Not enough distinct source queries for disjoint train/dev/test")
    order = np.random.default_rng(seed).permutation(len(examples))
    a, b, c = train_size, train_size + dev_size, train_size + dev_size + test_size
    parts = {"train": order[:a], "dev": order[a:b], "test": order[b:c]}
    parts["transfer"] = parts["test"][:transfer_size]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / f"{name}.jsonl").exists() for name in parts):
        raise FileExistsError("Split output exists; use a new directory")
    manifest = {"seed": seed, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "sampling": "numpy permutation; train/dev/test disjoint; transfer subset of test", "splits": {}}
    for name, indices in parts.items():
        selected = [examples[i] for i in indices]
        write_examples(destination / f"{name}.jsonl", selected)
        manifest["splits"][name] = [{"benchmark": e.benchmark, "id": e.id} for e in selected]
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def assert_disjoint(left, right):
    overlap = {(e.benchmark, e.id) for e in left} & {(e.benchmark, e.id) for e in right}
    if overlap:
        raise ValueError(f"Training/development query overlap: {sorted(overlap)[:5]}")
