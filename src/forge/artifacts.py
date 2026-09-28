from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .objectives import DatasetCostNormalizer, pareto_utility


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


class OutcomeTable:
    def __init__(self, prepared, rows, actions, *, split):
        self.prepared = list(prepared)
        self.actions = list(actions)
        self.split = split
        if not self.prepared or not self.actions:
            raise ValueError("Outcome table must contain queries and actions")
        self.datasets = [item.example.benchmark for item in self.prepared]
        self.features = np.stack([item.features for item in self.prepared])
        if len({(p.variant, p.feature_schema) for p in self.prepared}) != 1:
            raise ValueError("Prepared feature variants/schemas cannot be mixed")
        keys = [(item.example.benchmark, item.example.id) for item in self.prepared]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate query IDs in prepared data")
        positions = {key: i for i, key in enumerate(keys)}
        action_positions = {(a.support, a.thinking): j for j, a in enumerate(self.actions)}
        if len(action_positions) != len(self.actions):
            raise ValueError("Duplicate actions in manifest")
        shape = (len(keys), len(self.actions))
        self.f1, self.cin, self.cout = [np.full(shape, np.nan) for _ in range(3)]
        self.rows = {}
        for row in rows:
            key = (row["benchmark"], row["example_id"])
            action = (row["action"]["support"], row["action"]["thinking"])
            if key not in positions or action not in action_positions:
                raise ValueError("Outcome does not match the prepared query/action manifest")
            cell = (positions[key], action_positions[action])
            if cell in self.rows:
                raise ValueError("Duplicate query/action outcome")
            self.rows[cell] = row
            self.f1[cell] = row["f1"]
            self.cin[cell] = row["selected_usage"]["input_tokens"]
            self.cout[cell] = row["selected_usage"]["output_tokens"]
        if not all(np.isfinite(x).all() for x in (self.features, self.f1, self.cin, self.cout)):
            raise ValueError("Incomplete or nonfinite outcome table; resume Stage 0 first")

    def fit_normalizer(self):
        return DatasetCostNormalizer().fit(self.cin, self.cout, self.datasets, split=self.split)

    def utilities(self, normalizer, lambda_in=0.1, lambda_out=0.2):
        return pareto_utility(self.f1, self.cin, self.cout, self.datasets, normalizer, lambda_in, lambda_out)

    def select(self, indices):
        if len(indices) != len(self.prepared):
            raise ValueError("One selected action is required per query")
        return [self.rows[(i, int(action))] for i, action in enumerate(indices)]
