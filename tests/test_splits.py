import json

import pytest

from forge.splits import make_splits


def test_disjoint_deterministic_splits_and_matched_transfer(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps({"id": str(i), "question": f"Q{i}?", "answers": ["a"], "benchmark": "toy"}) for i in range(20)))
    first = make_splits(source, tmp_path / "a", train_size=10, dev_size=4, test_size=6, transfer_size=2)
    second = make_splits(source, tmp_path / "b", train_size=10, dev_size=4, test_size=6, transfer_size=2)
    assert first == second
    ids = {name: {e["id"] for e in values} for name, values in first["splits"].items()}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]
    assert ids["transfer"] <= ids["test"]
    with pytest.raises(FileExistsError):
        make_splits(source, tmp_path / "a", train_size=10, dev_size=4, test_size=6, transfer_size=2)
