import json

import pytest

from evalharness.datasets import load_dataset, load_predictions, dataset_sha256


def test_load_dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"id": "a", "question": "q?", "ground_truth": "gt", "contexts": ["c1"]})
        + "\n"
        + json.dumps({"question": "q2?"})
        + "\n"
    )
    cases = load_dataset(p)
    assert len(cases) == 2
    assert cases[0].id == "a"
    assert cases[0].contexts == ["c1"]
    assert cases[1].id == "2"  # falls back to line number
    assert dataset_sha256(p) == dataset_sha256(p)


def test_load_dataset_rejects_duplicates_and_empty(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": "x", "question": "q"}\n{"id": "x", "question": "q"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(p)
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(p)


def test_load_predictions(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"case_id": "a", "answer": "hi", "model": "m1"}\n{"id": "b", "answer": "yo"}\n')
    preds = load_predictions(p)
    assert preds["a"].model == "m1"
    assert preds["b"].answer == "yo"
