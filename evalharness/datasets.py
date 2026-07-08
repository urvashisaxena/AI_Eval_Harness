"""Golden dataset and prediction loading (JSONL)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import EvalCase, Prediction


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load a golden dataset from JSONL.

    Each line: {"id": ..., "question": ..., "ground_truth": ...,
                "contexts": [...], "metadata": {...}}
    """
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(_lines(path), start=1):
        row = json.loads(line)
        if "question" not in row:
            raise ValueError(f"{path}:{lineno}: missing required field 'question'")
        case_id = str(row.get("id", lineno))
        if case_id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {case_id!r}")
        seen.add(case_id)
        cases.append(
            EvalCase(
                id=case_id,
                question=row["question"],
                ground_truth=row.get("ground_truth", ""),
                contexts=list(row.get("contexts", [])),
                metadata=dict(row.get("metadata", {})),
            )
        )
    if not cases:
        raise ValueError(f"{path}: dataset is empty")
    return cases


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    """Load model predictions from JSONL.

    Each line: {"case_id": ..., "answer": ..., "model": ...}
    (accepts "id" as an alias for "case_id")
    """
    preds: dict[str, Prediction] = {}
    for lineno, line in enumerate(_lines(path), start=1):
        row = json.loads(line)
        case_id = str(row.get("case_id", row.get("id", "")))
        if not case_id:
            raise ValueError(f"{path}:{lineno}: missing 'case_id'")
        if "answer" not in row:
            raise ValueError(f"{path}:{lineno}: missing 'answer'")
        preds[case_id] = Prediction(
            case_id=case_id, answer=row["answer"], model=row.get("model", "")
        )
    return preds


def _lines(path: str | Path):
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            yield line
