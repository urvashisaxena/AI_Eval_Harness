"""Core data types shared across the harness."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """One row of a golden dataset."""

    id: str
    question: str
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A model answer for one eval case."""

    case_id: str
    answer: str
    model: str = ""


@dataclass
class MetricScore:
    """A single metric value for a single case."""

    metric: str
    value: float  # normalized to [0, 1]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    scores: dict[str, MetricScore] = field(default_factory=dict)


@dataclass
class RunResult:
    """A complete evaluation run — the unit stored in the registry."""

    run_id: str
    label: str
    model: str
    judge: str
    dataset_path: str
    dataset_sha256: str
    created_at: str  # ISO 8601 UTC
    metrics: dict[str, float] = field(default_factory=dict)  # aggregate per metric
    cases: list[CaseResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        cases = [
            CaseResult(
                case_id=c["case_id"],
                question=c["question"],
                answer=c["answer"],
                scores={
                    name: MetricScore(**s) for name, s in c.get("scores", {}).items()
                },
            )
            for c in d.get("cases", [])
        ]
        return cls(
            run_id=d["run_id"],
            label=d.get("label", ""),
            model=d.get("model", ""),
            judge=d.get("judge", ""),
            dataset_path=d.get("dataset_path", ""),
            dataset_sha256=d.get("dataset_sha256", ""),
            created_at=d.get("created_at", ""),
            metrics=d.get("metrics", {}),
            cases=cases,
            metadata=d.get("metadata", {}),
        )
