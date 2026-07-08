"""Harness configuration (YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_METRICS = [
    "llm_rubric",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "hallucination_rate",
]


@dataclass
class Threshold:
    """Governance gate for one metric.

    For higher-is-better metrics:  ``min`` is an absolute floor and
    ``max_regression`` the largest tolerated drop vs the baseline.
    For lower-is-better metrics (hallucination_rate): ``max`` is an absolute
    ceiling and ``max_regression`` the largest tolerated *rise* vs baseline.
    """

    min: float | None = None
    max: float | None = None
    max_regression: float | None = None


@dataclass
class HarnessConfig:
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    judge: str = "heuristic"
    judge_model: str = "claude-opus-4-8"
    thresholds: dict[str, Threshold] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "HarnessConfig":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        thresholds = {
            name: Threshold(
                min=t.get("min"),
                max=t.get("max"),
                max_regression=t.get("max_regression"),
            )
            for name, t in (raw.get("thresholds") or {}).items()
        }
        return cls(
            metrics=raw.get("metrics", list(DEFAULT_METRICS)),
            judge=raw.get("judge", "heuristic"),
            judge_model=raw.get("judge_model", "claude-opus-4-8"),
            thresholds=thresholds,
        )
