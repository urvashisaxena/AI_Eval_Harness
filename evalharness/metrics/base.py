"""Metric interface and aggregation."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..judges.base import Judge
from ..schema import EvalCase, MetricScore, Prediction


@dataclass(frozen=True)
class MetricInfo:
    name: str
    description: str
    higher_is_better: bool = True
    needs_contexts: bool = False


class Metric(abc.ABC):
    info: MetricInfo

    @abc.abstractmethod
    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        """Score one (case, prediction) pair. Value is normalized to [0, 1]."""

    def applicable(self, case: EvalCase) -> bool:
        return not (self.info.needs_contexts and not case.contexts)


# Registry of metric metadata, populated as metric classes are defined.
METRIC_INFO: dict[str, MetricInfo] = {}


def register(info: MetricInfo):
    def wrap(cls):
        cls.info = info
        METRIC_INFO[info.name] = info
        return cls

    return wrap


def aggregate(scores: list[MetricScore]) -> float:
    """Mean of per-case values (simple macro average)."""
    if not scores:
        return float("nan")
    return sum(s.value for s in scores) / len(scores)
