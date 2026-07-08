from .base import Metric, MetricInfo, METRIC_INFO, aggregate
from .llm_rubric import LLMRubricMetric
from .ragas import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
)
from .hallucination import HallucinationMetric

ALL_METRICS = {
    m.info.name: m
    for m in (
        LLMRubricMetric,
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextPrecisionMetric,
        ContextRecallMetric,
        HallucinationMetric,
    )
}


def make_metrics(names: list[str]) -> list[Metric]:
    metrics = []
    for name in names:
        if name not in ALL_METRICS:
            raise ValueError(
                f"unknown metric {name!r}; available: {sorted(ALL_METRICS)}"
            )
        metrics.append(ALL_METRICS[name]())
    return metrics


__all__ = [
    "Metric",
    "MetricInfo",
    "METRIC_INFO",
    "aggregate",
    "ALL_METRICS",
    "make_metrics",
]
