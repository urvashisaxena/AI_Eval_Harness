"""LLM-as-judge rubric grading."""

from __future__ import annotations

from ..judges.base import Judge
from ..schema import EvalCase, MetricScore, Prediction
from .base import Metric, MetricInfo, register


@register(
    MetricInfo(
        name="llm_rubric",
        description=(
            "Judge-graded rubric score (correctness, completeness, clarity, "
            "safety), averaged and normalized to 0–1."
        ),
        higher_is_better=True,
    )
)
class LLMRubricMetric(Metric):
    def __init__(self, rubric: dict[str, str] | None = None):
        self.rubric = rubric  # None → judge's default rubric

    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        grades = judge.grade_rubric(
            question=case.question,
            answer=pred.answer,
            ground_truth=case.ground_truth,
            rubric=self.rubric,
        )
        value = sum(g.normalized for g in grades) / len(grades) if grades else 0.0
        return MetricScore(
            metric=self.info.name,
            value=round(value, 4),
            details={
                "grades": [
                    {"criterion": g.criterion, "score": g.score, "reasoning": g.reasoning}
                    for g in grades
                ]
            },
        )
