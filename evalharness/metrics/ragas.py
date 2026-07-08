"""RAGAS-style RAG evaluation metrics.

Implements the four core RAGAS measures using the harness's Judge
abstraction:

* faithfulness       — fraction of answer claims supported by the contexts
* answer_relevancy   — how directly the answer addresses the question
* context_precision  — fraction of retrieved contexts relevant to the question
* context_recall     — fraction of ground-truth claims recoverable from contexts
"""

from __future__ import annotations

from ..judges.base import Judge
from ..schema import EvalCase, MetricScore, Prediction
from .base import Metric, MetricInfo, register


@register(
    MetricInfo(
        name="faithfulness",
        description="Fraction of claims in the answer supported by the retrieved contexts.",
        higher_is_better=True,
        needs_contexts=True,
    )
)
class FaithfulnessMetric(Metric):
    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        claims = judge.extract_claims(pred.answer)
        if not claims:
            return MetricScore(self.info.name, 0.0, {"claims": [], "note": "no claims extracted"})
        verdicts = judge.verify_claims(claims, case.contexts)
        supported = sum(v.supported for v in verdicts)
        return MetricScore(
            metric=self.info.name,
            value=round(supported / len(verdicts), 4),
            details={
                "claims_total": len(verdicts),
                "claims_supported": supported,
                "verdicts": [
                    {"claim": v.claim, "supported": v.supported, "reasoning": v.reasoning}
                    for v in verdicts
                ],
            },
        )


@register(
    MetricInfo(
        name="answer_relevancy",
        description="Judge-rated relevance of the answer to the question (0–1).",
        higher_is_better=True,
    )
)
class AnswerRelevancyMetric(Metric):
    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        value = judge.rate_relevance(case.question, pred.answer)
        return MetricScore(self.info.name, round(value, 4))


@register(
    MetricInfo(
        name="context_precision",
        description="Fraction of retrieved context passages relevant to the question.",
        higher_is_better=True,
        needs_contexts=True,
    )
)
class ContextPrecisionMetric(Metric):
    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        flags = [judge.rate_context_relevance(case.question, c) for c in case.contexts]
        value = sum(flags) / len(flags) if flags else 0.0
        return MetricScore(
            metric=self.info.name,
            value=round(value, 4),
            details={"contexts_total": len(flags), "contexts_relevant": sum(flags)},
        )


@register(
    MetricInfo(
        name="context_recall",
        description=(
            "Fraction of ground-truth claims attributable to the retrieved "
            "contexts (did retrieval surface what was needed?)."
        ),
        higher_is_better=True,
        needs_contexts=True,
    )
)
class ContextRecallMetric(Metric):
    def applicable(self, case: EvalCase) -> bool:
        return bool(case.contexts) and bool(case.ground_truth)

    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        claims = judge.extract_claims(case.ground_truth)
        if not claims:
            return MetricScore(self.info.name, 0.0, {"note": "no ground-truth claims"})
        verdicts = judge.verify_claims(claims, case.contexts)
        supported = sum(v.supported for v in verdicts)
        return MetricScore(
            metric=self.info.name,
            value=round(supported / len(verdicts), 4),
            details={"claims_total": len(verdicts), "claims_supported": supported},
        )
