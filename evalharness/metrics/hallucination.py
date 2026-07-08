"""Hallucination-rate measurement.

A claim is *hallucinated* when neither the retrieved contexts nor the
ground-truth reference supports it. The per-case value reported is the
hallucinated-claim rate (0 = fully grounded, 1 = entirely fabricated), so
**lower is better**. Case-level flags are kept in the details so governance
reports can also state "N% of answers contained at least one hallucination".
"""

from __future__ import annotations

from ..judges.base import Judge
from ..schema import EvalCase, MetricScore, Prediction
from .base import Metric, MetricInfo, register


@register(
    MetricInfo(
        name="hallucination_rate",
        description=(
            "Fraction of answer claims unsupported by contexts + ground truth "
            "(lower is better)."
        ),
        higher_is_better=False,
    )
)
class HallucinationMetric(Metric):
    def applicable(self, case: EvalCase) -> bool:
        # Needs *some* grounding signal to check claims against.
        return bool(case.contexts) or bool(case.ground_truth)

    def score_case(self, case: EvalCase, pred: Prediction, judge: Judge) -> MetricScore:
        claims = judge.extract_claims(pred.answer)
        if not claims:
            return MetricScore(self.info.name, 0.0, {"claims": [], "note": "no claims extracted"})
        evidence = list(case.contexts)
        if case.ground_truth:
            evidence.append(case.ground_truth)
        verdicts = judge.verify_claims(claims, evidence)
        hallucinated = [v for v in verdicts if not v.supported]
        return MetricScore(
            metric=self.info.name,
            value=round(len(hallucinated) / len(verdicts), 4),
            details={
                "claims_total": len(verdicts),
                "claims_hallucinated": len(hallucinated),
                "case_has_hallucination": bool(hallucinated),
                "hallucinated_claims": [
                    {"claim": v.claim, "reasoning": v.reasoning} for v in hallucinated
                ],
            },
        )
