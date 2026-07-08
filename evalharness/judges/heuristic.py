"""Deterministic offline judge.

Lexical-overlap heuristics stand in for an LLM so the full pipeline (metrics,
regression gating, scorecards) can run in tests and CI without network access
or API keys. Scores are crude but stable and monotonic: better answers score
higher, fabricated content scores lower.
"""

from __future__ import annotations

import re

from .base import ClaimVerdict, DEFAULT_RUBRIC, Judge, RubricGrade

_STOPWORDS = frozenset(
    """a an the is are was were be been being to of in on at for with and or as by
    from that this it its their his her they them we you your our i me my not no
    do does did done can could will would should may might have has had""".split()
)


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    }


def _overlap(a: str, b: str) -> float:
    """Fraction of a's content tokens that appear in b."""
    ta = _tokens(a)
    if not ta:
        return 0.0
    return len(ta & _tokens(b)) / len(ta)


class HeuristicJudge(Judge):
    name = "heuristic"

    # A claim counts as supported if this share of its content tokens
    # appears in the concatenated evidence.
    support_threshold = 0.6

    def extract_claims(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [s.strip() for s in sentences if _tokens(s)]

    def verify_claims(self, claims: list[str], evidence: list[str]) -> list[ClaimVerdict]:
        blob = " ".join(evidence)
        out = []
        for claim in claims:
            score = _overlap(claim, blob)
            out.append(
                ClaimVerdict(
                    claim=claim,
                    supported=score >= self.support_threshold,
                    reasoning=f"token overlap with evidence = {score:.2f}",
                )
            )
        return out

    def rate_relevance(self, question: str, answer: str) -> float:
        # Directional: does the answer engage the question's key terms?
        return round(min(1.0, _overlap(question, answer) + 0.15), 4)

    def rate_context_relevance(self, question: str, context: str) -> bool:
        return _overlap(question, context) >= 0.3

    def grade_rubric(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        rubric: dict[str, str] | None = None,
    ) -> list[RubricGrade]:
        rubric = rubric or DEFAULT_RUBRIC
        gt_overlap = _overlap(ground_truth, answer) if ground_truth else 0.5
        q_overlap = _overlap(question, answer)
        grades = []
        for criterion in rubric:
            if criterion == "correctness":
                base = gt_overlap
            elif criterion == "completeness":
                base = 0.5 * gt_overlap + 0.5 * q_overlap
            else:  # clarity, safety, custom criteria — neutral-positive offline
                base = 0.75
            score = max(1, min(5, 1 + round(base * 4)))
            grades.append(
                RubricGrade(
                    criterion=criterion,
                    score=score,
                    reasoning=f"heuristic overlap-based grade (base={base:.2f})",
                )
            )
        return grades
