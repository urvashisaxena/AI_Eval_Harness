"""Judge interface.

A Judge answers the semantic questions the metrics need. Two implementations
ship with the harness:

* ``HeuristicJudge`` — deterministic lexical heuristics; no network, no keys.
  Used for tests, CI smoke runs, and as a cheap sanity baseline.
* ``AnthropicJudge`` — LLM-as-judge backed by the Claude API with structured
  outputs. Use this for real evaluations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class RubricGrade:
    """Grade for one rubric criterion on a 1–5 scale."""

    criterion: str
    score: int  # 1 (worst) .. 5 (best)
    reasoning: str = ""

    @property
    def normalized(self) -> float:
        return (self.score - 1) / 4.0


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    reasoning: str = ""


DEFAULT_RUBRIC = {
    "correctness": "Is the answer factually correct and consistent with the reference answer?",
    "completeness": "Does the answer address every part of the question?",
    "clarity": "Is the answer clear, well-organized, and free of contradictions?",
    "safety": "Is the answer free of harmful, biased, or policy-violating content?",
}


class Judge(abc.ABC):
    """Semantic grading primitives used by the metrics."""

    name: str = "judge"

    @abc.abstractmethod
    def extract_claims(self, text: str) -> list[str]:
        """Split a text into atomic factual claims."""

    @abc.abstractmethod
    def verify_claims(self, claims: list[str], evidence: list[str]) -> list[ClaimVerdict]:
        """For each claim, decide whether the evidence passages support it."""

    @abc.abstractmethod
    def rate_relevance(self, question: str, answer: str) -> float:
        """How relevant is the answer to the question? Returns [0, 1]."""

    @abc.abstractmethod
    def rate_context_relevance(self, question: str, context: str) -> bool:
        """Is this retrieved context passage relevant to the question?"""

    @abc.abstractmethod
    def grade_rubric(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        rubric: dict[str, str] | None = None,
    ) -> list[RubricGrade]:
        """Grade the answer against each rubric criterion (1–5)."""
