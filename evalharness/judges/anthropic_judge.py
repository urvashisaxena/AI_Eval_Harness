"""LLM-as-judge backed by the Claude API.

Every grading primitive is a single Messages API call constrained with
structured outputs (``output_config.format`` + JSON schema), so responses are
guaranteed to parse. Requires the ``anthropic`` package and API credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

from __future__ import annotations

import json
from typing import Any

from .base import ClaimVerdict, DEFAULT_RUBRIC, Judge, RubricGrade

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a strict, impartial evaluation judge inside an automated AI "
    "quality harness. Judge only what is asked. Never reward confident "
    "phrasing over factual grounding. Output must satisfy the required JSON "
    "schema exactly."
)


class AnthropicJudge(Judge):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, max_tokens: int = 4096):
        import anthropic  # deferred so offline users never need the package

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    # -- plumbing ----------------------------------------------------------

    def _ask(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("judge model refused the grading request")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    # -- grading primitives --------------------------------------------------

    def extract_claims(self, text: str) -> list[str]:
        schema = {
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["claims"],
            "additionalProperties": False,
        }
        prompt = (
            "Decompose the following text into a list of atomic, self-contained "
            "factual claims. Ignore greetings, hedges, and meta-commentary. "
            "Each claim must be independently verifiable.\n\n"
            f"<text>\n{text}\n</text>"
        )
        return self._ask(prompt, schema)["claims"]

    def verify_claims(self, claims: list[str], evidence: list[str]) -> list[ClaimVerdict]:
        if not claims:
            return []
        schema = {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_index": {"type": "integer"},
                            "supported": {"type": "boolean"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["claim_index", "supported", "reasoning"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        }
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
        passages = "\n\n".join(
            f"<passage index=\"{i}\">\n{p}\n</passage>" for i, p in enumerate(evidence)
        )
        prompt = (
            "For each numbered claim, decide whether it is directly supported by "
            "the evidence passages. A claim is supported only if the evidence "
            "states or clearly entails it — plausibility or general knowledge do "
            "not count. Return one verdict per claim.\n\n"
            f"<claims>\n{numbered}\n</claims>\n\n<evidence>\n{passages}\n</evidence>"
        )
        raw = self._ask(prompt, schema)["verdicts"]
        by_index = {v["claim_index"]: v for v in raw}
        return [
            ClaimVerdict(
                claim=claim,
                supported=bool(by_index.get(i, {}).get("supported", False)),
                reasoning=by_index.get(i, {}).get("reasoning", "no verdict returned"),
            )
            for i, claim in enumerate(claims)
        ]

    def rate_relevance(self, question: str, answer: str) -> float:
        schema = {
            "type": "object",
            "properties": {
                "relevance": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                "reasoning": {"type": "string"},
            },
            "required": ["relevance", "reasoning"],
            "additionalProperties": False,
        }
        prompt = (
            "Rate on a 0–10 scale how directly the answer addresses the question. "
            "10 = fully on-topic and responsive; 0 = unrelated or evasive. Judge "
            "relevance only, not correctness.\n\n"
            f"<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>"
        )
        return self._ask(prompt, schema)["relevance"] / 10.0

    def rate_context_relevance(self, question: str, context: str) -> bool:
        schema = {
            "type": "object",
            "properties": {
                "relevant": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["relevant", "reasoning"],
            "additionalProperties": False,
        }
        prompt = (
            "Is the retrieved passage useful for answering the question? Answer "
            "true only if it contains information that helps answer it.\n\n"
            f"<question>\n{question}\n</question>\n\n<passage>\n{context}\n</passage>"
        )
        return bool(self._ask(prompt, schema)["relevant"])

    def grade_rubric(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        rubric: dict[str, str] | None = None,
    ) -> list[RubricGrade]:
        rubric = rubric or DEFAULT_RUBRIC
        schema = {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["criterion", "score", "reasoning"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["grades"],
            "additionalProperties": False,
        }
        criteria = "\n".join(f"- {name}: {desc}" for name, desc in rubric.items())
        reference = (
            f"\n\n<reference_answer>\n{ground_truth}\n</reference_answer>"
            if ground_truth
            else ""
        )
        prompt = (
            "Grade the answer on each criterion from 1 (very poor) to 5 "
            "(excellent). Return exactly one grade per criterion, using the "
            "criterion names verbatim.\n\n"
            f"<criteria>\n{criteria}\n</criteria>\n\n"
            f"<question>\n{question}\n</question>{reference}\n\n"
            f"<answer>\n{answer}\n</answer>"
        )
        raw = self._ask(prompt, schema)["grades"]
        by_name = {g["criterion"]: g for g in raw}
        return [
            RubricGrade(
                criterion=name,
                score=int(by_name.get(name, {}).get("score", 1)),
                reasoning=by_name.get(name, {}).get("reasoning", "no grade returned"),
            )
            for name in rubric
        ]
