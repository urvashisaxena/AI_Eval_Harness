from .base import Judge, RubricGrade
from .heuristic import HeuristicJudge


def make_judge(name: str, **kwargs) -> Judge:
    """Factory: 'heuristic' (offline, deterministic) or 'anthropic' (Claude)."""
    if name == "heuristic":
        return HeuristicJudge()
    if name == "anthropic":
        from .anthropic_judge import AnthropicJudge

        return AnthropicJudge(**kwargs)
    raise ValueError(f"unknown judge {name!r} (expected 'heuristic' or 'anthropic')")


__all__ = ["Judge", "RubricGrade", "HeuristicJudge", "make_judge"]
