"""Evaluation runner: dataset + predictions + judge → RunResult."""

from __future__ import annotations

import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import HarnessConfig
from .datasets import dataset_sha256, load_dataset, load_predictions
from .judges import make_judge
from .metrics import aggregate, make_metrics
from .schema import CaseResult, MetricScore, Prediction, RunResult


def generate_predictions(
    dataset_path: str | Path, model: str, max_tokens: int = 1024
) -> dict[str, Prediction]:
    """Generate answers with a Claude model under test (contexts included
    in the prompt when present, mimicking a RAG application)."""
    import anthropic

    client = anthropic.Anthropic()
    preds: dict[str, Prediction] = {}
    for case in load_dataset(dataset_path):
        if case.contexts:
            ctx = "\n\n".join(
                f"<context index=\"{i}\">\n{c}\n</context>"
                for i, c in enumerate(case.contexts)
            )
            content = (
                "Answer the question using only the provided context. If the "
                f"context is insufficient, say so.\n\n{ctx}\n\n"
                f"Question: {case.question}"
            )
        else:
            content = case.question
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        answer = next((b.text for b in response.content if b.type == "text"), "")
        preds[case.id] = Prediction(case_id=case.id, answer=answer, model=model)
    return preds


def run_eval(
    dataset_path: str | Path,
    predictions: dict[str, Prediction],
    config: HarnessConfig,
    model: str = "",
    label: str = "",
    verbose: bool = False,
) -> RunResult:
    cases = load_dataset(dataset_path)
    judge = make_judge(
        config.judge,
        **({"model": config.judge_model} if config.judge == "anthropic" else {}),
    )
    metrics = make_metrics(config.metrics)

    missing = [c.id for c in cases if c.id not in predictions]
    if missing:
        raise ValueError(f"predictions missing for case ids: {missing}")

    case_results: list[CaseResult] = []
    per_metric: dict[str, list[MetricScore]] = {m.info.name: [] for m in metrics}

    for i, case in enumerate(cases, start=1):
        pred = predictions[case.id]
        result = CaseResult(case_id=case.id, question=case.question, answer=pred.answer)
        for metric in metrics:
            if not metric.applicable(case):
                continue
            score = metric.score_case(case, pred, judge)
            result.scores[metric.info.name] = score
            per_metric[metric.info.name].append(score)
        case_results.append(result)
        if verbose:
            print(f"  [{i}/{len(cases)}] scored case {case.id}", file=sys.stderr)

    aggregates = {
        name: round(aggregate(scores), 4)
        for name, scores in per_metric.items()
        if scores and not math.isnan(aggregate(scores))
    }

    inferred_model = model or next(
        (p.model for p in predictions.values() if p.model), "unknown"
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    return RunResult(
        run_id=run_id,
        label=label or run_id,
        model=inferred_model,
        judge=judge.name,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256(dataset_path),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metrics=aggregates,
        cases=case_results,
        metadata={"num_cases": len(cases), "metrics_config": config.metrics},
    )


def run_from_files(
    dataset_path: str | Path,
    predictions_path: str | Path,
    config: HarnessConfig,
    model: str = "",
    label: str = "",
    verbose: bool = False,
) -> RunResult:
    predictions = load_predictions(predictions_path)
    return run_eval(dataset_path, predictions, config, model=model, label=label, verbose=verbose)
