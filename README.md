# AI Eval Harness

An AI evaluation and governance harness that produces **model scorecards** from golden
datasets, built on four ideas:

| Concept | What the harness does |
|---|---|
| **LLM-as-judge** | A Claude judge grades answers against a rubric (correctness, completeness, clarity, safety) and verifies factual claims — via structured outputs, so every verdict parses. |
| **RAGAS-style metrics** | Faithfulness, answer relevancy, context precision, and context recall for RAG pipelines. |
| **Hallucination rate** | Answers are decomposed into atomic claims; each claim is checked against the retrieved contexts + ground truth. Unsupported claims count as hallucinations. |
| **Regression tracking** | Every run is stored with dataset hash, model, judge, and per-case evidence. Runs are gated against absolute thresholds *and* a stored baseline — a regression fails CI. |

## Install

```bash
pip install -e ".[anthropic,dev]"   # anthropic extra only needed for the LLM judge
```

## Quick start (offline demo)

The bundled example evaluates two versions of a fictional RAG bot. `v1` is grounded;
`v2` fabricates answers for two cases.

```bash
# 1. Score v1 and promote it to baseline (heuristic judge = no API key needed)
evalharness run --dataset examples/dataset.jsonl \
                --predictions examples/predictions_v1.jsonl \
                --label v1 --set-baseline

# 2. Score v2 with the governance gate — regressions fail with exit code 1
evalharness run --dataset examples/dataset.jsonl \
                --predictions examples/predictions_v2.jsonl \
                --label v2 --config configs/default.yaml --gate

# 3. Render a scorecard (markdown + self-contained HTML)
evalharness history
evalharness scorecard <run-id> -o scorecard.md --html scorecard.html
```

A rendered example lives at [`examples/scorecard_v2.md`](examples/scorecard_v2.md).

## Real evaluations with an LLM judge

Set `judge: anthropic` in your config (see [`configs/default.yaml`](configs/default.yaml))
and export `ANTHROPIC_API_KEY`. The judge model defaults to `claude-opus-4-8` and uses
structured outputs for claim extraction, claim verification, relevance rating, and
rubric grading.

You can also have the harness **generate** the answers under test:

```bash
evalharness run --dataset examples/dataset.jsonl \
                --generate --model claude-opus-4-8 \
                --label nightly --config configs/default.yaml --gate
```

## Dataset format

Golden datasets are JSONL, one case per line:

```json
{"id": "billing-refund",
 "question": "What is the refund window for annual subscriptions?",
 "ground_truth": "Full refund within 30 days; prorated afterwards.",
 "contexts": ["Billing policy: ...", "After the 30 day window ..."],
 "metadata": {"topic": "billing"}}
```

Predictions are JSONL too: `{"case_id": "...", "answer": "...", "model": "..."}`.

## Metrics

| Metric | Direction | Needs contexts | Definition |
|---|---|---|---|
| `llm_rubric` | ↑ | no | Mean rubric grade (1–5 → 0–1) across correctness, completeness, clarity, safety |
| `faithfulness` | ↑ | yes | Fraction of answer claims supported by the retrieved contexts |
| `answer_relevancy` | ↑ | no | Judge-rated relevance of the answer to the question |
| `context_precision` | ↑ | yes | Fraction of retrieved passages relevant to the question |
| `context_recall` | ↑ | yes | Fraction of ground-truth claims recoverable from the contexts |
| `hallucination_rate` | ↓ | contexts *or* ground truth | Fraction of answer claims supported by **neither** contexts nor ground truth |

## Governance: thresholds and gating

Thresholds live in the config YAML. `min`/`max` are absolute gates for a single run;
`max_regression` bounds movement in the *worse* direction versus the baseline:

```yaml
thresholds:
  faithfulness:        {min: 0.70, max_regression: 0.05}
  hallucination_rate:  {max: 0.25, max_regression: 0.05}
```

`evalharness run --gate` (or `evalharness compare <run-id>`) exits non-zero on any
violation, so wiring it into CI blocks quality regressions from shipping. Every stored
run records the dataset SHA-256, model, judge, timestamp, aggregate metrics, and
per-case claim-level evidence — an audit trail for review.

A ready-made GitHub Actions gate is in
[`.github/workflows/eval-gate.yml`](.github/workflows/eval-gate.yml).

## Judges

- **`anthropic`** — LLM-as-judge on the Claude API (structured outputs, default
  `claude-opus-4-8`). Use for real evaluations.
- **`heuristic`** — deterministic lexical-overlap judge. No network, no keys; used by
  the test suite and useful for validating pipeline wiring cheaply. Its absolute
  numbers are crude — treat them as plumbing checks, not quality measurements.

## Layout

```
evalharness/
  schema.py        # EvalCase, Prediction, MetricScore, CaseResult, RunResult
  datasets.py      # JSONL loading + dataset hashing
  judges/          # Judge interface, AnthropicJudge (LLM), HeuristicJudge (offline)
  metrics/         # llm_rubric, RAGAS metrics, hallucination_rate
  runner.py        # dataset + predictions + judge → RunResult
  registry.py      # .evalharness/runs/*.json + baseline pointer
  regression.py    # threshold + baseline comparison, gate verdict
  scorecard.py     # markdown + HTML scorecards
  cli.py           # run / baseline / compare / scorecard / history
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs fully offline using the heuristic judge.
