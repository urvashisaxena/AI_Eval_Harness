from evalharness.judges import HeuristicJudge, make_judge
from evalharness.metrics import make_metrics, ALL_METRICS
from evalharness.schema import EvalCase, Prediction

JUDGE = HeuristicJudge()

CASE = EvalCase(
    id="c1",
    question="What is the refund window for annual subscriptions?",
    ground_truth="Annual subscriptions can be refunded in full within 30 days of purchase.",
    contexts=[
        "Billing policy: annual subscriptions may be refunded in full within 30 days of purchase."
    ],
)
GROUNDED = Prediction("c1", "Annual subscriptions may be refunded in full within 30 days of purchase.")
FABRICATED = Prediction("c1", "Refunds are handled by mailing a certified cheque to the CEO within one decade.")


def test_make_judge_factory():
    assert make_judge("heuristic").name == "heuristic"
    try:
        make_judge("nope")
        assert False
    except ValueError:
        pass


def test_claim_extraction_and_verification():
    claims = JUDGE.extract_claims("The sky is blue. Water boils at 100C.")
    assert len(claims) == 2
    verdicts = JUDGE.verify_claims(claims, ["The sky is blue on clear days."])
    assert verdicts[0].supported and not verdicts[1].supported


def test_faithfulness_grounded_beats_fabricated():
    (metric,) = make_metrics(["faithfulness"])
    good = metric.score_case(CASE, GROUNDED, JUDGE).value
    bad = metric.score_case(CASE, FABRICATED, JUDGE).value
    assert good > bad
    assert good == 1.0


def test_hallucination_rate_direction():
    (metric,) = make_metrics(["hallucination_rate"])
    good = metric.score_case(CASE, GROUNDED, JUDGE)
    bad = metric.score_case(CASE, FABRICATED, JUDGE)
    assert good.value < bad.value
    assert bad.details["case_has_hallucination"] is True
    assert good.details["case_has_hallucination"] is False


def test_rubric_and_relevancy_bounds():
    metrics = make_metrics(["llm_rubric", "answer_relevancy"])
    for metric in metrics:
        score = metric.score_case(CASE, GROUNDED, JUDGE)
        assert 0.0 <= score.value <= 1.0


def test_context_metrics():
    metrics = make_metrics(["context_precision", "context_recall"])
    for metric in metrics:
        score = metric.score_case(CASE, GROUNDED, JUDGE)
        assert score.value == 1.0


def test_applicability():
    no_ctx = EvalCase(id="x", question="q?", ground_truth="", contexts=[])
    assert not ALL_METRICS["faithfulness"]().applicable(no_ctx)
    assert not ALL_METRICS["hallucination_rate"]().applicable(no_ctx)
    assert ALL_METRICS["answer_relevancy"]().applicable(no_ctx)
