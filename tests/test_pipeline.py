"""End-to-end pipeline tests using the offline heuristic judge and the
bundled example dataset."""

from pathlib import Path

from evalharness.config import HarnessConfig, Threshold
from evalharness.registry import RunRegistry
from evalharness.regression import compare
from evalharness.runner import run_from_files
from evalharness.scorecard import render_html, render_markdown
from evalharness.cli import main as cli_main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DATASET = EXAMPLES / "dataset.jsonl"
PREDS_V1 = EXAMPLES / "predictions_v1.jsonl"
PREDS_V2 = EXAMPLES / "predictions_v2.jsonl"


def _config() -> HarnessConfig:
    cfg = HarnessConfig()  # heuristic judge, all metrics
    cfg.thresholds = {
        "faithfulness": Threshold(min=0.7, max_regression=0.05),
        "hallucination_rate": Threshold(max=0.25, max_regression=0.05),
    }
    return cfg


def test_run_and_regression_gate(tmp_path):
    cfg = _config()
    v1 = run_from_files(DATASET, PREDS_V1, cfg, label="v1")
    v2 = run_from_files(DATASET, PREDS_V2, cfg, label="v2")

    assert set(cfg.metrics) == set(v1.metrics)
    assert v1.metrics["faithfulness"] > v2.metrics["faithfulness"]
    assert v1.metrics["hallucination_rate"] < v2.metrics["hallucination_rate"]

    # v1 against itself as baseline: clean pass
    good = compare(v1, v1, cfg)
    assert good.passed

    # v2 against v1 baseline: hallucination regression must block
    report = compare(v2, v1, cfg)
    assert not report.passed
    blocking = {f.metric for f in report.findings if f.blocking}
    assert "hallucination_rate" in blocking
    assert "faithfulness" in blocking


def test_registry_roundtrip_and_baseline(tmp_path):
    cfg = _config()
    registry = RunRegistry(tmp_path / ".evalharness")
    run = run_from_files(DATASET, PREDS_V1, cfg, label="v1")
    registry.save(run)

    loaded = registry.load(run.run_id)
    assert loaded.metrics == run.metrics
    assert loaded.cases[0].scores.keys() == run.cases[0].scores.keys()

    registry.set_baseline(run.run_id)
    assert registry.get_baseline().run_id == run.run_id
    assert len(registry.list_runs()) == 1


def test_scorecard_rendering(tmp_path):
    cfg = _config()
    v1 = run_from_files(DATASET, PREDS_V1, cfg, label="v1")
    v2 = run_from_files(DATASET, PREDS_V2, cfg, label="v2")
    report = compare(v2, v1, cfg)

    md = render_markdown(v2, v1, report, cfg)
    assert "Model Scorecard" in md
    assert "hallucination_rate" in md
    assert "FAIL" in md  # gate failure is visible

    html_doc = render_html(v2, v1, report, cfg)
    assert html_doc.startswith("<!doctype html>")
    assert "<table>" in html_doc


def test_cli_end_to_end(tmp_path, capsys):
    home = str(tmp_path / ".evalharness")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "judge: heuristic\n"
        "thresholds:\n"
        "  hallucination_rate: {max: 0.25, max_regression: 0.05}\n"
        "  faithfulness: {min: 0.7, max_regression: 0.05}\n"
    )

    rc = cli_main([
        "run", "--dataset", str(DATASET), "--predictions", str(PREDS_V1),
        "--label", "v1", "--home", home, "--config", str(config_path),
        "--set-baseline",
    ])
    assert rc == 0
    run_id_v1 = capsys.readouterr().out.splitlines()[0].split()[1]

    # v2 run with gating: should fail the gate (exit 1)
    rc = cli_main([
        "run", "--dataset", str(DATASET), "--predictions", str(PREDS_V2),
        "--label", "v2", "--home", home, "--config", str(config_path), "--gate",
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAILED" in out

    # scorecard for the baseline run renders
    rc = cli_main(["scorecard", run_id_v1, "--home", home,
                   "-o", str(tmp_path / "card.md"), "--html", str(tmp_path / "card.html")])
    assert rc == 0
    assert (tmp_path / "card.md").exists()
    assert (tmp_path / "card.html").exists()

    rc = cli_main(["history", "--home", home])
    assert rc == 0
    assert "*baseline*" in capsys.readouterr().out
