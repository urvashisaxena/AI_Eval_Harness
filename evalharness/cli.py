"""evalharness command-line interface.

    evalharness run       --dataset d.jsonl --predictions p.jsonl [--label v1]
    evalharness run       --dataset d.jsonl --generate --model claude-...   (calls the API)
    evalharness baseline  <run-id>
    evalharness compare   <run-id> [--against <run-id>]        (exit 1 on gate failure)
    evalharness scorecard <run-id> [-o scorecard.md] [--html scorecard.html]
    evalharness history
"""

from __future__ import annotations

import argparse
import sys

from .config import HarnessConfig
from .registry import RunRegistry
from .regression import compare
from .runner import generate_predictions, run_eval, run_from_files
from .scorecard import render_html, render_markdown


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--home", default=".evalharness", help="registry directory (default: .evalharness)")
    p.add_argument("--config", default=None, help="path to a harness config YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalharness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="score a dataset and store the run")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--predictions", help="JSONL of model answers")
    p_run.add_argument("--generate", action="store_true",
                       help="generate answers with --model via the Claude API instead of --predictions")
    p_run.add_argument("--model", default="", help="model under test (label, or model id with --generate)")
    p_run.add_argument("--label", default="", help="human-readable run label")
    p_run.add_argument("--judge", choices=["heuristic", "anthropic"], default=None,
                       help="override the judge from config")
    p_run.add_argument("--set-baseline", action="store_true", help="mark this run as the new baseline")
    p_run.add_argument("--gate", action="store_true",
                       help="compare against the baseline after scoring; exit 1 on failure")
    p_run.add_argument("-v", "--verbose", action="store_true")
    _add_common(p_run)

    p_base = sub.add_parser("baseline", help="set (or show) the baseline run")
    p_base.add_argument("run_id", nargs="?", help="run id to promote; omit to show current baseline")
    _add_common(p_base)

    p_cmp = sub.add_parser("compare", help="compare a run against the baseline (CI gate)")
    p_cmp.add_argument("run_id")
    p_cmp.add_argument("--against", default=None, help="compare against this run instead of the baseline")
    _add_common(p_cmp)

    p_card = sub.add_parser("scorecard", help="render a scorecard for a run")
    p_card.add_argument("run_id")
    p_card.add_argument("-o", "--output", default=None, help="write markdown here (default: stdout)")
    p_card.add_argument("--html", default=None, help="also write a self-contained HTML scorecard here")
    _add_common(p_card)

    p_hist = sub.add_parser("history", help="list stored runs")
    _add_common(p_hist)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = RunRegistry(args.home)
    config = HarnessConfig.load(args.config)

    if args.command == "run":
        if getattr(args, "judge", None):
            config.judge = args.judge
        if args.generate:
            if not args.model:
                print("error: --generate requires --model <model-id>", file=sys.stderr)
                return 2
            print(f"Generating answers with {args.model} …", file=sys.stderr)
            predictions = generate_predictions(args.dataset, args.model)
            run = run_eval(args.dataset, predictions, config,
                           model=args.model, label=args.label, verbose=args.verbose)
        elif args.predictions:
            run = run_from_files(args.dataset, args.predictions, config,
                                 model=args.model, label=args.label, verbose=args.verbose)
        else:
            print("error: provide --predictions or --generate", file=sys.stderr)
            return 2
        path = registry.save(run)
        print(f"Run {run.run_id} saved to {path}")
        for metric, value in run.metrics.items():
            print(f"  {metric:20s} {value:.4f}")
        if args.set_baseline:
            registry.set_baseline(run.run_id)
            print(f"Baseline set to {run.run_id}")
        if args.gate:
            baseline = registry.get_baseline()
            report = compare(run, baseline, config)
            _print_report(report)
            return 0 if report.passed else 1
        return 0

    if args.command == "baseline":
        if args.run_id:
            run = registry.set_baseline(args.run_id)
            print(f"Baseline set to {run.run_id} ({run.label})")
        else:
            baseline = registry.get_baseline()
            print(baseline.run_id if baseline else "no baseline set")
        return 0

    if args.command == "compare":
        run = registry.load(args.run_id)
        baseline = registry.load(args.against) if args.against else registry.get_baseline()
        if baseline is None:
            print("error: no baseline set (use `evalharness baseline <run-id>`)", file=sys.stderr)
            return 2
        report = compare(run, baseline, config)
        _print_report(report)
        return 0 if report.passed else 1

    if args.command == "scorecard":
        run = registry.load(args.run_id)
        baseline = registry.get_baseline()
        if baseline is not None and baseline.run_id == run.run_id:
            baseline = None  # a run compared with itself is noise
        report = compare(run, baseline, config) if baseline else None
        md = render_markdown(run, baseline, report, config)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(md)
            print(f"Wrote {args.output}")
        else:
            print(md)
        if args.html:
            with open(args.html, "w", encoding="utf-8") as fh:
                fh.write(render_html(run, baseline, report, config))
            print(f"Wrote {args.html}")
        return 0

    if args.command == "history":
        runs = registry.list_runs()
        if not runs:
            print("no runs recorded")
            return 0
        baseline = registry.get_baseline()
        for run in runs:
            marker = " *baseline*" if baseline and run.run_id == baseline.run_id else ""
            summary = "  ".join(f"{m}={v:.3f}" for m, v in run.metrics.items())
            print(f"{run.run_id}  [{run.label}] model={run.model}{marker}\n    {summary}")
        return 0

    return 2  # unreachable


def _print_report(report) -> None:
    against = report.baseline_id or "(no baseline)"
    print(f"\nComparison: {report.run_id} vs {against}")
    if not report.findings:
        print("  no differences against thresholds or baseline")
    for f in report.findings:
        prefix = "FAIL" if f.blocking else ("OK  " if f.kind == "improvement" else "WARN")
        print(f"  [{prefix}] {f.message}")
    print(f"\nGate: {'PASSED' if report.passed else 'FAILED'}")


if __name__ == "__main__":
    raise SystemExit(main())
