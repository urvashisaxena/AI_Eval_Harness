"""Regression detection and governance gating.

Compares a run against absolute thresholds and against a baseline run, and
produces a list of findings. Any finding with ``blocking=True`` should fail
the gate (CI exits non-zero), which is how the harness enforces "no silent
quality regressions ship".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import HarnessConfig, Threshold
from .metrics.base import METRIC_INFO
from .schema import RunResult


@dataclass
class Finding:
    metric: str
    kind: str  # "below_min" | "above_max" | "regression" | "improvement" | "info"
    message: str
    blocking: bool = False


@dataclass
class ComparisonReport:
    run_id: str
    baseline_id: str | None
    findings: list[Finding] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)  # current - baseline

    @property
    def passed(self) -> bool:
        return not any(f.blocking for f in self.findings)


def _higher_is_better(metric: str) -> bool:
    info = METRIC_INFO.get(metric)
    return info.higher_is_better if info else True


def compare(
    run: RunResult,
    baseline: RunResult | None,
    config: HarnessConfig,
) -> ComparisonReport:
    report = ComparisonReport(
        run_id=run.run_id, baseline_id=baseline.run_id if baseline else None
    )

    if baseline is not None and baseline.dataset_sha256 != run.dataset_sha256:
        report.findings.append(
            Finding(
                metric="*",
                kind="info",
                message=(
                    "baseline was scored on a different dataset revision "
                    f"({baseline.dataset_sha256[:12]}… vs {run.dataset_sha256[:12]}…); "
                    "deltas may not be comparable"
                ),
            )
        )

    for metric, value in run.metrics.items():
        hib = _higher_is_better(metric)
        threshold: Threshold | None = config.thresholds.get(metric)

        # --- absolute gates -------------------------------------------------
        if threshold:
            if threshold.min is not None and value < threshold.min:
                report.findings.append(
                    Finding(
                        metric=metric,
                        kind="below_min",
                        message=f"{metric} = {value:.4f} is below the required minimum {threshold.min}",
                        blocking=True,
                    )
                )
            if threshold.max is not None and value > threshold.max:
                report.findings.append(
                    Finding(
                        metric=metric,
                        kind="above_max",
                        message=f"{metric} = {value:.4f} exceeds the allowed maximum {threshold.max}",
                        blocking=True,
                    )
                )

        # --- baseline comparison ----------------------------------------------
        if baseline is None or metric not in baseline.metrics:
            continue
        base_value = baseline.metrics[metric]
        delta = round(value - base_value, 4)
        report.deltas[metric] = delta

        # "worse" is a drop for higher-is-better metrics, a rise otherwise
        worsening = -delta if hib else delta
        tolerance = threshold.max_regression if threshold and threshold.max_regression is not None else None

        if tolerance is not None and worsening > tolerance:
            direction = "dropped" if hib else "rose"
            report.findings.append(
                Finding(
                    metric=metric,
                    kind="regression",
                    message=(
                        f"{metric} {direction} {abs(delta):.4f} vs baseline "
                        f"({base_value:.4f} → {value:.4f}), tolerance is {tolerance}"
                    ),
                    blocking=True,
                )
            )
        elif worsening > 0:
            report.findings.append(
                Finding(
                    metric=metric,
                    kind="regression",
                    message=(
                        f"{metric} moved worse by {abs(delta):.4f} vs baseline "
                        f"({base_value:.4f} → {value:.4f}) — within tolerance"
                    ),
                    blocking=False,
                )
            )
        elif worsening < 0:
            report.findings.append(
                Finding(
                    metric=metric,
                    kind="improvement",
                    message=(
                        f"{metric} improved by {abs(delta):.4f} vs baseline "
                        f"({base_value:.4f} → {value:.4f})"
                    ),
                )
            )

    return report
