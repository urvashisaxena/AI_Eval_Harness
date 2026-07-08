"""Run registry: persisted runs + baseline pointer for regression tracking.

Runs are stored as JSON under ``.evalharness/runs/`` (committed or artifact-
uploaded as the team prefers), giving an auditable history of every scored
evaluation: model, judge, dataset hash, timestamp, aggregate metrics, and
per-case evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import RunResult

DEFAULT_HOME = Path(".evalharness")


class RunRegistry:
    def __init__(self, home: str | Path = DEFAULT_HOME):
        self.home = Path(home)
        self.runs_dir = self.home / "runs"
        self.baseline_file = self.home / "baseline.json"

    # -- runs ---------------------------------------------------------------

    def save(self, run: RunResult) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{run.run_id}.json"
        path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> RunResult:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            matches = sorted(self.runs_dir.glob(f"{run_id}*.json")) if self.runs_dir.exists() else []
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                ids = ", ".join(m.stem for m in matches)
                raise KeyError(f"run id prefix {run_id!r} is ambiguous: {ids}")
            else:
                raise KeyError(f"no run found with id {run_id!r}")
        return RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self) -> list[RunResult]:
        if not self.runs_dir.exists():
            return []
        runs = [
            RunResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(self.runs_dir.glob("*.json"))
        ]
        return sorted(runs, key=lambda r: r.created_at)

    # -- baseline -------------------------------------------------------------

    def set_baseline(self, run_id: str) -> RunResult:
        run = self.load(run_id)  # validates existence, resolves prefixes
        self.home.mkdir(parents=True, exist_ok=True)
        self.baseline_file.write_text(
            json.dumps({"run_id": run.run_id}), encoding="utf-8"
        )
        return run

    def get_baseline(self) -> RunResult | None:
        if not self.baseline_file.exists():
            return None
        run_id = json.loads(self.baseline_file.read_text(encoding="utf-8"))["run_id"]
        try:
            return self.load(run_id)
        except KeyError:
            return None
