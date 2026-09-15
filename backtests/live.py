"""The live event stream a runner writes while it works, and its readers.

A runner that wants to be watched opens a :class:`LiveRun` and hands it the
things it already produces: progress messages, the priced trade streams, the
fold table, the outcome. Each becomes one JSON line in
``backtests/results/live/<name>.jsonl`` (gitignored output, not evidence),
with the trade and fold frames copied to CSVs beside it. The runner's own
result files are untouched; its console output is unchanged, because
:meth:`LiveRun.progress` wraps the callback it already prints through.

:class:`NullLive` has the same interface and writes nothing, so runner code
carries no ``if live:`` branches.

The readers at the bottom are what the viewer (``bots/liveview.py``) calls.
They validate the run name and resolve it only inside the live directory, so
nothing outside it can be read through them. Nothing in this module can
start, stop or parameterise a run.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = PROJECT_ROOT / "backtests" / "results" / "live"

NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")

Progress = Callable[[str], None]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _validate(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(
            f"live run name {name!r} must match {NAME_RE.pattern}: lowercase "
            f"letters, digits and underscores, no path separators"
        )
    return name


def validate_name(name: str) -> str:
    """The viewer's check: a bad name is a 404, not a file lookup."""
    return _validate(name)


def _plain(value):
    """A JSON-safe scalar: numpy types unwrapped, NaN and infinities to None."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _resolve(name: str, directory: Path, suffix: str) -> Path:
    """A path inside ``directory`` for a validated name, and nowhere else."""
    directory = Path(directory).resolve()
    path = (directory / f"{_validate(name)}{suffix}").resolve()
    if path.parent != directory:
        raise ValueError(f"{name!r} does not resolve inside {directory}")
    return path


class NullLive:
    """The do-nothing stand-in used when ``--live`` is off."""

    def progress(self, fallback: Progress | None = print) -> Progress:
        def emit(message: str) -> None:
            if fallback is not None:
                fallback(message)
        return emit

    def trades(self, frame: pd.DataFrame, basis: str = "standard") -> None:
        return None

    def folds(self, frame: pd.DataFrame) -> None:
        return None

    def finish(self, status: str) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullLive":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class LiveRun(NullLive):
    """Append-only event stream for one run, plus CSV copies of its frames."""

    def __init__(self, name: str, runner: str, directory: Path | None = None) -> None:
        self.name = _validate(name)
        # Resolved at call time, not import time, so a test can redirect it.
        self.directory = Path(directory) if directory is not None else LIVE_DIR
        self.runner = runner
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = _resolve(self.name, self.directory, ".jsonl")
        self._status: str | None = None
        self._closed = False
        # A fresh file per run: a re-run overwrites rather than appends
        # across runs, so a reader never sees two runs interleaved.
        self.path.write_text("", encoding="utf-8")
        self.event("start", run=self.name, runner=runner, pid=os.getpid())

    # -- writing ---------------------------------------------------------------

    def event(self, kind: str, **fields) -> None:
        record = {"t": _now(), "event": kind, **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def progress(self, fallback: Progress | None = print) -> Progress:
        def emit(message: str) -> None:
            if fallback is not None:
                fallback(message)
            self.event("stage", message=str(message))
        return emit

    def _copy(self, frame: pd.DataFrame, stem: str) -> Path:
        path = _resolve(f"{self.name}_{stem}", self.directory, ".csv")
        frame.to_csv(path, index=False)
        return path

    def trades(self, frame: pd.DataFrame, basis: str = "standard") -> None:
        basis = _validate(basis)
        path = self._copy(frame, f"{basis}_trades")
        self.event("trades", path=path.name, basis=basis, rows=int(len(frame)))

    def folds(self, frame: pd.DataFrame) -> None:
        path = self._copy(frame, "folds")
        self.event("folds", path=path.name, rows=int(len(frame)))

    def finish(self, status: str) -> None:
        self._status = str(status)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.event("done", status=self._status or "finished")

    def __enter__(self) -> "LiveRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and not self._closed:
            self._closed = True
            self.event("error", message=f"{exc_type.__name__}: {exc}")
        else:
            self.close()
        return False


# -- reading --------------------------------------------------------------------


def _events(path: Path) -> list[dict]:
    """Every complete line. A partial trailing line, mid-write, is skipped."""
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_events(name: str, after: int = 0, directory: Path = LIVE_DIR) -> dict:
    """Events with index >= ``after``, the cursor to pass next time, and the
    run's start time so a reader can tell a fresh run has replaced the file."""
    events = _events(_resolve(name, directory, ".jsonl"))
    after = max(0, int(after))
    started = events[0].get("t") if events and events[0].get("event") == "start" else None
    return {"events": events[after:], "next": len(events), "started": started,
            "run": _validate(name)}


def list_runs(directory: Path = LIVE_DIR) -> list[dict]:
    directory = Path(directory)
    if not directory.exists():
        return []
    runs: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        if not NAME_RE.match(path.stem):
            continue
        events = _events(path)
        if not events or events[0].get("event") != "start":
            continue
        last = events[-1]
        if last.get("event") == "done":
            status = last.get("status", "finished")
        elif last.get("event") == "error":
            status = "error"
        else:
            status = "running"
        runs.append({
            "name": path.stem,
            "runner": events[0].get("runner", ""),
            "started": events[0].get("t", ""),
            "status": status,
            "events": len(events),
        })
    runs.sort(key=lambda r: r["started"], reverse=True)
    return runs


def read_trades(name: str, basis: str = "standard", directory: Path = LIVE_DIR) -> list[dict]:
    """Cumulative net P&L in exit order, one point per trade."""
    path = _resolve(f"{_validate(name)}_{_validate(basis)}", directory, "_trades.csv")
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    if "exit_time" in frame.columns:
        frame = frame.sort_values("exit_time", kind="stable")
    pnl = frame["net_pnl"].astype(float)
    cum = pnl.fillna(0.0).cumsum()
    return [
        {
            "i": i,
            "exit_time": str(row.get("exit_time", "")),
            "net_pnl": _plain(p),
            "cum_pnl": _plain(c),
        }
        for i, ((_, row), p, c) in enumerate(zip(frame.iterrows(), pnl, cum), start=1)
    ]


def read_folds(name: str, directory: Path = LIVE_DIR) -> list[dict]:
    path = _resolve(f"{_validate(name)}_folds", directory, ".csv")
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [{k: _plain(v) for k, v in r.items()} for r in frame.to_dict(orient="records")]
