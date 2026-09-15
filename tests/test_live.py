"""Tests for backtests/live.py — the live event stream a runner writes and the
readers the viewer uses.

Everything here runs against a temporary directory. The module never touches
the runners' own result files, and nothing in it can start a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import live


def trades_frame(n: int = 4) -> pd.DataFrame:
    times = pd.date_range("2020-01-02 10:00", periods=n, freq="1D", tz="America/New_York")
    return pd.DataFrame({
        "entry_time": times,
        "exit_time": times + pd.Timedelta(hours=1),
        "net_pnl": [10.0, -5.0, 20.0, -2.5][:n],
        "session_date": [t.date() for t in times],
    })


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- LiveRun -----------------------------------------------------------------


class TestLiveRun:
    def test_opening_writes_start_with_runner_and_pid(self, tmp_path):
        run = live.LiveRun("demo", runner="run_entry12 --reproduce", directory=tmp_path)
        events = read_lines(tmp_path / "demo.jsonl")
        assert [e["event"] for e in events] == ["start"]
        assert events[0]["run"] == "demo"
        assert events[0]["runner"] == "run_entry12 --reproduce"
        assert isinstance(events[0]["pid"], int)
        assert "t" in events[0]
        run.close()

    def test_progress_calls_the_fallback_and_writes_a_stage(self, tmp_path):
        seen: list[str] = []
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            progress = run.progress(seen.append)
            progress("[progress] loading bars")
        assert seen == ["[progress] loading bars"]
        events = read_lines(tmp_path / "demo.jsonl")
        stage = [e for e in events if e["event"] == "stage"]
        assert stage == [{"t": stage[0]["t"], "event": "stage", "message": "[progress] loading bars"}]

    def test_trades_writes_a_copy_and_an_event(self, tmp_path):
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            run.trades(trades_frame(), basis="standard")
            run.trades(trades_frame(2), basis="comparable")
        events = read_lines(tmp_path / "demo.jsonl")
        trade_events = [e for e in events if e["event"] == "trades"]
        assert [(e["basis"], e["rows"], e["path"]) for e in trade_events] == [
            ("standard", 4, "demo_standard_trades.csv"),
            ("comparable", 2, "demo_comparable_trades.csv"),
        ]
        assert len(pd.read_csv(tmp_path / "demo_standard_trades.csv")) == 4

    def test_folds_writes_a_copy_and_an_event(self, tmp_path):
        folds = pd.DataFrame({"test_year": [2020, 2021], "test_net_pnl": [1.0, -1.0]})
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            run.folds(folds)
        events = read_lines(tmp_path / "demo.jsonl")
        assert [e for e in events if e["event"] == "folds"][0]["rows"] == 2
        assert (tmp_path / "demo_folds.csv").exists()

    def test_exit_writes_done_with_the_finished_status(self, tmp_path):
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            run.finish("REJECTED")
        events = read_lines(tmp_path / "demo.jsonl")
        assert events[-1]["event"] == "done"
        assert events[-1]["status"] == "REJECTED"

    def test_exit_without_finish_still_writes_done(self, tmp_path):
        with live.LiveRun("demo", runner="x", directory=tmp_path):
            pass
        events = read_lines(tmp_path / "demo.jsonl")
        assert events[-1] == {"t": events[-1]["t"], "event": "done", "status": "finished"}

    def test_an_exception_writes_error_and_propagates(self, tmp_path):
        with pytest.raises(ValueError, match="boom"):
            with live.LiveRun("demo", runner="x", directory=tmp_path):
                raise ValueError("boom")
        events = read_lines(tmp_path / "demo.jsonl")
        assert events[-1]["event"] == "error"
        assert "boom" in events[-1]["message"]

    def test_a_new_run_with_the_same_name_starts_a_fresh_file(self, tmp_path):
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            run.progress(lambda _: None)("first")
        with live.LiveRun("demo", runner="y", directory=tmp_path):
            pass
        events = read_lines(tmp_path / "demo.jsonl")
        assert [e["event"] for e in events] == ["start", "done"]
        assert events[0]["runner"] == "y"

    @pytest.mark.parametrize("bad", ["../x", "Demo", "a b", "", "x" * 65, "demo.jsonl"])
    def test_names_are_validated(self, tmp_path, bad):
        with pytest.raises(ValueError):
            live.LiveRun(bad, runner="x", directory=tmp_path)
        assert list(tmp_path.iterdir()) == []


# --- NullLive ----------------------------------------------------------------


class TestNullLive:
    def test_accepts_every_call_and_writes_nothing(self, tmp_path):
        seen: list[str] = []
        with live.NullLive() as run:
            run.progress(seen.append)("hello")
            run.trades(trades_frame(), basis="standard")
            run.folds(pd.DataFrame({"a": [1]}))
            run.finish("ACCEPTED")
        assert seen == ["hello"]
        assert list(tmp_path.iterdir()) == []

    def test_progress_with_no_fallback_is_silent(self, capsys):
        live.NullLive().progress(None)("quiet")
        assert capsys.readouterr().out == ""


# --- readers -------------------------------------------------------------------


class TestReaders:
    def _write_run(self, tmp_path: Path, name: str = "demo", status: str | None = "ACCEPTED"):
        with live.LiveRun(name, runner="run_generated", directory=tmp_path) as run:
            p = run.progress(lambda _: None)
            p("stage one")
            p("stage two")
            run.trades(trades_frame(), basis="standard")
            run.folds(pd.DataFrame({"test_year": [2020], "test_net_pnl": [1.0]}))
            if status:
                run.finish(status)

    def test_list_runs_reports_name_runner_start_and_status(self, tmp_path):
        self._write_run(tmp_path)
        self._write_run(tmp_path, name="other", status=None)
        runs = {r["name"]: r for r in live.list_runs(tmp_path)}
        assert runs["demo"]["runner"] == "run_generated"
        assert runs["demo"]["status"] == "ACCEPTED"
        assert runs["other"]["status"] == "finished"
        assert "started" in runs["demo"]

    def test_a_run_still_going_has_status_running(self, tmp_path):
        run = live.LiveRun("going", runner="x", directory=tmp_path)
        run.progress(lambda _: None)("stage")
        assert live.list_runs(tmp_path)[0]["status"] == "running"
        run.close()

    def test_read_events_paginates_by_cursor(self, tmp_path):
        self._write_run(tmp_path)
        first = live.read_events("demo", after=0, directory=tmp_path)
        assert first["events"][0]["event"] == "start"
        assert first["next"] == len(first["events"])
        rest = live.read_events("demo", after=first["next"], directory=tmp_path)
        assert rest["events"] == []
        assert rest["next"] == first["next"]
        partial = live.read_events("demo", after=2, directory=tmp_path)
        assert partial["events"] == first["events"][2:]

    def test_read_events_skips_a_partial_last_line(self, tmp_path):
        self._write_run(tmp_path)
        path = tmp_path / "demo.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"t": "2026-01-01T00:00:00", "event": "stage", "mess')
        out = live.read_events("demo", after=0, directory=tmp_path)
        assert all(e["event"] != "stage" or "message" in e for e in out["events"])
        assert out["events"][-1]["event"] == "done"

    def test_read_trades_returns_cumulative_pnl_in_exit_order(self, tmp_path):
        self._write_run(tmp_path)
        points = live.read_trades("demo", basis="standard", directory=tmp_path)
        assert [p["i"] for p in points] == [1, 2, 3, 4]
        assert [p["cum_pnl"] for p in points] == [10.0, 5.0, 25.0, 22.5]
        assert points[0]["exit_time"] < points[-1]["exit_time"]

    def test_read_trades_for_a_missing_basis_is_empty(self, tmp_path):
        self._write_run(tmp_path)
        assert live.read_trades("demo", basis="comparable", directory=tmp_path) == []

    def test_read_folds_returns_rows(self, tmp_path):
        self._write_run(tmp_path)
        rows = live.read_folds("demo", directory=tmp_path)
        assert rows == [{"test_year": 2020, "test_net_pnl": 1.0}]

    def test_read_folds_turns_nan_into_none_so_json_is_valid(self, tmp_path):
        """A fold year the halt emptied has NaN Sharpe; JSON has no NaN."""
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            run.folds(pd.DataFrame({"test_year": [2022, 2023], "test_sharpe": [float("nan"), 1.5],
                                    "low_confidence": [True, False]}))
        rows = live.read_folds("demo", directory=tmp_path)
        assert rows[0]["test_sharpe"] is None
        assert rows[1]["test_sharpe"] == 1.5
        assert rows[0]["low_confidence"] is True
        json.dumps(rows, allow_nan=False)

    def test_read_trades_turns_nan_into_none(self, tmp_path):
        frame = trades_frame(2)
        frame.loc[frame.index[1], "net_pnl"] = float("nan")
        with live.LiveRun("demo", runner="x", directory=tmp_path) as run:
            run.trades(frame, basis="standard")
        pts = live.read_trades("demo", basis="standard", directory=tmp_path)
        assert pts[1]["net_pnl"] is None
        json.dumps(pts, allow_nan=False)

    def test_read_events_reports_when_the_run_started(self, tmp_path):
        """The page resets when a new run replaces the file; ``started`` is how it knows."""
        self._write_run(tmp_path)
        out = live.read_events("demo", after=3, directory=tmp_path)
        assert out["started"] == live.read_events("demo", after=0, directory=tmp_path)["events"][0]["t"]
        assert out["run"] == "demo"
        assert live.read_events("nothing", after=0, directory=tmp_path) == {
            "events": [], "next": 0, "started": None, "run": "nothing"}

    def test_validate_name_is_public(self):
        assert live.validate_name("entry12_a") == "entry12_a"
        with pytest.raises(ValueError):
            live.validate_name("../x")

    @pytest.mark.parametrize("bad", ["../demo", "DEMO", "demo/../x", "demo\\x"])
    def test_readers_refuse_bad_names(self, tmp_path, bad):
        self._write_run(tmp_path)
        with pytest.raises(ValueError):
            live.read_events(bad, after=0, directory=tmp_path)
        with pytest.raises(ValueError):
            live.read_trades(bad, basis="standard", directory=tmp_path)
        with pytest.raises(ValueError):
            live.read_folds(bad, directory=tmp_path)

    def test_unknown_run_reads_as_empty_not_error(self, tmp_path):
        out = live.read_events("nothing", after=0, directory=tmp_path)
        assert (out["events"], out["next"]) == ([], 0)
        assert live.read_trades("nothing", basis="standard", directory=tmp_path) == []
        assert live.read_folds("nothing", directory=tmp_path) == []

    def test_default_directory_is_under_results(self):
        assert live.LIVE_DIR.parent.name == "results"
        assert live.LIVE_DIR.name == "live"
