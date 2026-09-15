"""The runner hooks for the live view.

No test here runs a runner for real: the heavy pieces inside each ``run`` are
replaced with small fakes, and the tests assert what the runner hands to its
``LiveRun`` and in what order, and that the runner's own output files are
byte-identical with the live stream on and off.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import live
import rules

ET = rules.ET


def trades_df(n: int = 4) -> pd.DataFrame:
    t = pd.date_range("2020-01-02 10:00", periods=n, freq="1D", tz=ET)
    return pd.DataFrame({
        "entry_time": t, "exit_time": t + pd.Timedelta(hours=1),
        "direction": "long", "net_pnl": [10.0, -5.0, 20.0, -2.5][:n],
        "gross_pnl": [12.0, -3.0, 22.0, -0.5][:n], "commission": 1.0,
        "slippage_cost": 2.5, "duration_seconds": 3600.0,
        "session_date": [x.date() for x in t],
    })


def folds_df() -> pd.DataFrame:
    return pd.DataFrame({"test_year": [2020, 2021], "test_net_pnl": [5.0, -1.0]})


def events_of(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def kinds(path: Path) -> list[str]:
    return [e["event"] for e in events_of(path)]


def files_bytes(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


class Recorder:
    """Stands in for LiveRun at a CLI seam; records how it was constructed."""
    calls: list[tuple] = []

    def __init__(self, name, runner, directory=None):
        Recorder.calls.append((name, runner))
        self.inner = live.NullLive()

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _reset_recorder():
    Recorder.calls = []


# --- run_generated ----------------------------------------------------------


@pytest.fixture
def generated(monkeypatch, tmp_path):
    import run_generated

    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
        index=pd.date_range("2020-01-02 09:30", periods=5, freq="1D", tz=ET))
    parquet = tmp_path / "bars.parquet"
    parquet.write_bytes(b"")

    class Strategy:
        def generate_signals(self, frame):
            return pd.DataFrame({"entry_long": True}, index=frame.index)

    monkeypatch.setattr(run_generated, "parquet_for", lambda symbol: parquet)
    monkeypatch.setattr(run_generated.loader, "load_bars", lambda p: bars)
    monkeypatch.setattr(run_generated.loader, "detect_roll_dates", lambda b: set())
    monkeypatch.setattr(run_generated.loader, "detect_early_close_dates", lambda b: set())
    monkeypatch.setattr(run_generated, "build_strategy", lambda *a: Strategy())
    monkeypatch.setattr(run_generated, "resolve_contracts", lambda s, c: (1, "1 contract"))
    monkeypatch.setattr(run_generated, "guarded_streams",
                        lambda *a: (trades_df(), [], [], trades_df(3)))
    monkeypatch.setattr(run_generated, "fold_frame", lambda t, p: folds_df())
    monkeypatch.setattr(run_generated.eval_sim, "daily_pnl_from_trades",
                        lambda t: pd.Series([1.0]))
    monkeypatch.setattr(run_generated.eval_sim, "simulate",
                        lambda d, paths: SimpleNamespace(pass_probability=0.1,
                                                         payout_probability=0.05))
    monkeypatch.setattr(run_generated, "count_evaluation_blowups", lambda t: {"blowups": 0})
    monkeypatch.setattr(run_generated, "compute_metrics", lambda t: {
        "sharpe": 0.1, "profit_factor": 1.0, "max_drawdown": -5.0, "max_daily_loss": -5.0,
        "avg_duration_seconds": 3600.0, "microscalp_profit_pct": 0.0,
        "min_hold_violation_count": 0, "microscalp_trade_count": 0})
    monkeypatch.setattr(run_generated, "summarise_basis", lambda t, p: None)
    return run_generated


class TestRunGenerated:
    def _run(self, mod, monkeypatch, results: Path, live_run):
        results.mkdir()
        monkeypatch.setattr(mod, "RESULTS", results)
        return mod.run("demo", "m:C", progress=None, live=live_run)

    def test_hands_the_live_run_every_event_in_order(self, generated, monkeypatch, tmp_path):
        with live.LiveRun("demo", runner="test", directory=tmp_path / "live") as lr:
            result = self._run(generated, monkeypatch, tmp_path / "results", lr)
        ev = events_of(tmp_path / "live" / "demo.jsonl")
        ks = [e["event"] for e in ev]
        assert ks[0] == "start"
        assert ks.count("stage") >= 5
        assert [e["basis"] for e in ev if e["event"] == "trades"] == ["standard", "comparable"]
        assert ks.index("trades") < ks.index("folds") < ks.index("done")
        assert ev[-1]["status"] == ("ACCEPTED" if result.accepted else "REJECTED")

    def test_output_files_are_identical_with_live_on_and_off(self, generated, monkeypatch, tmp_path):
        self._run(generated, monkeypatch, tmp_path / "off", live.NullLive())
        with live.LiveRun("demo", runner="test", directory=tmp_path / "live") as lr:
            self._run(generated, monkeypatch, tmp_path / "on", lr)
        assert files_bytes(tmp_path / "off") == files_bytes(tmp_path / "on")
        assert set(files_bytes(tmp_path / "off")) == {
            "demo_folds.csv", "demo_trades.csv", "demo_folds_nohalt.csv", "demo_trades_nohalt.csv"}

    def test_default_live_is_null_and_progress_still_prints(self, generated, monkeypatch, tmp_path, capsys):
        seen = []
        results = tmp_path / "results"; results.mkdir()
        monkeypatch.setattr(generated, "RESULTS", results)
        generated.run("demo", "m:C", progress=seen.append)
        assert any("loading" in m for m in seen)

    def test_cli_live_flag_opens_a_live_run_named_after_the_strategy(self, generated, monkeypatch):
        captured = {}

        def fake_run(name, class_path, symbol, contracts, paths, progress=None,
                     commission=None, live=None):
            captured["live"] = live
            return SimpleNamespace()

        monkeypatch.setattr(generated, "run", fake_run)
        monkeypatch.setattr(generated, "verdict_block", lambda r, n: "")
        monkeypatch.setattr(generated, "LiveRun", Recorder)
        assert generated.main(["vwap_fade", "--class-path", "m:C", "--live"]) == 0
        assert Recorder.calls == [("vwap_fade", "run_generated vwap_fade")]
        assert isinstance(captured["live"], Recorder)

    def test_cli_without_the_flag_passes_a_null_live(self, generated, monkeypatch):
        captured = {}
        monkeypatch.setattr(generated, "run",
                            lambda *a, **k: captured.update(k) or SimpleNamespace())
        monkeypatch.setattr(generated, "verdict_block", lambda r, n: "")
        generated.main(["vwap_fade", "--class-path", "m:C"])
        assert isinstance(captured["live"], live.NullLive)
        assert not isinstance(captured["live"], live.LiveRun)


# --- run_entry11 and run_entry12 -----------------------------------------------


def diag_frame() -> pd.DataFrame:
    idx = [date(2020, 1, 2), date(2020, 2, 3)]
    return pd.DataFrame({"entry_bar_breach": [False, False], "entered": [True, False],
                         "label": ["T+1", "T-1"], "skipped_reason": [None, "early_close"]},
                        index=idx)


def _patch_entry(monkeypatch, mod, results: Path):
    results.mkdir(exist_ok=True)
    monkeypatch.setattr(mod, "RESULTS", results)
    monkeypatch.setattr(mod, "scored_returns", lambda *a: pd.DataFrame({"window": [True, False]}))
    monkeypatch.setattr(mod, "mechanism_test", lambda r: {})
    monkeypatch.setattr(mod, "generate", lambda *a: (pd.DataFrame(), diag_frame()))
    monkeypatch.setattr(mod, "guarded_streams", lambda *a: (trades_df(), trades_df(3), 0, 0))
    monkeypatch.setattr(mod, "stream_stats", lambda *a: {})
    monkeypatch.setattr(mod, "fold_frame", lambda t, p: folds_df())
    monkeypatch.setattr(mod, "verdict_status", lambda c: "REJECTED")
    monkeypatch.setattr(mod, "verdict_block", lambda r, d: "block")
    monkeypatch.setattr(mod, "round_turn_points", lambda *a: 0.7)


@pytest.fixture
def entry11(monkeypatch, tmp_path):
    import run_entry11
    _patch_entry(monkeypatch, run_entry11, tmp_path / "results")
    monkeypatch.setattr(run_entry11, "load", lambda *a: SimpleNamespace())
    monkeypatch.setattr(run_entry11, "evaluate_criteria", lambda *a: [])
    return run_entry11


@pytest.fixture
def entry12(monkeypatch, tmp_path):
    import run_entry12
    _patch_entry(monkeypatch, run_entry12, tmp_path / "results")
    monkeypatch.setattr(run_entry12, "load", lambda *a: SimpleNamespace())
    monkeypatch.setattr(run_entry12, "mes_control_sd_check", lambda: 40.99)
    monkeypatch.setattr(run_entry12, "control_sd", lambda r: 50.0)
    monkeypatch.setattr(run_entry12, "derive_stop_points", lambda sd, spec: 18.25)
    monkeypatch.setattr(run_entry12, "criteria_for", lambda *a: [])
    return run_entry12


class TestEntryRunners:
    def test_entry11_hands_over_the_stop_arm_base_streams_and_folds(self, entry11, tmp_path):
        with live.LiveRun("entry11", runner="t", directory=tmp_path / "live") as lr:
            _, status, _ = entry11.run(paths=10, progress=lambda m: None, live=lr)
        ev = events_of(tmp_path / "live" / "entry11.jsonl")
        assert [e["basis"] for e in ev if e["event"] == "trades"] == ["standard", "comparable"]
        assert [e["event"] for e in ev if e["event"] == "folds"] == ["folds"]
        assert ev[-1] == {"t": ev[-1]["t"], "event": "done", "status": status}
        assert any("pricing" in e.get("message", "") for e in ev if e["event"] == "stage")

    def test_entry11_files_identical_with_live_on_and_off(self, entry11, monkeypatch, tmp_path):
        monkeypatch.setattr(entry11, "RESULTS", tmp_path / "off"); (tmp_path / "off").mkdir()
        entry11.run(paths=10, progress=lambda m: None)
        monkeypatch.setattr(entry11, "RESULTS", tmp_path / "on"); (tmp_path / "on").mkdir()
        with live.LiveRun("entry11", runner="t", directory=tmp_path / "live") as lr:
            entry11.run(paths=10, progress=lambda m: None, live=lr)
        off, on = files_bytes(tmp_path / "off"), files_bytes(tmp_path / "on")
        # the verdict carries today's date in its text; everything else is exact
        assert set(off) == set(on)
        assert {k: v for k, v in off.items() if k.endswith(".csv")} == \
               {k: v for k, v in on.items() if k.endswith(".csv")}

    def test_entry11_cli_live_flag(self, entry11, monkeypatch):
        captured = {}
        monkeypatch.setattr(entry11, "run",
                            lambda *a, **k: captured.update(k) or ({}, "REJECTED", ""))
        monkeypatch.setattr(entry11, "LiveRun", Recorder)
        assert entry11.main(["--run", "--live"]) == 0
        assert Recorder.calls == [("entry11", "run_entry11 --run")]
        assert isinstance(captured["live"], Recorder)

    def test_entry12_run_hands_over_streams_folds_and_status(self, entry12, tmp_path):
        part = entry12.PARTS["A"]
        with live.LiveRun("entry12_a", runner="t", directory=tmp_path / "live") as lr:
            _, status, _ = entry12.run(part, None, paths=10, progress=lambda m: None, live=lr)
        ev = events_of(tmp_path / "live" / "entry12_a.jsonl")
        assert [e["basis"] for e in ev if e["event"] == "trades"] == ["standard", "comparable"]
        assert "folds" in [e["event"] for e in ev]
        assert ev[-1]["status"] == status

    def test_entry12_reproduce_reports_identical_and_finishes_reproduced(self, entry12, monkeypatch, tmp_path):
        monkeypatch.setattr(entry12, "reproduction_differences", lambda *a: [])
        with live.LiveRun("entry12_a", runner="t", directory=tmp_path / "live") as lr:
            assert entry12.reproduce(entry12.PARTS["A"], None, live=lr) is True
        ev = events_of(tmp_path / "live" / "entry12_a.jsonl")
        assert any("IDENTICAL" in e.get("message", "") for e in ev)
        assert ev[-1]["status"] == "REPRODUCED"

    def test_entry12_reproduce_with_differences_finishes_diverged(self, entry12, monkeypatch, tmp_path):
        monkeypatch.setattr(entry12, "reproduction_differences", lambda *a: ["x differs"])
        with live.LiveRun("entry12_a", runner="t", directory=tmp_path / "live") as lr:
            assert entry12.reproduce(entry12.PARTS["A"], None, live=lr) is False
        ev = events_of(tmp_path / "live" / "entry12_a.jsonl")
        assert any("DIVERGES" in e.get("message", "") for e in ev)
        assert ev[-1]["status"] == "DIVERGED"

    def test_entry12_cli_live_flag_names_the_part_and_mode(self, entry12, monkeypatch):
        monkeypatch.setattr(entry12, "reproduce", lambda *a, **k: True)
        monkeypatch.setattr(entry12, "LiveRun", Recorder)
        assert entry12.main(["--part", "A", "--reproduce", "--live"]) == 0
        assert Recorder.calls == [("entry12_a", "run_entry12 --part A --reproduce")]


# --- the bot's call path ---------------------------------------------------------


class TestBotRunner:
    @pytest.fixture
    def wired(self, monkeypatch, tmp_path):
        import run_generated
        import runners
        import submissions

        captured = {}
        record = SimpleNamespace(status="testing", class_path="m:C", hypothesis_entry=14)
        fake_registry = SimpleNamespace(names=lambda: ["vwap_fade"], get=lambda n: record)
        monkeypatch.setattr(submissions, "is_generated", lambda n: True)
        monkeypatch.setattr(runners, "registry", lambda: fake_registry)
        monkeypatch.setattr(run_generated, "run",
                            lambda *a, **k: captured.update(k) or SimpleNamespace())
        monkeypatch.setattr(live, "LIVE_DIR", tmp_path / "live")
        return runners, captured

    def test_env_var_off_passes_a_null_live(self, wired, monkeypatch):
        runners, captured = wired
        monkeypatch.delenv("FUTURES_LIVE", raising=False)
        runners.run_walkforward_generated("vwap_fade", progress=None)
        assert isinstance(captured["live"], live.NullLive)
        assert not isinstance(captured["live"], live.LiveRun)

    def test_env_var_on_opens_a_live_run_in_the_live_directory(self, wired, monkeypatch, tmp_path):
        runners, captured = wired
        monkeypatch.setenv("FUTURES_LIVE", "1")
        runners.run_walkforward_generated("vwap_fade", progress=None)
        assert isinstance(captured["live"], live.LiveRun)
        assert captured["live"].name == "vwap_fade"
        assert (tmp_path / "live" / "vwap_fade.jsonl").exists()
        assert "walkforward" in captured["live"].runner
