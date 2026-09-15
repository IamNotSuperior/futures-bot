"""bots/liveview.py — the read-only page that follows a live run.

Runs through FastAPI's test client against a temporary live directory. The
viewer must be unable to start, stop or re-run anything: every route is GET,
and the module never imports a runner or the subprocess module.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

import live


def _write_run(directory: Path, name: str = "demo") -> None:
    t = pd.date_range("2020-01-02 10:00", periods=3, freq="1D", tz="America/New_York")
    trades = pd.DataFrame({"exit_time": t + pd.Timedelta(hours=1), "net_pnl": [10.0, -5.0, 20.0]})
    with live.LiveRun(name, runner="run_entry12 --part A --reproduce", directory=directory) as run:
        say = run.progress(lambda _: None)
        say("[progress] loading bars")
        say("[progress] pricing")
        run.trades(trades, basis="standard")
        run.folds(pd.DataFrame({"test_year": [2020, 2021], "test_net_pnl": [5.0, -1.0]}))
        run.finish("REPRODUCED")


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    import liveview

    _write_run(tmp_path)
    return TestClient(liveview.build_app(tmp_path))


class TestRoutes:
    def test_health(self, client):
        assert client.get("/health").json() == {"ok": True}

    def test_page_is_html_and_says_it_is_read_only(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        body = r.text.lower()
        assert "read-only" in body
        assert "<form" not in body
        assert "replay of the finished stream" in body

    def test_runs_lists_the_run_with_status(self, client):
        runs = client.get("/runs").json()
        assert runs == [{"name": "demo", "runner": "run_entry12 --part A --reproduce",
                         "started": runs[0]["started"], "status": "REPRODUCED",
                         "events": runs[0]["events"]}]

    def test_events_paginate_by_cursor(self, client):
        first = client.get("/runs/demo/events", params={"after": 0}).json()
        assert first["events"][0]["event"] == "start"
        assert first["events"][-1]["event"] == "done"
        rest = client.get("/runs/demo/events", params={"after": first["next"]}).json()
        assert (rest["events"], rest["next"]) == ([], first["next"])

    def test_trades_are_cumulative(self, client):
        pts = client.get("/runs/demo/trades", params={"basis": "standard"}).json()
        assert [p["cum_pnl"] for p in pts] == [10.0, 5.0, 25.0]
        assert client.get("/runs/demo/trades", params={"basis": "comparable"}).json() == []

    def test_folds(self, client):
        assert client.get("/runs/demo/folds").json() == [
            {"test_year": 2020, "test_net_pnl": 5.0}, {"test_year": 2021, "test_net_pnl": -1.0}]

    def test_folds_with_nan_are_served_as_null_not_404(self, tmp_path):
        """The real entry 12 table has NaN Sharpe in halted years; it must render."""
        from fastapi.testclient import TestClient

        import liveview

        with live.LiveRun("nan_run", runner="x", directory=tmp_path) as run:
            run.folds(pd.DataFrame({"test_year": [2022], "test_sharpe": [float("nan")]}))
        r = TestClient(liveview.build_app(tmp_path)).get("/runs/nan_run/folds")
        assert r.status_code == 200
        assert r.json() == [{"test_year": 2022, "test_sharpe": None}]

    def test_events_carry_the_start_marker(self, client):
        out = client.get("/runs/demo/events", params={"after": 2}).json()
        assert out["run"] == "demo"
        assert out["started"] == client.get("/runs/demo/events").json()["events"][0]["t"]

    def test_unknown_run_is_empty_not_an_error(self, client):
        out = client.get("/runs/nothing/events").json()
        assert (out["events"], out["next"]) == ([], 0)
        assert client.get("/runs/nothing/trades").json() == []
        assert client.get("/runs/nothing/folds").json() == []

    @pytest.mark.parametrize("bad", ["DEMO", "..", "demo.jsonl", "a%20b", "x" * 70])
    def test_bad_names_are_404(self, client, bad):
        for route in ("events", "trades", "folds"):
            assert client.get(f"/runs/{bad}/{route}").status_code == 404

    def test_bad_basis_is_404(self, client):
        assert client.get("/runs/demo/trades", params={"basis": "../x"}).status_code == 404


class TestReadOnly:
    def test_every_route_is_get_only(self, tmp_path):
        import liveview

        app = liveview.build_app(tmp_path)
        for route in app.routes:
            methods = getattr(route, "methods", None)
            if methods:
                assert set(methods) <= {"GET", "HEAD"}, (route.path, methods)

    def test_post_is_rejected_everywhere(self, client):
        for path in ("/", "/runs", "/runs/demo/events", "/health"):
            assert client.post(path).status_code == 405

    def test_module_cannot_start_a_process_or_reach_a_runner(self):
        import liveview

        source = inspect.getsource(liveview)
        assert "subprocess" not in source
        assert "import run_" not in source
        assert "os.system" not in source

    def test_directory_defaults_to_the_live_directory(self):
        import liveview

        assert liveview.build_app().state.directory == live.LIVE_DIR

    def test_default_port_does_not_collide_with_the_desk(self):
        import liveview

        assert liveview.DEFAULT_PORT == 8790

    def test_module_never_calls_a_runner_function(self):
        """It may read runners' table of saved files; it may not run anything."""
        import liveview

        source = inspect.getsource(liveview)
        for forbidden in ("runners.run", "run_walkforward", "run_backtest", "run_evalsim"):
            assert forbidden not in source, forbidden


# --- saved verdicts ------------------------------------------------------------


def _stream(n: int, start: float = 10.0) -> pd.DataFrame:
    t = pd.date_range("2020-01-02 10:00", periods=n, freq="1D", tz="America/New_York")
    return pd.DataFrame({"exit_time": t + pd.Timedelta(hours=1),
                         "net_pnl": [start, -5.0, 20.0, -2.5, 7.0][:n]})


class FakeRegistry:
    def __init__(self, records):
        self._records = records

    def names(self):
        return list(self._records)

    def get(self, name):
        return self._records[name]


@pytest.fixture
def saved(tmp_path):
    """A synthetic results directory, runner table and registry."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import liveview
    from runners import Runner

    results = tmp_path / "results"
    results.mkdir()
    _stream(4).to_csv(results / "tom_x_stop_slip1.csv", index=False)
    _stream(5).to_csv(results / "tom_x_stop_slip1_nohalt.csv", index=False)
    pd.DataFrame({"test_year": [2020, 2021], "test_net_pnl": [5.0, float("nan")]}).to_csv(
        results / "tom_x_folds.csv", index=False)
    _stream(3).to_csv(results / "gen_x_trades.csv", index=False)
    pd.DataFrame({"test_year": [2020], "test_net_pnl": [1.0]}).to_csv(
        results / "gen_x_folds.csv", index=False)
    _stream(2).to_csv(results / "nofolds_stream.csv", index=False)

    table = {
        "tom_x": Runner(1, 4, None, "tom_x_folds.csv", "tom_x_stop_slip1.csv",
                        note="entry 11 stop arm, 1 tick, halt ON"),
        "nofolds": Runner(1, 4, None, None, "nofolds_stream.csv", note="no fold table"),
        "absent": Runner(1, 1, None, "absent_folds.csv", "absent_stream.csv", note="regenerable"),
    }
    rec = lambda n, e, s: SimpleNamespace(name=n, hypothesis_entry=e, status=s,  # noqa: E731
                                          verdict_commit="abc1234", class_path=None)
    registry = FakeRegistry({
        "tom_x": rec("tom_x", 11, "rejected"),
        "gen_x": rec("gen_x", 9, "rejected"),
        "nofolds": rec("nofolds", 5, "rejected"),
        "absent": rec("absent", 2, "rejected"),
        "manual": rec("manual", 3, "proposed"),
    })
    app = liveview.build_app(tmp_path / "live", results=results, table=table,
                             registry_loader=lambda: registry)
    return TestClient(app)


class TestSaved:
    def test_lists_every_registry_name_with_status_and_files(self, saved):
        rows = {r["name"]: r for r in saved.get("/saved").json()}
        assert rows["tom_x"]["entry"] == 11
        assert rows["tom_x"]["status"] == "rejected"
        assert rows["tom_x"]["note"] == "entry 11 stop arm, 1 tick, halt ON"
        assert rows["tom_x"]["trades_file"] == "tom_x_stop_slip1.csv"
        assert rows["tom_x"]["folds_file"] == "tom_x_folds.csv"
        assert rows["tom_x"]["has_comparable"] is True
        assert rows["tom_x"]["available"] is True
        assert rows["tom_x"]["produced"]

    def test_generated_strategies_follow_the_fixed_naming(self, saved):
        rows = {r["name"]: r for r in saved.get("/saved").json()}
        assert rows["gen_x"]["trades_file"] == "gen_x_trades.csv"
        assert rows["gen_x"]["folds_file"] == "gen_x_folds.csv"
        assert rows["gen_x"]["has_comparable"] is False
        assert rows["gen_x"]["available"] is True

    def test_missing_files_are_listed_as_unavailable_not_hidden(self, saved):
        rows = {r["name"]: r for r in saved.get("/saved").json()}
        assert rows["absent"]["available"] is False
        assert rows["absent"]["produced"] is None
        assert rows["manual"]["available"] is False
        assert rows["manual"]["trades_file"] is None

    def test_trades_are_cumulative_and_the_comparable_sibling_is_served(self, saved):
        std = saved.get("/saved/tom_x/trades").json()
        assert [p["cum_pnl"] for p in std] == [10.0, 5.0, 25.0, 22.5]
        cmp_ = saved.get("/saved/tom_x/trades", params={"basis": "comparable"}).json()
        assert len(cmp_) == 5
        assert saved.get("/saved/gen_x/trades", params={"basis": "comparable"}).json() == []

    def test_folds_with_nan_render_as_null(self, saved):
        assert saved.get("/saved/tom_x/folds").json() == [
            {"test_year": 2020, "test_net_pnl": 5.0}, {"test_year": 2021, "test_net_pnl": None}]

    def test_no_fold_table_is_empty_not_an_error(self, saved):
        assert saved.get("/saved/nofolds/folds").json() == []
        assert len(saved.get("/saved/nofolds/trades").json()) == 2

    def test_absent_files_read_as_empty(self, saved):
        assert saved.get("/saved/absent/trades").json() == []
        assert saved.get("/saved/absent/folds").json() == []

    @pytest.mark.parametrize("bad", ["nope", "../tom_x", "TOM_X", "tom_x_stop_slip1.csv"])
    def test_unknown_names_are_404(self, saved, bad):
        assert saved.get(f"/saved/{bad}/trades").status_code == 404
        assert saved.get(f"/saved/{bad}/folds").status_code == 404

    def test_bad_basis_is_404(self, saved):
        assert saved.get("/saved/tom_x/trades", params={"basis": "../x"}).status_code == 404
        assert saved.get("/saved/tom_x/trades", params={"basis": "other"}).status_code == 404

    def test_page_offers_the_saved_group(self, saved):
        assert "saved verdicts" in saved.get("/").text.lower()

    def test_everything_is_still_get_only(self, saved):
        for path in ("/saved", "/saved/tom_x/trades", "/saved/tom_x/folds"):
            assert saved.post(path).status_code == 405
