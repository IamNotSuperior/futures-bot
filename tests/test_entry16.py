"""Tests for entry 16's runner: the real-against-placebo and real-against-zero
tests, the per-year and reported breakdowns, the five kill criteria, the
reproduction check and the CLI wiring. Frozen specification at the entry 16
freeze commit.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import live
import run_entry16 as r16
from sweep import CONTINUATION, PLACEBO, REAL


def events(real_m, placebo_m, real_years=None, sides=None, depths=None) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(real_m):
        y = (real_years or [2020 + (i % 7) for i in range(len(real_m))])[i]
        rows.append({"date": date(y, 1, 1 + i % 28), "year": y, "kind": REAL,
                     "side": (sides or ["high", "low"] * len(real_m))[i],
                     "level": 100.0, "extreme": 101.0, "depth": (depths or [1.0] * len(real_m))[i],
                     "decision_ts": pd.NaT, "m30": m, "m1555": m * 1.5})
    for i, m in enumerate(placebo_m):
        y = 2020 + (i % 7)
        rows.append({"date": date(y, 2, 1 + i % 28), "year": y, "kind": PLACEBO, "side": "high",
                     "level": 95.0, "extreme": 96.0, "depth": 1.0, "decision_ts": pd.NaT,
                     "m30": m, "m1555": m})
    rows.append({"date": date(2020, 3, 2), "year": 2020, "kind": CONTINUATION, "side": "high",
                 "level": 100.0, "extreme": np.nan, "depth": np.nan, "decision_ts": pd.NaT,
                 "m30": np.nan, "m1555": -5.0})
    return pd.DataFrame(rows)


class TestMechanismTest:
    def test_real_above_placebo_and_above_zero(self):
        rng = np.random.default_rng(5)
        real = list(rng.normal(3.0, 8.0, 400))
        placebo = list(rng.normal(0.5, 8.0, 400))
        m = r16.mechanism_test(events(real, placebo))
        assert m["real_vs_placebo"]["t"] > 2.0
        assert m["real_vs_placebo"]["p_one_sided"] < 0.01
        assert m["real_vs_placebo"]["n_a"] == 400 and m["real_vs_placebo"]["n_b"] == 400
        assert m["real_vs_zero"]["t"] > 2.0
        assert m["real_vs_zero"]["n"] == 400
        assert m["years_real_positive"] >= 6
        assert m["continuations"]["n"] == 1
        assert set(m["by_side"]) == {"high", "low"}
        assert len(m["by_depth_tercile"]) == 3
        assert "real_vs_placebo_1555" in m

    def test_no_difference_reads_flat(self):
        rng = np.random.default_rng(6)
        both = list(rng.normal(0.0, 8.0, 300))
        m = r16.mechanism_test(events(both, both))
        assert abs(m["real_vs_placebo"]["t"]) < 1e-9
        assert abs(m["real_vs_zero"]["t"]) < 2.0

    def test_one_sample_t_by_hand(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert r16.one_sample_t(x)["t"] == pytest.approx(x.mean() / (x.std(ddof=1) / 2))
        assert r16.one_sample_t(np.array([1.0]))["t"] != r16.one_sample_t(np.array([1.0]))["t"]  # nan


def stream_stats(profitable_years=4, net=100.0, pass_probability=0.3, blowups=0, trading_days=300):
    return {"profitable_years": profitable_years, "net_pnl": net,
            "pass_probability": pass_probability, "blowups": blowups, "trading_days": trading_days}


def mechanism(t_vs_placebo=2.5, t_vs_zero=3.0, years=5):
    return {"real_vs_placebo": {"t": t_vs_placebo, "p_one_sided": 0.01},
            "real_vs_zero": {"t": t_vs_zero, "p_one_sided": 0.001},
            "years_real_positive": years}


class TestCriteria:
    def test_all_pass(self):
        crit = r16.evaluate_criteria(mechanism(), stream_stats(), stream_stats(), stream_stats())
        assert [c.passed for c in crit] == [True] * 5

    @pytest.mark.parametrize("mech, idx, expect", [
        (mechanism(t_vs_placebo=1.9), 0, False),
        (mechanism(t_vs_zero=1.9), 1, False),
        (mechanism(years=3), 1, False),
        (mechanism(), 1, True),
    ])
    def test_mechanism_lines(self, mech, idx, expect):
        crit = r16.evaluate_criteria(mech, stream_stats(), stream_stats(), stream_stats())
        assert crit[idx].passed is expect

    def test_thresholds_are_the_entrys(self):
        assert r16.MIN_T == 2.0
        assert r16.MIN_YEARS_POSITIVE == 4
        assert r16.CONTRACTS == 1
        assert r16.STOP_BUFFER_TICKS == 4


class TestReproduction:
    def _diag(self, dates, entered, depths):
        return pd.DataFrame({"entered": entered, "depth": depths,
                             "skipped_reason": [None if e else "roll_day" for e in entered]},
                            index=pd.Index(dates, name="date"))

    def test_consistent(self):
        real = pd.DataFrame({"date": [date(2020, 1, 6), date(2020, 1, 7)], "depth": [1.0, 2.5]})
        diag = self._diag([date(2020, 1, 6), date(2020, 1, 7)], [True, True], [1.0, 2.5])
        assert r16.reproduction_differences(real, diag) == []

    def test_entry_on_a_non_event_day_and_missing_day_and_depth_mismatch(self):
        real = pd.DataFrame({"date": [date(2020, 1, 6), date(2020, 1, 7)], "depth": [1.0, 2.5]})
        diag = self._diag([date(2020, 1, 6), date(2020, 1, 8)], [True, True], [1.0, 2.5])
        problems = r16.reproduction_differences(real, diag)
        assert any("2020-01-07" in p for p in problems) and any("2020-01-08" in p for p in problems)
        diag2 = self._diag([date(2020, 1, 6), date(2020, 1, 7)], [True, True], [1.0, 9.0])
        assert r16.reproduction_differences(real, diag2)


class TestCli:
    class Recorder:
        calls: list = []

        def __init__(self, name, runner, directory=None):
            TestCli.Recorder.calls.append((name, runner))
            self.inner = live.NullLive()

        def __getattr__(self, item):
            return getattr(self.inner, item)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_run_with_live(self, monkeypatch):
        TestCli.Recorder.calls = []
        captured = {}
        monkeypatch.setattr(r16, "run", lambda *a, **k: captured.update(k) or ({}, "REJECTED", ""))
        monkeypatch.setattr(r16, "LiveRun", TestCli.Recorder)
        assert r16.main(["--run", "--live"]) == 0
        assert TestCli.Recorder.calls == [("entry16", "run_entry16 --run")]
        assert isinstance(captured["live"], TestCli.Recorder)

    def test_reproduce_with_live(self, monkeypatch):
        TestCli.Recorder.calls = []
        monkeypatch.setattr(r16, "reproduce", lambda *a, **k: True)
        monkeypatch.setattr(r16, "LiveRun", TestCli.Recorder)
        assert r16.main(["--reproduce", "--live"]) == 0
        assert TestCli.Recorder.calls == [("entry16", "run_entry16 --reproduce")]

    def test_no_action_prints_help(self):
        assert r16.main([]) == 2
