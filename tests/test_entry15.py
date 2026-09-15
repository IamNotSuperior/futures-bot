"""Tests for entry 15's runner: the opening-impact and reversal tests, the
derived ``sd_open``, the five kill criteria, and the CLI wiring. Frozen
specification at the entry 15 freeze commit.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import live
import run_entry15 as r15
from opex import CONTROL_FRIDAY, CONTROL_OTHER
from quarterly import MONTHLY_EXPIRY, QUARTERLY


def population(q_open, f_open, q_rev=None, f_rev=None, years=None) -> pd.DataFrame:
    """A session_moves-shaped frame: opening moves per label, reversal flags."""
    rows = []
    q_rev = q_rev if q_rev is not None else [i % 2 == 0 for i in range(len(q_open))]
    f_rev = f_rev if f_rev is not None else [i % 2 == 0 for i in range(len(f_open))]
    for i, (o, r) in enumerate(zip(q_open, q_rev)):
        y = (years or [2020 + (i % 7) for i in range(len(q_open))])[i]
        rows.append({"date": date(y, 3, 1 + i % 28), "year": y, "label": QUARTERLY,
                     "opening_move": o, "abs_open": abs(o), "afternoon_move": -o if r else o,
                     "reversal": r})
    for i, (o, r) in enumerate(zip(f_open, f_rev)):
        y = 2020 + (i % 7)
        rows.append({"date": date(y, 4, 1 + i % 28), "year": y, "label": CONTROL_FRIDAY,
                     "opening_move": o, "abs_open": abs(o), "afternoon_move": -o if r else o,
                     "reversal": r})
    rows.append({"date": date(2020, 5, 1), "year": 2020, "label": MONTHLY_EXPIRY,
                 "opening_move": 99.0, "abs_open": 99.0, "afternoon_move": 0.0, "reversal": True})
    rows.append({"date": date(2020, 5, 4), "year": 2020, "label": CONTROL_OTHER,
                 "opening_move": 99.0, "abs_open": 99.0, "afternoon_move": 0.0, "reversal": True})
    return pd.DataFrame(rows)


class TestSdOpen:
    def test_is_the_friday_controls_opening_move_sd_and_ignores_other_labels(self):
        pop = population([50.0] * 7, [3.0, -3.0, 6.0, -6.0, 9.0, -9.0, 12.0])
        fri = pop[pop["label"] == CONTROL_FRIDAY]["opening_move"]
        assert r15.sd_open_from(pop) == pytest.approx(float(fri.std(ddof=1)))


class TestMechanismTest:
    def test_larger_expiry_impact_reads_positive_and_reversal_share_difference_in_points(self):
        rng = np.random.default_rng(3)
        q = list(rng.normal(0, 30.0, 26))
        f = list(rng.normal(0, 10.0, 250))
        m = r15.mechanism_test(population(q, f, q_rev=[True] * 20 + [False] * 6,
                                          f_rev=[True] * 125 + [False] * 125))
        assert m["impact"]["t"] > 2.0
        assert m["impact"]["p_one_sided"] < 0.01
        assert m["impact"]["n_a"] == 26 and m["impact"]["n_b"] == 250
        assert m["reversal"]["share_expiry"] == pytest.approx(20 / 26)
        assert m["reversal"]["share_control"] == pytest.approx(0.5)
        assert m["reversal"]["diff_points"] == pytest.approx((20 / 26 - 0.5) * 100)
        assert m["reversal"]["z"] > 1.64
        assert m["years_expiry_impact_above"] == 7
        assert set(m["conditional"]) == {"expiry", "control"}

    def test_monthly_expiry_and_other_sessions_are_in_neither_population(self):
        m = r15.mechanism_test(population([1.0] * 26, [1.0] * 250))
        assert m["impact"]["n_a"] == 26 and m["impact"]["n_b"] == 250   # the 99-point rows excluded

    def test_two_proportion_z_by_hand(self):
        assert r15.two_proportion_z(20, 26, 125, 250) == pytest.approx(
            ((20 / 26) - 0.5) / math.sqrt((145 / 276) * (1 - 145 / 276) * (1 / 26 + 1 / 250)))
        assert r15.two_proportion_z(0, 0, 10, 20) == 0.0


def stream_stats(profitable_years=4, net=100.0, pass_probability=0.3, blowups=0, trading_days=10):
    return {"profitable_years": profitable_years, "net_pnl": net,
            "pass_probability": pass_probability, "blowups": blowups, "trading_days": trading_days}


def mechanism(t=2.5, diff_points=20.0, z=2.0):
    return {"impact": {"t": t, "p_one_sided": 0.01},
            "reversal": {"diff_points": diff_points, "z": z}}


class TestCriteria:
    def test_all_pass(self):
        crit = r15.evaluate_criteria(mechanism(), stream_stats(), stream_stats(), stream_stats())
        assert [c.passed for c in crit] == [True] * 5

    @pytest.mark.parametrize("mech, idx, expect", [
        (mechanism(t=1.9), 0, False),
        (mechanism(t=-2.5), 0, False),
        (mechanism(diff_points=14.9), 1, False),
        (mechanism(z=1.5), 1, False),
        (mechanism(), 1, True),
    ])
    def test_mechanism_lines(self, mech, idx, expect):
        crit = r15.evaluate_criteria(mech, stream_stats(), stream_stats(), stream_stats())
        assert crit[idx].passed is expect

    def test_thresholds_are_the_entrys(self):
        assert r15.MIN_T == 2.0
        assert r15.MIN_REVERSAL_POINTS == 15.0
        assert r15.MIN_Z == 1.64
        assert r15.CONTRACTS == 1
        assert r15.PARAMS_K == 1.0 and r15.PARAMS_S == 1.0


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
        monkeypatch.setattr(r15, "run", lambda *a, **k: captured.update(k) or ({}, "REJECTED", ""))
        monkeypatch.setattr(r15, "LiveRun", TestCli.Recorder)
        assert r15.main(["--run", "--live"]) == 0
        assert TestCli.Recorder.calls == [("entry15", "run_entry15 --run")]
        assert isinstance(captured["live"], TestCli.Recorder)

    def test_reproduce_with_live(self, monkeypatch):
        TestCli.Recorder.calls = []
        monkeypatch.setattr(r15, "reproduce", lambda *a, **k: True)
        monkeypatch.setattr(r15, "LiveRun", TestCli.Recorder)
        assert r15.main(["--reproduce", "--live"]) == 0
        assert TestCli.Recorder.calls == [("entry15", "run_entry15 --reproduce")]

    def test_no_action_prints_help(self):
        assert r15.main([]) == 2
