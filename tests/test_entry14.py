"""Tests for entry 14's runner: the mechanism test on the range population,
the derived ``sd_move``, the five kill criteria, the reproduction check and
the CLI wiring. Frozen specification at ``f9f129c``.

The runner's heavy pieces are not exercised on the cache here; each piece is
tested on small synthetic populations so a wrong sign or a swapped control
would be caught before a run.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import live
import run_entry14 as r14
from opex import CONTROL_FRIDAY, CONTROL_OTHER, EXPIRY


def population(expiry_ranges, friday_ranges, other_ranges=(), years=None,
               quarterly_every=4) -> pd.DataFrame:
    """A session_ranges-shaped frame with the given ranges per label."""
    rows = []
    y = years or [2020 + (i % 7) for i in range(len(expiry_ranges))]
    for i, r in enumerate(expiry_ranges):
        rows.append({"date": date(y[i], 1, 1 + i % 28), "year": y[i], "label": EXPIRY,
                     "quarterly": i % quarterly_every == 0, "range_points": r,
                     "log_range": math.log(r), "abs_move": r / 2, "morning_move": (-1) ** i * r / 4})
    for i, r in enumerate(friday_ranges):
        yr = 2020 + (i % 7)
        rows.append({"date": date(yr, 2, 1 + i % 28), "year": yr, "label": CONTROL_FRIDAY,
                     "quarterly": False, "range_points": r, "log_range": math.log(r),
                     "abs_move": r / 2, "morning_move": (-1) ** i * r / 3})
    for i, r in enumerate(other_ranges):
        yr = 2020 + (i % 7)
        rows.append({"date": date(yr, 3, 1 + i % 28), "year": yr, "label": CONTROL_OTHER,
                     "quarterly": False, "range_points": r, "log_range": math.log(r),
                     "abs_move": r / 2, "morning_move": 0.0})
    return pd.DataFrame(rows)


class TestSdMove:
    def test_is_the_friday_controls_morning_move_sd(self):
        pop = population([10.0] * 7, [12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0])
        fri = pop[pop["label"] == CONTROL_FRIDAY]["morning_move"]
        assert r14.sd_move_from(pop) == pytest.approx(float(fri.std(ddof=1)))

    def test_never_reads_an_expiry_day(self):
        base = population([10.0] * 7, [20.0] * 14)
        shifted = base.copy()
        shifted.loc[shifted["label"] == EXPIRY, "morning_move"] = 999.0
        assert r14.sd_move_from(base) == r14.sd_move_from(shifted)


class TestMechanismTest:
    def test_expiry_smaller_gives_negative_t_and_negative_median_ratio(self):
        rng = np.random.default_rng(0)
        expiry = list(rng.normal(20.0, 1.0, 84).clip(1))
        friday = list(rng.normal(40.0, 2.0, 250).clip(1))
        m = r14.mechanism_test(population(expiry, friday, other_ranges=[38.0] * 50))
        assert m["primary"]["t"] < -2.0
        assert m["primary"]["p_one_sided"] < 0.01
        assert m["primary"]["median_ratio_pct"] < -40
        assert m["primary"]["n_a"] == 84 and m["primary"]["n_b"] == 250
        assert m["years_expiry_below_friday"] == 7
        assert m["secondary"]["n_b"] == 50
        assert set(m["quarterly"]) == {"quarterly", "monthly"}
        assert m["abs_move"]["n_a"] == 84

    def test_no_difference_gives_t_near_zero(self):
        m = r14.mechanism_test(population([20.0] * 84, [20.0] * 250))
        assert abs(m["primary"]["t"]) < 1e-9 or math.isnan(m["primary"]["t"])
        assert m["primary"]["median_ratio_pct"] == 0.0

    def test_t_is_expiry_minus_control(self):
        """A larger expiry range must read as a positive t, the entry's sign convention."""
        rng = np.random.default_rng(1)
        m = r14.mechanism_test(population(list(rng.normal(40.0, 2.0, 84)),
                                          list(rng.normal(20.0, 1.0, 250))))
        assert m["primary"]["t"] > 2.0
        assert m["primary"]["p_one_sided"] > 0.99   # p is for expiry *smaller*


def stream_stats(profitable_years=4, net=100.0, pass_probability=0.3, blowups=0,
                 trading_days=80):
    return {"profitable_years": profitable_years, "net_pnl": net,
            "pass_probability": pass_probability, "blowups": blowups,
            "trading_days": trading_days}


def mechanism(t=-2.5, ratio=-8.0, years=5):
    return {"primary": {"t": t, "median_ratio_pct": ratio, "p_one_sided": 0.01},
            "years_expiry_below_friday": years}


class TestCriteria:
    def test_all_pass(self):
        crit = r14.evaluate_criteria(mechanism(), stream_stats(), stream_stats(blowups=1),
                                     stream_stats())
        assert [c.passed for c in crit] == [True] * 5
        assert r14.verdict_status(crit) == "ACCEPTED"

    @pytest.mark.parametrize("mech, expect", [
        (mechanism(t=-1.9), False),           # t short of the line
        (mechanism(ratio=-4.9), False),       # median damping under 5%
        (mechanism(t=2.5, ratio=-8.0), False),  # wrong sign
        (mechanism(), True),
    ])
    def test_criterion_1_needs_both_the_t_and_the_median_line(self, mech, expect):
        crit = r14.evaluate_criteria(mech, stream_stats(), stream_stats(), stream_stats())
        assert crit[0].passed is expect

    def test_criterion_2_needs_four_of_seven_years(self):
        assert r14.evaluate_criteria(mechanism(years=3), stream_stats(), stream_stats(),
                                     stream_stats())[1].passed is False

    def test_account_and_rule_13_criteria(self):
        crit = r14.evaluate_criteria(mechanism(), stream_stats(pass_probability=0.2),
                                     stream_stats(blowups=2), stream_stats(profitable_years=3))
        assert [c.passed for c in crit[2:]] == [False, False, False]
        assert r14.verdict_status(crit) == "REJECTED"

    def test_thresholds_are_the_entrys(self):
        assert r14.MAX_T == -2.0
        assert r14.MIN_MEDIAN_DAMPING_PCT == 5.0
        assert r14.MIN_YEARS_EXPIRY_BELOW == 4
        assert r14.CONTRACTS == 3
        assert r14.PARAMS_K == 0.5 and r14.PARAMS_S == 1.0


class TestReproduction:
    def _diag(self, dates, entered, moves, threshold=5.0):
        return pd.DataFrame({
            "entered": entered, "morning_move": moves,
            "skipped_reason": [None if e else ("inside_threshold" if abs(m) < threshold else "roll_day")
                               for e, m in zip(entered, moves)],
        }, index=pd.Index(dates, name="date"))

    def test_consistent_run_has_no_differences(self):
        expiries = [date(2020, 1, 17), date(2020, 2, 21)]
        diag = self._diag(expiries, [True, False], [8.0, 2.0])
        assert r14.reproduction_differences(set(expiries), diag, threshold=5.0) == []

    def test_entry_on_a_non_expiry_day_is_a_difference(self):
        diag = self._diag([date(2020, 1, 17), date(2020, 1, 24)], [True, True], [8.0, 8.0])
        problems = r14.reproduction_differences({date(2020, 1, 17)}, diag, threshold=5.0)
        assert any("2020-01-24" in p for p in problems)

    def test_entry_inside_the_threshold_is_a_difference(self):
        diag = self._diag([date(2020, 1, 17)], [True], [2.0])
        assert r14.reproduction_differences({date(2020, 1, 17)}, diag, threshold=5.0)

    def test_missing_expiry_day_is_a_difference(self):
        diag = self._diag([date(2020, 1, 17)], [True], [8.0])
        assert r14.reproduction_differences({date(2020, 1, 17), date(2020, 2, 21)}, diag,
                                            threshold=5.0)


class TestVerdictBlock:
    def test_block_names_the_status_the_rule_and_the_regression_line(self):
        results = {
            "status": "REJECTED",
            "criteria": r14.evaluate_criteria(mechanism(t=-1.0, ratio=-2.0), stream_stats(),
                                              stream_stats(), stream_stats()),
            "mechanism": {**mechanism(t=-1.0, ratio=-2.0),
                          "primary": {"n_a": 80, "n_b": 251, "mean_a": 3.0, "mean_b": 3.1,
                                      "median_a": 20.0, "median_b": 21.0, "diff": -0.1,
                                      "t": -1.0, "df": 100.0, "p_one_sided": 0.16,
                                      "median_ratio_pct": -2.0, "sd_a": 0.4, "sd_b": 0.4},
                          "secondary": {"n_a": 80, "n_b": 1315, "t": -0.5, "p_one_sided": 0.3,
                                        "median_ratio_pct": -1.0, "diff": -0.05},
                          "abs_move": {"n_a": 80, "n_b": 251, "t": -0.2, "p_one_sided": 0.4,
                                       "median_ratio_pct": -0.5, "diff": -0.01},
                          "quarterly": {"quarterly": {"n": 26, "median": 22.0},
                                        "monthly": {"n": 54, "median": 19.0}},
                          "per_year": [{"year": 2020, "expiry_median": 20.0,
                                        "friday_median": 21.0, "below": True}]},
            "sd_move": 14.2, "threshold_points": 7.1, "stop_points": 14.2,
            "arms": {r14.BASE_SLIPPAGE_TICKS: {"standard": {**stream_stats(), "trades": 30,
                     "mean_per_trade": 3.3, "sharpe": 0.1, "profit_factor": 1.1,
                     "max_drawdown": -100.0, "max_daily_loss": -50.0, "worst_day_pct": 10.0,
                     "best_day_pct": 12.0, "payout_probability": 0.1,
                     "dd_sessions_blocked": 0, "loss_limit_exits": 0, "avg_duration_min": 60.0,
                     "microscalp_pct": 0.0, "by_reason": {}, "year_pnl": {}, "loss_halts": 0},
                     "comparable": {**stream_stats(), "trades": 30, "mean_per_trade": 3.3,
                     "sharpe": 0.1, "profit_factor": 1.1, "max_drawdown": -100.0,
                     "max_daily_loss": -50.0, "worst_day_pct": 10.0, "best_day_pct": 12.0,
                     "payout_probability": 0.1, "dd_sessions_blocked": 0, "loss_limit_exits": 0,
                     "avg_duration_min": 60.0, "microscalp_pct": 0.0, "by_reason": {},
                     "year_pnl": {}, "loss_halts": 0}}},
            "diagnostics": {"skipped": pd.DataFrame(columns=["date", "quarterly", "skipped_reason"]),
                            "entry_bar_breaches": 0, "inside_threshold": 50,
                            "thursday_expiries": [date(2022, 4, 14)]},
            "metrics_standard_1t": {"min_hold_violation_count": 0, "microscalp_trade_count": 0},
        }
        results["arms"][2.0] = results["arms"][r14.BASE_SLIPPAGE_TICKS]
        block = r14.verdict_block(results, "2026-09-15")
        assert "### Verdict: REJECTED" in block
        assert "sd_move" in block and "14.20" in block
        assert "Rule 6/7 regression check" in block
        assert "k = 0.5" in block


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

    def test_run_with_live_opens_a_live_run(self, monkeypatch):
        TestCli.Recorder.calls = []
        captured = {}
        monkeypatch.setattr(r14, "run", lambda *a, **k: captured.update(k) or ({}, "REJECTED", ""))
        monkeypatch.setattr(r14, "LiveRun", TestCli.Recorder)
        assert r14.main(["--run", "--live"]) == 0
        assert TestCli.Recorder.calls == [("entry14", "run_entry14 --run")]
        assert isinstance(captured["live"], TestCli.Recorder)

    def test_reproduce_with_live(self, monkeypatch):
        TestCli.Recorder.calls = []
        monkeypatch.setattr(r14, "reproduce", lambda *a, **k: True)
        monkeypatch.setattr(r14, "LiveRun", TestCli.Recorder)
        assert r14.main(["--reproduce", "--live"]) == 0
        assert TestCli.Recorder.calls == [("entry14", "run_entry14 --reproduce")]

    def test_no_action_prints_help(self):
        assert r14.main([]) == 2
