"""Tests for entry 11's runner: the frozen constants, the mechanism test, the
kill criteria and the verdict block. The run over real bars is the script's
job; what is pinned here is every piece that decides something."""

import inspect
from datetime import date

import numpy as np
import pandas as pd
import pytest

import rules
from engine import MES, CostModel
from walkforward import build_folds
from run_entry11 import (
    BASE_SLIPPAGE_TICKS, CONTRACTS, END, MAX_BLOWUPS, MIN_PASS_PROBABILITY,
    MIN_PROFITABLE_FOLDS, MIN_T, MIN_YEARS_WINDOW_BEATS_CONTROL,
    SENSITIVITY_TICKS, START, STOP_POINTS, YEARS, evaluate_criteria,
    mechanism_test, round_turn_points, verdict_block, verdict_status, welch,
)


class TestFrozenConstants:
    def test_the_operator_specification(self):
        assert CONTRACTS == 4
        assert STOP_POINTS == 15.0
        assert BASE_SLIPPAGE_TICKS == 1.0
        assert SENSITIVITY_TICKS == (2.0,)

    def test_the_kill_lines(self):
        assert MIN_T == 2.0
        assert MIN_YEARS_WINDOW_BEATS_CONTROL == 4
        assert MIN_PASS_PROBABILITY == 0.25
        assert MAX_BLOWUPS == 1
        assert MIN_PROFITABLE_FOLDS == 4

    def test_the_span_is_imported_from_walkforward(self):
        folds = build_folds()
        assert START == folds[0].test_start
        assert YEARS == [f.test_year for f in folds]
        assert END == folds[-1].test_end
        source = inspect.getsource(__import__("run_entry11"))
        assert "date(2020" not in source

    def test_contracts_are_inside_the_cap(self):
        assert CONTRACTS < rules.POSITION_CAP

    def test_one_stop_cannot_reach_the_daily_loss_limit(self):
        costs = CostModel(slippage_ticks=BASE_SLIPPAGE_TICKS)
        worst = (STOP_POINTS * MES.point_value * CONTRACTS
                 + costs.commission_round_turn(CONTRACTS)
                 + costs.slippage_round_turn(MES, CONTRACTS))
        assert worst < rules.DAILY_LOSS_LIMIT


class TestRoundTurn:
    def test_one_tick_is_seventy_hundredths_of_a_point(self):
        assert round_turn_points(CostModel(slippage_ticks=1.0)) == pytest.approx(0.70)

    def test_two_ticks_is_one_point_twenty(self):
        assert round_turn_points(CostModel(slippage_ticks=2.0)) == pytest.approx(1.20)

    def test_it_is_computed_from_the_cost_model_not_written_down(self):
        dear = CostModel(commission_per_side=2.0, slippage_ticks=1.0)
        assert round_turn_points(dear) == pytest.approx((4.0 + 2.5) / 5.0)


class TestWelch:
    def test_identical_samples_have_zero_t_and_half_p(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = welch(a, a)
        assert w["diff"] == 0.0
        assert w["t"] == pytest.approx(0.0)
        assert w["p_one_sided"] == pytest.approx(0.5)

    def test_a_clearly_higher_window_has_a_large_positive_t(self):
        rng = np.random.default_rng(0)
        window = rng.normal(5.0, 1.0, 300)
        control = rng.normal(0.0, 1.0, 1300)
        w = welch(window, control)
        assert w["diff"] == pytest.approx(window.mean() - control.mean())
        assert w["t"] > 20
        assert w["p_one_sided"] < 1e-6
        assert w["n_a"] == 300 and w["n_b"] == 1300

    def test_a_lower_window_has_a_negative_t_and_p_above_half(self):
        w = welch(np.array([0.0, -1.0, -2.0, 1.0]), np.array([3.0, 4.0, 5.0, 2.0]))
        assert w["t"] < 0
        assert w["p_one_sided"] > 0.5


def returns_frame(rows):
    """(date, label, points) rows into the frame ``session_returns`` produces."""
    return pd.DataFrame([{"date": d, "year": d.year, "label": lab,
                          "window": lab != "control", "points": p}
                         for d, lab, p in rows])


class TestMechanismTest:
    def test_pooled_per_year_and_per_label_tables(self):
        rows = []
        for y in (2024, 2025):
            for m in range(1, 13):
                rows.append((date(y, m, 2), "T+1", 3.0 if y == 2024 else -1.0))
                rows.append((date(y, m, 3), "T+2", 1.0))
                rows.append((date(y, m, 10), "control", 0.0))
                rows.append((date(y, m, 11), "control", 0.5))
        out = mechanism_test(returns_frame(rows))
        pooled = out["pooled"]
        assert pooled["n_a"] == 48 and pooled["n_b"] == 48
        assert pooled["diff"] == pytest.approx((3 + 1 - 1 + 1) / 4 - 0.25)
        years = out["by_year"].set_index("year")
        assert bool(years.loc[2024, "window_beats_control"]) is True
        assert bool(years.loc[2025, "window_beats_control"]) is False
        assert out["years_window_beats_control"] == 1
        labels = out["by_label"].set_index("label")
        assert labels.loc["T+2", "mean"] == pytest.approx(1.0)
        assert labels.loc["T+2", "n"] == 24
        assert set(labels.index) == {"T+1", "T+2"}


def stats(**over):
    base = {"trades": 300, "net_pnl": 500.0, "profitable_years": 5,
            "pass_probability": 0.40, "blowups": 0, "trading_days": 300}
    base.update(over)
    return base


def mech(diff=2.0, t=3.0, years=5):
    return {"pooled": {"diff": diff, "t": t}, "years_window_beats_control": years}


class TestCriteria:
    def all_pass(self):
        return evaluate_criteria(mech(), stats(), stats(), stats(), round_turn=0.70)

    def test_everything_passing_is_accepted(self):
        crit = self.all_pass()
        assert len(crit) == 5
        assert all(c.passed for c in crit)
        assert verdict_status(crit) == "ACCEPTED"

    def test_excess_under_one_round_turn_fails_criterion_one(self):
        crit = evaluate_criteria(mech(diff=0.69), stats(), stats(), stats(), 0.70)
        assert not crit[0].passed and verdict_status(crit) == "REJECTED"

    def test_t_under_two_fails_criterion_one_even_with_a_large_excess(self):
        crit = evaluate_criteria(mech(diff=5.0, t=1.99), stats(), stats(), stats(), 0.70)
        assert not crit[0].passed

    def test_three_of_seven_years_fails_criterion_two(self):
        crit = evaluate_criteria(mech(years=3), stats(), stats(), stats(), 0.70)
        assert not crit[1].passed and crit[0].passed

    def test_pass_probability_is_read_on_the_standard_stream(self):
        crit = evaluate_criteria(mech(), stats(pass_probability=0.24),
                                 stats(pass_probability=0.9), stats(), 0.70)
        assert not crit[2].passed
        crit = evaluate_criteria(mech(), stats(pass_probability=0.25),
                                 stats(pass_probability=0.0), stats(), 0.70)
        assert crit[2].passed

    def test_blowups_are_read_on_the_comparable_stream(self):
        crit = evaluate_criteria(mech(), stats(blowups=5), stats(blowups=2), stats(), 0.70)
        assert not crit[3].passed
        crit = evaluate_criteria(mech(), stats(blowups=5), stats(blowups=1), stats(), 0.70)
        assert crit[3].passed

    def test_rule_13_needs_both_cost_levels(self):
        crit = evaluate_criteria(mech(), stats(), stats(),
                                 stats(profitable_years=3), 0.70)
        assert not crit[4].passed
        crit = evaluate_criteria(mech(), stats(net_pnl=-1.0), stats(), stats(), 0.70)
        assert not crit[4].passed
        crit = evaluate_criteria(mech(), stats(profitable_years=4, net_pnl=1.0),
                                 stats(), stats(profitable_years=4, net_pnl=1.0), 0.70)
        assert crit[4].passed

    def test_the_threshold_text_names_the_lines(self):
        crit = self.all_pass()
        assert "0.70" in crit[0].threshold and "2.0" in crit[0].threshold
        assert "4 of 7" in crit[1].threshold
        assert "25%" in crit[2].threshold
        assert "<= 1" in crit[3].threshold


class TestVerdictBlock:
    def test_renders_the_status_and_every_criterion(self):
        crit = evaluate_criteria(mech(diff=0.1, t=0.2, years=3), stats(pass_probability=0.1),
                                 stats(blowups=2), stats(profitable_years=2), 0.70)
        results = {
            "criteria": crit, "status": verdict_status(crit),
            "mechanism": {
                "pooled": {"n_a": 314, "n_b": 1332, "mean_a": 0.5, "mean_b": 0.4,
                           "sd_a": 30.0, "sd_b": 28.0, "median_a": 1.0,
                           "median_b": 0.8, "diff": 0.1, "t": 0.2,
                           "p_one_sided": 0.42, "df": 500.0},
                "by_year": pd.DataFrame([{"year": y, "n_window": 44, "mean_window": 1.0,
                                          "n_control": 200, "mean_control": 0.5,
                                          "diff": 0.5, "t": 0.3,
                                          "window_beats_control": True} for y in YEARS]),
                "by_label": pd.DataFrame([{"label": lab, "n": 78, "mean": 1.0, "sd": 30.0,
                                           "diff": 0.6, "t": 0.2}
                                          for lab in ("T-1", "T+1", "T+2", "T+3")]),
                "years_window_beats_control": 3,
            },
            "arms": {
                arm: {ticks: {"standard": full_stats(), "comparable": full_stats(),
                              "dd_halts": 0, "loss_halts": 0}
                      for ticks in (1.0, 2.0)}
                for arm in ("signal", "stop")
            },
            "diagnostics": {"skipped": pd.DataFrame(columns=["date", "label", "skipped_reason"]),
                            "entry_bar_breaches": 0},
        }
        text = verdict_block(results, "2026-09-11")
        assert text.startswith("### Verdict: REJECTED")
        for c in crit:
            assert c.label in text
        assert "T+3" in text
        assert "Worst day as % of total profit" in text
        assert "Profit from <=5s holds" in text
        assert "2 ticks" in text


def full_stats():
    return {
        "trades": 100, "net_pnl": -50.0, "mean_per_trade": -0.5,
        "profitable_years": 3, "sharpe": -0.1, "profit_factor": 0.98,
        "max_drawdown": -900.0, "max_daily_loss": -400.0,
        "worst_day_pct": None, "best_day_pct": None, "best_day": 500.0,
        "pass_probability": 0.1, "payout_probability": 0.2, "blowups": 1,
        "dd_sessions_blocked": 0, "avg_duration_min": 385.0, "microscalp_pct": 0.0,
        "trading_days": 100, "year_pnl": {y: 0.0 for y in YEARS},
        "by_reason": {"flatten_1555": (90, -1.0, -90.0), "stop": (10, 4.0, 40.0)},
        "loss_limit_exits": 0, "loss_halts": 0,
    }


class TestScoredReturns:
    """The first reproduction run diverged by three sessions: the population
    was built on bars sliced to the scored span, which cannot see the
    December 2019 boundary, so 2, 3 and 6 January 2020 were unlabelled while
    the strategy, generating over the whole file, traded them. The population
    is built on the whole file and sliced, like the signals."""

    def test_a_boundary_just_before_the_span_start_labels_the_first_days(self):
        from run_entry11 import Loaded, scored_returns
        from test_tom import session

        days = [date(2019, 12, 30), date(2019, 12, 31), date(2020, 1, 2),
                date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]
        bars = pd.concat(session(d, 100.0, 101.0) for d in days)
        loaded = Loaded(bars=bars, win_bars=bars[bars.index.date >= START],
                        rolls=set(), early=set())
        out = scored_returns(loaded).set_index("date")
        assert out.loc[date(2020, 1, 2), "label"] == "T+1"
        assert out.loc[date(2020, 1, 6), "label"] == "T+3"
        assert date(2019, 12, 31) not in out.index, "the span starts in 2020"


class TestGuardParity:
    def test_the_runner_uses_the_engine_guards_and_restates_nothing(self):
        import run_entry11

        source = inspect.getsource(run_entry11)
        assert "apply_internal_guards(" in source
        assert "def enforce_trailing_drawdown_halt" not in source
        assert "def enforce_daily_loss_limit" not in source
        assert str(int(rules.TRAILING_DD_STOP)) not in source.replace(
            f"{rules.TRAILING_DD_STOP:,.0f}", "")
        assert str(int(rules.DAILY_LOSS_LIMIT)) not in source.replace(
            f"{rules.DAILY_LOSS_LIMIT:,.0f}", "")
