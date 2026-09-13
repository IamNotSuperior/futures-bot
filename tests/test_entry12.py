"""Tests for entry 12's runner: the instrument-aware stop, the Part B trigger,
and the pieces it reuses from entry 11 with MNQ's round turn."""

import inspect
from datetime import date

import pandas as pd
import pytest

import rules
from engine import MES, MNQ, CostModel
from run_entry11 import STOP_POINTS
from run_entry12 import (
    CONTRACTS, ENTRY11_MES_CONTROL_SD, MIN_FORWARD_WINDOW_SESSIONS, PARTS,
    Loaded, control_sd, criteria_for, derive_stop_points, forward_ready,
    round_to_tick, stop_fraction, verdict_block,
)
from test_tom import session


class TestFrozenConstants:
    def test_one_contract(self):
        assert CONTRACTS == 1
        assert CONTRACTS < rules.POSITION_CAP

    def test_the_parts(self):
        assert set(PARTS) == {"A", "B"}
        assert PARTS["A"].spec is MNQ and PARTS["A"].label == "MNQ"
        assert PARTS["B"].spec is MES and PARTS["B"].label == "MES forward"
        assert PARTS["A"].start == date(2020, 1, 1)
        assert PARTS["B"].start == date(2026, 9, 1), "forward data only"

    def test_the_forward_trigger(self):
        assert MIN_FORWARD_WINDOW_SESSIONS == 96

    def test_entry_11_control_sd_is_recorded_with_its_source(self):
        assert ENTRY11_MES_CONTROL_SD == pytest.approx(40.99, abs=0.005)
        source = inspect.getsource(__import__("run_entry12"))
        assert "2bf4fc4" in source, "the verdict it was read from"


class TestStop:
    def test_fraction_is_entry_11s_stop_over_mes_control_sd(self):
        assert stop_fraction() == pytest.approx(STOP_POINTS / ENTRY11_MES_CONTROL_SD)
        assert stop_fraction() == pytest.approx(0.366, abs=0.001)

    def test_round_to_tick(self):
        assert round_to_tick(53.11, MNQ) == 53.0
        assert round_to_tick(53.13, MNQ) == 53.25
        assert round_to_tick(15.0, MES) == 15.0

    def test_derived_stop_scales_with_the_control_sd(self):
        # A control sd of 150 MNQ points gives 0.366 * 150 = 54.9 -> 55.0
        assert derive_stop_points(150.0, MNQ) == 55.0
        # Applied back to MES's own control sd it recovers entry 11's stop.
        assert derive_stop_points(ENTRY11_MES_CONTROL_SD, MES) == STOP_POINTS

    def test_control_sd_reads_only_the_control_population(self):
        returns = pd.DataFrame({
            "date": [date(2025, 1, d) for d in range(2, 12)],
            "year": 2025,
            "label": ["T+1", "control", "control", "control", "T+2",
                      "control", "control", "control", "control", "T+3"],
            "points": [100.0, 1.0, -1.0, 2.0, 100.0, -2.0, 1.0, -1.0, 2.0, 100.0],
        })
        returns["window"] = returns["label"] != "control"
        assert control_sd(returns) == pytest.approx(
            pd.Series([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0]).std(ddof=1))


class TestForwardTrigger:
    def _loaded(self, months):
        days = []
        for m in months:
            for d in (1, 2, 3, 4, 5, 28, 29, 30):
                try:
                    day = date(2026, m, d) if m <= 12 else date(2027, m - 12, d)
                except ValueError:
                    continue
                if day.weekday() < 5:
                    days.append(day)
        bars = pd.concat(session(d, 100.0, 100.0) for d in sorted(set(days)))
        return Loaded(bars=bars, win_bars=bars, rolls=set(), early=set())

    def test_too_few_forward_sessions_refuses(self):
        loaded = self._loaded([9, 10, 11])
        ready, n = forward_ready(loaded)
        assert ready is False and n < MIN_FORWARD_WINDOW_SESSIONS


class TestCriteria:
    def test_round_turn_is_the_instruments_own(self):
        crit = criteria_for(
            {"pooled": {"diff": 0.9, "t": 3.0}, "years_window_beats_control": 5},
            {"trades": 300, "net_pnl": 1.0, "profitable_years": 5,
             "pass_probability": 0.3, "blowups": 0, "trading_days": 300},
            {"trades": 300, "net_pnl": 1.0, "profitable_years": 5,
             "pass_probability": 0.3, "blowups": 0, "trading_days": 300},
            {"trades": 300, "net_pnl": 1.0, "profitable_years": 5,
             "pass_probability": 0.3, "blowups": 0, "trading_days": 300},
            MNQ, CostModel(slippage_ticks=1.0))
        # 0.9 MNQ points is under MNQ's 1.00-point round turn; on MES it would pass.
        assert not crit[0].passed
        assert "1.00" in crit[0].threshold


class TestVerdictBlock:
    def test_renders_part_a_with_the_derived_stop(self):
        from run_entry11 import YEARS, evaluate_criteria
        crit = evaluate_criteria(
            {"pooled": {"diff": 0.1, "t": 0.2}, "years_window_beats_control": 3},
            {"trades": 10, "net_pnl": -1.0, "profitable_years": 0,
             "pass_probability": 0.01, "blowups": 0, "trading_days": 10},
            {"trades": 300, "net_pnl": 1.0, "profitable_years": 4,
             "pass_probability": 0.1, "blowups": 0, "trading_days": 300},
            {"trades": 10, "net_pnl": -1.0, "profitable_years": 0,
             "pass_probability": 0.01, "blowups": 0, "trading_days": 10},
            1.0)
        from test_entry11 import full_stats
        results = {
            "part": "A", "label": "MNQ", "status": "REJECTED", "criteria": crit,
            "stop_points": 55.0, "control_sd": 150.0, "round_turn": 1.0,
            "mechanism": {
                "pooled": {"n_a": 314, "n_b": 1327, "mean_a": 1.0, "mean_b": 0.5,
                           "sd_a": 150.0, "sd_b": 150.0, "median_a": 2.0,
                           "median_b": 1.0, "diff": 0.5, "t": 0.2,
                           "p_one_sided": 0.42, "df": 500.0},
                "by_year": pd.DataFrame([{"year": y, "n_window": 44, "mean_window": 1.0,
                                          "n_control": 200, "mean_control": 0.5,
                                          "diff": 0.5, "t": 0.3,
                                          "window_beats_control": True} for y in YEARS]),
                "by_label": pd.DataFrame([{"label": lab, "n": 78, "mean": 1.0, "sd": 150.0,
                                           "diff": 0.6, "t": 0.2}
                                          for lab in ("T-1", "T+1", "T+2", "T+3")]),
                "years_window_beats_control": 3,
            },
            "arms": {arm: {ticks: {"standard": full_stats(), "comparable": full_stats(),
                                   "dd_halts": 0, "loss_halts": 0}
                           for ticks in (1.0, 2.0)} for arm in ("signal", "stop")},
            "diagnostics": {"skipped": pd.DataFrame(columns=["date", "label", "skipped_reason"]),
                            "entry_bar_breaches": 0},
        }
        text = verdict_block(results, "2026-09-12")
        assert text.startswith("### Part A result: REJECTED")
        assert "55.00" in text and "150.00" in text
        assert "Part B is not run" in text
        for c in crit:
            assert c.label in text


class TestGuardParity:
    def test_the_runner_uses_the_engine_guards_and_restates_nothing(self):
        import run_entry12

        source = inspect.getsource(run_entry12)
        assert "apply_internal_guards(" in source
        assert "def enforce_trailing_drawdown_halt" not in source
        assert "def enforce_daily_loss_limit" not in source
        assert str(int(rules.TRAILING_DD_STOP)) not in source.replace(
            f"{rules.TRAILING_DD_STOP:,.0f}", "")
        assert "date(2020" not in source, "the span is imported from walkforward"
