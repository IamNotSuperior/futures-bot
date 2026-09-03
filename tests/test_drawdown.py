"""Tests for the end-of-day trailing drawdown tracking.

The trail keys on end-of-day balances only, which is the whole distinction:
an intraday spike must not raise the peak and an intraday dip must not lower
the account. Every figure below is checkable by hand.
"""

from datetime import date

import pandas as pd
import pytest

import rules
from engine import (
    count_evaluation_blowups, equity_curve_by_day, trailing_drawdown_summary,
)

ET = "America/New_York"
START = 50_000.0


def trades_from_daily(daily_pnl: dict[str, float]) -> pd.DataFrame:
    """One trade per day, netting exactly the given P&L."""
    rows = []
    for day, pnl in daily_pnl.items():
        rows.append(
            {
                "entry_time": pd.Timestamp(f"{day} 10:00", tz=ET),
                "exit_time": pd.Timestamp(f"{day} 10:30", tz=ET),
                "direction": "long",
                "net_pnl": float(pnl),
                "session_date": pd.Timestamp(day).date(),
                "duration_seconds": 1800.0,
            }
        )
    return pd.DataFrame(rows)


class TestEquityCurve:
    def test_balance_accumulates(self):
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": 300.0, "2025-01-03": -100.0, "2025-01-06": 250.0}
        ))
        assert list(eq["balance"]) == [50_300.0, 50_200.0, 50_450.0]

    def test_peak_only_rises_on_a_close(self):
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": 500.0, "2025-01-03": -200.0, "2025-01-06": 100.0}
        ))
        # Peak is the prior day's close, so day 1 is measured against the start.
        assert list(eq["peak_eod_balance"]) == [50_000.0, 50_500.0, 50_500.0]

    def test_drawdown_is_measured_from_the_peak(self):
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": 1_000.0, "2025-01-03": -400.0, "2025-01-06": -300.0}
        ))
        assert list(eq["drawdown_from_peak"]) == [0.0, 400.0, 700.0]

    def test_floors_trail_the_peak(self):
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": 1_000.0, "2025-01-03": -100.0}
        ))
        # After a 51,000 close the firm floor is 49,000 and internal is 49,500.
        assert eq.iloc[1]["firm_floor"] == 49_000.0
        assert eq.iloc[1]["internal_floor"] == 49_500.0

    def test_states(self):
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": -500.0, "2025-01-03": -600.0, "2025-01-06": -500.0}
        ))
        # cumulative -500, -1,100, -1,600 against a peak of 50,000
        assert list(eq["state"]) == ["ok", "warn", "stop"]

    def test_empty_input(self):
        assert equity_curve_by_day(pd.DataFrame()).empty


class TestTermination:
    def test_not_terminated_just_inside_the_line(self):
        eq = equity_curve_by_day(trades_from_daily({"2025-01-02": -1_999.0}))
        assert bool(eq.iloc[0]["terminated"]) is False

    def test_terminated_exactly_at_the_line(self):
        eq = equity_curve_by_day(trades_from_daily({"2025-01-02": -2_000.0}))
        assert bool(eq.iloc[0]["terminated"]) is True

    def test_a_profitable_account_can_still_be_terminated(self):
        """Up $2,500 then down $2,000 kills it while still above the start."""
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": 2_500.0, "2025-01-03": -2_000.0}
        ))
        assert eq.iloc[1]["balance"] == 50_500.0  # still $500 up on the start
        assert bool(eq.iloc[1]["terminated"]) is True

    def test_intraday_dip_that_closes_flat_survives(self):
        """The trail is end-of-day: only the close is scored.

        A day that traded $2,500 against us but closed flat leaves the account
        alive, which is exactly what "EOD trailing" buys over an intraday trail.
        """
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": -2_500.0, "2025-01-03": 2_500.0}
        ))
        assert bool(eq.iloc[0]["terminated"]) is True  # it closed down 2,500
        # But two trades in one day netting zero never register a drawdown:
        same_day = pd.DataFrame([
            {"entry_time": pd.Timestamp("2025-01-02 10:00", tz=ET),
             "exit_time": pd.Timestamp("2025-01-02 10:30", tz=ET),
             "direction": "long", "net_pnl": -2_500.0,
             "session_date": date(2025, 1, 2), "duration_seconds": 1800.0},
            {"entry_time": pd.Timestamp("2025-01-02 11:00", tz=ET),
             "exit_time": pd.Timestamp("2025-01-02 11:30", tz=ET),
             "direction": "long", "net_pnl": 2_500.0,
             "session_date": date(2025, 1, 2), "duration_seconds": 1800.0},
        ])
        eq2 = equity_curve_by_day(same_day)
        assert eq2.iloc[0]["drawdown_from_peak"] == 0.0
        assert bool(eq2.iloc[0]["terminated"]) is False

    def test_summary_reports_the_first_termination_date(self):
        eq = equity_curve_by_day(trades_from_daily(
            {"2025-01-02": -1_000.0, "2025-01-03": -1_100.0, "2025-01-06": -100.0}
        ))
        s = trailing_drawdown_summary(eq)
        assert s["terminated"] is True
        assert s["termination_date"] == date(2025, 1, 3)


class TestBlowupCount:
    def test_no_blowup_on_a_quiet_series(self):
        got = count_evaluation_blowups(trades_from_daily(
            {"2025-01-02": 100.0, "2025-01-03": -50.0}
        ))
        assert got["blowups"] == 0
        assert got["passes"] == 0

    def test_single_blowup(self):
        got = count_evaluation_blowups(trades_from_daily(
            {"2025-01-02": -1_000.0, "2025-01-03": -1_000.0}
        ))
        assert got["blowups"] == 1
        assert got["dates"] == [date(2025, 1, 3)]
        assert got["days_survived"] == [2]

    def test_account_restarts_after_a_blowup(self):
        """A second $2,000 hole after the reset is a second dead account."""
        got = count_evaluation_blowups(trades_from_daily(
            {"2025-01-02": -2_000.0, "2025-01-03": -2_000.0, "2025-01-06": -2_000.0}
        ))
        assert got["blowups"] == 3
        assert got["days_survived"] == [1, 1, 1]

    def test_pass_is_counted_and_resets(self):
        got = count_evaluation_blowups(trades_from_daily(
            {"2025-01-02": 3_000.0, "2025-01-03": -2_000.0}
        ))
        assert got["passes"] == 1
        assert got["blowups"] == 1  # the fresh account then died

    def test_profit_target_reached_before_the_drawdown(self):
        got = count_evaluation_blowups(trades_from_daily(
            {"2025-01-02": 1_500.0, "2025-01-03": 1_500.0}
        ))
        assert got["passes"] == 1
        assert got["blowups"] == 0

    def test_empty(self):
        got = count_evaluation_blowups(pd.DataFrame())
        assert got["blowups"] == 0


class TestUsesRulesConstants:
    def test_firm_line_comes_from_the_rules_module(self):
        """Restating 2,000 here would be the duplicated-constant bug again."""
        eq = equity_curve_by_day(trades_from_daily({"2025-01-02": -100.0}))
        assert eq.iloc[0]["firm_floor"] == START - rules.FIRM.max_trailing_drawdown
        assert eq.iloc[0]["internal_floor"] == (
            START - rules.INTERNAL.trailing_drawdown_stop
        )

    def test_starting_balance_defaults_to_the_account_size(self):
        eq = equity_curve_by_day(trades_from_daily({"2025-01-02": 0.0}))
        assert eq.iloc[0]["balance"] == rules.ACCOUNT_SIZE
