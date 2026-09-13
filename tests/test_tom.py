"""Tests for entry 11: the turn-of-month calendar, session returns and the
two arms of the strategy. Frozen specification at ``a6d9af1``.

The calendar is the part that decides which sessions are in the window, so it
is pinned first and on real dates: Labor Day 2025 is a CME session with the
cash market shut, the day after Thanksgiving is a half-day with it open, and
the labels must come out as the entry says.
"""

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

import rules
from base import validate_signals
from engine import MES, CostModel, build_trades, price_trades
from tom import (
    EXIT_FLATTEN, EXIT_STOP, FLATTEN_BAR, OPEN_BAR, WINDOW_LABELS, TOMParams,
    TurnOfMonth, cash_half_days, label_window_days, session_returns,
    trading_days,
)

ET = "America/New_York"


# ---------------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------------


class TestCashHalfDays:
    def test_day_after_thanksgiving_is_a_half_day(self):
        assert cash_half_days({date(2024, 11, 29)}) == {date(2024, 11, 29)}

    def test_thanksgiving_itself_is_not(self):
        assert cash_half_days({date(2024, 11, 28)}) == set()

    def test_christmas_eve_on_a_weekday_before_a_weekday_christmas(self):
        # 2024-12-25 is a Wednesday: the 24th is a half-day.
        assert cash_half_days({date(2024, 12, 24)}) == {date(2024, 12, 24)}

    def test_july_third_when_the_fourth_is_a_weekday(self):
        # 2025-07-04 is a Friday: the 3rd is a half-day.
        assert cash_half_days({date(2025, 7, 3)}) == {date(2025, 7, 3)}

    def test_july_third_when_the_fourth_is_a_saturday_is_the_observed_holiday(self):
        # 2020-07-04 is a Saturday; the cash market was closed Friday the 3rd.
        assert cash_half_days({date(2020, 7, 3)}) == set()

    def test_full_holiday_globex_sessions_are_not_half_days(self):
        holidays = {date(2025, 9, 1), date(2025, 1, 20), date(2025, 2, 17),
                    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4)}
        assert cash_half_days(holidays) == set()


class TestTradingDays:
    def test_cash_closed_holiday_sessions_are_removed(self):
        sessions = [date(2025, 8, 29), date(2025, 9, 1), date(2025, 9, 2),
                    date(2025, 9, 3), date(2025, 9, 4)]
        assert trading_days(sessions, {date(2025, 9, 1)}) == [
            date(2025, 8, 29), date(2025, 9, 2), date(2025, 9, 3), date(2025, 9, 4)]

    def test_half_days_stay_in_the_calendar(self):
        sessions = [date(2024, 11, 27), date(2024, 11, 28), date(2024, 11, 29),
                    date(2024, 12, 2)]
        early = {date(2024, 11, 28), date(2024, 11, 29)}
        assert trading_days(sessions, early) == [
            date(2024, 11, 27), date(2024, 11, 29), date(2024, 12, 2)]

    def test_output_is_sorted_and_deduplicated(self):
        sessions = [date(2025, 3, 4), date(2025, 3, 3), date(2025, 3, 4)]
        assert trading_days(sessions, set()) == [date(2025, 3, 3), date(2025, 3, 4)]


class TestLabelWindowDays:
    def test_labor_day_month_labels_land_on_cash_trading_days(self):
        days = [date(2025, 8, 28), date(2025, 8, 29), date(2025, 9, 2),
                date(2025, 9, 3), date(2025, 9, 4), date(2025, 9, 5)]
        assert label_window_days(days) == {
            date(2025, 8, 29): "T-1", date(2025, 9, 2): "T+1",
            date(2025, 9, 3): "T+2", date(2025, 9, 4): "T+3",
        }

    def test_a_half_day_can_be_t_minus_one(self):
        """2024-11-29 is the last cash trading day of November. It is labelled
        even though it will later be skipped as a trade day: the label is the
        calendar's, the skip is the trade rule's."""
        days = [date(2024, 11, 27), date(2024, 11, 29), date(2024, 12, 2),
                date(2024, 12, 3), date(2024, 12, 4), date(2024, 12, 5)]
        labels = label_window_days(days)
        assert labels[date(2024, 11, 29)] == "T-1"
        assert labels[date(2024, 12, 2)] == "T+1"
        assert labels[date(2024, 12, 4)] == "T+3"
        assert date(2024, 11, 27) not in labels
        assert date(2024, 12, 5) not in labels

    def test_an_unobserved_boundary_is_not_labelled(self):
        """The file starts mid-May 2019; nothing in May is T+k. It ends on
        2026-08-31 with no September; that day is not T-1."""
        first = [date(2019, 5, 6), date(2019, 5, 7), date(2019, 5, 8)]
        assert label_window_days(first) == {}
        last = [date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)]
        assert label_window_days(last) == {}

    def test_every_label_is_one_of_the_four(self):
        days = [date(2025, 1, 31), date(2025, 2, 3), date(2025, 2, 4),
                date(2025, 2, 5), date(2025, 2, 6), date(2025, 2, 28),
                date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)]
        labels = label_window_days(days)
        assert set(labels.values()) <= set(WINDOW_LABELS)
        assert len(labels) == 8


# ---------------------------------------------------------------------------
# Session returns and the strategy
# ---------------------------------------------------------------------------


def session(day: date, open_px: float, close_px: float, low_path=None,
            end: str = "15:59") -> pd.DataFrame:
    """One RTH session of flat 1-minute bars from ``open_px`` at 09:30 to
    ``close_px`` from 15:55 onward. ``low_path`` maps a bar time to a low."""
    idx = pd.date_range(f"{day} 09:30", f"{day} {end}", freq="1min", tz=ET)
    px = np.where(idx.time >= FLATTEN_BAR, close_px, open_px).astype(float)
    frame = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                          "volume": 10, "instrument_id": 1}, index=idx)
    for t, low in (low_path or {}).items():
        stamp = pd.Timestamp(f"{day} {t}", tz=ET)
        frame.loc[stamp, "low"] = low
    return frame


# A month boundary with the labels the calendar gives it.
D_CTRL = date(2025, 8, 28)     # control
D_TM1 = date(2025, 8, 29)      # T-1
D_LABOR = date(2025, 9, 1)     # Labor Day: early close, cash market shut
D_TP1 = date(2025, 9, 2)       # T+1
D_TP2 = date(2025, 9, 3)       # T+2
D_TP3 = date(2025, 9, 4)       # T+3
D_CTRL2 = date(2025, 9, 5)     # control

EARLY = {D_LABOR}


def fixture_bars(**overrides) -> pd.DataFrame:
    spec = {
        D_CTRL: (100.0, 101.0), D_TM1: (100.0, 103.0),
        D_TP1: (100.0, 98.0), D_TP2: (100.0, 100.5), D_TP3: (100.0, 96.0),
        D_CTRL2: (100.0, 99.0),
    }
    spec.update(overrides)
    frames = [session(d, o, c) for d, (o, c) in sorted(spec.items())]
    frames.append(session(D_LABOR, 100.0, 100.0, end="12:59"))
    return pd.concat(frames).sort_index()


class TestSessionReturns:
    def test_return_is_the_1555_open_less_the_0930_open(self):
        out = session_returns(fixture_bars(), roll_dates=set(), early_close_dates=EARLY)
        by_date = out.set_index("date")["points"]
        assert by_date[D_TM1] == pytest.approx(3.0)
        assert by_date[D_TP1] == pytest.approx(-2.0)
        assert by_date[D_CTRL2] == pytest.approx(-1.0)

    def test_labels_and_control_flag(self):
        out = session_returns(fixture_bars(), roll_dates=set(), early_close_dates=EARLY)
        by_date = out.set_index("date")
        assert by_date.loc[D_TM1, "label"] == "T-1"
        assert by_date.loc[D_TP3, "label"] == "T+3"
        assert by_date.loc[D_CTRL, "label"] == "control"
        assert bool(by_date.loc[D_TP1, "window"]) is True
        assert bool(by_date.loc[D_CTRL, "window"]) is False

    def test_early_close_and_roll_sessions_are_excluded_from_both_groups(self):
        out = session_returns(fixture_bars(), roll_dates={D_TP2},
                              early_close_dates=EARLY)
        assert D_LABOR not in set(out["date"])
        assert D_TP2 not in set(out["date"])
        assert D_TP3 in set(out["date"])

    def test_a_session_without_a_1555_bar_is_excluded(self):
        bars = fixture_bars()
        bars = bars[~((bars.index.date == D_CTRL2) & (bars.index.time >= time(15, 50)))]
        out = session_returns(bars, roll_dates=set(), early_close_dates=EARLY)
        assert D_CTRL2 not in set(out["date"])


class TestParams:
    def test_defaults_are_the_frozen_specification(self):
        p = TOMParams()
        assert p.stop_points is None
        assert TOMParams(stop_points=15.0).stop_points == 15.0

    def test_a_non_positive_stop_is_refused(self):
        with pytest.raises(ValueError):
            TOMParams(stop_points=0.0)


class TestSignalArm:
    def test_enters_long_at_0930_on_window_days_only(self):
        strat = TurnOfMonth(TOMParams(), early_close_dates=EARLY)
        sig = validate_signals(strat.generate_signals(fixture_bars()))
        entries = sig[sig["entry_long"]]
        assert set(entries.index.date) == {D_TM1, D_TP1, D_TP2, D_TP3}
        assert all(t == OPEN_BAR for t in entries.index.time)
        assert not sig["entry_short"].any()

    def test_exits_at_the_1555_bar_open_labelled_flatten(self):
        strat = TurnOfMonth(TOMParams(), early_close_dates=EARLY)
        sig = strat.generate_signals(fixture_bars())
        exits = sig[sig["exit_long"]]
        assert set(exits.index.date) == {D_TM1, D_TP1, D_TP2, D_TP3}
        assert all(t == FLATTEN_BAR for t in exits.index.time)
        assert (exits["exit_reason"] == EXIT_FLATTEN).all()
        assert exits.loc[pd.Timestamp(f"{D_TM1} 15:55", tz=ET), "exit_price"] == 103.0

    def test_roll_and_early_close_window_days_are_skipped_and_counted(self):
        strat = TurnOfMonth(TOMParams(), roll_dates={D_TP1}, early_close_dates=EARLY)
        sig = strat.generate_signals(fixture_bars())
        assert set(sig[sig["entry_long"]].index.date) == {D_TM1, D_TP2, D_TP3}
        skipped = strat.diagnostics
        assert skipped.loc[D_TP1, "skipped_reason"] == "roll_day"
        assert skipped.loc[D_TM1, "entered"] == True  # noqa: E712

    def test_a_half_day_in_the_window_is_labelled_but_not_traded(self):
        bars = pd.concat([
            session(date(2024, 11, 27), 100.0, 100.0),
            session(date(2024, 11, 29), 100.0, 100.0, end="12:59"),
            session(date(2024, 12, 2), 100.0, 100.0),
            session(date(2024, 12, 3), 100.0, 100.0),
            session(date(2024, 12, 4), 100.0, 100.0),
            session(date(2024, 12, 5), 100.0, 100.0),
        ])
        early = {date(2024, 11, 29)}
        strat = TurnOfMonth(TOMParams(), early_close_dates=early)
        sig = strat.generate_signals(bars)
        assert set(sig[sig["entry_long"]].index.date) == {
            date(2024, 12, 2), date(2024, 12, 3), date(2024, 12, 4)}
        assert strat.diagnostics.loc[date(2024, 11, 29), "label"] == "T-1"
        assert strat.diagnostics.loc[date(2024, 11, 29), "skipped_reason"] == "early_close"

    def test_gross_points_reproduce_the_session_returns(self):
        """Pre-registered test 1: the signal arm's trade stream *is* the
        bar-derived window return, session for session."""
        bars = fixture_bars()
        strat = TurnOfMonth(TOMParams(), early_close_dates=EARLY)
        trades = price_trades(build_trades(strat.generate_signals(bars), bars),
                              MES, CostModel(commission_per_side=0.0, slippage_ticks=0.0), 1)
        returns = session_returns(bars, set(), EARLY)
        window = returns[returns["window"]].set_index("date")["points"]
        assert list(trades["session_date"]) == list(window.index)
        assert trades["gross_points"].to_numpy() == pytest.approx(window.to_numpy())
        assert (trades["net_pnl"] == trades["gross_pnl"]).all()

    def test_signals_are_session_independent(self):
        bars = fixture_bars()
        strat = TurnOfMonth(TOMParams(), early_close_dates=EARLY)
        whole = strat.generate_signals(bars)
        part = strat.generate_signals(bars[bars.index.date >= D_TM1])
        cols = ["entry_long", "exit_long", "exit_price"]
        pd.testing.assert_frame_equal(whole.loc[part.index, cols], part[cols])


class TestStopArm:
    def test_stop_at_fifteen_points_fills_at_the_level_stop_first(self):
        bars = fixture_bars()
        bars.loc[pd.Timestamp(f"{D_TP1} 10:15", tz=ET), "low"] = 84.5
        strat = TurnOfMonth(TOMParams(stop_points=15.0), early_close_dates=EARLY)
        sig = strat.generate_signals(bars)
        exits = sig[sig["exit_long"] & (sig.index.date == D_TP1)]
        assert len(exits) == 1
        assert exits.index[0] == pd.Timestamp(f"{D_TP1} 10:15", tz=ET)
        assert exits["exit_reason"].iloc[0] == EXIT_STOP
        assert exits["exit_price"].iloc[0] == 85.0
        assert sig.loc[pd.Timestamp(f"{D_TP1} 09:30", tz=ET), "stop_price"] == 85.0

    def test_a_low_that_does_not_reach_the_stop_flattens_at_1555(self):
        bars = fixture_bars()
        bars.loc[pd.Timestamp(f"{D_TP1} 10:15", tz=ET), "low"] = 85.25
        strat = TurnOfMonth(TOMParams(stop_points=15.0), early_close_dates=EARLY)
        sig = strat.generate_signals(bars)
        exits = sig[sig["exit_long"] & (sig.index.date == D_TP1)]
        assert exits["exit_reason"].iloc[0] == EXIT_FLATTEN

    def test_an_entry_bar_breach_is_not_acted_on_and_is_counted(self):
        bars = fixture_bars()
        bars.loc[pd.Timestamp(f"{D_TP1} 09:30", tz=ET), "low"] = 80.0
        strat = TurnOfMonth(TOMParams(stop_points=15.0), early_close_dates=EARLY)
        sig = strat.generate_signals(bars)
        exits = sig[sig["exit_long"] & (sig.index.date == D_TP1)]
        assert exits["exit_reason"].iloc[0] == EXIT_FLATTEN
        assert strat.diagnostics.loc[D_TP1, "entry_bar_breach"] == True  # noqa: E712
        assert strat.diagnostics["entry_bar_breach"].fillna(False).sum() == 1

    def test_entries_are_identical_to_the_signal_arm(self):
        bars = fixture_bars()
        a = TurnOfMonth(TOMParams(), early_close_dates=EARLY).generate_signals(bars)
        b = TurnOfMonth(TOMParams(stop_points=15.0), early_close_dates=EARLY).generate_signals(bars)
        pd.testing.assert_series_equal(a["entry_long"], b["entry_long"])

    def test_stop_and_flatten_never_mark_the_same_bar_twice(self):
        bars = fixture_bars()
        bars.loc[pd.Timestamp(f"{D_TP1} 15:55", tz=ET), "low"] = 80.0
        strat = TurnOfMonth(TOMParams(stop_points=15.0), early_close_dates=EARLY)
        sig = strat.generate_signals(bars)
        exits = sig[sig["exit_long"] & (sig.index.date == D_TP1)]
        assert len(exits) == 1
        assert exits["exit_reason"].iloc[0] == EXIT_FLATTEN


class TestPowerCheckScript:
    """Entry 12 runs the calendar-only power check on MNQ. The script must
    take the parquet as an argument rather than hard-coding MES's."""

    def test_parquet_flag_selects_the_file(self, tmp_path, capsys):
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "research"))
        import power_check_tom

        bars = fixture_bars()
        bars.index.name = "ts_event_et"
        path = tmp_path / "synthetic.parquet"
        bars.to_parquet(path)
        rc = power_check_tom.main(["--parquet", str(path)])
        out = capsys.readouterr().out
        assert "synthetic.parquet" in out
        assert "2025" in out
        assert "POWER CHECK" in out
        assert rc in (0, 1)


class TestRulesCompatibility:
    def test_every_entry_and_exit_is_inside_the_rules_window(self):
        bars = fixture_bars()
        strat = TurnOfMonth(TOMParams(stop_points=15.0), early_close_dates=EARLY)
        sig = strat.generate_signals(bars)
        for ts in sig[sig["entry_long"]].index:
            assert rules.is_entry_allowed(ts, EARLY, set())
        for ts in sig[sig["exit_long"]].index:
            assert not rules.must_flatten(ts, EARLY)

    def test_constants_are_the_frozen_bars(self):
        assert OPEN_BAR == time(9, 30)
        assert FLATTEN_BAR == time(15, 55)
        assert WINDOW_LABELS == ("T-1", "T+1", "T+2", "T+3")
