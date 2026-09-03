"""Tests for the opening-range breakout strategy.

Bars are built by hand so the opening range, breakout bars and stop/target
levels are all known by arithmetic.
"""

from datetime import date, time

import pandas as pd
import pytest

from base import REQUIRED_SIGNAL_COLUMNS, validate_signals
from orb import ORBParams, OpeningRangeBreakout, resample_bars

ET = "America/New_York"
DAY = "2025-07-16"  # a regular Wednesday


def bars_from_closes(closes, day: str = DAY, start: str = "09:30") -> pd.DataFrame:
    """1-minute bars whose OHLC is derived from a close series.

    open = previous close, high/low = the enclosing range of open and close, so
    a 5-minute resample has predictable extremes.
    """
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq="1min", tz=ET)
    close = pd.Series([float(c) for c in closes], index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    pair = pd.concat([open_, close], axis=1)
    return pd.DataFrame(
        {
            "open": open_,
            "high": pair.max(axis=1),
            "low": pair.min(axis=1),
            "close": close,
            "volume": 100,
            "instrument_id": 1,
        },
        index=idx,
    )


# Opening range: minutes 0-14 oscillate between 100 and 102.
OPENING = [100.0, 102.0] * 7 + [100.0]
OR_HIGH, OR_LOW = 102.0, 100.0
OR_HEIGHT = 2.0


def flat(value, count):
    return [float(value)] * count


class TestOpeningRange:
    def test_range_is_the_first_15_minutes(self):
        bars = bars_from_closes(OPENING + flat(101, 45))
        strat = OpeningRangeBreakout()
        bars5 = resample_bars(bars, 5)
        opening = bars5[bars5.index.time < time(9, 45)]
        assert len(opening) == 3
        assert opening["high"].max() == OR_HIGH
        assert opening["low"].min() == OR_LOW

    def test_no_signal_when_price_stays_inside_the_range(self):
        bars = bars_from_closes(OPENING + flat(101, 60))
        signals = OpeningRangeBreakout().generate_signals(bars)
        assert not signals[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()


class TestEntryArithmetic:
    """Breakout at 09:45, entry at the 09:50 open, levels off the OR height."""

    def build(self, stop_multiple=1.0, target_multiple=2.0):
        closes = (
            OPENING                       # 0-14   opening range
            + flat(102.5, 4) + [103.0]    # 15-19  5m bar 09:45 closes at 103 > 102
            + flat(103.0, 40)             # 20-59  drift, no stop/target touch
        )
        bars = bars_from_closes(closes)
        strat = OpeningRangeBreakout(
            ORBParams(stop_multiple=stop_multiple, target_multiple=target_multiple)
        )
        return strat.generate_signals(bars)

    def test_entry_fires_on_the_bar_after_the_breakout(self):
        sig = self.build()
        entries = sig.index[sig["entry_long"]]
        assert len(entries) == 1
        assert entries[0].time() == time(9, 50)

    def test_stop_and_target_levels(self):
        sig = self.build(stop_multiple=1.0, target_multiple=2.0)
        row = sig.loc[sig["entry_long"]].iloc[0]
        # entry is the 09:50 open, which is the 09:49 close = 103.0
        assert float(row["stop_price"]) == pytest.approx(103.0 - OR_HEIGHT)
        assert float(row["target_price"]) == pytest.approx(103.0 + 2 * OR_HEIGHT)

    def test_stop_multiple_scales_the_distance(self):
        sig = self.build(stop_multiple=3.0, target_multiple=1.0)
        row = sig.loc[sig["entry_long"]].iloc[0]
        assert float(row["stop_price"]) == pytest.approx(103.0 - 3 * OR_HEIGHT)
        assert float(row["target_price"]) == pytest.approx(103.0 + 3 * OR_HEIGHT)

    def test_no_lookahead_entry_is_after_its_trigger(self):
        sig = self.build()
        bars5 = resample_bars(bars_from_closes(
            OPENING + flat(102.5, 4) + [103.0] + flat(103.0, 40)
        ), 5)
        entry_ts = sig.index[sig["entry_long"]][0]
        trigger = bars5[bars5["close"] > OR_HIGH].index[0]
        assert trigger < entry_ts


class TestExits:
    def test_stop_exit(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]     # breakout, entry at 09:50 open = 103
            + flat(103.0, 5)               # 09:50 bar, no touch
            + flat(100.5, 5)               # 09:55 bar dips to 100.5 <= stop 101
            + flat(101.0, 30)
        )
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        exits = sig.loc[sig["exit_long"]]
        assert len(exits) == 1
        assert exits.index[0].time() == time(9, 55)
        assert exits.iloc[0]["exit_reason"] == "stop"
        assert float(exits.iloc[0]["exit_price"]) == pytest.approx(101.0)

    def test_target_exit(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]
            + flat(103.0, 5)
            + flat(107.5, 5)               # reaches target 107
            + flat(107.0, 30)
        )
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        exits = sig.loc[sig["exit_long"]]
        assert exits.iloc[0]["exit_reason"] == "target"
        assert float(exits.iloc[0]["exit_price"]) == pytest.approx(107.0)

    def test_session_end_exit_when_neither_level_is_hit(self):
        closes = OPENING + flat(102.5, 4) + [103.0] + flat(103.0, 40)
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        exits = sig.loc[sig["exit_long"]]
        assert exits.iloc[0]["exit_reason"] == "session_end"

    def test_stop_wins_when_one_bar_holds_both_levels(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]
            + flat(103.0, 5)
            + [99.0, 108.0, 103.0, 103.0, 103.0]  # one 5m bar spans stop and target
            + flat(103.0, 30)
        )
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        exits = sig.loc[sig["exit_long"]]
        assert exits.iloc[0]["exit_reason"] == "stop"


class TestOnePositionAtATime:
    """A breakout the other way while a trade is open must be ignored."""

    #: A wide stop keeps the long alive while price falls through the OR low.
    WIDE = ORBParams(stop_multiple=5.0, target_multiple=2.0)

    def test_opposite_breakout_is_suppressed_while_a_trade_is_open(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]     # long entry at 09:50, stop 93, target 123
            + flat(103.0, 5)               # 09:50
            + flat(99.0, 5)                # 09:55 closes below OR low - would be a short
            + flat(99.0, 30)               # long never stops out
        )
        sig = OpeningRangeBreakout(self.WIDE).generate_signals(bars_from_closes(closes))
        assert sig["entry_long"].sum() == 1
        assert sig["entry_short"].sum() == 0

    def test_no_long_and_short_open_at_the_same_time(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]
            + flat(103.0, 5)
            + flat(99.0, 40)
        )
        sig = OpeningRangeBreakout(self.WIDE).generate_signals(bars_from_closes(closes))
        assert_no_overlap(sig)

    def test_other_direction_may_trade_after_the_first_closes(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]     # long entry 09:50, stop 101 (default params)
            + flat(103.0, 5)               # 09:50
            + flat(100.5, 5)               # 09:55 stop hit at 101, now flat
            + flat(99.0, 10)               # 10:00 closes below OR low -> short breakout
            + flat(99.0, 25)
        )
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        assert sig["entry_long"].sum() == 1
        assert sig["entry_short"].sum() == 1
        long_entry = sig.index[sig["entry_long"]][0]
        long_exit = sig.index[sig["exit_long"]][0]
        short_entry = sig.index[sig["entry_short"]][0]
        assert long_exit < short_entry
        assert long_entry < long_exit
        assert_no_overlap(sig)

    def test_at_most_one_trade_per_direction_per_session(self):
        closes = (
            OPENING
            + flat(102.5, 4) + [103.0]
            + flat(103.0, 5)
            + flat(100.5, 5)               # stopped out
            + flat(103.0, 5)               # breaks above the range again
            + flat(103.0, 25)
        )
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        assert sig["entry_long"].sum() == 1


def assert_no_overlap(signals: pd.DataFrame) -> None:
    """No long and short trade may be open at the same moment."""
    for _, day in signals.groupby(signals.index.date):
        le = list(day.index[day["entry_long"]])
        lx = list(day.index[day["exit_long"]])
        se = list(day.index[day["entry_short"]])
        sx = list(day.index[day["exit_short"]])
        if not le or not se:
            continue
        assert not (le[0] < se[0] <= lx[0]), "short opened inside an open long"
        assert not (se[0] < le[0] <= sx[0]), "long opened inside an open short"


class TestGuards:
    def test_roll_day_blocks_all_entries(self):
        closes = OPENING + flat(102.5, 4) + [103.0] + flat(103.0, 40)
        bars = bars_from_closes(closes)
        strat = OpeningRangeBreakout(roll_dates={date(2025, 7, 16)})
        sig = strat.generate_signals(bars)
        assert not sig[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()

    def test_trade_window_end_blocks_late_entries(self):
        """A breakout after the window closes produces nothing."""
        closes = OPENING + flat(101.0, 105) + flat(103.0, 60)
        bars = bars_from_closes(closes)
        strat = OpeningRangeBreakout(ORBParams(trade_window_end=time(10, 0)))
        sig = strat.generate_signals(bars)
        assert sig["entry_long"].sum() == 0

    def test_signals_pass_interface_validation(self):
        closes = OPENING + flat(102.5, 4) + [103.0] + flat(103.0, 40)
        sig = OpeningRangeBreakout().generate_signals(bars_from_closes(closes))
        validate_signals(sig)
        for col in REQUIRED_SIGNAL_COLUMNS:
            assert sig[col].dtype == bool


class TestParams:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"opening_range_minutes": 0},
            {"opening_range_minutes": 7},  # not a whole number of 5m bars
            {"bar_minutes": 0},
            {"stop_multiple": 0},
            {"target_multiple": -1},
        ],
    )
    def test_invalid_params_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ORBParams(**kwargs)
