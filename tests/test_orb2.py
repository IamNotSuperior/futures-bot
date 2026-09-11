"""Tests for ORB-2, the fixed-bracket trend-filtered opening-range break.

Bars are built by hand so every level is known by arithmetic. The opening range
is fixed at 100/102 throughout, which puts the buy stop at 103.0 and the sell
stop at 99.0 with the entry offset of 1.0.

The boundary cases here are the ones entry 4 names: 09:45 placement, the 11:30
cancel, the 15:55 flatten, the 10-point ceiling edge, an open exactly on the
EMA, and both stops reached inside one bar.
"""

from datetime import date, time, timedelta

import pandas as pd
import pytest

import rules
from base import REQUIRED_SIGNAL_COLUMNS, validate_signals
from engine import MES, CostModel, build_trades, price_trades
from orb2 import (
    ORB2, ORB2Params, SKIP_NO_FILL, SKIP_ON_EMA, SKIP_ROLL, SKIP_UNSEEDED,
    SKIP_WIDE_RANGE,
)

ET = "America/New_York"
DAY = date(2025, 7, 16)  # a regular Wednesday

BASE = 101.0
OR_HIGH, OR_LOW = 102.0, 100.0
BUY_STOP, SELL_STOP = 103.0, 99.0
EMA_LONG, EMA_SHORT = 100.0, 102.0  # either side of the 09:45 open of 101


def session(day: date = DAY, base: float = BASE, minutes: int = 390) -> pd.DataFrame:
    """A flat RTH session, 09:30 to 15:59 inclusive."""
    idx = pd.date_range(f"{day} 09:30", periods=minutes, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "open": base, "high": base, "low": base, "close": base,
            "volume": 100, "instrument_id": 1,
        },
        index=idx,
        dtype="float64",
    ).assign(volume=100, instrument_id=1)


def put(bars: pd.DataFrame, hhmm: str, **values) -> pd.DataFrame:
    """Overwrite one bar's OHLC fields. Returns ``bars`` for chaining."""
    ts = pd.Timestamp(f"{bars.index[0].date()} {hhmm}", tz=ET)
    for col, value in values.items():
        bars.loc[ts, col] = float(value)
    return bars


def with_range(bars: pd.DataFrame, high: float = OR_HIGH, low: float = OR_LOW):
    """Set the 09:30-09:45 opening range by widening its first bar."""
    return put(bars, "09:30", open=BASE, high=high, low=low, close=BASE)


def touch(bars: pd.DataFrame, hhmm: str, high=None, low=None, open_=None, close=None):
    """A bar that reaches a level without changing the opening range."""
    values = {}
    if open_ is not None:
        values["open"] = open_
    if close is not None:
        values["close"] = close
    if high is not None:
        values["high"] = high
    if low is not None:
        values["low"] = low
    return put(bars, hhmm, **values)


def ema_series(value: float, day: date = DAY) -> pd.Series:
    return pd.Series({day: float(value)}, name="trend_ema")


def run(bars, ema=EMA_LONG, params=None, **kwargs):
    """Generate signals and return ``(strategy, signals)``."""
    trend = None if ema is None else ema_series(ema, bars.index[0].date())
    strat = ORB2(params or ORB2Params(), trend_ema=trend, **kwargs)
    return strat, strat.generate_signals(bars)


def only_trade(signals, bars):
    trades = build_trades(signals, bars)
    assert len(trades) == 1, f"expected exactly one trade, got {len(trades)}"
    return trades.iloc[0]


class TestInterface:
    def test_signals_satisfy_the_contract(self):
        bars = touch(with_range(session()), "10:00", high=104.0)
        _, signals = run(bars)
        validate_signals(signals)
        for col in REQUIRED_SIGNAL_COLUMNS:
            assert signals[col].dtype == bool

    def test_filter_on_requires_an_ema(self):
        with pytest.raises(ValueError, match="needs a trend_ema series"):
            ORB2(ORB2Params(use_trend_filter=True), trend_ema=None)

    def test_filter_off_does_not(self):
        ORB2(ORB2Params(use_trend_filter=False), trend_ema=None)

    def test_rejects_incoherent_params(self):
        with pytest.raises(ValueError, match="must be positive"):
            ORB2Params(stop_points=0)
        with pytest.raises(ValueError, match="must precede"):
            ORB2Params(opening_range_end=time(12, 0))


class TestOpeningRangeAndPlacement:
    def test_range_is_0930_to_0945(self):
        bars = with_range(session())
        # A spike after 09:45 must not enter the range, or the buy stop moves.
        bars = touch(bars, "09:50", high=110.0)
        strat, signals = run(bars)
        assert strat.diagnostics.loc[DAY, "range_high"] == OR_HIGH
        assert strat.diagnostics.loc[DAY, "range_low"] == OR_LOW

    def test_opening_range_bars_are_never_searched_for_a_fill(self):
        """The order is placed at 09:45; earlier bars cannot fill it.

        With the real 1.0-point offset this is true by arithmetic - the buy
        stop sits above the range high, so no opening bar can reach it and the
        test would pass even on a broken fill search. Setting the offset to
        zero removes that accident: the 09:30 bar's high now sits exactly on
        the level, so a search that wrongly included the opening range would
        fill there.
        """
        params = ORB2Params(entry_offset=0.0)
        bars = with_range(session())
        bars = touch(bars, "10:00", high=OR_HIGH)
        _, signals = run(bars, params=params)
        entries = signals.index[signals["entry_long"] | signals["entry_short"]]
        assert len(entries) == 1
        assert entries[0].time() == time(10, 0)

    def test_buy_stop_sits_one_point_above_the_range(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        _, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["entry_price"] == pytest.approx(BUY_STOP)

    def test_a_touch_just_below_the_stop_does_not_fill(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP - 0.25)
        strat, signals = run(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_NO_FILL

    def test_gap_through_the_level_fills_at_the_open(self):
        bars = touch(with_range(session()), "10:00", open_=107.0, high=108.0, close=107.0)
        _, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["entry_price"] == pytest.approx(107.0)

    def test_short_side_mirrors(self):
        bars = touch(with_range(session()), "10:00", low=SELL_STOP)
        _, signals = run(bars, ema=EMA_SHORT)
        trade = only_trade(signals, bars)
        assert trade["direction"] == "short"
        assert trade["entry_price"] == pytest.approx(SELL_STOP)


class TestTrendFilter:
    def test_open_above_ema_goes_long(self):
        bars = touch(touch(with_range(session()), "10:00", high=BUY_STOP), "10:30", low=SELL_STOP)
        _, signals = run(bars, ema=EMA_LONG)
        assert signals["entry_long"].any()
        assert not signals["entry_short"].any()

    def test_open_below_ema_goes_short(self):
        bars = touch(touch(with_range(session()), "10:00", high=BUY_STOP), "10:30", low=SELL_STOP)
        _, signals = run(bars, ema=EMA_SHORT)
        assert signals["entry_short"].any()
        assert not signals["entry_long"].any()

    def test_open_exactly_on_the_ema_does_not_trade(self):
        """Pre-registered boundary: equality is a no-trade."""
        bars = touch(with_range(session()), "10:00", high=BUY_STOP, low=SELL_STOP)
        strat, signals = run(bars, ema=BASE)
        assert not signals[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_ON_EMA

    def test_unseeded_ema_does_not_trade(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        strat = ORB2(ORB2Params(), trend_ema=pd.Series({DAY: float("nan")}))
        signals = strat.generate_signals(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_UNSEEDED

    def test_missing_ema_date_does_not_trade(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        strat = ORB2(ORB2Params(), trend_ema=pd.Series({date(2020, 1, 2): 50.0}))
        signals = strat.generate_signals(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_UNSEEDED


class TestRangeCeiling:
    def test_range_exactly_at_the_ceiling_still_trades(self):
        """The rule skips when height *exceeds* 10; 10.0 itself is allowed."""
        bars = with_range(session(), high=105.0, low=95.0)  # height 10.0
        bars = touch(bars, "10:00", high=106.0)
        strat, signals = run(bars, ema=90.0)
        assert strat.diagnostics.loc[DAY, "range_height"] == pytest.approx(10.0)
        assert signals["entry_long"].any()

    def test_range_over_the_ceiling_skips_the_session(self):
        bars = with_range(session(), high=105.25, low=95.0)  # height 10.25
        bars = touch(bars, "10:00", high=120.0)
        strat, signals = run(bars, ema=90.0)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_WIDE_RANGE


class TestEntryWindow:
    def test_fill_just_before_1130_is_taken(self):
        bars = touch(with_range(session()), "11:29", high=BUY_STOP)
        _, signals = run(bars)
        entries = signals.index[signals["entry_long"]]
        assert len(entries) == 1
        assert entries[0].time() == time(11, 29)

    def test_fill_at_1130_is_cancelled(self):
        """The cancel time itself is already too late."""
        bars = touch(with_range(session()), "11:30", high=BUY_STOP)
        strat, signals = run(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_NO_FILL

    def test_fill_after_1130_is_cancelled(self):
        bars = touch(with_range(session()), "14:00", high=120.0)
        strat, signals = run(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_NO_FILL


class TestBracket:
    def test_target_is_eighteen_points_from_the_fill(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        bars = touch(bars, "10:30", high=BUY_STOP + 18.0)
        _, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["exit_reason"] == "target"
        assert trade["exit_price"] == pytest.approx(BUY_STOP + 18.0)

    def test_stop_is_ten_points_from_the_fill(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        bars = touch(bars, "10:30", low=BUY_STOP - 10.0)
        _, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["exit_reason"] == "stop"
        assert trade["exit_price"] == pytest.approx(BUY_STOP - 10.0)

    def test_stop_wins_when_one_bar_holds_both(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        bars = touch(bars, "10:30", high=BUY_STOP + 18.0, low=BUY_STOP - 10.0)
        _, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["exit_reason"] == "stop"

    def test_the_entry_bar_can_stop_the_trade_out(self):
        """The fill happens inside the bar, so the rest of it still counts."""
        bars = touch(with_range(session()), "10:00", high=BUY_STOP, low=BUY_STOP - 10.0)
        strat, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["exit_reason"] == "stop"
        assert trade["entry_time"] == trade["exit_time"]
        assert bool(strat.diagnostics.loc[DAY, "same_bar_entry_exit"])

    def test_same_bar_flag_is_false_for_an_ordinary_trade(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        bars = touch(bars, "10:30", high=BUY_STOP + 18.0)
        strat, _ = run(bars)
        assert not bool(strat.diagnostics.loc[DAY, "same_bar_entry_exit"])


class TestFlatten:
    def test_untouched_trade_exits_at_1554(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        _, signals = run(bars)
        trade = only_trade(signals, bars)
        assert trade["exit_reason"] == "session_end"
        assert trade["exit_time"].time() == time(15, 54)

    def test_nothing_is_held_past_the_flatten_time(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        _, signals = run(bars)
        exits = signals.index[signals["exit_long"] | signals["exit_short"]]
        assert (exits.time < time(15, 55)).all()

    def test_early_close_pulls_the_flatten_forward(self):
        """Rule 2's 13:00 deadline binds ahead of the strategy's 15:55."""
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        _, signals = run(bars, early_close_dates={DAY})
        trade = only_trade(signals, bars)
        assert trade["exit_time"].time() == time(12, 59)


class TestOneTradePerDay:
    def test_only_one_trade_even_with_many_touches(self):
        bars = with_range(session())
        for hhmm in ("10:00", "10:30", "11:00"):
            bars = touch(bars, hhmm, high=BUY_STOP + 0.5, low=BUY_STOP - 10.0)
        _, signals = run(bars)
        assert int(signals["entry_long"].sum() + signals["entry_short"].sum()) == 1

    def test_no_reentry_after_a_stop(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        bars = touch(bars, "10:05", low=BUY_STOP - 10.0)
        bars = touch(bars, "11:00", high=BUY_STOP + 5.0)
        _, signals = run(bars)
        assert int(signals["entry_long"].sum()) == 1

    def test_roll_days_are_skipped(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        strat, signals = run(bars, roll_dates={DAY})
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_ROLL


class TestFilterOffArm:
    """The OFF arm places both brackets; the first fill wins."""

    def _params(self):
        return ORB2Params(use_trend_filter=False)

    def test_first_fill_wins_when_they_are_on_different_bars(self):
        bars = touch(with_range(session()), "10:00", low=SELL_STOP)
        bars = touch(bars, "10:30", high=BUY_STOP)
        strat = ORB2(self._params())
        signals = strat.generate_signals(bars)
        trade = only_trade(signals, bars)
        assert trade["direction"] == "short"
        assert trade["entry_time"].time() == time(10, 0)

    def test_opposite_order_is_cancelled_on_the_first_fill(self):
        bars = touch(with_range(session()), "10:00", low=SELL_STOP)
        bars = touch(bars, "10:30", high=BUY_STOP)
        strat = ORB2(self._params())
        signals = strat.generate_signals(bars)
        assert int(signals["entry_long"].sum()) == 0
        assert int(signals["entry_short"].sum()) == 1

    def test_both_stops_in_one_bar_resolves_pessimistically(self):
        """Unobservable at this resolution, so take the worse outcome.

        The long would win 18 points here and the short would lose 10, so the
        pre-registered rule must select the short.
        """
        bars = touch(with_range(session()), "10:00",
                     open_=BASE, high=BUY_STOP + 0.5, low=SELL_STOP - 0.5, close=BASE)
        bars = touch(bars, "10:05", high=BUY_STOP + 18.0)
        strat = ORB2(self._params())
        signals = strat.generate_signals(bars)
        trade = only_trade(signals, bars)

        assert bool(strat.diagnostics.loc[DAY, "ambiguous_both_stops"])
        assert trade["direction"] == "short"
        assert trade["exit_reason"] == "stop"

    def test_ambiguity_flag_is_false_otherwise(self):
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        strat = ORB2(self._params())
        strat.generate_signals(bars)
        assert not bool(strat.diagnostics.loc[DAY, "ambiguous_both_stops"])

    def test_filter_off_trades_a_day_the_filter_would_skip(self):
        """The two arms must actually differ, or the A/B measures nothing."""
        bars = touch(with_range(session()), "10:00", high=BUY_STOP)
        _, on_signals = run(bars, ema=EMA_SHORT)  # filter says short; only a long fills
        off = ORB2(self._params())
        off_signals = off.generate_signals(bars)
        assert not on_signals["entry_long"].any()
        assert off_signals["entry_long"].any()


class TestSessionIndependence:
    def test_generating_over_a_span_equals_generating_per_session(self):
        """The property the walk-forward slicing relies on."""
        days = [date(2025, 7, 14), date(2025, 7, 15), date(2025, 7, 16)]
        frames = []
        for day in days:
            bars = touch(with_range(session(day)), "10:00", high=BUY_STOP)
            frames.append(bars)
        combined = pd.concat(frames)
        ema = pd.Series({d: EMA_LONG for d in days})

        whole = ORB2(ORB2Params(), trend_ema=ema).generate_signals(combined)
        for day, frame in zip(days, frames):
            piece = ORB2(ORB2Params(), trend_ema=ema).generate_signals(frame)
            pd.testing.assert_frame_equal(
                whole.loc[whole.index.date == day], piece, check_freq=False
            )


class TestSizeLinearity:
    """Ruling 3: the 1-contract series must be the 4-contract series over four.

    Commission and slippage are both strictly per-contract, so P&L is exactly
    linear in size. The 1-contract run therefore carries no independent
    evidence about the strategy - only the geometry against fixed-dollar eval
    limits differs. If these ever disagree it is a bug, not a finding.
    """

    def _trades(self):
        days = [date(2025, 7, 14), date(2025, 7, 15), date(2025, 7, 16)]
        frames = []
        for i, day in enumerate(days):
            bars = touch(with_range(session(day)), "10:00", high=BUY_STOP)
            # One winner, one loser, one time exit.
            if i == 0:
                bars = touch(bars, "10:30", high=BUY_STOP + 18.0)
            elif i == 1:
                bars = touch(bars, "10:30", low=BUY_STOP - 10.0)
            frames.append(bars)
        bars = pd.concat(frames)
        ema = pd.Series({d: EMA_LONG for d in days})
        signals = ORB2(ORB2Params(), trend_ema=ema).generate_signals(bars)
        return build_trades(signals, bars), bars

    def test_four_contracts_is_exactly_four_times_one(self):
        raw, _ = self._trades()
        assert len(raw) == 3
        one = price_trades(raw, MES, CostModel(), 1)
        four = price_trades(raw, MES, CostModel(), 4)

        for col in ("gross_pnl", "commission", "slippage_cost", "net_pnl"):
            pd.testing.assert_series_equal(
                four[col], one[col] * 4, check_names=False, rtol=1e-12,
            )

    def test_per_trade_and_not_just_in_total(self):
        raw, _ = self._trades()
        one = price_trades(raw, MES, CostModel(), 1)
        four = price_trades(raw, MES, CostModel(), 4)
        for i in range(len(raw)):
            assert four["net_pnl"].iloc[i] == pytest.approx(
                one["net_pnl"].iloc[i] * 4
            ), f"trade {i} is not linear in size"

    def test_prices_and_points_do_not_depend_on_size(self):
        raw, _ = self._trades()
        one = price_trades(raw, MES, CostModel(), 1)
        four = price_trades(raw, MES, CostModel(), 4)
        for col in ("entry_fill", "exit_fill", "gross_points", "net_points"):
            pd.testing.assert_series_equal(four[col], one[col], check_names=False)

    #: Entry 4 was scored at $1.25 a side. Its arithmetic is pinned at that
    #: model by name; the default commission is now Lucid's verified $0.50.
    HISTORICAL = CostModel(commission_per_side=rules.ASSUMED_COMMISSION_PER_SIDE)

    def test_the_bracket_arithmetic_entry_four_fixes(self):
        """+$85.00 a winner, -$55.00 a loser, per contract, after costs."""
        raw, _ = self._trades()
        one = price_trades(raw, MES, self.HISTORICAL, 1)
        by_reason = dict(zip(raw["exit_reason"], one["net_pnl"]))
        assert by_reason["target"] == pytest.approx(85.0)
        assert by_reason["stop"] == pytest.approx(-55.0)

    def test_the_bracket_arithmetic_at_the_verified_commission(self):
        """+$86.50 a winner, -$53.50 a loser at $0.50 a side."""
        raw, _ = self._trades()
        one = price_trades(raw, MES, CostModel(), 1)
        by_reason = dict(zip(raw["exit_reason"], one["net_pnl"]))
        assert by_reason["target"] == pytest.approx(86.5)
        assert by_reason["stop"] == pytest.approx(-53.5)

    def test_break_even_hit_rate_is_the_same_at_both_sizes(self):
        from engine import bracket_break_even

        raw, _ = self._trades()
        for costs, expected in ((self.HISTORICAL, 55.0 / 140.0),
                                (CostModel(), 53.5 / 140.0)):
            rates = []
            for contracts in (1, 4):
                priced = price_trades(raw, MES, costs, contracts)
                by_reason = dict(zip(raw["exit_reason"], priced["net_pnl"]))
                win, loss = by_reason["target"], -by_reason["stop"]
                rates.append(loss / (win + loss))
            assert rates[0] == pytest.approx(rates[1])
            assert rates[0] == pytest.approx(expected)
            assert rates[0] == pytest.approx(bracket_break_even(10.0, 18.0, MES, costs))


class TestPayoutInThePooledBlock:
    """Entry 4's pooled figures carry the payout milestone next to the pass."""

    def _stream(self):
        raw = pd.DataFrame([
            {"entry_time": pd.Timestamp(f"2025-07-{d:02d} 09:46", tz=ET),
             "exit_time": pd.Timestamp(f"2025-07-{d:02d} 11:00", tz=ET),
             "direction": "long", "entry_price": 100.0,
             "exit_price": 100.0 + move, "exit_reason": reason}
            for d, move, reason in ((14, 18.0, "target"), (15, -10.0, "stop"),
                                    (16, 18.0, "target"), (17, -10.0, "stop"))
        ])
        return price_trades(raw, MES, CostModel(), 4)

    def test_pooled_block_reports_payout_under_pass(self):
        from run_orb2 import pooled_block

        text, stats = pooled_block(self._stream(), 4, "fixture", paths=200)
        assert "payout_probability" in stats
        assert stats["payout_probability"] >= stats["pass_probability"]
        assert "Pass probability" in text
        assert "Payout probability" in text
        assert text.index("Payout probability") > text.index("Pass probability")

    def test_pooled_block_states_the_break_even_for_the_run_cost(self):
        """39.29% was a literal in the report. It follows the cost model now,
        so the saved $1.25 reports still read 39.29% when reproduced and a
        run at the verified $0.50 reads 38.21%."""
        from run_orb2 import pooled_block

        historical = CostModel(commission_per_side=rules.ASSUMED_COMMISSION_PER_SIDE)
        old, _ = pooled_block(self._stream(), 4, "fixture", paths=200, costs=historical)
        new, _ = pooled_block(self._stream(), 4, "fixture", paths=200)
        assert "break-even 39.29%" in old
        assert "break-even 38.21%" in new
