"""Tests for entry 6, the London breakout of the overnight range.

The test that matters most is :class:`TestRangeSpansTwoCalendarDates`. Entry 6
records that ``rules.session_date`` and the range window have different
definitions and that the code must not assume they agree, so that is asserted
directly rather than trusted.
"""

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import pytest

import rules
from base import REQUIRED_SIGNAL_COLUMNS, validate_signals
from engine import MES, CostModel, build_trades, price_trades
from london import (
    SKIP_FILTERED, SKIP_NO_BREAK, SKIP_ROLL, SKIP_STOP_OVER_BUDGET,
    SKIP_THIN_RANGE, SKIP_WIDE_RANGE, LondonBreakout, LondonParams,
    contracts_for, contracts_for_stop, london_date, resample_continuous,
)
from trend import intraday_ema

ET = "America/New_York"
DAY = date(2025, 7, 16)          # a Wednesday
PREV = date(2025, 7, 15)


def block(day: date, start: str, periods: int, base: float,
          instrument_id: int = 1) -> pd.DataFrame:
    idx = pd.date_range(f"{day} {start}", periods=periods, freq="1min", tz=ET)
    return pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base,
         "volume": 100, "instrument_id": instrument_id},
        index=idx, dtype="float64",
    ).assign(volume=100, instrument_id=instrument_id)


def session(day: date = DAY, base: float = 100.0) -> pd.DataFrame:
    """Evening of the prior day through 10:00 on ``day``, flat at ``base``."""
    evening = block(day - timedelta(days=1), "19:00", 300, base)   # 19:00-23:59
    morning = block(day, "00:00", 601, base)                       # 00:00-10:00
    return pd.concat([evening, morning])


def put(bars: pd.DataFrame, when: str, **values) -> pd.DataFrame:
    ts = pd.Timestamp(when, tz=ET)
    for col, value in values.items():
        bars.loc[ts, col] = float(value)
    return bars


def close_5m(bars, when: str, value: float):
    """Make the 5-minute candle labelled ``when`` close at ``value``.

    A 5-minute candle's close is the close of the *last* one-minute bar in its
    bucket, so setting the bar at the label does nothing to the candle.
    """
    last = pd.Timestamp(when, tz=ET) + pd.Timedelta(minutes=4)
    bars.loc[last, "close"] = float(value)
    bars.loc[last, "high"] = max(float(value), float(bars.loc[last, "high"]))
    bars.loc[last, "low"] = min(float(value), float(bars.loc[last, "low"]))
    return bars


def with_range(bars, high=110.0, low=100.0, day=PREV):
    """Set the overnight range from a bar in the prior evening."""
    return put(bars, f"{day} 20:00", open=100.0, high=high, low=low, close=100.0)


def run(bars, params=None, ema=None, **kwargs):
    p = params or LondonParams(use_trend_filter=False)
    strat = LondonBreakout(p, trend_ema=ema, **kwargs)
    return strat, strat.generate_signals(bars)


class TestLondonDate:
    def test_evening_bars_belong_to_the_next_day(self):
        assert london_date(pd.Timestamp(f"{PREV} 19:00", tz=ET)) == DAY
        assert london_date(pd.Timestamp(f"{PREV} 23:59", tz=ET)) == DAY

    def test_morning_bars_belong_to_their_own_day(self):
        assert london_date(pd.Timestamp(f"{DAY} 00:00", tz=ET)) == DAY
        assert london_date(pd.Timestamp(f"{DAY} 18:59", tz=ET)) == DAY

    def test_the_boundary_is_1900(self):
        assert london_date(pd.Timestamp(f"{DAY} 18:59", tz=ET)) == DAY
        assert london_date(pd.Timestamp(f"{DAY} 19:00", tz=ET)) == DAY + timedelta(days=1)

    def test_it_is_not_rules_session_date(self):
        """The two definitions differ, which is the whole point."""
        ts = pd.Timestamp(f"{PREV} 20:00", tz=ET)
        assert rules.session_date(ts) == PREV
        assert london_date(ts) == DAY
        assert london_date(ts) != rules.session_date(ts)


class TestRangeSpansTwoCalendarDates:
    """Entry 6's explicit requirement, asserted rather than assumed."""

    def test_the_range_uses_the_prior_evenings_bars(self):
        bars = with_range(session(), high=110.0, low=100.0)
        strat, _ = run(bars)
        d = strat.diagnostics.loc[DAY]
        assert d["range_high"] == pytest.approx(110.0)
        assert d["range_low"] == pytest.approx(100.0)

    def test_moving_the_prior_evening_moves_the_range(self):
        base = with_range(session(), high=110.0, low=100.0)
        strat_a, _ = run(base)
        moved = with_range(session(), high=130.0, low=100.0)
        strat_b, _ = run(moved)
        assert strat_a.diagnostics.loc[DAY, "range_high"] == pytest.approx(110.0)
        assert strat_b.diagnostics.loc[DAY, "range_high"] == pytest.approx(130.0)

    def test_bars_after_0255_are_not_in_the_range(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = put(bars, f"{DAY} 03:00", high=500.0)     # inside the entry window
        strat, _ = run(bars)
        assert strat.diagnostics.loc[DAY, "range_high"] == pytest.approx(110.0)

    def test_bars_before_1900_are_not_in_the_range(self):
        bars = with_range(session(), high=110.0, low=100.0)
        early = block(PREV, "17:00", 60, 100.0)
        early = put(early, f"{PREV} 17:30", high=900.0)
        strat, _ = run(pd.concat([early, bars]).sort_index())
        assert strat.diagnostics.loc[DAY, "range_high"] == pytest.approx(110.0)

    def test_a_session_with_no_evening_is_skipped(self):
        strat, signals = run(block(DAY, "00:00", 601, 100.0))
        assert not signals[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_THIN_RANGE


class TestSizing:
    @pytest.mark.parametrize("range_pts,expected", [
        (2.0, 5), (8.0, 5), (10.0, 4), (13.5, 2), (20.0, 2),
        (20.25, 1), (40.0, 1), (41.0, 1),
    ])
    def test_contract_arithmetic(self, range_pts, expected):
        assert contracts_for(range_pts, LondonParams()) == expected

    def test_never_exceeds_the_position_cap(self):
        for pts in np.arange(0.25, 41.0, 0.25):
            n = contracts_for(float(pts), LondonParams())
            assert 1 <= n <= rules.POSITION_CAP

    def test_range_over_forty_points_is_skipped(self):
        bars = with_range(session(), high=141.0, low=100.0)   # 41 points
        bars = close_5m(bars, f"{DAY} 03:00", 200.0)
        strat, signals = run(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_WIDE_RANGE

    def test_exactly_forty_points_still_trades(self):
        bars = with_range(session(), high=140.0, low=100.0)   # 40 points
        bars = close_5m(bars, f"{DAY} 03:00", 145.0)
        strat, signals = run(bars)
        assert signals["entry_long"].any()
        assert int(strat.diagnostics.loc[DAY, "contracts"]) == 1

    def test_size_is_published_on_the_entry_bar(self):
        bars = with_range(session(), high=110.0, low=100.0)   # 10 points -> 4
        bars = close_5m(bars, f"{DAY} 03:00", 115.0)
        _, signals = run(bars)
        entry = signals.index[signals["entry_long"]][0]
        assert int(signals.loc[entry, "contracts"]) == 4


class TestBreakAndEntry:
    def _breaking(self, close=115.0, at="03:00"):
        bars = with_range(session(), high=110.0, low=100.0)
        return close_5m(bars, f"{DAY} {at}", close)

    def test_a_close_outside_the_range_triggers(self):
        _, signals = run(self._breaking())
        assert signals["entry_long"].any()

    def test_the_trade_is_taken_at_the_next_candle_open(self):
        bars = self._breaking(at="03:00")
        bars = put(bars, f"{DAY} 03:05", open=117.0)
        _, signals = run(bars)
        entry = signals.index[signals["entry_long"]][0]
        assert entry.time() == time(3, 5)
        assert float(signals.loc[entry, "entry_price"]) == pytest.approx(117.0)

    def test_a_touch_without_a_close_outside_does_not_trigger(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = put(bars, f"{DAY} 03:00", high=120.0)
        bars = close_5m(bars, f"{DAY} 03:00", 105.0)
        strat, signals = run(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_NO_BREAK

    def test_no_break_before_0300(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 02:00", 120.0)
        strat, signals = run(bars)
        # 02:00 is inside the range window, so it redefines the range instead.
        assert not signals["entry_long"].any()

    def test_no_break_after_0500(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 05:00", 200.0)
        strat, signals = run(bars)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_NO_BREAK

    def test_a_trigger_at_0455_is_honoured_though_it_fills_at_0500(self):
        """The window governs the trigger, not the fill."""
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 04:55", 115.0)
        _, signals = run(bars)
        entry = signals.index[signals["entry_long"]][0]
        assert entry.time() == time(5, 0)

    def test_first_break_only_and_one_trade_per_day(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 115.0)
        bars = close_5m(bars, f"{DAY} 04:00", 95.0)   # opposite side later
        _, signals = run(bars)
        assert int(signals["entry_long"].sum() + signals["entry_short"].sum()) == 1

    def test_short_side_mirrors(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 95.0)
        _, signals = run(bars)
        assert signals["entry_short"].any()

    def test_roll_days_are_skipped_on_the_trade_date(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 115.0)
        strat, signals = run(bars, roll_dates={DAY})
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_ROLL


class TestBracket:
    def _entered(self, target_multiple=1.0):
        # Base 105 sits inside the 100-110 range, so a flat bar touches
        # neither the stop at 100 nor the target at 121.
        bars = with_range(session(base=105.0), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 115.0)
        bars = put(bars, f"{DAY} 03:05", open=111.0)
        return bars, LondonParams(use_trend_filter=False,
                                  target_multiple=target_multiple)

    def test_stop_is_the_far_side_of_the_range(self):
        bars, p = self._entered()
        _, signals = run(bars, p)
        entry = signals.index[signals["entry_long"]][0]
        assert float(signals.loc[entry, "stop_price"]) == pytest.approx(100.0)

    def test_target_is_one_range_height_in_the_1x_arm(self):
        bars, p = self._entered(1.0)
        _, signals = run(bars, p)
        entry = signals.index[signals["entry_long"]][0]
        assert float(signals.loc[entry, "target_price"]) == pytest.approx(121.0)

    def test_target_is_two_range_heights_in_the_2x_arm(self):
        bars, p = self._entered(2.0)
        _, signals = run(bars, p)
        entry = signals.index[signals["entry_long"]][0]
        assert float(signals.loc[entry, "target_price"]) == pytest.approx(131.0)

    def test_the_overshoot_is_recorded(self):
        bars, p = self._entered()
        strat, _ = run(bars, p)
        assert float(strat.diagnostics.loc[DAY, "overshoot"]) == pytest.approx(1.0)

    def test_the_stop_is_further_than_the_range_height(self):
        """Entry 6 keeps this deliberately; it must be visible."""
        bars, p = self._entered()
        strat, _ = run(bars, p)
        d = strat.diagnostics.loc[DAY]
        assert float(d["stop_distance"]) > float(d["range_pts"])
        assert float(d["risk_dollars"]) > 200.0

    def test_stop_wins_when_one_bar_holds_both(self):
        bars, p = self._entered()
        bars = put(bars, f"{DAY} 03:10", high=125.0, low=99.0)
        _, signals = run(bars, p)
        trades = build_trades(signals, bars)
        assert trades.iloc[0]["exit_reason"] == "stop"

    def test_flatten_lands_on_the_0925_bar_open(self):
        bars, p = self._entered()
        bars = put(bars, f"{DAY} 09:25", open=107.0)
        _, signals = run(bars, p)
        trades = build_trades(signals, bars)
        row = trades.iloc[0]
        assert row["exit_reason"] == "flatten_0925"
        assert row["exit_time"].time() == time(9, 25)
        assert row["exit_price"] == pytest.approx(107.0)

    def test_nothing_is_held_past_0925(self):
        bars, p = self._entered()
        _, signals = run(bars, p)
        exits = signals.index[signals["exit_long"] | signals["exit_short"]]
        assert (exits.time <= time(9, 25)).all()


class TestTrendFilter:
    def _bars(self, close=115.0):
        bars = with_range(session(), high=110.0, low=100.0)
        return close_5m(bars, f"{DAY} 03:00", close)

    def test_a_long_break_below_the_ema_is_blocked(self):
        bars = self._bars(115.0)
        ema = pd.Series({pd.Timestamp(f"{DAY} 03:00", tz=ET): 500.0})
        strat, signals = run(bars, LondonParams(use_trend_filter=True), ema=ema)
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_FILTERED

    def test_a_long_break_above_the_ema_is_taken(self):
        bars = self._bars(115.0)
        ema = pd.Series({pd.Timestamp(f"{DAY} 03:00", tz=ET): 50.0})
        _, signals = run(bars, LondonParams(use_trend_filter=True), ema=ema)
        assert signals["entry_long"].any()

    def test_filter_on_requires_an_ema(self):
        with pytest.raises(ValueError, match="needs a trend_ema"):
            LondonBreakout(LondonParams(use_trend_filter=True), trend_ema=None)


class TestIntradayEma:
    def _bars(self, n=1200):
        idx = pd.date_range("2025-07-14 00:00", periods=n, freq="1min", tz=ET)
        close = pd.Series(np.arange(n, dtype=float) + 100.0, index=idx)
        return pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": 100}, index=idx)

    def test_it_does_not_filter_to_rth(self):
        """Entry 6 needs a value at 03:00, which RTH filtering would remove."""
        out = intraday_ema(self._bars(), 5, 10)
        assert any(ts.time() < time(9, 30) for ts in out.dropna().index)

    def test_a_bar_never_sees_its_own_close(self):
        bars = self._bars()
        before = intraday_ema(bars, 5, 10)
        target = before.dropna().index[50]
        bumped = bars.copy()
        mask = (bumped.index >= target) & (bumped.index < target + pd.Timedelta("5min"))
        for col in ("open", "high", "low", "close"):
            bumped.loc[mask, col] += 500.0
        after = intraday_ema(bumped, 5, 10)
        assert after.loc[target] == pytest.approx(before.loc[target])

    def test_a_later_bar_does_move(self):
        bars = self._bars()
        before = intraday_ema(bars, 5, 10)
        idx = before.dropna().index
        target, later = idx[50], idx[51]
        bumped = bars.copy()
        mask = (bumped.index >= target) & (bumped.index < target + pd.Timedelta("5min"))
        for col in ("open", "high", "low", "close"):
            bumped.loc[mask, col] += 500.0
        after = intraday_ema(bumped, 5, 10)
        assert after.loc[later] != pytest.approx(before.loc[later])

    def test_resample_continuous_does_not_restart_at_sessions(self):
        bars = pd.concat([block(PREV, "19:00", 30, 100.0),
                          block(DAY, "00:00", 30, 100.0)])
        five = resample_continuous(bars, 5)
        assert len(five) == 12
        assert five.index[0].time() == time(19, 0)


class TestInterface:
    def test_signals_satisfy_the_contract(self):
        bars = with_range(session(), high=110.0, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 115.0)
        _, signals = run(bars)
        validate_signals(signals)
        for col in REQUIRED_SIGNAL_COLUMNS:
            assert signals[col].dtype == bool

    def test_rejects_incoherent_params(self):
        with pytest.raises(ValueError, match="target_multiple must be positive"):
            LondonParams(target_multiple=0)
        with pytest.raises(ValueError, match="position cap"):
            LondonParams(max_contracts=99)

    def test_rules_permit_an_0300_entry(self):
        """Recorded in entry 6; asserted here so a rule change surfaces it."""
        ts = pd.Timestamp(f"{DAY} 03:00", tz=ET)
        assert rules.is_entry_allowed(ts)
        assert not rules.must_flatten(ts)
        assert not rules.must_flatten(pd.Timestamp(f"{DAY} 09:25", tz=ET))

    def test_empty_bars(self):
        empty = pd.DataFrame(
            {c: [] for c in ("open", "high", "low", "close", "volume")},
            index=pd.DatetimeIndex([], tz=ET))
        _, signals = run(empty)
        assert signals.empty


# ---------------------------------------------------------------------------
# Entry 10: the same signal held through the US session
# ---------------------------------------------------------------------------


def full_day(day: date = DAY, base: float = 100.0) -> pd.DataFrame:
    """Evening of the prior day through 16:05 on ``day``, flat at ``base``."""
    evening = block(day - timedelta(days=1), "19:00", 300, base)   # 19:00-23:59
    trading = block(day, "00:00", 966, base)                       # 00:00-16:05
    return pd.concat([evening, trading])


def entered_full_day(**params):
    """Range 100-110, break at 03:00, filled 111 at 03:05; flat at 105 after.

    Flat at 105 the bars touch neither the stop at 100 nor the target at 121,
    so the trade lives until whatever flatten the params say.
    """
    bars = with_range(full_day(base=105.0), high=110.0, low=100.0)
    bars = close_5m(bars, f"{DAY} 03:00", 115.0)
    bars = put(bars, f"{DAY} 03:05", open=111.0)
    return bars, LondonParams(use_trend_filter=False, **params)


class TestDefaultsAreEntrySix:
    """Entries 6 and 7 are frozen; the defaults must still describe them."""

    def test_default_flatten_and_sizing_are_entry_sixs(self):
        p = LondonParams()
        assert p.flatten_time == time(9, 25)
        assert p.early_close_flatten_time is None
        assert p.size_on == "range"

    def test_default_flatten_reason_is_unchanged(self):
        bars, p = entered_full_day()
        _, signals = run(bars, p)
        trades = build_trades(signals, bars)
        assert trades.iloc[0]["exit_reason"] == "flatten_0925"
        assert trades.iloc[0]["exit_time"].time() == time(9, 25)


class TestFlattenThroughTheUSSession:
    def test_flatten_lands_on_the_1555_bar_open(self):
        bars, p = entered_full_day(flatten_time=time(15, 55))
        bars = put(bars, f"{DAY} 15:55", open=108.0)
        _, signals = run(bars, p)
        row = build_trades(signals, bars).iloc[0]
        assert row["exit_reason"] == "flatten_1555"
        assert row["exit_time"].time() == time(15, 55)
        assert row["exit_price"] == pytest.approx(108.0)

    def test_a_position_open_at_0925_is_held(self):
        bars, p = entered_full_day(flatten_time=time(15, 55))
        _, signals = run(bars, p)
        exits = signals.index[signals["exit_long"] | signals["exit_short"]]
        assert len(exits) == 1
        assert exits[0].time() == time(15, 55)

    def test_nothing_is_held_past_1555(self):
        bars, p = entered_full_day(flatten_time=time(15, 55))
        _, signals = run(bars, p)
        exits = signals.index[signals["exit_long"] | signals["exit_short"]]
        assert (exits.time <= time(15, 55)).all()

    def test_the_us_session_can_resolve_the_trade(self):
        """A stop touched at 10:30 is taken at 10:30 - that is the mechanism."""
        bars, p = entered_full_day(flatten_time=time(15, 55))
        bars = put(bars, f"{DAY} 10:30", low=99.0)
        _, signals = run(bars, p)
        row = build_trades(signals, bars).iloc[0]
        assert row["exit_reason"] == "stop"
        assert row["exit_time"].time() == time(10, 30)

    def test_early_close_day_flattens_at_the_configured_time(self):
        bars, p = entered_full_day(flatten_time=time(15, 55),
                                   early_close_flatten_time=time(12, 55))
        bars = put(bars, f"{DAY} 12:55", open=106.0)
        _, signals = run(bars, p, early_close_dates={DAY})
        row = build_trades(signals, bars).iloc[0]
        assert row["exit_reason"] == "flatten_1255"
        assert row["exit_time"].time() == time(12, 55)
        assert row["exit_price"] == pytest.approx(106.0)

    def test_early_close_time_is_ignored_on_a_normal_day(self):
        bars, p = entered_full_day(flatten_time=time(15, 55),
                                   early_close_flatten_time=time(12, 55))
        _, signals = run(bars, p)
        row = build_trades(signals, bars).iloc[0]
        assert row["exit_time"].time() == time(15, 55)

    def test_without_a_configured_early_time_the_rules_deadline_binds(self):
        """Rule 2 backstop: no position may live past the exchange close."""
        bars, p = entered_full_day(flatten_time=time(15, 55))
        _, signals = run(bars, p, early_close_dates={DAY})
        row = build_trades(signals, bars).iloc[0]
        assert row["exit_time"].time() <= rules.EARLY_SESSION_CLOSE
        assert not rules.must_flatten(row["exit_time"] - pd.Timedelta(minutes=1),
                                      early_close_dates={DAY})

    def test_early_flatten_must_precede_the_early_close(self):
        with pytest.raises(ValueError, match="early"):
            LondonParams(flatten_time=time(15, 55),
                         early_close_flatten_time=rules.EARLY_SESSION_CLOSE)

    def test_rules_permit_1555_and_force_1300_on_an_early_close(self):
        """Recorded in entry 10; asserted so a rule change surfaces it."""
        assert not rules.must_flatten(pd.Timestamp(f"{DAY} 15:55", tz=ET))
        assert not rules.must_flatten(pd.Timestamp(f"{DAY} 12:55", tz=ET),
                                      early_close_dates={DAY})
        assert rules.must_flatten(pd.Timestamp(f"{DAY} 13:00", tz=ET),
                                  early_close_dates={DAY})


class TestStopBasedSizing:
    """The log's standing rule for new strategies: size off the realised stop,
    capped by the daily loss limit, and skip when one contract is over budget."""

    @pytest.mark.parametrize("stop_pts,expected", [
        (2.0, 5), (8.0, 5), (10.0, 4), (11.0, 3), (13.5, 2), (20.0, 2),
        (20.25, 1), (40.0, 1), (40.25, 0), (41.0, 0),
    ])
    def test_contract_arithmetic(self, stop_pts, expected):
        """Zero means the session is skipped: one contract already risks more
        than the budget."""
        assert contracts_for_stop(stop_pts, LondonParams()) == expected

    def test_the_daily_loss_limit_caps_a_larger_budget(self):
        """A $1,000 budget on a 30-point stop wants 6 contracts, the cap allows
        5, and 5 x 30 x $5 = $750 breaches the $400 limit: 2 is the most that
        fits."""
        n = contracts_for_stop(30.0, LondonParams(risk_dollars=1000.0))
        assert n == 2
        assert n * 30.0 * 5.0 <= rules.DAILY_LOSS_LIMIT

    def test_never_exceeds_the_cap_or_the_limit(self):
        p = LondonParams(risk_dollars=1000.0)
        for pts in np.arange(0.25, 45.0, 0.25):
            n = contracts_for_stop(float(pts), p)
            assert 0 <= n <= rules.POSITION_CAP
            assert n * float(pts) * p.point_value <= rules.DAILY_LOSS_LIMIT

    def test_size_on_stop_uses_the_realised_stop_not_the_range(self):
        """Range 10, overshoot 1: the stop is 11 points away. Range-based
        sizing says 4 contracts; the realised stop says 3, risking $165."""
        bars, p = entered_full_day(size_on="stop")
        strat, signals = run(bars, p)
        entry = signals.index[signals["entry_long"]][0]
        assert int(signals.loc[entry, "contracts"]) == 3
        d = strat.diagnostics.loc[DAY]
        assert float(d["stop_distance"]) == pytest.approx(11.0)
        assert float(d["risk_dollars"]) == pytest.approx(165.0)
        assert float(d["risk_dollars"]) <= p.risk_dollars

    def test_a_stop_over_the_budget_skips_the_session(self):
        """A 39.5-point range passes the 40-point cap; a 1.5-point overshoot
        puts the realised stop at 41 points, $205 on one contract: skipped."""
        bars = with_range(full_day(base=120.0), high=139.5, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 145.0)
        bars = put(bars, f"{DAY} 03:05", open=141.0)
        strat, signals = run(bars, LondonParams(use_trend_filter=False,
                                                size_on="stop"))
        assert not signals["entry_long"].any()
        assert strat.diagnostics.loc[DAY, "skipped_reason"] == SKIP_STOP_OVER_BUDGET

    def test_range_based_sizing_takes_the_same_session(self):
        """The same bars under entry 6's rule trade 1 contract, risking $205 -
        the breach entry 6's addendum measured."""
        bars = with_range(full_day(base=120.0), high=139.5, low=100.0)
        bars = close_5m(bars, f"{DAY} 03:00", 145.0)
        bars = put(bars, f"{DAY} 03:05", open=141.0)
        strat, signals = run(bars, LondonParams(use_trend_filter=False))
        assert signals["entry_long"].any()
        assert float(strat.diagnostics.loc[DAY, "risk_dollars"]) == pytest.approx(205.0)

    def test_size_on_is_validated(self):
        with pytest.raises(ValueError, match="size_on"):
            LondonParams(size_on="guess")


class TestEntryTenParams:
    def test_the_frozen_configuration_is_constructible(self):
        p = LondonParams(target_multiple=1.0, use_trend_filter=True,
                         flatten_time=time(15, 55),
                         early_close_flatten_time=time(12, 55), size_on="stop")
        assert p.flatten_time == time(15, 55)
        assert p.early_close_flatten_time == time(12, 55)
        assert p.size_on == "stop"
