"""Tests for the daily trend filter.

The test that matters here is :class:`TestNoLookahead`. Everything else checks
arithmetic; that class checks the property the module exists to guarantee.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trend import (
    EMA_PERIOD, daily_closes, ema, largest_roll_gap, trend_filter,
)

ET = "America/New_York"


def session_bars(day: date, closes: list[float], instrument_id: int = 1) -> pd.DataFrame:
    """1-minute RTH bars for one session, ending at the given closes."""
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="1min", tz=ET)
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
            "instrument_id": instrument_id,
        },
        index=idx,
    )


def multi_session_bars(closes_per_day: list[list[float]], start=date(2025, 1, 6),
                       ids: list[int] | None = None):
    """Consecutive weekday sessions, one entry per day in ``closes_per_day``.

    ``ids`` gives each session's contract, so a roll can be modelled the way
    the real series carries one: a change in ``instrument_id``.
    """
    frames = []
    day = start
    for i, closes in enumerate(closes_per_day):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        frames.append(session_bars(day, closes, 1 if ids is None else ids[i]))
        day += timedelta(days=1)
    return pd.concat(frames)


def ramp(n: int, start: float = 100.0, step: float = 1.0) -> list[list[float]]:
    """``n`` sessions whose closes rise by ``step`` a day."""
    return [[start + i * step] * 10 for i in range(n)]


class TestDailyCloses:
    def test_close_is_the_last_bar_of_the_session(self):
        bars = multi_session_bars([[100.0, 101.0, 105.0], [200.0, 199.0]])
        closes = daily_closes(bars)
        assert list(closes) == [105.0, 199.0]

    def test_indexed_by_session_date(self):
        bars = multi_session_bars(ramp(3))
        closes = daily_closes(bars)
        assert list(closes.index) == [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]

    def test_empty_bars_give_an_empty_series(self):
        empty = pd.DataFrame(
            {c: [] for c in ("open", "high", "low", "close", "volume")},
            index=pd.DatetimeIndex([], tz=ET),
        )
        assert daily_closes(empty).empty


class TestEma:
    def test_seed_is_the_simple_mean_of_the_first_period(self):
        values = np.arange(1.0, 61.0)
        out = ema(values, period=50)
        assert out[49] == pytest.approx(values[:50].mean())

    def test_nothing_before_the_seed(self):
        out = ema(np.arange(1.0, 61.0), period=50)
        assert np.isnan(out[:49]).all()

    def test_recursion_uses_alpha_two_over_period_plus_one(self):
        values = np.arange(1.0, 61.0)
        out = ema(values, period=50)
        alpha = 2.0 / 51.0
        expected = alpha * values[50] + (1 - alpha) * out[49]
        assert out[50] == pytest.approx(expected)

    def test_flat_series_gives_the_flat_value(self):
        out = ema(np.full(80, 42.0), period=50)
        assert out[79] == pytest.approx(42.0)

    def test_too_short_a_series_is_all_nan(self):
        assert np.isnan(ema(np.arange(10.0), period=50)).all()

    def test_rejects_a_nonpositive_period(self):
        with pytest.raises(ValueError, match="period must be positive"):
            ema(np.arange(10.0), period=0)


class TestTrendFilter:
    def test_value_for_a_session_is_the_ema_through_the_previous_close(self):
        bars = multi_session_bars(ramp(EMA_PERIOD + 5))
        closes = daily_closes(bars)
        filt = trend_filter(bars)
        raw = ema(closes.to_numpy(), EMA_PERIOD)
        # Session at position i reads the EMA computed through position i-1.
        assert filt.iloc[EMA_PERIOD] == pytest.approx(raw[EMA_PERIOD - 1])
        assert filt.iloc[-1] == pytest.approx(raw[-2])

    def test_unseeded_sessions_are_nan(self):
        bars = multi_session_bars(ramp(EMA_PERIOD + 5))
        filt = trend_filter(bars)
        # Positions 0..period-1 cannot have a value: the seed needs `period`
        # completed closes and the shift costs one more.
        assert filt.iloc[:EMA_PERIOD].isna().all()
        assert filt.iloc[EMA_PERIOD:].notna().all()

    def test_first_tradeable_session_is_the_fifty_first(self):
        bars = multi_session_bars(ramp(EMA_PERIOD + 5))
        filt = trend_filter(bars)
        assert filt.notna().idxmax() == filt.index[EMA_PERIOD]

    def test_rising_series_sits_above_its_ema(self):
        bars = multi_session_bars(ramp(EMA_PERIOD + 20))
        closes = daily_closes(bars)
        filt = trend_filter(bars)
        live = filt.dropna()
        assert (closes.loc[live.index] > live).all()

    def test_empty_bars_give_an_empty_series(self):
        empty = pd.DataFrame(
            {c: [] for c in ("open", "high", "low", "close", "volume")},
            index=pd.DatetimeIndex([], tz=ET),
        )
        assert trend_filter(empty).empty


class TestNoLookahead:
    """The filter value for a session must not depend on that session's bars.

    This is the guarantee the whole module exists for. Each test perturbs a
    session and asserts that the session's *own* filter value is unmoved; the
    last one walks every tradeable session so a lookahead that only appears at
    one point in the series cannot hide.
    """

    def _bars(self):
        return multi_session_bars(ramp(EMA_PERIOD + 10))

    def test_moving_a_sessions_close_does_not_move_its_own_filter_value(self):
        bars = self._bars()
        target = sorted(set(bars.index.date))[EMA_PERIOD + 5]
        before = trend_filter(bars).loc[target]

        bumped = bars.copy()
        mask = bumped.index.date == target
        for col in ("open", "high", "low", "close"):
            bumped.loc[mask, col] += 500.0

        assert trend_filter(bumped).loc[target] == pytest.approx(before)

    def test_a_later_session_does_move(self):
        """The mutation must actually be capable of moving something.

        Without this, a filter that returned a constant would pass the test
        above and look correct.
        """
        bars = self._bars()
        days = sorted(set(bars.index.date))
        target = days[EMA_PERIOD + 5]
        after = days[EMA_PERIOD + 6]
        before = trend_filter(bars).loc[after]

        bumped = bars.copy()
        mask = bumped.index.date == target
        for col in ("open", "high", "low", "close"):
            bumped.loc[mask, col] += 500.0

        assert trend_filter(bumped).loc[after] != pytest.approx(before)

    def test_no_tradeable_session_depends_on_its_own_bars(self):
        bars = self._bars()
        baseline = trend_filter(bars)
        tradeable = baseline.dropna().index

        for target in tradeable:
            bumped = bars.copy()
            mask = bumped.index.date == target
            for col in ("open", "high", "low", "close"):
                bumped.loc[mask, col] += 500.0
            assert trend_filter(bumped).loc[target] == pytest.approx(
                baseline.loc[target]
            ), f"filter value for {target} moved when {target}'s own bars changed"

    def test_truncating_the_future_does_not_change_the_past(self):
        """Values must not depend on bars that had not happened yet."""
        bars = self._bars()
        full = trend_filter(bars)
        days = sorted(set(bars.index.date))
        cut = days[EMA_PERIOD + 4]

        truncated = trend_filter(bars[bars.index.date <= cut])
        shared = truncated.dropna().index
        assert len(shared)
        pd.testing.assert_series_equal(truncated.loc[shared], full.loc[shared])


class TestLargestRollGap:
    """The measure keys on the contract change, not on a roll-date label.

    On the real series 12 of 29 roll dates fall on a Sunday Globex reopen with
    no RTH session, so the contaminated close-to-close step actually lands on
    the following Monday. Reading the change off ``instrument_id`` catches it
    wherever it falls.
    """

    def _bars(self, closes, ids):
        return multi_session_bars(closes, ids=ids)

    def test_reports_the_biggest_step_on_a_contract_change(self):
        # ramp() closes are 100, 101, 102, ... so day 3 jumping to 140 is a
        # 38-point step away from day 2's close of 102.
        closes = ramp(5)
        closes[3] = [140.0] * 10
        bars = self._bars(closes, ids=[1, 1, 1, 2, 2])
        days = sorted(set(bars.index.date))

        out = largest_roll_gap(bars, set())
        assert out["date"] == days[3]
        assert out["points"] == pytest.approx(38.0)
        assert out["n_rolls"] == 1

    def test_picks_the_largest_of_several(self):
        closes = ramp(6)
        closes[2] = [110.0] * 10
        closes[4] = [200.0] * 10
        bars = self._bars(closes, ids=[1, 1, 2, 2, 3, 3])
        days = sorted(set(bars.index.date))

        out = largest_roll_gap(bars, set())
        assert out["date"] == days[4]
        assert out["n_rolls"] == 2

    def test_no_contract_change_reports_zero(self):
        bars = multi_session_bars(ramp(5))
        assert largest_roll_gap(bars, set())["points"] == 0.0

    def test_compares_roll_steps_against_ordinary_ones(self):
        """The maximum alone is misleading; the medians are the honest read."""
        closes = ramp(8)
        closes[3] = [140.0] * 10
        bars = self._bars(closes, ids=[1, 1, 1, 2, 2, 2, 2, 2])
        out = largest_roll_gap(bars, set())
        assert out["median_ordinary"] == pytest.approx(1.0)
        assert out["median_roll"] > out["median_ordinary"]

    def test_the_step_is_found_where_it_lands_not_where_it_is_labelled(self):
        """A roll whose label has no session still has its step counted.

        This is the Sunday-reopen case that the roll-date version missed: the
        contract changes over a non-session day, so the contaminated step shows
        up on the next session and must still be attributed to the roll.
        """
        closes = ramp(6)
        closes[4] = [150.0] * 10          # the jump lands on session index 4
        bars = self._bars(closes, ids=[1, 1, 1, 1, 2, 2])
        days = sorted(set(bars.index.date))

        # A caller passing an unrelated roll-date label must not change this.
        out = largest_roll_gap(bars, {date(1999, 1, 1)})
        assert out["date"] == days[4]
        assert out["points"] == pytest.approx(47.0)

    def test_falls_back_to_roll_dates_without_an_instrument_column(self):
        bars = multi_session_bars(ramp(5)).drop(columns=["instrument_id"])
        days = sorted(set(bars.index.date))
        out = largest_roll_gap(bars, {days[3]})
        assert out["n_rolls"] == 1
