import numpy as np
import pandas as pd
import pytest

import orborb_flat_1030 as strat

REQUIRED = ("entry_long", "entry_short", "exit_long", "exit_short")
FLAT_TIME = pd.Timestamp("2000-01-01 10:30").time()


# -- helpers ---------------------------------------------------------------


def bars_from(rows, date="2024-01-02"):
    """Build 1-minute OHLCV bars starting 09:30 on ``date`` from (o,h,l,c) rows."""
    idx = pd.date_range(f"{date} 09:30", periods=len(rows), freq="1min")
    df = pd.DataFrame(
        np.asarray(rows, dtype=float), columns=["open", "high", "low", "close"], index=idx
    )
    df["volume"] = 100.0
    return df


def flat(price, n=1, spread=0.5):
    return [(price, price + spread, price - spread, price)] * n


def pad_to(rows, n_bars, price):
    """Extend ``rows`` with quiet bars until it has ``n_bars`` rows."""
    return rows + flat(price, n_bars - len(rows))


def opening_rows(high=5005.0, low=5000.0):
    """15 bars, 09:30-09:44, establishing a range of ``high``/``low``."""
    rows = [(5002.0, high, low, 5003.0)]
    rows += flat(5003.0, 14)
    return rows


def long_day(date="2024-01-02"):
    """Range 5000-5005, upside touch at 09:50, entry 09:51, target hit 09:55."""
    rows = opening_rows()
    rows += flat(5003.0, 5)                              # 09:45-09:49 quiet
    rows.append((5003.0, 5007.0, 5002.5, 5006.5))        # 09:50 touches 5006
    rows.append((5006.5, 5007.0, 5006.0, 5006.5))        # 09:51 entry bar
    rows += flat(5010.0, 3)                              # 09:52-09:54
    rows.append((5010.0, 5025.0, 5009.0, 5024.0))        # 09:55 through target
    return bars_from(pad_to(rows, 66, 5020.0), date=date)


def ts(hhmm, date="2024-01-02"):
    return pd.Timestamp(f"{date} {hhmm}")


# -- interface -------------------------------------------------------------


def test_signals_frame_satisfies_the_interface():
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(long_day())

    for col in REQUIRED:
        assert col in signals.columns
        assert signals[col].dtype == bool
    assert not (signals["entry_long"] & signals["entry_short"]).any()
    assert len(signals) == len(strat.rth_minute_bars(long_day()))


def test_name_matches_the_submission():
    assert strat.OpeningRangeBreakoutFlat1030.name == "orborb_flat_1030"


def test_empty_bars_give_an_empty_valid_frame():
    empty = pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ["open", "high", "low", "close", "volume"]},
        index=pd.DatetimeIndex([]),
    )
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(empty)
    assert len(signals) == 0
    for col in REQUIRED:
        assert col in signals.columns


# -- lookahead -------------------------------------------------------------


def test_no_lookahead_a_future_spike_cannot_create_an_earlier_signal():
    """Bars are identical through 09:49. In one frame 09:50 breaks out.

    If the strategy peeked at bar 09:50 while deciding bar 09:50, the two
    frames would differ at 09:50. They must not: the earliest the breakout
    can be acted on is the 09:51 open.
    """
    quiet = pad_to(opening_rows() + flat(5003.0, 5) + flat(5003.0, 1), 66, 5003.0)
    spike = opening_rows() + flat(5003.0, 5)
    spike.append((5003.0, 5007.0, 5002.5, 5006.5))       # 09:50 breaks out
    spike = pad_to(spike, 66, 5006.5)

    s = strat.OpeningRangeBreakoutFlat1030()
    a = s.generate_signals(bars_from(quiet))
    b = s.generate_signals(bars_from(spike))

    upto = a.index <= ts("09:50")
    pd.testing.assert_series_equal(
        a.loc[upto, "entry_long"], b.loc[upto, "entry_long"]
    )
    pd.testing.assert_series_equal(
        a.loc[upto, "entry_short"], b.loc[upto, "entry_short"]
    )
    assert not a["entry_long"].any()
    assert b.loc[ts("09:51"), "entry_long"]


def test_no_lookahead_truncating_the_future_leaves_earlier_signals_unchanged():
    full = long_day()
    prefix = full.iloc[:22]  # ends on the 09:51 entry bar

    s = strat.OpeningRangeBreakoutFlat1030()
    a = s.generate_signals(full)
    b = s.generate_signals(prefix)

    common = b.index
    pd.testing.assert_series_equal(
        a.loc[common, "entry_long"], b.loc[common, "entry_long"]
    )
    pd.testing.assert_series_equal(
        a.loc[common, "entry_short"], b.loc[common, "entry_short"]
    )
    assert b.loc[ts("09:51"), "entry_long"]


def test_entry_is_marked_on_the_bar_after_the_touch():
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(long_day())
    entries = signals.index[signals["entry_long"]]
    assert list(entries) == [ts("09:51")]
    assert not signals.loc[ts("09:50"), "entry_long"]


# -- mechanism -------------------------------------------------------------


def test_long_brackets_are_fixed_points_from_the_fill():
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(long_day())
    entry = ts("09:51")
    assert np.isclose(float(signals.loc[entry, "stop_price"]), 5006.5 - 10.0)
    assert np.isclose(float(signals.loc[entry, "target_price"]), 5006.5 + 18.0)
    assert np.isclose(float(signals.loc[entry, "opening_range_height"]), 5.0)


def test_target_exit_is_marked_at_the_target_level():
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(long_day())
    exits = signals.index[signals["exit_long"]]
    assert list(exits) == [ts("09:55")]
    assert signals.loc[ts("09:55"), "exit_reason"] == "target"
    assert np.isclose(float(signals.loc[ts("09:55"), "exit_price"]), 5024.5)


def test_short_side_mirrors_the_long_side():
    rows = opening_rows()
    rows += flat(5003.0, 5)
    rows.append((5003.0, 5003.5, 4998.0, 4998.5))        # 09:50 touches 4999
    rows.append((4998.5, 4999.0, 4998.0, 4998.5))        # 09:51 entry bar
    rows += flat(4990.0, 3)
    rows.append((4990.0, 4991.0, 4979.0, 4980.0))        # 09:55 through target
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 66, 4985.0))
    )

    assert list(signals.index[signals["entry_short"]]) == [ts("09:51")]
    assert not signals["entry_long"].any()
    assert np.isclose(float(signals.loc[ts("09:51"), "stop_price"]), 4998.5 + 10.0)
    assert np.isclose(float(signals.loc[ts("09:51"), "target_price"]), 4998.5 - 18.0)
    assert list(signals.index[signals["exit_short"]]) == [ts("09:55")]


def test_stop_wins_when_one_bar_holds_both_stop_and_target():
    rows = opening_rows()
    rows += flat(5003.0, 5)
    rows.append((5003.0, 5007.0, 5002.5, 5006.5))        # 09:50 touch
    rows.append((5006.5, 5007.0, 5006.0, 5006.5))        # 09:51 entry
    rows.append((5006.5, 5030.0, 4990.0, 5000.0))        # 09:52 spans both
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 66, 5000.0))
    )
    assert signals.loc[ts("09:52"), "exit_long"]
    assert signals.loc[ts("09:52"), "exit_reason"] == "stop"


def test_wide_opening_range_is_skipped():
    rows = opening_rows(high=5012.0, low=5000.0)         # 12-point range
    rows += flat(5006.0, 5)
    rows.append((5006.0, 5014.0, 5005.0, 5013.5))        # would trigger
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 66, 5013.0))
    )
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_one_trade_per_day_even_if_the_other_side_breaks_later():
    rows = opening_rows()
    rows += flat(5003.0, 5)
    rows.append((5003.0, 5007.0, 5002.5, 5006.5))        # 09:50 upside touch
    rows.append((5006.5, 5007.0, 5006.0, 5006.5))        # 09:51 entry
    rows += flat(5010.0, 3)
    rows.append((5010.0, 5025.0, 5009.0, 5024.0))        # 09:55 target
    rows += flat(4990.0, 5)                              # collapse below 4999
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 66, 4990.0))
    )
    assert int(signals["entry_long"].sum()) == 1
    assert int(signals["entry_short"].sum()) == 0


def test_a_bar_touching_both_triggers_stands_the_day_aside():
    rows = opening_rows()
    rows += flat(5003.0, 5)
    rows.append((5003.0, 5007.0, 4998.0, 5000.0))        # 09:50 touches both
    rows.append((5000.0, 5008.0, 4997.0, 5007.0))
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 66, 5005.0))
    )
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_unfilled_orders_are_cancelled_at_1030():
    rows = opening_rows()
    rows += flat(5003.0, 55)                             # quiet through 10:39
    rows.append((5003.0, 5020.0, 5002.0, 5019.0))        # 10:40 breakout, too late
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 80, 5019.0))
    )
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_no_entry_is_marked_at_or_after_the_flat_time():
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(long_day())
    late = signals.index.time >= FLAT_TIME
    assert not signals.loc[late, "entry_long"].any()
    assert not signals.loc[late, "entry_short"].any()


def test_open_position_is_flattened_at_1030():
    rows = opening_rows()
    rows += flat(5003.0, 5)
    rows.append((5003.0, 5007.0, 5002.5, 5006.5))        # 09:50 touch
    rows.append((5006.5, 5007.0, 5006.0, 5006.5))        # 09:51 entry
    signals = strat.OpeningRangeBreakoutFlat1030().generate_signals(
        bars_from(pad_to(rows, 66, 5006.5))              # never reaches 10 or 18
    )
    exits = signals.index[signals["exit_long"]]
    assert list(exits) == [ts("10:30")]
    assert signals.loc[ts("10:30"), "exit_reason"] == "time_flat"


def test_sessions_are_independent():
    day1 = long_day("2024-01-02")
    day2 = long_day("2024-01-03")
    s = strat.OpeningRangeBreakoutFlat1030()

    combined = s.generate_signals(pd.concat([day1, day2]))
    separate = pd.concat([s.generate_signals(day1), s.generate_signals(day2)])

    for col in REQUIRED:
        pd.testing.assert_series_equal(combined[col], separate[col])


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        strat.ORBFlat1030Params(opening_range_minutes=0)
    with pytest.raises(ValueError):
        strat.ORBFlat1030Params(stop_points=0)
    with pytest.raises(ValueError):
        strat.ORBFlat1030Params(target_points=-1)
    with pytest.raises(ValueError):
        strat.ORBFlat1030Params(max_range_points=0)