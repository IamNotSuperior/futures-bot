import numpy as np
import pandas as pd
import pytest

import orb_full_day_test as strat

NY = "America/New_York"
REQUIRED = ("entry_long", "entry_short", "exit_long", "exit_short")


def flat_session(date="2024-03-05", price=5000.0, n=390):
    """A featureless RTH day: 09:30 to 15:59 inclusive, 1-minute bars."""
    idx = pd.date_range(f"{date} 09:30", periods=n, freq="1min", tz=NY)
    return pd.DataFrame(
        {
            "open": float(price),
            "high": float(price) + 0.25,
            "low": float(price) - 0.25,
            "close": float(price),
            "volume": 100.0,
        },
        index=idx,
    )


def ts_of(df, hhmm):
    return pd.Timestamp(f"{df.index[0].date()} {hhmm}", tz=NY)


def set_bar(df, hhmm, open_=None, high=None, low=None, close=None):
    ts = ts_of(df, hhmm)
    if open_ is not None:
        df.at[ts, "open"] = float(open_)
    if high is not None:
        df.at[ts, "high"] = float(high)
    if low is not None:
        df.at[ts, "low"] = float(low)
    if close is not None:
        df.at[ts, "close"] = float(close)
    return df


def signals(bars, **kwargs):
    return strat.ORBFullDayTest(**kwargs).generate_signals(bars)


def long_break_day(date="2024-03-05"):
    """Range 4999.75-5000.25; long trigger 5001.25 touched on the 09:50 bar."""
    bars = flat_session(date=date)
    set_bar(bars, "09:50", high=5002.0, close=5001.5)
    set_bar(bars, "09:51", open_=5002.0, high=5002.5, low=5001.0, close=5001.5)
    return bars


# -- interface ------------------------------------------------------------


def test_signals_frame_satisfies_interface():
    bars = pd.concat(
        [
            long_break_day("2024-03-05"),
            set_bar(flat_session("2024-03-06"), "10:10", low=4995.0, close=4996.0),
            flat_session("2024-03-07"),
        ]
    )
    sig = signals(bars)

    for col in REQUIRED:
        assert col in sig.columns
        assert sig[col].dtype == bool
    assert not (sig["entry_long"] & sig["entry_short"]).any()
    assert len(sig) == len(bars)
    assert sig.index.equals(bars.index)


def test_name_and_contracts_are_transcribed():
    s = strat.ORBFullDayTest()
    assert s.name == "orb_full_day_test"
    assert s.contracts == 4
    assert isinstance(s.contracts, int)


def test_empty_bars_give_empty_valid_frame():
    empty = pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([], tz=NY),
    )
    sig = signals(empty)
    assert len(sig) == 0
    for col in REQUIRED:
        assert col in sig.columns


# -- lookahead ------------------------------------------------------------


def test_no_lookahead_future_bars_do_not_change_earlier_signals():
    """Everything after the entry bar is rewritten; earlier signals must not move."""
    base_bars = long_break_day()
    quiet = base_bars.copy()

    violent = base_bars.copy()
    cut = ts_of(violent, "09:52")
    after = violent.index >= cut
    violent.loc[after, "open"] = 4900.0
    violent.loc[after, "high"] = 4901.0
    violent.loc[after, "low"] = 4850.0
    violent.loc[after, "close"] = 4860.0

    sig_quiet = signals(quiet)
    sig_violent = signals(violent)

    head = sig_quiet.index < cut
    for col in REQUIRED:
        pd.testing.assert_series_equal(
            sig_quiet.loc[head, col], sig_violent.loc[head, col], check_names=False
        )
    # And the entry itself is present in both, unchanged in timing.
    assert sig_quiet["entry_long"].sum() == 1
    assert sig_violent["entry_long"].sum() == 1
    assert sig_quiet.index[sig_quiet["entry_long"]][0] == ts_of(quiet, "09:51")
    assert sig_violent.index[sig_violent["entry_long"]][0] == ts_of(violent, "09:51")


def test_no_lookahead_truncating_after_entry_keeps_the_entry():
    bars = long_break_day()
    full = signals(bars)
    truncated = signals(bars.loc[: ts_of(bars, "09:51")])

    entry_ts = ts_of(bars, "09:51")
    assert bool(full.loc[entry_ts, "entry_long"])
    assert bool(truncated.loc[entry_ts, "entry_long"])
    assert truncated["entry_long"].sum() == 1
    assert truncated["entry_short"].sum() == 0


def test_trigger_bar_itself_never_carries_the_entry():
    """A trigger touched inside bar T is acted on at T+1, never at T."""
    bars = flat_session()
    set_bar(bars, "09:51", high=5002.0, close=5001.5)
    set_bar(bars, "09:52", open_=5002.0, high=5002.5, low=5001.0, close=5001.5)

    sig = signals(bars)
    assert not bool(sig.loc[ts_of(bars, "09:51"), "entry_long"])
    assert bool(sig.loc[ts_of(bars, "09:52"), "entry_long"])
    assert not sig.loc[: ts_of(bars, "09:51"), "entry_long"].any()


# -- mechanism ------------------------------------------------------------


def test_long_break_enters_at_next_open_with_fixed_stop_and_target():
    bars = long_break_day()
    sig = signals(bars)
    entry_ts = ts_of(bars, "09:51")

    assert bool(sig.loc[entry_ts, "entry_long"])
    assert not sig["entry_short"].any()
    assert float(sig.loc[entry_ts, "stop_price"]) == pytest.approx(5002.0 - 10.0)
    assert float(sig.loc[entry_ts, "target_price"]) == pytest.approx(5002.0 + 18.0)
    assert float(sig.loc[entry_ts, "trigger_price"]) == pytest.approx(5001.25)


def test_short_break_enters_short():
    bars = flat_session()
    set_bar(bars, "09:50", low=4998.0, close=4998.5)
    set_bar(bars, "09:51", open_=4998.0, high=4999.0, low=4997.5, close=4998.0)

    sig = signals(bars)
    entry_ts = ts_of(bars, "09:51")
    assert bool(sig.loc[entry_ts, "entry_short"])
    assert not sig["entry_long"].any()
    assert float(sig.loc[entry_ts, "stop_price"]) == pytest.approx(4998.0 + 10.0)
    assert float(sig.loc[entry_ts, "target_price"]) == pytest.approx(4998.0 - 18.0)


def test_target_exit_is_marked():
    bars = long_break_day()
    set_bar(bars, "10:05", high=5021.0, close=5020.0)
    sig = signals(bars)

    exits = sig.index[sig["exit_long"]]
    assert len(exits) == 1
    assert exits[0] == ts_of(bars, "10:05")
    assert sig.loc[exits[0], "exit_reason"] == "target"
    assert float(sig.loc[exits[0], "exit_price"]) == pytest.approx(5020.0)


def test_stop_exit_is_marked_and_only_one_trade_per_day():
    bars = long_break_day()
    set_bar(bars, "10:05", low=4990.0, close=4991.0)
    # A second, later breakout that must be ignored.
    set_bar(bars, "11:00", high=5030.0, close=5029.0)
    set_bar(bars, "11:01", open_=5030.0, high=5031.0, low=5029.0, close=5030.0)

    sig = signals(bars)
    assert sig["entry_long"].sum() == 1
    assert sig["entry_short"].sum() == 0
    exits = sig.index[sig["exit_long"]]
    assert len(exits) == 1
    assert sig.loc[exits[0], "exit_reason"] == "stop"


def test_stop_and_target_in_one_bar_books_the_stop():
    bars = long_break_day()
    set_bar(bars, "10:05", high=5021.0, low=4990.0, close=5000.0)
    sig = signals(bars)
    exits = sig.index[sig["exit_long"]]
    assert len(exits) == 1
    assert sig.loc[exits[0], "exit_reason"] == "stop"


def test_open_position_gets_a_session_end_exit():
    bars = long_break_day()  # nothing hits stop or target afterwards
    sig = signals(bars)
    exits = sig.index[sig["exit_long"]]
    assert len(exits) == 1
    assert sig.loc[exits[0], "exit_reason"] == "session_end"


def test_wide_opening_range_skips_the_day():
    bars = flat_session()
    set_bar(bars, "09:30", high=5012.0, low=5000.0, close=5006.0)  # 12-point range
    set_bar(bars, "10:00", high=5020.0, close=5019.0)
    set_bar(bars, "10:01", open_=5020.0, high=5021.0, low=5019.0, close=5020.0)

    sig = signals(bars)
    assert not sig["entry_long"].any()
    assert not sig["entry_short"].any()


def test_both_triggers_in_one_bar_stands_down():
    bars = flat_session()
    set_bar(bars, "09:50", high=5002.0, low=4995.0, close=5000.0)
    sig = signals(bars)
    assert not sig["entry_long"].any()
    assert not sig["entry_short"].any()


def test_no_entry_from_a_trigger_after_the_cancel_time():
    bars = flat_session()
    set_bar(bars, "12:00", high=5002.0, close=5001.5)
    set_bar(bars, "12:01", open_=5002.0, high=5002.5, low=5001.0, close=5001.5)
    sig = signals(bars)
    assert not sig["entry_long"].any()
    assert not sig["entry_short"].any()


def test_cancel_time_boundary():
    live = flat_session()
    set_bar(live, "11:29", high=5002.0, close=5001.5)
    set_bar(live, "11:30", open_=5002.0, high=5002.5, low=5001.0, close=5001.5)
    sig_live = signals(live)
    assert bool(sig_live.loc[ts_of(live, "11:30"), "entry_long"])

    dead = flat_session()
    set_bar(dead, "11:30", high=5002.0, close=5001.5)
    set_bar(dead, "11:31", open_=5002.0, high=5002.5, low=5001.0, close=5001.5)
    sig_dead = signals(dead)
    assert not sig_dead["entry_long"].any()


def test_quiet_day_produces_no_signals():
    sig = signals(flat_session())
    for col in REQUIRED:
        assert not sig[col].any()


def test_sessions_are_independent():
    day_a = long_break_day("2024-03-05")
    day_b = long_break_day("2024-03-06")
    joint = signals(pd.concat([day_a, day_b]))
    apart = pd.concat([signals(day_a), signals(day_b)])

    for col in REQUIRED:
        pd.testing.assert_series_equal(joint[col], apart[col], check_names=False)
    assert joint["entry_long"].sum() == 2


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        strat.ORBFullDayParams(opening_range_minutes=0)
    with pytest.raises(ValueError):
        strat.ORBFullDayParams(stop_points=0)
    with pytest.raises(ValueError):
        strat.ORBFullDayParams(target_points=-1)
    with pytest.raises(ValueError):
        strat.ORBFullDayParams(max_range_points=0)
    with pytest.raises(ValueError):
        strat.ORBFullDayParams(entry_window_end_minute=9 * 60 + 40)