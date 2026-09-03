"""Tests for the end-of-day rebalance strategy (hypothesis 2).

The timing is the whole hypothesis, so it gets the most attention: the signal
must come from the bar labelled 15:25 (closing 15:29:59, the price as of
15:30), and the fill must come from the open of the bar labelled 15:35. The
no-lookahead assertion below is the one that matters - it fails if the entry
price is ever knowable from the signal bar.
"""

from datetime import date, time

import pandas as pd
import pytest

from base import REQUIRED_SIGNAL_COLUMNS, validate_signals
from eod_rebalance import (
    ENTRY_BAR, OPEN_BAR, SIGNAL_BAR, EODParams, EODRebalanceDrift, parameter_grid,
)
from orb import resample_bars

ET = "America/New_York"
DAY = "2025-07-16"  # a regular Wednesday


def session_1m(day: str, closes: dict[str, float], base: float = 5000.0,
               last: str = "15:59") -> pd.DataFrame:
    """One session of 1-minute bars, flat at `base` except where overridden.

    `closes` maps HH:MM to the close for that minute; the level then persists
    until the next override, so a session is described by its turning points.
    """
    idx = pd.date_range(f"{day} 09:30", f"{day} {last}", freq="1min", tz=ET)
    px = pd.Series(float("nan"), index=idx, dtype=float)
    px.iloc[0] = base
    for clock, value in closes.items():
        stamp = pd.Timestamp(f"{day} {clock}", tz=ET)
        if stamp in px.index:
            px.loc[stamp] = value
    px = px.ffill()
    open_ = px.shift(1)
    open_.iloc[0] = base
    pair = pd.concat([open_, px], axis=1)
    return pd.DataFrame(
        {
            "open": open_,
            "high": pair.max(axis=1),
            "low": pair.min(axis=1),
            "close": px,
            "volume": 100,
            "instrument_id": 1,
        },
        index=idx,
    )


def signals_for(bars, params=None, **kwargs):
    strat = EODRebalanceDrift(params or EODParams(), **kwargs)
    return strat.generate_signals(bars), resample_bars(bars, 5)


class TestSignalAndEntryTiming:
    def test_signal_comes_from_the_1525_bar_not_the_1530_bar(self):
        """A move that happens at 15:30-15:34 must not count toward the signal.

        Price is flat until 15:30, then jumps 1%. The signal window closes at
        15:29:59, so nothing should fire: the strategy cannot see that move.
        """
        bars = session_1m(DAY, {"15:30": 5050.0})
        sig, _ = signals_for(bars, EODParams(threshold=0.005))
        assert not sig[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()

    def test_move_completed_by_1529_does_fire(self):
        bars = session_1m(DAY, {"10:00": 5050.0})  # +1% by 10:00, held
        sig, _ = signals_for(bars, EODParams(threshold=0.005))
        entries = sig.index[sig["entry_long"]]
        assert len(entries) == 1
        assert entries[0].time() == ENTRY_BAR

    def test_entry_is_the_1535_bar_open(self):
        bars = session_1m(DAY, {"10:00": 5050.0, "15:35": 5060.0})
        sig, bars5 = signals_for(bars, EODParams(threshold=0.005))
        entry_ts = sig.index[sig["entry_long"]][0]
        assert entry_ts.time() == ENTRY_BAR
        # The fill price the engine will use is that bar's open, which is the
        # 15:34 close - not the 15:35 close.
        assert bars5.loc[entry_ts, "open"] == pytest.approx(5050.0)

    def test_no_lookahead_signal_bar_strictly_precedes_entry_bar(self):
        """The explicit assertion: every input to the decision predates the fill.

        The signal bar closes at 15:29:59 and the entry bar opens at 15:35:00,
        so the decision uses only information at least five minutes old.
        """
        bars = session_1m(DAY, {"10:00": 5050.0})
        sig, bars5 = signals_for(bars, EODParams(threshold=0.005))
        entry_ts = sig.index[sig["entry_long"]][0]

        session = bars5[bars5.index.date == pd.Timestamp(DAY).date()]
        signal_ts = session.index[session.index.time == SIGNAL_BAR][0]
        signal_close_time = signal_ts + pd.Timedelta(minutes=5)

        assert signal_ts < entry_ts
        assert signal_close_time <= entry_ts
        assert (entry_ts - signal_ts) == pd.Timedelta(minutes=10)

    def test_signal_uses_the_0930_open_not_the_previous_close(self):
        bars = session_1m(DAY, {"10:00": 5050.0})
        session = resample_bars(bars, 5)
        open_bar = session[session.index.time == OPEN_BAR]
        assert float(open_bar["open"].iloc[0]) == pytest.approx(5000.0)


class TestThresholdGate:
    @pytest.mark.parametrize(
        "close_px, threshold, fires",
        [
            (5030.0, 0.005, True),    # +0.60%
            (5024.0, 0.005, False),   # +0.48%
            (5025.5, 0.005, True),    # +0.51%, just clear of the boundary
            (5060.0, 0.010, True),    # +1.20%
            (5040.0, 0.010, False),   # +0.80%
        ],
    )
    def test_threshold_boundary(self, close_px, threshold, fires):
        bars = session_1m(DAY, {"10:00": close_px})
        sig, _ = signals_for(bars, EODParams(threshold=threshold))
        assert bool(sig["entry_long"].any()) is fires

    def test_the_exact_boundary_is_not_a_meaningful_case(self):
        """A "exactly 0.5%" move is not exactly 0.5% in binary floating point.

        5025/5000 - 1 evaluates to 0.004999999999999893, about 1e-16 below the
        threshold, so it does not fire. The comparison is left exact rather than
        given a tolerance: a difference of one part in 1e16 is roughly 1e-11
        index points, which is economically meaningless either way. Recorded so
        a future reader does not mistake this for an off-by-one.
        """
        assert (5025.0 / 5000.0 - 1.0) < 0.005
        bars = session_1m(DAY, {"10:00": 5025.0})
        sig, _ = signals_for(bars, EODParams(threshold=0.005))
        assert not sig["entry_long"].any()

    def test_direction_follows_the_day(self):
        up = session_1m(DAY, {"10:00": 5050.0})
        down = session_1m(DAY, {"10:00": 4950.0})
        sig_up, _ = signals_for(up, EODParams(threshold=0.005))
        sig_dn, _ = signals_for(down, EODParams(threshold=0.005))
        assert sig_up["entry_long"].sum() == 1 and sig_up["entry_short"].sum() == 0
        assert sig_dn["entry_short"].sum() == 1 and sig_dn["entry_long"].sum() == 0


class TestExits:
    def test_exits_at_the_rth_close_when_no_stop_is_hit(self):
        bars = session_1m(DAY, {"10:00": 5050.0})
        sig, bars5 = signals_for(bars, EODParams(threshold=0.005, stop_pct=None))
        exits = sig.loc[sig["exit_long"]]
        assert len(exits) == 1
        assert exits.index[0].time() == time(15, 55)  # closes 15:59:59
        assert exits.iloc[0]["exit_reason"] == "cash_close"

    def test_stop_is_taken_when_touched(self):
        # +1% by 10:00 so we go long at 5050, then a slide through the stop.
        bars = session_1m(DAY, {"10:00": 5050.0, "15:40": 5000.0})
        sig, _ = signals_for(bars, EODParams(threshold=0.005, stop_pct=0.005))
        exits = sig.loc[sig["exit_long"]]
        assert exits.iloc[0]["exit_reason"] == "stop"
        # stop sits 0.5% below the 5050 entry
        assert float(exits.iloc[0]["exit_price"]) == pytest.approx(5050.0 * 0.995)

    def test_no_stop_means_the_slide_is_ridden_to_the_close(self):
        bars = session_1m(DAY, {"10:00": 5050.0, "15:40": 5000.0})
        sig, _ = signals_for(bars, EODParams(threshold=0.005, stop_pct=None))
        assert sig.loc[sig["exit_long"]].iloc[0]["exit_reason"] == "cash_close"


class TestEarlyCloseSessions:
    """A 13:00 session has no 15:25 or 15:35 bar, so nothing can fire."""

    def test_no_entries_on_an_early_close_session(self):
        bars = session_1m(DAY, {"10:00": 5050.0}, last="12:59")
        early = {pd.Timestamp(DAY).date()}
        sig, _ = signals_for(
            bars, EODParams(threshold=0.005), early_close_dates=early
        )
        assert not sig[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()

    def test_early_close_produces_nothing_even_without_the_calendar(self):
        """The bars themselves are missing, so it holds with no calendar too."""
        bars = session_1m(DAY, {"10:00": 5050.0}, last="12:59")
        sig, _ = signals_for(bars, EODParams(threshold=0.005))
        assert not sig[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()

    def test_a_normal_session_alongside_an_early_close_still_trades(self):
        early_day, normal_day = "2025-07-03", "2025-07-07"
        bars = pd.concat([
            session_1m(early_day, {"10:00": 5050.0}, last="12:59"),
            session_1m(normal_day, {"10:00": 5050.0}),
        ])
        sig, _ = signals_for(
            bars, EODParams(threshold=0.005),
            early_close_dates={pd.Timestamp(early_day).date()},
        )
        entries = sig.index[sig["entry_long"]]
        assert len(entries) == 1
        assert entries[0].date() == pd.Timestamp(normal_day).date()


class TestGuards:
    def test_roll_day_blocks_entries(self):
        bars = session_1m(DAY, {"10:00": 5050.0})
        sig, _ = signals_for(
            bars, EODParams(threshold=0.005),
            roll_dates={pd.Timestamp(DAY).date()},
        )
        assert not sig[list(REQUIRED_SIGNAL_COLUMNS)].to_numpy().any()

    def test_entry_is_inside_the_rule_windows(self):
        """15:35 precedes the 16:20 cutoff and the 16:30 flatten."""
        import rules
        entry = pd.Timestamp(f"{DAY} 15:35", tz=ET)
        assert rules.is_entry_allowed(entry) is True
        assert rules.must_flatten(entry) is False

    def test_signals_pass_interface_validation(self):
        bars = session_1m(DAY, {"10:00": 5050.0})
        sig, _ = signals_for(bars, EODParams(threshold=0.005))
        validate_signals(sig)
        for col in REQUIRED_SIGNAL_COLUMNS:
            assert sig[col].dtype == bool

    def test_one_trade_per_session(self):
        bars = session_1m(DAY, {"10:00": 5050.0})
        sig, _ = signals_for(bars, EODParams(threshold=0.005))
        assert sig["entry_long"].sum() + sig["entry_short"].sum() == 1


class TestGrid:
    def test_grid_is_the_frozen_twelve(self):
        grid = parameter_grid()
        assert len(grid) == 12
        assert {p.threshold for p in grid} == {0.005, 0.0075, 0.010, 0.015}
        assert {p.stop_pct for p in grid} == {0.0025, 0.005, None}

    @pytest.mark.parametrize(
        "kwargs", [{"threshold": 0}, {"threshold": -1}, {"stop_pct": 0},
                   {"stop_pct": -0.01}, {"bar_minutes": 0}],
    )
    def test_invalid_params_rejected(self, kwargs):
        with pytest.raises(ValueError):
            EODParams(**kwargs)
