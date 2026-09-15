"""Tests for entry 16: prior-session levels, the sweep event, the real and
placebo populations, and the fade arm. Frozen specification at the entry 16
freeze commit.

Everything runs on synthetic one-minute sessions. A sweep is pinned bar by
bar: the cross bar, the rejection bar, the extreme, the decision bar.
"""

from datetime import date, time

import pandas as pd
import pytest

from base import validate_signals
from engine import MES, CostModel, build_trades, price_trades
from sweep import (
    EXIT_FLATTEN, EXIT_STOP, EXIT_TARGET, HORIZON_BARS, LAST_EVENT_BAR, PLACEBO,
    PLACEBO_FRACTION, REAL, STOP_BUFFER_TICKS, SweepFade, SweepParams, find_sweep,
    prior_session_levels, session_events,
)

ET = "America/New_York"


def session(day: str, open_: float = 100.0, high: float = 101.0, low: float = 99.0,
            last_bar: str = "15:59") -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:30", f"{day} {last_bar}", freq="1min", tz=ET)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": open_,
                         "volume": 1, "instrument_id": 1}, index=index)


def at(bars: pd.DataFrame, day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz=ET)


def set_bar(bars: pd.DataFrame, ts: pd.Timestamp, open_=None, high=None, low=None, close=None) -> None:
    for col, v in (("open", open_), ("high", high), ("low", low), ("close", close)):
        if v is not None:
            bars.loc[ts, col] = v


def high_sweep_day(day: str, pdh: float = 110.0, pdl: float = 90.0) -> pd.DataFrame:
    """Opens inside [pdl, pdh]; crosses pdh at 10:05 with a high of 112; keeps a
    high of 113 at 10:06; closes back below pdh at 10:07; drifts down after."""
    bars = session(day, open_=100.0, high=100.5, low=99.5)
    set_bar(bars, at(bars, day, "10:05"), open_=109.0, high=112.0, low=108.5, close=111.0)
    set_bar(bars, at(bars, day, "10:06"), open_=111.0, high=113.0, low=110.5, close=111.5)
    set_bar(bars, at(bars, day, "10:07"), open_=111.5, high=111.6, low=108.0, close=109.0)
    later = bars.index > at(bars, day, "10:07")
    bars.loc[later, ["open", "close"]] = 105.0
    bars.loc[later, "high"] = 105.5
    bars.loc[later, "low"] = 104.5
    return bars


class TestPriorSessionLevels:
    def test_levels_are_the_previous_rth_sessions_high_and_low(self):
        d1 = session("2026-03-02", high=120.0, low=80.0)
        d2 = session("2026-03-03", high=105.0, low=95.0)
        d3 = session("2026-03-04")
        levels = prior_session_levels(pd.concat([d1, d2, d3]))
        assert levels.loc[date(2026, 3, 3), "pdh"] == 120.0
        assert levels.loc[date(2026, 3, 3), "pdl"] == 80.0
        assert levels.loc[date(2026, 3, 4), "pdh"] == 105.0
        assert date(2026, 3, 2) not in levels.index

    def test_only_rth_bars_count(self):
        d1 = session("2026-03-02", high=120.0, low=80.0)
        evening = pd.DataFrame({"open": 100.0, "high": 500.0, "low": 1.0, "close": 100.0,
                                "volume": 1, "instrument_id": 1},
                               index=pd.date_range("2026-03-02 18:00", "2026-03-02 18:05",
                                                   freq="1min", tz=ET))
        d2 = session("2026-03-03")
        levels = prior_session_levels(pd.concat([d1, evening, d2]))
        assert levels.loc[date(2026, 3, 3), "pdh"] == 120.0


class TestFindSweep:
    def test_high_sweep_is_pinned_bar_by_bar(self):
        day = "2026-03-03"
        ev = find_sweep(high_sweep_day(day), high_level=110.0, low_level=90.0)
        assert ev["side"] == "high"
        assert ev["cross_ts"] == at(None, day, "10:05")
        assert ev["reject_ts"] == at(None, day, "10:07")
        assert ev["extreme"] == 113.0
        assert ev["decision_ts"] == at(None, day, "10:08")
        assert ev["level"] == 110.0
        assert ev["depth"] == 3.0

    def test_low_sweep_mirrors(self):
        day = "2026-03-03"
        bars = session(day, open_=100.0, high=100.5, low=99.5)
        set_bar(bars, at(bars, day, "11:00"), open_=91.0, high=91.5, low=88.0, close=89.0)
        set_bar(bars, at(bars, day, "11:01"), open_=89.0, high=92.0, low=88.5, close=91.0)
        ev = find_sweep(bars, high_level=110.0, low_level=90.0)
        assert ev["side"] == "low" and ev["extreme"] == 88.0
        assert ev["reject_ts"] == at(None, day, "11:01")
        assert ev["decision_ts"] == at(None, day, "11:02")

    def test_a_cross_with_no_rejection_is_a_continuation_not_a_sweep(self):
        day = "2026-03-03"
        bars = session(day, open_=100.0, high=100.5, low=99.5)
        later = bars.index >= at(bars, day, "10:05")
        bars.loc[later, ["open", "close"]] = 111.0
        bars.loc[later, "high"] = 112.0
        bars.loc[later, "low"] = 110.5
        ev = find_sweep(bars, high_level=110.0, low_level=90.0)
        assert ev is None or ev.get("side") is None
        cont = find_sweep(bars, high_level=110.0, low_level=90.0, continuations=True)
        assert cont["continuation"] == "high"

    def test_first_rejection_of_either_side_wins(self):
        day = "2026-03-03"
        bars = high_sweep_day(day)   # high rejection at 10:07
        # an earlier low sweep: cross at 09:40, reject at 09:41
        set_bar(bars, at(bars, day, "09:40"), open_=91.0, high=91.5, low=88.0, close=89.0)
        set_bar(bars, at(bars, day, "09:41"), open_=89.0, high=92.0, low=88.5, close=91.0)
        ev = find_sweep(bars, high_level=110.0, low_level=90.0)
        assert ev["side"] == "low"

    def test_crosses_after_the_last_event_bar_are_ignored(self):
        day = "2026-03-03"
        bars = session(day, open_=100.0, high=100.5, low=99.5)
        set_bar(bars, at(bars, day, "15:00"), high=112.0, close=111.0)
        set_bar(bars, at(bars, day, "15:01"), close=109.0)
        assert find_sweep(bars, high_level=110.0, low_level=90.0) is None
        assert LAST_EVENT_BAR == time(14, 59)

    def test_rejection_after_the_last_event_bar_does_not_count(self):
        day = "2026-03-03"
        bars = session(day, open_=100.0, high=100.5, low=99.5)
        set_bar(bars, at(bars, day, "14:58"), high=112.0, close=111.0)
        set_bar(bars, at(bars, day, "14:59"), high=111.5, close=111.0)
        set_bar(bars, at(bars, day, "15:00"), close=109.0)
        assert find_sweep(bars, high_level=110.0, low_level=90.0) is None


class TestSessionEvents:
    def _bars(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)         # sets pdh 110 / pdl 90 for d2
        d2 = high_sweep_day("2026-03-03")                         # real high sweep, then to 105
        d3 = session("2026-03-04", open_=120.0, high=121.0, low=119.0)   # opens above d2's high: ineligible
        return pd.concat([d1, d2, d3])

    def test_real_event_with_sign_adjusted_reversion(self):
        ev = session_events(self._bars(), roll_dates=set(), early_close_dates=set())
        real = ev[(ev["kind"] == REAL) & (ev["date"] == date(2026, 3, 3))]
        assert len(real) == 1
        r = real.iloc[0]
        assert r["side"] == "high" and r["depth"] == 3.0
        # decision open 105.0 at 10:08; 30 bars later open 105.0 -> reversion 0 for the drift;
        # sign-adjusted: down after a high sweep reads positive
        assert r["m30"] == pytest.approx(105.0 - 105.0)
        assert r["m1555"] == pytest.approx(0.0)
        assert r["year"] == 2026

    def test_placebo_event_uses_the_inner_level(self):
        # placebo high level = 110 - 0.25*20 = 105; the drift to 105.0 does not exceed it,
        # but the 10:05 bar (high 112) does, and the 10:07 close 109 is not below 105:
        # the rejection comes when close < 105 — never in this fixture, so no placebo event.
        ev = session_events(self._bars(), roll_dates=set(), early_close_dates=set())
        assert (ev["kind"] == PLACEBO).sum() == 0
        assert PLACEBO_FRACTION == 0.25

    def test_placebo_event_when_price_closes_back_below_the_inner_level(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)
        day = "2026-03-03"
        d2 = session(day, open_=100.0, high=100.5, low=99.5)
        set_bar(d2, at(d2, day, "10:05"), open_=104.0, high=106.0, low=103.5, close=105.5)   # crosses 105
        set_bar(d2, at(d2, day, "10:06"), open_=105.5, high=105.8, low=103.0, close=104.0)   # closes back below
        ev = session_events(pd.concat([d1, d2]), roll_dates=set(), early_close_dates=set())
        p = ev[ev["kind"] == PLACEBO]
        assert len(p) == 1 and p.iloc[0]["level"] == 105.0 and p.iloc[0]["extreme"] == 106.0
        assert (ev["kind"] == REAL).sum() == 0   # never crossed 110

    def test_session_opening_outside_the_band_is_not_eligible(self):
        ev = session_events(self._bars(), roll_dates=set(), early_close_dates=set())
        assert date(2026, 3, 4) not in set(ev["date"])

    def test_skips_roll_and_early_close_sessions(self):
        bars = self._bars()
        ev = session_events(bars, roll_dates={date(2026, 3, 3)}, early_close_dates=set())
        assert date(2026, 3, 3) not in set(ev["date"])

    def test_continuations_are_counted(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)
        day = "2026-03-03"
        d2 = session(day, open_=100.0, high=100.5, low=99.5)
        later = d2.index >= at(d2, day, "10:05")
        d2.loc[later, ["open", "close"]] = 111.0
        d2.loc[later, "high"] = 112.0
        d2.loc[later, "low"] = 110.5
        ev = session_events(pd.concat([d1, d2]), roll_dates=set(), early_close_dates=set())
        cont = ev[ev["kind"] == "continuation"]
        assert len(cont) == 1 and cont.iloc[0]["side"] == "high"

    def test_horizon_is_thirty_bars(self):
        assert HORIZON_BARS == 30


class TestSweepFade:
    def _run(self, bars, buffer_ticks=STOP_BUFFER_TICKS):
        strat = SweepFade(SweepParams(stop_buffer_ticks=buffer_ticks))
        return strat, strat.generate_signals(bars)

    def test_params(self):
        assert SweepParams().stop_buffer_ticks == 4
        assert SweepParams().buffer_points == 1.0
        with pytest.raises(ValueError):
            SweepParams(stop_buffer_ticks=0)

    def test_short_at_the_decision_bar_with_stop_beyond_the_extreme_and_a_one_to_one_target(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)
        d2 = high_sweep_day("2026-03-03")
        strat, sig = self._run(pd.concat([d1, d2]))
        ts = at(None, "2026-03-03", "10:08")
        assert bool(sig.loc[ts, "entry_short"]) is True
        assert sig.loc[ts, "entry_price"] == 105.0
        assert sig.loc[ts, "stop_price"] == 114.0          # extreme 113 + 1 point
        assert sig.loc[ts, "target_price"] == 96.0         # 105 - (114 - 105)
        d = strat.diagnostics.loc[date(2026, 3, 3)]
        assert d["side"] == "high" and bool(d["entered"]) is True

    def test_target_stop_and_flatten_exits(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)
        target = high_sweep_day("2026-03-03")
        set_bar(target, at(None, "2026-03-03", "11:00"), low=95.0)
        _, sig = self._run(pd.concat([d1, target]))
        ex = sig[sig["exit_short"]]
        assert ex["exit_reason"].iloc[0] == EXIT_TARGET and ex["exit_price"].iloc[0] == 96.0

        d1b = session("2026-03-04", high=110.0, low=90.0)
        stop = high_sweep_day("2026-03-05")
        set_bar(stop, at(None, "2026-03-05", "11:00"), high=115.0)
        _, sig2 = self._run(pd.concat([d1b, stop]))
        ex2 = sig2[sig2["exit_short"]]
        assert ex2["exit_reason"].iloc[0] == EXIT_STOP and ex2["exit_price"].iloc[0] == 114.0

        d1c = session("2026-03-09", high=110.0, low=90.0)
        flat = high_sweep_day("2026-03-10")
        _, sig3 = self._run(pd.concat([d1c, flat]))
        ex3 = sig3[sig3["exit_short"]]
        assert ex3["exit_reason"].iloc[0] == EXIT_FLATTEN
        assert ex3.index[0] == at(None, "2026-03-10", "15:55")

    def test_no_trade_on_placebo_only_sessions_and_one_trade_per_session(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)
        day = "2026-03-03"
        d2 = session(day, open_=100.0, high=100.5, low=99.5)
        set_bar(d2, at(d2, day, "10:05"), open_=104.0, high=106.0, low=103.5, close=105.5)
        set_bar(d2, at(d2, day, "10:06"), open_=105.5, high=105.8, low=103.0, close=104.0)
        _, sig = self._run(pd.concat([d1, d2]))
        assert not sig["entry_short"].any() and not sig["entry_long"].any()

    def test_signals_validate_and_price(self):
        d1 = session("2026-03-02", high=110.0, low=90.0)
        d2 = high_sweep_day("2026-03-03")
        set_bar(d2, at(None, "2026-03-03", "11:00"), low=95.0)
        _, sig = self._run(pd.concat([d1, d2]))
        validate_signals(sig)
        trades = price_trades(build_trades(sig, pd.concat([d1, d2])), MES, CostModel(slippage_ticks=0.0), 1)
        assert len(trades) == 1
        assert trades.iloc[0]["direction"] == "short"
        assert trades.iloc[0]["gross_points"] == pytest.approx(9.0)
