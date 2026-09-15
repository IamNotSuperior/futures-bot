"""Tests for entry 15: the quarterly-expiry calendar, the opening-move
population and the 10:00 fade arm. Frozen specification at the entry 15
freeze commit.

Monthly expiry days are excluded from both populations; only the four
settlement Fridays a year (or the Thursday before, when the cash market is
shut) are the treatment, and the same other Fridays as entry 14 are the
control.
"""

from datetime import date, time

import pandas as pd
import pytest

from base import validate_signals
from engine import MES, CostModel, build_trades, price_trades
from opex import CONTROL_FRIDAY, CONTROL_OTHER, EXIT_FLATTEN, EXIT_TARGET
from quarterly import (
    DECISION_BAR, MONTHLY_EXPIRY, QUARTERLY, QuarterlyFade, QuarterlyParams,
    quarterly_expiry_days, session_calendar, session_moves,
)

ET = "America/New_York"


def session_bars(day: str, last_bar: str = "15:59", open_: float = 100.0,
                 high: float = 101.0, low: float = 99.0) -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:30", f"{day} {last_bar}", freq="1min", tz=ET)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": open_,
                         "volume": 1, "instrument_id": 1}, index=index)


def shaped_session(day: str, open_: float, at_1000: float, at_1555: float,
                   pad: float = 0.5) -> pd.DataFrame:
    """Flat at ``open_`` until 09:59, ``at_1000`` from 10:00, ``at_1555`` from 15:55."""
    bars = session_bars(day, open_=open_, high=open_ + pad, low=open_ - pad)
    mid = bars.index.time >= time(10, 0)
    bars.loc[mid, ["open", "close"]] = at_1000
    bars.loc[mid, "high"] = at_1000 + pad
    bars.loc[mid, "low"] = at_1000 - pad
    late = bars.index.time >= time(15, 55)
    bars.loc[late, ["open", "close"]] = at_1555
    bars.loc[late, "high"] = at_1555 + pad
    bars.loc[late, "low"] = at_1555 - pad
    return bars


class TestCalendar:
    def test_quarterly_days_are_the_settlement_months_only(self):
        sessions = [date(2026, 1, 16), date(2026, 2, 20), date(2026, 3, 20), date(2026, 6, 19),
                    date(2026, 6, 18), date(2026, 9, 18), date(2026, 12, 18)]
        # June 2026: the 19th is a Globex-only holiday session; expiry is the 18th.
        out = quarterly_expiry_days(sessions, early_close_dates={date(2026, 6, 19)})
        assert out == {date(2026, 3, 20), date(2026, 6, 18), date(2026, 9, 18), date(2026, 12, 18)}

    def test_labels(self):
        bars = pd.concat([
            session_bars("2026-02-20"),   # monthly expiry
            session_bars("2026-03-18"),   # Wednesday
            session_bars("2026-03-20"),   # quarterly expiry
            session_bars("2026-03-27"),   # Friday control
        ])
        cal = session_calendar(bars, roll_dates=set(), early_close_dates=set()).set_index("date")
        assert cal.loc[date(2026, 2, 20), "label"] == MONTHLY_EXPIRY
        assert cal.loc[date(2026, 3, 20), "label"] == QUARTERLY
        assert cal.loc[date(2026, 3, 27), "label"] == CONTROL_FRIDAY
        assert cal.loc[date(2026, 3, 18), "label"] == CONTROL_OTHER

    def test_eligibility_reasons_carry_over(self):
        bars = pd.concat([session_bars("2026-03-20", last_bar="12:59"), session_bars("2026-03-27")])
        cal = session_calendar(bars, roll_dates={date(2026, 3, 27)},
                               early_close_dates={date(2026, 3, 20)}).set_index("date")
        assert cal.loc[date(2026, 3, 20), "skipped_reason"] == "early_close"
        assert cal.loc[date(2026, 3, 27), "skipped_reason"] == "roll_day"

    def test_decision_bar_is_ten_oclock(self):
        assert DECISION_BAR == time(10, 0)


class TestSessionMoves:
    def test_opening_move_afternoon_move_and_reversal(self):
        up_then_back = shaped_session("2026-03-20", 100.0, 110.0, 104.0)   # o=+10, a=-6, reversal
        up_then_on = shaped_session("2026-03-27", 100.0, 105.0, 112.0)     # o=+5, a=+7, no reversal
        pop = session_moves(pd.concat([up_then_back, up_then_on]), roll_dates=set(),
                            early_close_dates=set()).set_index("date")
        q = pop.loc[date(2026, 3, 20)]
        assert q["label"] == QUARTERLY
        assert q["opening_move"] == 10.0 and q["afternoon_move"] == -6.0
        assert bool(q["reversal"]) is True
        assert q["abs_open"] == 10.0
        f = pop.loc[date(2026, 3, 27)]
        assert bool(f["reversal"]) is False

    def test_zero_opening_move_counts_as_reversal(self):
        flat = shaped_session("2026-03-27", 100.0, 100.0, 103.0)
        pop = session_moves(flat, roll_dates=set(), early_close_dates=set())
        assert bool(pop["reversal"].iloc[0]) is True

    def test_monthly_expiry_days_are_present_but_labelled_out(self):
        bars = pd.concat([shaped_session("2026-02-20", 100.0, 101.0, 102.0),
                          shaped_session("2026-03-27", 100.0, 101.0, 102.0)])
        pop = session_moves(bars, roll_dates=set(), early_close_dates=set()).set_index("date")
        assert pop.loc[date(2026, 2, 20), "label"] == MONTHLY_EXPIRY


class TestQuarterlyFade:
    def _signals(self, bars, sd_open=10.0, k=1.0, s=1.0):
        strat = QuarterlyFade(QuarterlyParams(sd_open=sd_open, k=k, s=s))
        return strat, strat.generate_signals(bars)

    def test_params_thresholds(self):
        p = QuarterlyParams(sd_open=12.0, k=1.0, s=1.0)
        assert p.threshold_points == 12.0 and p.stop_points == 12.0

    def test_short_at_ten_on_a_quarterly_day_beyond_the_threshold(self):
        bars = shaped_session("2026-03-20", 100.0, 112.0, 112.0)   # o=+12 >= 10
        strat, sig = self._signals(bars)
        ts = pd.Timestamp("2026-03-20 10:00", tz=ET)
        assert bool(sig.loc[ts, "entry_short"]) is True
        assert sig.loc[ts, "entry_price"] == 112.0
        assert sig.loc[ts, "target_price"] == 100.0
        assert sig.loc[ts, "stop_price"] == 122.0
        assert strat.diagnostics.loc[date(2026, 3, 20), "direction"] == "short"

    def test_inside_threshold_is_no_trade(self):
        bars = shaped_session("2026-03-20", 100.0, 105.0, 105.0)   # o=+5 < 10
        strat, sig = self._signals(bars)
        assert not sig["entry_short"].any() and not sig["entry_long"].any()

    def test_monthly_expiry_and_control_fridays_never_trade(self):
        bars = pd.concat([shaped_session("2026-02-20", 100.0, 115.0, 115.0),   # monthly expiry
                          shaped_session("2026-03-27", 100.0, 115.0, 115.0)])  # Friday control
        strat, sig = self._signals(bars)
        assert not sig["entry_short"].any()
        assert list(strat.diagnostics.index) == []

    def test_target_hit_and_flatten_are_inherited(self):
        hit = shaped_session("2026-03-20", 100.0, 112.0, 112.0)
        hit.loc[hit.index.time == time(10, 1), "low"] = 99.5   # touches the target at 100
        strat, sig = self._signals(hit)
        ex = sig[sig["exit_short"]]
        assert ex["exit_reason"].iloc[0] == EXIT_TARGET and ex["exit_price"].iloc[0] == 100.0

        stay = shaped_session("2026-06-19", 100.0, 112.0, 112.0)   # 2026-06-19 is a session here (no early close passed)
        strat2, sig2 = self._signals(stay)
        ex2 = sig2[sig2["exit_short"]]
        assert ex2["exit_reason"].iloc[0] == EXIT_FLATTEN
        assert ex2.index[0] == pd.Timestamp("2026-06-19 15:55", tz=ET)

    def test_engine_prices_a_one_contract_short(self):
        hit = shaped_session("2026-03-20", 100.0, 112.0, 112.0)
        hit.loc[hit.index.time == time(10, 1), "low"] = 99.5
        _, sig = self._signals(hit)
        validate_signals(sig)
        trades = price_trades(build_trades(sig, hit), MES, CostModel(slippage_ticks=0.0), 1)
        assert len(trades) == 1
        assert trades.iloc[0]["gross_points"] == pytest.approx(12.0)
        assert trades.iloc[0]["duration_seconds"] == 60.0
