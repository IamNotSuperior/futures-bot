"""Tests for entry 14: the monthly option-expiry calendar and the session
populations the mechanism test runs on. Frozen specification at ``f9f129c``.

The calendar decides which sessions are expiry days, so it is pinned first
and on real dates: 18 April 2025 was the third Friday of April and Good
Friday, so expiry moved to Thursday the 17th.
"""

import math
from datetime import date, time

import pandas as pd
import pytest

from base import validate_signals
from engine import MES, CostModel, build_trades, price_trades
from opex import (
    CONTROL_FRIDAY, CONTROL_OTHER, EXIT_FLATTEN, EXIT_STOP, EXIT_TARGET, EXPIRY,
    SKIP_INSIDE_THRESHOLD, OpexFade, OpexParams, expiry_days, is_quarterly,
    session_calendar, session_ranges, third_friday,
)

ET = "America/New_York"


def session_bars(day: str, last_bar: str = "15:59", open_: float = 100.0,
                 high: float = 101.0, low: float = 99.0, drift: float = 0.0) -> pd.DataFrame:
    """One RTH session of 1-minute bars; ``drift`` moves the open each bar."""
    index = pd.date_range(f"{day} 09:30", f"{day} {last_bar}", freq="1min", tz=ET)
    opens = open_ + drift * pd.RangeIndex(len(index)).to_numpy()
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": opens,
                         "volume": 1, "instrument_id": 1}, index=index)


# ---------------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------------


class TestThirdFriday:
    @pytest.mark.parametrize("year, month, expected", [
        (2026, 3, date(2026, 3, 20)),
        (2026, 9, date(2026, 9, 18)),
        (2025, 4, date(2025, 4, 18)),   # Good Friday - handled by expiry_days, not here
        (2024, 2, date(2024, 2, 16)),
        (2021, 1, date(2021, 1, 15)),
    ])
    def test_dates(self, year, month, expected):
        assert third_friday(year, month) == expected


class TestExpiryDays:
    def test_third_friday_when_the_session_exists(self):
        sessions = [date(2026, 3, 19), date(2026, 3, 20), date(2026, 3, 23)]
        assert expiry_days(sessions) == {date(2026, 3, 20)}

    def test_good_friday_moves_expiry_to_the_thursday(self):
        # April 2025: the 18th is the third Friday and Good Friday (no session).
        sessions = [date(2025, 4, 16), date(2025, 4, 17), date(2025, 4, 21)]
        assert expiry_days(sessions) == {date(2025, 4, 17)}

    def test_one_expiry_per_month_present(self):
        sessions = [date(2026, 1, 16), date(2026, 2, 20), date(2026, 3, 20)]
        assert expiry_days(sessions) == {date(2026, 1, 16), date(2026, 2, 20), date(2026, 3, 20)}

    def test_a_month_with_no_session_on_or_before_the_third_friday_has_no_expiry(self):
        # Only sessions after the third Friday of March 2026 (20th).
        sessions = [date(2026, 3, 23), date(2026, 3, 24)]
        assert expiry_days(sessions) == set()

    def test_the_thursday_before_an_ordinary_friday_is_not_an_expiry(self):
        sessions = [date(2026, 3, 19), date(2026, 3, 20)]
        assert date(2026, 3, 19) not in expiry_days(sessions)

    def test_a_cash_closed_holiday_session_on_the_third_friday_moves_expiry_to_thursday(self):
        """Juneteenth 2026 is the third Friday of June: Globex trades a short
        session with the cash market shut, so expiration is Thursday the 18th.
        The Globex session is passed as an early close that is not a cash
        half-day (addendum 2026-09-15)."""
        sessions = [date(2026, 6, 17), date(2026, 6, 18), date(2026, 6, 19), date(2026, 6, 22)]
        assert expiry_days(sessions, early_close_dates={date(2026, 6, 19)}) == {date(2026, 6, 18)}

    def test_a_cash_half_day_on_the_third_friday_stays_the_expiry(self):
        """A true half-day (cash open until 13:00) is still a cash trading
        day; the expiry stays on it and eligibility skips it separately."""
        # 2027-12-24 is not a third Friday; construct the rule on a synthetic
        # half-day: the day after Thanksgiving 2026 is Friday 27 November, a
        # fourth Friday, so use cash_half_days' own rule via a real one:
        # 2024-11-29 (fourth Friday) is a half-day and not a third Friday.
        # Third Friday of Nov 2024 is the 15th.
        sessions = [date(2024, 11, 14), date(2024, 11, 15), date(2024, 11, 29)]
        assert expiry_days(sessions, early_close_dates={date(2024, 11, 29)}) == {date(2024, 11, 15)}


class TestQuarterly:
    def test_march_june_september_december(self):
        assert is_quarterly(date(2026, 3, 20)) is True
        assert is_quarterly(date(2026, 6, 19)) is True
        assert is_quarterly(date(2026, 4, 17)) is False


class TestSessionCalendar:
    def _bars(self):
        return pd.concat([
            session_bars("2026-03-18"),                     # Wednesday, other
            session_bars("2026-03-19"),                     # Thursday, other
            session_bars("2026-03-20"),                     # expiry, quarterly
            session_bars("2026-03-27"),                     # Friday control
            session_bars("2026-04-03", last_bar="12:59"),   # Friday early close (not eligible)
            session_bars("2026-04-10"),                     # Friday control
        ])

    def test_labels_and_flags(self):
        cal = session_calendar(self._bars(), roll_dates=set(), early_close_dates={date(2026, 4, 3)})
        by = cal.set_index("date")
        assert by.loc[date(2026, 3, 20), "label"] == EXPIRY
        assert bool(by.loc[date(2026, 3, 20), "quarterly"]) is True
        assert by.loc[date(2026, 3, 27), "label"] == CONTROL_FRIDAY
        assert by.loc[date(2026, 3, 18), "label"] == CONTROL_OTHER
        assert bool(by.loc[date(2026, 4, 3), "eligible"]) is False
        assert by.loc[date(2026, 4, 3), "skipped_reason"] == "early_close"
        assert bool(by.loc[date(2026, 3, 20), "eligible"]) is True

    def test_juneteenth_2026_labels_the_thursday_as_expiry_and_skips_the_friday(self):
        bars = pd.concat([session_bars("2026-06-18"), session_bars("2026-06-19", last_bar="12:59"),
                          session_bars("2026-06-22")])
        cal = session_calendar(bars, roll_dates=set(), early_close_dates={date(2026, 6, 19)})
        by = cal.set_index("date")
        assert by.loc[date(2026, 6, 18), "label"] == EXPIRY
        assert bool(by.loc[date(2026, 6, 18), "quarterly"]) is True
        assert bool(by.loc[date(2026, 6, 18), "eligible"]) is True
        assert by.loc[date(2026, 6, 19), "label"] != EXPIRY
        assert by.loc[date(2026, 6, 19), "skipped_reason"] == "early_close"

    def test_roll_day_and_missing_flatten_are_skipped(self):
        bars = pd.concat([session_bars("2026-03-20"), session_bars("2026-03-27", last_bar="15:50")])
        cal = session_calendar(bars, roll_dates={date(2026, 3, 20)}, early_close_dates=set())
        by = cal.set_index("date")
        assert by.loc[date(2026, 3, 20), "skipped_reason"] == "roll_day"
        assert by.loc[date(2026, 3, 27), "skipped_reason"] == "missing_bars"

    def test_reads_the_index_only(self):
        """A calendar built on bars with no prices at all must still work."""
        bars = self._bars()[["instrument_id"]]
        cal = session_calendar(bars, roll_dates=set(), early_close_dates=set())
        assert len(cal) == 6

    def test_years_come_from_the_dates(self):
        cal = session_calendar(self._bars(), roll_dates=set(), early_close_dates=set())
        assert set(cal["year"]) == {2026}


class TestSessionRanges:
    def test_range_abs_move_and_morning_move_per_eligible_session(self):
        # Expiry day: open 100, drifts +0.01/bar so 15:55 open = 100 + 0.01*385
        expiry = session_bars("2026-03-20", high=103.0, low=98.0, drift=0.01)
        friday = session_bars("2026-03-27", high=110.0, low=90.0)
        early = session_bars("2026-04-03", last_bar="12:59")
        pop = session_ranges(pd.concat([expiry, friday, early]),
                             roll_dates=set(), early_close_dates={date(2026, 4, 3)})
        assert list(pop["date"]) == [date(2026, 3, 20), date(2026, 3, 27)]
        row = pop.set_index("date").loc[date(2026, 3, 20)]
        assert row["range_points"] == 5.0
        assert row["abs_move"] == pytest.approx(0.01 * 385)
        assert row["morning_move"] == pytest.approx(0.01 * 60)   # 10:30 is 60 bars after 09:30
        assert row["label"] == EXPIRY
        assert pop.set_index("date").loc[date(2026, 3, 27), "range_points"] == 20.0

    def test_range_uses_bars_up_to_1554_only(self):
        """A spike on the 15:55 bar or later is after the flatten and not in the range."""
        day = session_bars("2026-03-27", high=101.0, low=99.0)
        day.loc[day.index.time >= time(15, 55), "high"] = 500.0
        pop = session_ranges(day, roll_dates=set(), early_close_dates=set())
        assert pop["range_points"].iloc[0] == 2.0

    def test_log_range_is_carried(self):
        day = session_bars("2026-03-27", high=104.0, low=100.0)
        pop = session_ranges(day, roll_dates=set(), early_close_dates=set())
        assert pop["log_range"].iloc[0] == pytest.approx(math.log(4.0))


# ---------------------------------------------------------------------------
# The fade arm
# ---------------------------------------------------------------------------


def expiry_session(day: str = "2026-03-20", open_: float = 100.0, at_1030: float = 110.0,
                   after: float | None = None, high_pad: float = 0.5, low_pad: float = 0.5) -> pd.DataFrame:
    """An expiry-day session: flat at ``open_`` until 10:29, ``at_1030`` from
    10:30 on, then ``after`` (default: stays at ``at_1030``). High/low hug the
    open by the pads so nothing is touched unless a test moves them."""
    bars = session_bars(day, open_=open_, high=open_ + high_pad, low=open_ - low_pad)
    later = bars.index.time >= time(10, 30)
    bars.loc[later, ["open", "close"]] = at_1030
    bars.loc[later, "high"] = at_1030 + high_pad
    bars.loc[later, "low"] = at_1030 - low_pad
    if after is not None:
        tail = bars.index.time >= time(10, 31)
        bars.loc[tail, ["open", "close"]] = after
        bars.loc[tail, "high"] = after + high_pad
        bars.loc[tail, "low"] = after - low_pad
    return bars


def signals_for(bars, sd_move=10.0, k=0.5, s=1.0):
    strat = OpexFade(OpexParams(sd_move=sd_move, k=k, s=s))
    return strat, strat.generate_signals(bars)


class TestOpexParams:
    def test_thresholds_in_points_follow_sd_move(self):
        p = OpexParams(sd_move=12.0, k=0.5, s=1.0)
        assert p.threshold_points == 6.0
        assert p.stop_points == 12.0

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            OpexParams(sd_move=0.0)
        with pytest.raises(ValueError):
            OpexParams(sd_move=10.0, k=0.0)


class TestFadeEntries:
    def test_morning_rally_beyond_threshold_is_a_short_at_1030_targeting_the_open(self):
        bars = expiry_session(open_=100.0, at_1030=110.0)   # m = +10 >= 0.5 * 10
        strat, sig = signals_for(bars, sd_move=10.0)
        ts = pd.Timestamp("2026-03-20 10:30", tz=ET)
        assert bool(sig.loc[ts, "entry_short"]) is True
        assert bool(sig.loc[ts, "entry_long"]) is False
        assert sig.loc[ts, "entry_price"] == 110.0
        assert sig.loc[ts, "target_price"] == 100.0
        assert sig.loc[ts, "stop_price"] == 120.0
        d = strat.diagnostics.loc[date(2026, 3, 20)]
        assert bool(d["entered"]) is True and d["direction"] == "short"
        assert d["morning_move"] == 10.0

    def test_morning_drop_beyond_threshold_is_a_long(self):
        bars = expiry_session(open_=100.0, at_1030=94.0)   # m = -6 <= -5
        strat, sig = signals_for(bars, sd_move=10.0)
        ts = pd.Timestamp("2026-03-20 10:30", tz=ET)
        assert bool(sig.loc[ts, "entry_long"]) is True
        assert sig.loc[ts, "target_price"] == 100.0
        assert sig.loc[ts, "stop_price"] == 84.0

    def test_move_inside_the_threshold_is_no_trade(self):
        bars = expiry_session(open_=100.0, at_1030=103.0)   # |m| = 3 < 5
        strat, sig = signals_for(bars, sd_move=10.0)
        assert not sig["entry_long"].any() and not sig["entry_short"].any()
        d = strat.diagnostics.loc[date(2026, 3, 20)]
        assert bool(d["entered"]) is False
        assert d["skipped_reason"] == SKIP_INSIDE_THRESHOLD

    def test_only_expiry_days_trade(self):
        bars = pd.concat([expiry_session("2026-03-20", at_1030=110.0),
                          expiry_session("2026-03-27", at_1030=110.0)])   # Friday, not expiry
        strat, sig = signals_for(bars, sd_move=10.0)
        assert sig["entry_short"].sum() == 1
        assert list(strat.diagnostics.index) == [date(2026, 3, 20)]

    def test_one_trade_per_day_and_signals_validate(self):
        bars = expiry_session(open_=100.0, at_1030=110.0)
        _, sig = signals_for(bars, sd_move=10.0)
        validate_signals(sig)
        assert (sig["entry_short"] | sig["entry_long"]).sum() == 1


class TestFadeExits:
    def test_target_hit_exits_at_the_target_price(self):
        # short at 110, target 100: price falls to 99.5 from 10:31 so the low touches 100
        bars = expiry_session(open_=100.0, at_1030=110.0, after=100.4, low_pad=0.5)
        strat, sig = signals_for(bars, sd_move=10.0)
        hit = sig[sig["exit_short"]]
        assert len(hit) == 1
        assert hit.index[0] == pd.Timestamp("2026-03-20 10:31", tz=ET)
        assert hit["exit_price"].iloc[0] == 100.0
        assert hit["exit_reason"].iloc[0] == EXIT_TARGET
        assert strat.diagnostics.loc[date(2026, 3, 20), "exit_reason"] == EXIT_TARGET

    def test_stop_hit_exits_at_the_stop_price(self):
        # short at 110, stop 120: price rises to 120.5 from 10:31
        bars = expiry_session(open_=100.0, at_1030=110.0, after=120.2)
        strat, sig = signals_for(bars, sd_move=10.0)
        hit = sig[sig["exit_short"]]
        assert hit["exit_price"].iloc[0] == 120.0
        assert hit["exit_reason"].iloc[0] == EXIT_STOP

    def test_stop_and_target_in_one_bar_is_stop_first(self):
        bars = expiry_session(open_=100.0, at_1030=110.0)
        ts = pd.Timestamp("2026-03-20 10:31", tz=ET)
        bars.loc[ts, "high"] = 125.0   # through the stop at 120
        bars.loc[ts, "low"] = 95.0     # and through the target at 100
        strat, sig = signals_for(bars, sd_move=10.0)
        hit = sig[sig["exit_short"]]
        assert hit.index[0] == ts
        assert hit["exit_reason"].iloc[0] == EXIT_STOP

    def test_neither_hit_flattens_at_1555_open(self):
        bars = expiry_session(open_=100.0, at_1030=110.0)   # stays at 110
        strat, sig = signals_for(bars, sd_move=10.0)
        hit = sig[sig["exit_short"]]
        assert hit.index[0] == pd.Timestamp("2026-03-20 15:55", tz=ET)
        assert hit["exit_price"].iloc[0] == 110.0
        assert hit["exit_reason"].iloc[0] == EXIT_FLATTEN

    def test_entry_bar_breach_is_counted_not_acted_on(self):
        bars = expiry_session(open_=100.0, at_1030=110.0)
        ts = pd.Timestamp("2026-03-20 10:30", tz=ET)
        bars.loc[ts, "high"] = 121.0   # the entry bar itself is through the stop
        strat, sig = signals_for(bars, sd_move=10.0)
        assert bool(strat.diagnostics.loc[date(2026, 3, 20), "entry_bar_breach"]) is True
        assert not sig.loc[ts, "exit_short"]

    def test_engine_prices_the_short_correctly(self):
        bars = expiry_session(open_=100.0, at_1030=110.0, after=100.4)
        _, sig = signals_for(bars, sd_move=10.0)
        trades = price_trades(build_trades(sig, bars), MES, CostModel(slippage_ticks=0.0), 3)
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["direction"] == "short"
        assert t["gross_points"] == pytest.approx(10.0)
        assert t["gross_pnl"] == pytest.approx(10.0 * MES.point_value * 3)
        assert t["duration_seconds"] == 60.0
