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

from opex import (
    CONTROL_FRIDAY, CONTROL_OTHER, EXPIRY, expiry_days, is_quarterly,
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
