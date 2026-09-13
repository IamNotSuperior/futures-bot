"""Tests for the CME holiday calendar and its effect on the pre-trade gate.

The case that matters most is the one the calendar exists for: a half-day the
cached bar data cannot know about, where without the calendar a late entry
looks perfectly legal.
"""

from datetime import date

import pandas as pd
import pytest

import cme_calendar as cal
import rules
from pretrade import AccountState, TicketRequest, evaluate

ET = rules.ET


def et(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz=ET)


def req(**kwargs) -> TicketRequest:
    base = dict(instrument="MES", direction="long", entry_price=6800.0,
                stop_price=6795.0, contracts=1, thesis="range reclaim")
    base.update(kwargs)
    return TicketRequest(**base)


class TestCalendarContents:
    def test_coverage_span(self):
        assert cal.COVERAGE_START == date(2026, 9, 1)
        assert cal.COVERAGE_END == date(2027, 12, 31)

    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 9, 7),    # Labor Day
            date(2026, 11, 26),  # Thanksgiving
            date(2026, 11, 27),  # Black Friday
            date(2026, 12, 24),  # Christmas Eve
            date(2027, 1, 18),   # MLK
            date(2027, 2, 15),   # Presidents
            date(2027, 5, 31),   # Memorial
            date(2027, 6, 18),   # Juneteenth observed
            date(2027, 7, 5),    # July 4 observed: a 13:00 ET halt, not a closure
            date(2027, 9, 6),    # Labor Day
            date(2027, 11, 25),  # Thanksgiving
            date(2027, 11, 26),  # Black Friday
        ],
    )
    def test_early_closes(self, day):
        assert cal.is_early_close(day) is True
        assert cal.is_closed(day) is False

    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 12, 25),  # Christmas
            date(2027, 1, 1),    # New Year's Day
            date(2027, 3, 26),   # Good Friday
            date(2027, 12, 24),  # Christmas observed (Dec 25 is a Saturday)
        ],
    )
    def test_full_closures(self, day):
        assert cal.is_closed(day) is True
        assert cal.is_early_close(day) is False

    def test_a_date_is_never_both(self):
        assert cal.early_close_dates() & cal.closed_dates() == set()

    @pytest.mark.parametrize(
        "day", [
            date(2026, 9, 8), date(2026, 11, 25), date(2027, 3, 25),
            date(2026, 12, 31),  # New Year's Eve 2026: regular 16:00 CT close
            date(2027, 7, 2),    # Friday before the observed July 4: regular close
            date(2027, 12, 23),  # Thursday before the observed Christmas: regular close
            date(2027, 12, 31),  # New Year's Eve 2027: regular close
        ],
    )
    def test_normal_sessions(self, day):
        """Each of these was once modelled as, or flagged as possibly, an
        early close. CME's Globex schedule for ES shows a regular 16:00 CT
        close on every one (verified 2026-09-12)."""
        assert cal.describe(day) is None


class TestVerification:
    """The calendar was built from holiday rules and verified against CME's
    own schedule service. The record of that check lives in the module so the
    next reader knows which dates were confirmed and when."""

    def test_the_verification_record_names_its_source_and_date(self):
        assert cal.VERIFIED_ON == date(2026, 9, 12)
        assert "cmegroup.com" in cal.VERIFIED_SOURCE

    def test_every_2026_date_the_desk_reaches_first_is_verified(self):
        for day in (date(2026, 9, 7), date(2026, 11, 26), date(2026, 11, 27),
                    date(2026, 12, 24), date(2026, 12, 25)):
            assert day in cal.VERIFIED_DATES

    def test_nothing_remains_flagged_for_verification(self):
        assert cal.NEEDS_VERIFICATION == {}

    def test_half_day_close_times_are_never_modelled_later_than_published(self):
        """Rule 2 fails in the dangerous direction if the model closes later
        than the exchange. CME publishes 12:00 CT (13:00 ET) halts on holiday
        Globex sessions and 12:15 CT (13:15 ET) closes on the day after
        Thanksgiving and Christmas Eve. The model's single 13:00 ET close is
        equal to or earlier than every one of them."""
        for day, published_et in cal.PUBLISHED_CLOSE_ET.items():
            assert cal.is_early_close(day), day
            assert rules.EARLY_SESSION_CLOSE <= published_et, day

    def test_good_friday_2027_is_march_26(self):
        """Easter 2027 falls on March 28, so Good Friday is the 26th."""
        assert date(2027, 3, 26).weekday() == 4
        assert cal.is_closed(date(2027, 3, 26))

    def test_observed_holidays_land_on_weekdays(self):
        for day in cal.early_close_dates() | cal.closed_dates():
            assert day.weekday() < 5, f"{day} is a weekend"


class TestCoverageWarnings:
    def test_beyond_coverage_warns(self):
        note = cal.coverage_warning(date(2028, 1, 3))
        assert note is not None and "past this calendar's coverage" in note

    def test_inside_coverage_is_quiet(self):
        assert cal.coverage_warning(date(2026, 10, 1)) is None

    def test_ambiguous_dates_are_flagged(self):
        for day in cal.NEEDS_VERIFICATION:
            assert cal.coverage_warning(day) is not None

    def test_historical_dates_defer_to_the_bar_data(self):
        assert cal.coverage_warning(date(2025, 7, 3)) is None


class TestBlackFriday2026:
    """The case named in the request: 2026-11-27, a half-day."""

    DAY = "2026-11-27"

    def test_1255_is_blocked_without_the_flag(self):
        """Blocked because the calendar knows, not because a flag was passed."""
        d = evaluate(req(), AccountState(), et(f"{self.DAY} 12:55"),
                     early_close_dates=cal.early_close_dates())
        assert d.allowed is False
        assert any("cutoff" in b for b in d.blocks)

    def test_without_the_calendar_it_would_have_been_allowed(self):
        """The exact failure the calendar removes."""
        assert evaluate(req(), AccountState(),
                        et(f"{self.DAY} 12:55")).allowed is True

    def test_1249_still_trades(self):
        d = evaluate(req(), AccountState(), et(f"{self.DAY} 12:49"),
                     early_close_dates=cal.early_close_dates())
        assert d.allowed is True

    def test_boundary_is_1250(self):
        early = cal.early_close_dates()
        assert evaluate(req(), AccountState(), et(f"{self.DAY} 12:49:59"),
                        early_close_dates=early).allowed is True
        assert evaluate(req(), AccountState(), et(f"{self.DAY} 12:50"),
                        early_close_dates=early).allowed is False

    def test_deadlines_match_the_rules_module(self):
        day = date(2026, 11, 27)
        assert rules.flatten_deadline(day, cal.early_close_dates()) == \
            rules.EARLY_SESSION_CLOSE
        assert rules.entry_deadline(day, cal.early_close_dates()).strftime("%H:%M") \
            == "12:50"


class TestClosedDatesBlock:
    def test_christmas_2026_blocks_at_any_hour(self):
        for clock in ("09:35", "11:00", "14:00"):
            d = evaluate(req(), AccountState(), et(f"2026-12-25 {clock}"),
                         closed_dates=cal.closed_dates())
            assert d.allowed is False
            assert any("exchange holiday" in b for b in d.blocks)

    def test_good_friday_2027_blocks(self):
        d = evaluate(req(), AccountState(), et("2027-03-26 10:00"),
                     closed_dates=cal.closed_dates())
        assert d.allowed is False

    def test_a_normal_day_is_unaffected(self):
        d = evaluate(req(), AccountState(), et("2026-12-23 10:00"),
                     early_close_dates=cal.early_close_dates(),
                     closed_dates=cal.closed_dates())
        assert d.allowed is True


class TestManualOverrideStillWorks:
    def test_flag_covers_a_date_the_calendar_misses(self):
        """Past coverage, the flag is the only defence - it must still work."""
        unknown = "2028-11-24"
        assert evaluate(req(), AccountState(), et(f"{unknown} 12:55")).allowed is True
        d = evaluate(req(), AccountState(), et(f"{unknown} 12:55"),
                     early_close_dates={date(2028, 11, 24)})
        assert d.allowed is False

    def test_flag_and_calendar_union_cleanly(self):
        early = cal.early_close_dates() | {date(2026, 10, 5)}
        assert evaluate(req(), AccountState(), et("2026-10-05 12:55"),
                        early_close_dates=early).allowed is False
        assert evaluate(req(), AccountState(), et("2026-11-27 12:55"),
                        early_close_dates=early).allowed is False
