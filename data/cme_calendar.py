"""CME equity-index futures holiday calendar.

Covers 2026-09-01 through 2027-12-31 - the span the cached bar data cannot
reach. Historical early closes are detected from the bars themselves by
:func:`data.loader.detect_early_close_dates`; this module exists only for dates
in the future, where no data can tell us.

**Verified against CME's own schedule service on 2026-09-12** (see
:data:`VERIFIED_SOURCE`): every date below was checked against the E-mini S&P
500 Globex schedule, which MES follows, for each holiday range through
2028-01-02. The check corrected three 2027 dates the holiday rules had
guessed wrong, all in the safe (over-blocking) direction: 2027-07-02 and
2027-12-23 are regular sessions, and 2027-07-05 is a 13:00 ET halt rather
than a closure. CME finalises holiday hours about two weeks before each
holiday and says they can change; re-check a date in the fortnight before
trading it. A wrong date here fails in the dangerous direction: a half-day
treated as a full day allows an entry that should be blocked.

On the close time
-----------------
CME equity index actually closes at 13:00 ET on most half-days and 13:15 ET on
a few - the day after Thanksgiving and Christmas Eve among them. The rules
module models a single early close at 13:00 (``rules.EARLY_SESSION_CLOSE``).
That is deliberately the stricter reading: modelling a 13:15 close as 13:00
moves the entry cutoff from 13:05 back to 12:50, which blocks trades that would
have been legal. Over-blocking is the safe error for a risk control;
under-blocking is not.
"""

from __future__ import annotations

from datetime import date, time

COVERAGE_START = date(2026, 9, 1)
COVERAGE_END = date(2027, 12, 31)

#: Half-days. Equity index stops early; the rules module treats these as 13:00.
EARLY_CLOSES: dict[date, str] = {
    # --- 2026 ---
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving Day",
    date(2026, 11, 27): "Day after Thanksgiving",
    date(2026, 12, 24): "Christmas Eve",
    # --- 2027 ---
    date(2027, 1, 18): "Martin Luther King Jr. Day",
    date(2027, 2, 15): "Presidents Day",
    date(2027, 5, 31): "Memorial Day",
    date(2027, 6, 18): "Juneteenth (observed; June 19 is a Saturday)",
    date(2027, 7, 5): "Independence Day (observed; July 4 is a Sunday) - Globex halt 13:00 ET",
    date(2027, 9, 6): "Labor Day",
    date(2027, 11, 25): "Thanksgiving Day",
    date(2027, 11, 26): "Day after Thanksgiving",
}

#: Full closures. No RTH session at all, so no entry is ever legal.
CLOSED: dict[date, str] = {
    # --- 2026 ---
    date(2026, 12, 25): "Christmas Day",
    # --- 2027 ---
    date(2027, 1, 1): "New Year's Day",
    date(2027, 3, 26): "Good Friday",
    date(2027, 12, 24): "Christmas Day (observed; Dec 25 is a Saturday)",
}

#: Dates whose observance rule is ambiguous and still need checking. Empty
#: since the 2026-09-12 verification; a date added to the calendar without
#: a source belongs here until it has one.
NEEDS_VERIFICATION: dict[date, str] = {}

# ---------------------------------------------------------------------------
# The verification record
# ---------------------------------------------------------------------------

VERIFIED_ON = date(2026, 9, 12)
VERIFIED_SOURCE = (
    "https://www.cmegroup.com/trading-hours.html - the Globex holiday "
    "schedule service (services/trading-hours-by-product, product id 133, "
    "E-mini S&P 500), queried per holiday range 2026-09-06 .. 2028-01-02"
)

#: Every date the verification confirmed, early closes and closures alike,
#: plus the regular sessions the rules had been unsure of.
VERIFIED_DATES: frozenset[date] = frozenset(EARLY_CLOSES) | frozenset(CLOSED) | frozenset({
    date(2026, 12, 31), date(2027, 7, 2), date(2027, 12, 23), date(2027, 12, 31),
})

#: What CME actually publishes for each early close, in ET. Holiday Globex
#: sessions (the cash market shut) halt at 12:00 CT; the day after
#: Thanksgiving and Christmas Eve (the cash market open) close at 12:15 CT.
#: The rules module models all of them as ``EARLY_SESSION_CLOSE`` (13:00 ET),
#: which is equal to or earlier than every published time - the safe side.
#: ``tests/test_cme_calendar.py`` holds that inequality.
_HALT_1300 = time(13, 0)
_CLOSE_1315 = time(13, 15)
PUBLISHED_CLOSE_ET: dict[date, time] = {
    date(2026, 9, 7): _HALT_1300,
    date(2026, 11, 26): _HALT_1300,
    date(2026, 11, 27): _CLOSE_1315,
    date(2026, 12, 24): _CLOSE_1315,
    date(2027, 1, 18): _HALT_1300,
    date(2027, 2, 15): _HALT_1300,
    date(2027, 5, 31): _HALT_1300,
    date(2027, 6, 18): _HALT_1300,
    date(2027, 7, 5): _HALT_1300,
    date(2027, 9, 6): _HALT_1300,
    date(2027, 11, 25): _HALT_1300,
    date(2027, 11, 26): _CLOSE_1315,
}


def early_close_dates() -> set[date]:
    return set(EARLY_CLOSES)


def closed_dates() -> set[date]:
    return set(CLOSED)


def is_early_close(day: date) -> bool:
    return day in EARLY_CLOSES


def is_closed(day: date) -> bool:
    return day in CLOSED


def describe(day: date) -> str | None:
    """Holiday label for a date, or None if it is a normal session."""
    if day in CLOSED:
        return f"CLOSED - {CLOSED[day]}"
    if day in EARLY_CLOSES:
        return f"early close - {EARLY_CLOSES[day]}"
    return None


def covers(day: date) -> bool:
    return COVERAGE_START <= day <= COVERAGE_END

def coverage_warning(day: date) -> str | None:
    """A warning if this date is outside the calendar, or flagged for checking.

    A stale calendar is worse than none: it answers confidently about a date it
    knows nothing about. Past ``COVERAGE_END`` every day looks like a normal
    session, so the caller is told rather than left to assume.
    """
    if day > COVERAGE_END:
        return (
            f"{day} is past this calendar's coverage (ends {COVERAGE_END}). "
            f"Holidays are NOT being checked - extend data/cme_calendar.py or "
            f"pass --early-close manually."
        )
    if day < COVERAGE_START:
        return None  # historical dates come from the bar data instead
    if day in NEEDS_VERIFICATION:
        return f"{day}: {NEEDS_VERIFICATION[day]}"
    return None
