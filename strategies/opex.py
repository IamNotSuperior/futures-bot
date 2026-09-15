"""Entry 14: monthly option expiry, dealer gamma hedging, intraday on MES.
Frozen at ``f9f129c``.

The calendar
------------
An expiry day is the last session on or before the **third Friday** of each
calendar month: the third Friday itself when the exchange is open, the
preceding Thursday when it is Good Friday. Quarterly expiries (March, June,
September, December) are labelled as such and reported separately; they are
never selected on. Every other eligible Friday is the primary control; every
other eligible session is the secondary control.

Eligibility, for expiry and control alike: a 09:30 bar, a 15:55 bar, not a
roll day, not an early close.

Populations
-----------
:func:`session_calendar` reads the bar index only, so the power check can
run on it before any outcome exists. :func:`session_ranges` adds, per
eligible session, the range (highest high minus lowest low over the bars
labelled 09:30 through 15:54), its log, the absolute open-to-15:55 move and
the 09:30-to-10:30 morning move the fade arm keys on.

Timing convention
-----------------
Bars are labelled by opening minute and closed left, as in entry 11. The
range stops at the 15:54 bar because the flatten fills at the 15:55 open.
"""

from __future__ import annotations

import math
from datetime import date as date_type
from datetime import time, timedelta
from typing import Collection, Iterable

import pandas as pd

import rules
from tom import trading_days

#: Bar labelled 09:30 - its open is the RTH open.
OPEN_BAR = time(9, 30)
#: Bar labelled 10:30 - its open is the fade arm's decision price.
DECISION_BAR = time(10, 30)
#: Bar labelled 15:55 - its open is the price as of 15:55:00 and the exit fill.
FLATTEN_BAR = time(15, 55)
#: The last bar inside the range window.
LAST_RANGE_BAR = time(15, 54)

EXPIRY = "expiry"
CONTROL_FRIDAY = "friday"
CONTROL_OTHER = "other"
QUARTERLY_MONTHS = (3, 6, 9, 12)

SKIP_ROLL = "roll_day"
SKIP_EARLY = "early_close"
SKIP_MISSING = "missing_bars"


# ---------------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------------


def third_friday(year: int, month: int) -> date_type:
    first = date_type(year, month, 1)
    offset = (4 - first.weekday()) % 7  # Friday is weekday 4
    return first + timedelta(days=offset + 14)


def is_quarterly(day: date_type) -> bool:
    return day.month in QUARTERLY_MONTHS


def expiry_days(session_dates: Iterable[date_type],
                early_close_dates: Iterable[date_type] = ()) -> set[date_type]:
    """The last **cash trading day** on or before each month's third Friday.

    Sessions on which the cash equity market was shut - Globex-only holiday
    sessions such as Juneteenth on a Friday - are removed first through
    entry 11's ``trading_days`` (the three true half-days stay), so Good
    Friday and a cash-closed third Friday both move expiry to the Thursday
    (addendum of 2026-09-15). A month contributes an expiry only when it has
    a cash trading day on or before its third Friday, and only days in the
    same month count, so a partial month at the start of a file cannot
    borrow a day from before it.
    """
    sessions = trading_days(session_dates, early_close_dates)
    by_month: dict[tuple[int, int], list[date_type]] = {}
    for day in sessions:
        by_month.setdefault((day.year, day.month), []).append(day)
    out: set[date_type] = set()
    for (year, month), days in by_month.items():
        target = third_friday(year, month)
        candidates = [d for d in days if d <= target]
        if candidates:
            out.add(max(candidates))
    return out


def _bar_at(bars: pd.DataFrame, at: time, column: str = "open") -> pd.Series:
    """One value per session for the bar labelled ``at``, indexed by date."""
    rows = bars[bars.index.time == at]
    return pd.Series(rows[column].to_numpy(float), index=rows.index.date)


def session_calendar(bars: pd.DataFrame, roll_dates: Collection[date_type],
                     early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """One row per session with a 09:30 bar: its label, the quarterly flag,
    and why it is or is not eligible. Reads the bar index only."""
    rolls, early = set(roll_dates), set(early_close_dates)
    index = bars.index
    sessions = sorted(set(index[index.time == OPEN_BAR].date))
    has_flatten = set(index[index.time == FLATTEN_BAR].date)
    expiries = expiry_days(sessions, early)

    rows = []
    for day in sessions:
        if day in expiries:
            label = EXPIRY
        elif day.weekday() == 4:
            label = CONTROL_FRIDAY
        else:
            label = CONTROL_OTHER
        reason = None
        if rules.is_roll_day(day, rolls):
            reason = SKIP_ROLL
        elif day in early:
            reason = SKIP_EARLY
        elif day not in has_flatten:
            reason = SKIP_MISSING
        rows.append({"date": day, "year": day.year, "label": label,
                     "quarterly": label == EXPIRY and is_quarterly(day),
                     "eligible": reason is None, "skipped_reason": reason})
    return pd.DataFrame(rows, columns=["date", "year", "label", "quarterly",
                                       "eligible", "skipped_reason"])


# ---------------------------------------------------------------------------
# Populations: calendar first, prices second
# ---------------------------------------------------------------------------


def session_ranges(bars: pd.DataFrame, roll_dates: Collection[date_type],
                   early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """Eligible sessions with their range, log range, absolute open-to-15:55
    move and 09:30-to-10:30 morning move, all in points."""
    cal = session_calendar(bars, roll_dates, early_close_dates)
    cal = cal[cal["eligible"]].copy()

    window = bars[(bars.index.time >= OPEN_BAR) & (bars.index.time <= LAST_RANGE_BAR)]
    highs = window["high"].groupby(window.index.date).max()
    lows = window["low"].groupby(window.index.date).min()
    opens = _bar_at(bars, OPEN_BAR)
    decisions = _bar_at(bars, DECISION_BAR)
    flattens = _bar_at(bars, FLATTEN_BAR)

    cal["range_points"] = (cal["date"].map(highs) - cal["date"].map(lows)).astype(float)
    cal["log_range"] = cal["range_points"].map(lambda r: math.log(r) if r > 0 else float("nan"))
    cal["abs_move"] = (cal["date"].map(flattens) - cal["date"].map(opens)).abs().astype(float)
    cal["morning_move"] = (cal["date"].map(decisions) - cal["date"].map(opens)).astype(float)
    return cal.drop(columns=["eligible", "skipped_reason"]).reset_index(drop=True)
