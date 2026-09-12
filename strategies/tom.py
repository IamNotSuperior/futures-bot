"""Entry 11: turn-of-month institutional flows, intraday. Frozen at ``a6d9af1``.

Long MES at the 09:30 open on the four window days around each month
boundary (T-1, T+1, T+2, T+3 by the cash trading calendar), flat at the
15:55 open. Two arms with identical entries: the signal arm carries no stop
and its gross points *are* the session's open-to-15:55 return; the stop arm
carries a fixed 15-point stop, the Lucid-safe form. Everything else - sizing,
the guards, costs - is the engine's.

The calendar
------------
A trading day is a session with a 09:30 RTH bar, excluding sessions on which
the cash equity market is closed. CME trades a shortened Globex session on
the cash market's full holidays (Labor Day, Thanksgiving Day, Independence
Day and the rest) and the bar data reports those as early closes, exactly as
it reports the three true half-days on which the cash market is open until
13:00. The entry's mechanism runs on cash trading days, so the two kinds are
told apart by date rule (:func:`cash_half_days`) and the holiday sessions are
removed from the calendar before the window days are counted. Half-days stay
in the count - T+2 stays T+2 - and are skipped as trade days because they
have no 15:55.

Timing convention
-----------------
Bars are labelled by opening minute and closed left. The entry is marked on
the bar labelled 09:30 and fills at its open; the flatten is marked on the
bar labelled 15:55 and fills at its open, the price as of 15:55:00, as entry
10's ``flatten_1555`` did. A stop is checked from the bar *after* the entry
bar; a 09:30 bar whose low is already through the stop is counted in the
diagnostics and not acted on, because filling a stop inside the entry bar
would need a sequence the bar does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import time, timedelta
from typing import Collection, Iterable

import numpy as np
import pandas as pd

from base import Strategy, empty_signals, validate_signals
import rules

#: Bar labelled 09:30 - its open is the RTH open and the entry fill.
OPEN_BAR = time(9, 30)
#: Bar labelled 15:55 - its open is the price as of 15:55:00 and the exit fill.
FLATTEN_BAR = time(15, 55)

WINDOW_LABELS = ("T-1", "T+1", "T+2", "T+3")
CONTROL_LABEL = "control"

EXIT_FLATTEN = "flatten_1555"
EXIT_STOP = "stop"

SKIP_ROLL = "roll_day"
SKIP_EARLY = "early_close"
SKIP_MISSING = "missing_bars"


# ---------------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------------


def _thanksgiving(year: int) -> date_type:
    """Fourth Thursday of November."""
    first = date_type(year, 11, 1)
    offset = (3 - first.weekday()) % 7  # Thursday is weekday 3
    return first + timedelta(days=offset + 21)


def _weekday_holiday_eve(day: date_type, holiday: date_type) -> bool:
    """A half-day the day before ``holiday`` only when the holiday itself is
    a Tuesday to Friday. When it falls on a Saturday the eve is the observed
    closure, and when it falls on a Monday the eve is a Sunday."""
    return day == holiday - timedelta(days=1) and holiday.weekday() in (1, 2, 3, 4)


def cash_half_days(early_close_dates: Iterable[date_type]) -> set[date_type]:
    """The early-close sessions on which the cash equity market was open.

    Three date rules, and nothing else: the day after Thanksgiving; December
    24 when Christmas falls Tuesday to Friday; July 3 when Independence Day
    falls Tuesday to Friday. Every other early close in the bar data is a
    Globex-only session on a cash-market holiday.
    """
    out: set[date_type] = set()
    for day in early_close_dates:
        if day == _thanksgiving(day.year) + timedelta(days=1):
            out.add(day)
        elif _weekday_holiday_eve(day, date_type(day.year, 12, 25)):
            out.add(day)
        elif _weekday_holiday_eve(day, date_type(day.year, 7, 4)):
            out.add(day)
    return out


def trading_days(session_dates: Iterable[date_type],
                 early_close_dates: Iterable[date_type]) -> list[date_type]:
    """Cash trading days: every session less the cash-market holidays."""
    early = set(early_close_dates)
    closed = early - cash_half_days(early)
    return sorted(set(session_dates) - closed)


def label_window_days(days: Iterable[date_type]) -> dict[date_type, str]:
    """T-1 and T+1..T+3 for every month boundary the calendar observes.

    A boundary is observed when two consecutive trading days fall in
    different months. The first partial month of a file therefore has no T+k
    and its last day is no T-1, which is what keeps a mid-month start from
    inventing labels.
    """
    ordered = sorted(set(days))
    labels: dict[date_type, str] = {}
    for i in range(len(ordered) - 1):
        before, after = ordered[i], ordered[i + 1]
        if (before.year, before.month) == (after.year, after.month):
            continue
        labels[before] = "T-1"
        for k in range(1, 4):
            j = i + k
            if j >= len(ordered):
                break
            day = ordered[j]
            if (day.year, day.month) != (after.year, after.month):
                break
            labels[day] = f"T+{k}"
    return labels


# ---------------------------------------------------------------------------
# Sessions: calendar first, prices second
# ---------------------------------------------------------------------------


def _bar_at(bars: pd.DataFrame, at: time) -> pd.Series:
    """The ``open`` of the bar labelled ``at`` on each session, indexed by date."""
    rows = bars[bars.index.time == at]
    return pd.Series(rows["open"].to_numpy(float), index=rows.index.date)


def session_calendar(bars: pd.DataFrame, roll_dates: Collection[date_type],
                     early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """One row per session with a 09:30 bar: its label and why it is or is
    not eligible. Reads the bar index only - no price is touched - so the
    power check can run on it before any outcome exists."""
    rolls, early = set(roll_dates), set(early_close_dates)
    opens = _bar_at(bars, OPEN_BAR)
    has_flatten = set(bars.index[bars.index.time == FLATTEN_BAR].date)
    sessions = sorted(set(opens.index))
    labels = label_window_days(trading_days(sessions, early))

    rows = []
    for day in sessions:
        label = labels.get(day, CONTROL_LABEL)
        reason = None
        if rules.is_roll_day(day, rolls):
            reason = SKIP_ROLL
        elif day in early:
            reason = SKIP_EARLY
        elif day not in has_flatten:
            reason = SKIP_MISSING
        rows.append({"date": day, "year": day.year, "label": label,
                     "window": label != CONTROL_LABEL,
                     "eligible": reason is None, "skipped_reason": reason})
    return pd.DataFrame(rows, columns=["date", "year", "label", "window",
                                       "eligible", "skipped_reason"])


def session_returns(bars: pd.DataFrame, roll_dates: Collection[date_type],
                    early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """Eligible sessions with their open-to-15:55 return in points.

    This is the population the pre-registered mechanism test runs on: the
    window rows against the control rows, same bars, same horizon, same
    exclusions. The signal arm's trade stream must reproduce the window rows
    exactly (pre-registered test 1).
    """
    cal = session_calendar(bars, roll_dates, early_close_dates)
    cal = cal[cal["eligible"]].copy()
    opens = _bar_at(bars, OPEN_BAR)
    flattens = _bar_at(bars, FLATTEN_BAR)
    cal["points"] = (cal["date"].map(flattens) - cal["date"].map(opens)).astype(float)
    return cal.drop(columns=["eligible", "skipped_reason"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TOMParams:
    """Entry 11's two arms. ``None`` is the signal arm; 15.0 the stop arm."""

    stop_points: float | None = None

    def __post_init__(self) -> None:
        if self.stop_points is not None and self.stop_points <= 0:
            raise ValueError("stop_points must be positive or None")

    @property
    def label(self) -> str:
        return "signal" if self.stop_points is None else f"stop{self.stop_points:g}"


DIAGNOSTIC_COLUMNS = ["label", "entered", "skipped_reason", "entry_price",
                      "stop_price", "entry_bar_breach", "exit_reason"]


class TurnOfMonth(Strategy):
    """Long the window days from the 09:30 open to the 15:55 open."""

    name = "tom_intraday"

    def __init__(self, params: TOMParams | None = None,
                 roll_dates: Collection[date_type] = (),
                 early_close_dates: Collection[date_type] = ()) -> None:
        self.params = params or TOMParams()
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)
        self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = empty_signals(bars.index)
        for col in ("entry_price", "stop_price", "exit_price"):
            out[col] = pd.Series(np.nan, index=bars.index, dtype="float64")
        out["exit_reason"] = pd.Series(pd.NA, index=bars.index, dtype="string")

        cal = session_calendar(bars, self.roll_dates, self.early_close_dates)
        window = cal[cal["window"]]
        by_day = {ts.date(): g for ts, g in bars.groupby(bars.index.normalize())}

        diag: list[dict] = []
        for _, row in window.iterrows():
            day, label = row["date"], row["label"]
            record = {"date": day, "label": label, "entered": False,
                      "skipped_reason": row["skipped_reason"],
                      "entry_price": np.nan, "stop_price": np.nan,
                      "entry_bar_breach": False, "exit_reason": None}
            diag.append(record)
            if not row["eligible"]:
                continue
            session = by_day[day]
            self._session_signals(session, out, record)

        self.diagnostics = (pd.DataFrame(diag, columns=["date", *DIAGNOSTIC_COLUMNS])
                            .set_index("date") if diag
                            else pd.DataFrame(columns=DIAGNOSTIC_COLUMNS))
        return validate_signals(out)

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame,
                         record: dict) -> None:
        p = self.params
        times = session.index.time
        entry_rows = session[times == OPEN_BAR]
        flatten_rows = session[times == FLATTEN_BAR]
        if entry_rows.empty or flatten_rows.empty:
            record["skipped_reason"] = SKIP_MISSING
            return
        entry_ts, flatten_ts = entry_rows.index[0], flatten_rows.index[0]

        # The calendar already excludes these; the rules module is asked
        # anyway so the strategy cannot outrun a guard by construction.
        if not rules.is_entry_allowed(entry_ts, self.early_close_dates, self.roll_dates):
            record["skipped_reason"] = SKIP_ROLL if rules.is_roll_day(
                entry_ts, self.roll_dates) else SKIP_EARLY
            return

        entry_price = float(entry_rows["open"].iloc[0])
        out.loc[entry_ts, "entry_long"] = True
        out.loc[entry_ts, "entry_price"] = entry_price
        record.update(entered=True, entry_price=entry_price)

        stop = None
        if p.stop_points is not None:
            stop = entry_price - p.stop_points
            out.loc[entry_ts, "stop_price"] = stop
            record["stop_price"] = stop
            record["entry_bar_breach"] = bool(float(entry_rows["low"].iloc[0]) <= stop)

        holdable = session[(session.index > entry_ts) & (session.index < flatten_ts)]
        if stop is not None and not holdable.empty:
            hit = holdable["low"].to_numpy(float) <= stop
            if hit.any():
                ts = holdable.index[int(np.argmax(hit))]
                out.loc[ts, "exit_long"] = True
                out.loc[ts, "exit_price"] = float(stop)
                out.loc[ts, "exit_reason"] = EXIT_STOP
                record["exit_reason"] = EXIT_STOP
                return

        out.loc[flatten_ts, "exit_long"] = True
        out.loc[flatten_ts, "exit_price"] = float(flatten_rows["open"].iloc[0])
        out.loc[flatten_ts, "exit_reason"] = EXIT_FLATTEN
        record["exit_reason"] = EXIT_FLATTEN
