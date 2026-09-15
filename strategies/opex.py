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
from dataclasses import dataclass
from datetime import date as date_type
from datetime import time, timedelta
from typing import Collection, Iterable

import numpy as np
import pandas as pd

import rules
from base import Strategy, empty_signals, validate_signals
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


# ---------------------------------------------------------------------------
# The fade arm
# ---------------------------------------------------------------------------

EXIT_TARGET = "target"
EXIT_STOP = "stop"
EXIT_FLATTEN = "flatten_1555"
SKIP_INSIDE_THRESHOLD = "inside_threshold"

DIAGNOSTIC_COLUMNS = ["label", "quarterly", "entered", "skipped_reason", "direction",
                      "morning_move", "entry_price", "stop_price", "target_price",
                      "entry_bar_breach", "exit_reason"]


@dataclass(frozen=True)
class OpexParams:
    """The fade rule as frozen: ``k`` and ``s`` in units of ``sd_move``, the
    standard deviation of the primary control's 09:30-to-10:30 move, which
    the runner derives at run time and reports."""

    sd_move: float
    k: float = 0.5
    s: float = 1.0

    def __post_init__(self) -> None:
        if self.sd_move <= 0 or self.k <= 0 or self.s <= 0:
            raise ValueError("sd_move, k and s must all be positive")

    @property
    def threshold_points(self) -> float:
        return self.k * self.sd_move

    @property
    def stop_points(self) -> float:
        return self.s * self.sd_move


class OpexFade(Strategy):
    """Fade the morning move on an expiry day, back toward the day's open."""

    name = "opex_fade"

    def __init__(self, params: OpexParams,
                 roll_dates: Collection[date_type] = (),
                 early_close_dates: Collection[date_type] = ()) -> None:
        self.params = params
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)
        self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = empty_signals(bars.index)
        for col in ("entry_price", "stop_price", "target_price", "exit_price"):
            out[col] = pd.Series(np.nan, index=bars.index, dtype="float64")
        out["exit_reason"] = pd.Series(pd.NA, index=bars.index, dtype="string")

        cal = session_calendar(bars, self.roll_dates, self.early_close_dates)
        expiries = cal[cal["label"] == EXPIRY]
        by_day = {ts.date(): g for ts, g in bars.groupby(bars.index.normalize())}

        diag: list[dict] = []
        for _, row in expiries.iterrows():
            day = row["date"]
            record = {"date": day, "label": EXPIRY, "quarterly": bool(row["quarterly"]),
                      "entered": False, "skipped_reason": row["skipped_reason"],
                      "direction": None, "morning_move": np.nan, "entry_price": np.nan,
                      "stop_price": np.nan, "target_price": np.nan,
                      "entry_bar_breach": False, "exit_reason": None}
            diag.append(record)
            if not row["eligible"]:
                continue
            self._session_signals(by_day[day], out, record)

        self.diagnostics = (pd.DataFrame(diag, columns=["date", *DIAGNOSTIC_COLUMNS])
                            .set_index("date") if diag
                            else pd.DataFrame(columns=DIAGNOSTIC_COLUMNS))
        return validate_signals(out)

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame,
                         record: dict) -> None:
        p = self.params
        times = session.index.time
        open_rows = session[times == OPEN_BAR]
        entry_rows = session[times == DECISION_BAR]
        flatten_rows = session[times == FLATTEN_BAR]
        if open_rows.empty or entry_rows.empty or flatten_rows.empty:
            record["skipped_reason"] = SKIP_MISSING
            return
        entry_ts, flatten_ts = entry_rows.index[0], flatten_rows.index[0]

        # The calendar already excludes these; the rules module is asked
        # anyway so the strategy cannot outrun a guard by construction.
        if not rules.is_entry_allowed(entry_ts, self.early_close_dates, self.roll_dates):
            record["skipped_reason"] = SKIP_ROLL if rules.is_roll_day(
                entry_ts, self.roll_dates) else SKIP_EARLY
            return

        day_open = float(open_rows["open"].iloc[0])
        entry_price = float(entry_rows["open"].iloc[0])
        move = entry_price - day_open
        record["morning_move"] = move
        if abs(move) < p.threshold_points:
            record["skipped_reason"] = SKIP_INSIDE_THRESHOLD
            return

        short = move > 0
        direction = "short" if short else "long"
        stop = entry_price + p.stop_points if short else entry_price - p.stop_points
        target = day_open
        entry_col, exit_col = ("entry_short", "exit_short") if short else ("entry_long", "exit_long")

        out.loc[entry_ts, entry_col] = True
        out.loc[entry_ts, "entry_price"] = entry_price
        out.loc[entry_ts, "stop_price"] = stop
        out.loc[entry_ts, "target_price"] = target
        record.update(entered=True, direction=direction, entry_price=entry_price,
                      stop_price=stop, target_price=target)
        entry_high = float(entry_rows["high"].iloc[0])
        entry_low = float(entry_rows["low"].iloc[0])
        record["entry_bar_breach"] = bool(entry_high >= stop) if short else bool(entry_low <= stop)

        holdable = session[(session.index > entry_ts) & (session.index < flatten_ts)]
        if not holdable.empty:
            highs = holdable["high"].to_numpy(float)
            lows = holdable["low"].to_numpy(float)
            if short:
                stop_hit, target_hit = highs >= stop, lows <= target
            else:
                stop_hit, target_hit = lows <= stop, highs >= target
            either = stop_hit | target_hit
            if either.any():
                i = int(np.argmax(either))
                ts = holdable.index[i]
                # Stop-first when both are touched inside one bar.
                reason, price = (EXIT_STOP, stop) if stop_hit[i] else (EXIT_TARGET, target)
                out.loc[ts, exit_col] = True
                out.loc[ts, "exit_price"] = float(price)
                out.loc[ts, "exit_reason"] = reason
                record["exit_reason"] = reason
                return

        out.loc[flatten_ts, exit_col] = True
        out.loc[flatten_ts, "exit_price"] = float(flatten_rows["open"].iloc[0])
        out.loc[flatten_ts, "exit_reason"] = EXIT_FLATTEN
        record["exit_reason"] = EXIT_FLATTEN
