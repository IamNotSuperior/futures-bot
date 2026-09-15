"""Entry 15: quarterly futures expiry, index-arbitrage unwind at the
settlement open, on MES. Frozen at the entry 15 freeze commit.

The calendar
------------
Entry 14's expiry calendar (``opex.expiry_days``: the last cash trading day
on or before the third Friday) restricted to March, June, September and
December. The monthly expiry days of the other eight months are labelled
``monthly_expiry`` and belong to **neither** population: they carry entry
14's option-hedge effect and are neither the treatment nor a clean control.
The control is every other eligible Friday, entry 14's primary control.

Populations
-----------
:func:`session_moves` gives, per eligible session, the opening move
``o = open(10:00) - open(09:30)``, the afternoon move
``a = open(15:55) - open(10:00)``, ``abs_open = |o|`` and the reversal flag
(``a`` has the opposite sign to ``o``, or ``o`` is zero).

The fade arm
------------
:class:`QuarterlyFade` is entry 14's :class:`opex.OpexFade` with the
decision bar at 10:00 and quarterly days only; entries, target, stop-first
and flatten are inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import time
from typing import Collection, Iterable

import pandas as pd

from opex import (
    EXPIRY, FLATTEN_BAR, OPEN_BAR, OpexFade, expiry_days, is_quarterly,
)
from opex import session_calendar as _opex_calendar

#: Bar labelled 10:00 - its open is the decision price and the entry fill.
DECISION_BAR = time(10, 0)

QUARTERLY = "quarterly"
MONTHLY_EXPIRY = "monthly_expiry"


def quarterly_expiry_days(session_dates: Iterable[date_type],
                          early_close_dates: Iterable[date_type] = ()) -> set[date_type]:
    return {d for d in expiry_days(session_dates, early_close_dates) if is_quarterly(d)}


def session_calendar(bars: pd.DataFrame, roll_dates: Collection[date_type],
                     early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """Entry 14's calendar with expiry days split into quarterly and monthly."""
    cal = _opex_calendar(bars, roll_dates, early_close_dates).copy()
    is_expiry = cal["label"] == EXPIRY
    cal.loc[is_expiry & cal["quarterly"], "label"] = QUARTERLY
    cal.loc[is_expiry & ~cal["quarterly"], "label"] = MONTHLY_EXPIRY
    return cal


def _bar_at(bars: pd.DataFrame, at: time) -> pd.Series:
    rows = bars[bars.index.time == at]
    return pd.Series(rows["open"].to_numpy(float), index=rows.index.date)


def session_moves(bars: pd.DataFrame, roll_dates: Collection[date_type],
                  early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """Eligible sessions with their opening move, afternoon move and reversal flag."""
    cal = session_calendar(bars, roll_dates, early_close_dates)
    cal = cal[cal["eligible"]].copy()
    opens = _bar_at(bars, OPEN_BAR)
    decisions = _bar_at(bars, DECISION_BAR)
    flattens = _bar_at(bars, FLATTEN_BAR)
    o = (cal["date"].map(decisions) - cal["date"].map(opens)).astype(float)
    a = (cal["date"].map(flattens) - cal["date"].map(decisions)).astype(float)
    cal["opening_move"] = o
    cal["afternoon_move"] = a
    cal["abs_open"] = o.abs()
    cal["reversal"] = (o == 0) | ((o > 0) & (a < 0)) | ((o < 0) & (a > 0))
    return cal.drop(columns=["eligible", "skipped_reason"]).reset_index(drop=True)


@dataclass(frozen=True)
class QuarterlyParams:
    """The fade rule as frozen: ``k`` and ``s`` in units of ``sd_open``, the
    standard deviation of the control's 09:30-to-10:00 move."""

    sd_open: float
    k: float = 1.0
    s: float = 1.0

    def __post_init__(self) -> None:
        if self.sd_open <= 0 or self.k <= 0 or self.s <= 0:
            raise ValueError("sd_open, k and s must all be positive")

    @property
    def threshold_points(self) -> float:
        return self.k * self.sd_open

    @property
    def stop_points(self) -> float:
        return self.s * self.sd_open


class QuarterlyFade(OpexFade):
    """Fade the opening move on a quarterly expiry day from the 10:00 open."""

    name = "quarterly_fade"
    decision_bar = DECISION_BAR

    def _trade_days(self, cal: pd.DataFrame) -> pd.DataFrame:
        rows = cal[(cal["label"] == EXPIRY) & cal["quarterly"]].copy()
        rows["label"] = QUARTERLY
        return rows
