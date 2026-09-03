"""Daily trend filter: an EMA of *completed* daily closes.

The filter answers one question, once per session, before the session opens:
was yesterday's close above or below the 50-day EMA of the closes up to and
including yesterday?

Lookahead
---------
This module exists because the obvious implementation is wrong in a way that
does not announce itself. A daily EMA computed on a series that includes
*today* moves as today trades, so a filter read at 09:45 would be using the
09:45 price - and a backtest built that way quietly knows the future. The
source script this strategy was transcribed from repaints for exactly that
reason; this does not.

The guarantee here is structural, not a matter of care: :func:`trend_filter`
computes the EMA over daily closes and then **shifts it forward by one
session**, so the value published for session ``D`` is a function of closes
strictly before ``D``. No bar belonging to ``D`` can reach it.
``tests/test_trend.py`` asserts this by mutating a session's bars and checking
that its own filter value does not move.

Roll gaps
---------
The daily closes come from the unadjusted ``MES.v.0`` continuous series, so
each contract roll injects a small step that then propagates through a 50-day
average for roughly fifty sessions. Using the unadjusted series is consistent
with the rest of the project - every other module reads the same bars - and
the step is expected to be small next to the EMA-to-price distance the filter
actually keys on. :func:`largest_roll_gap` measures it rather than assuming it,
because entry 4 requires the size to be on record.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Collection

import numpy as np
import pandas as pd

RTH_START = "09:30"
RTH_LAST_BAR = "15:59"

#: Entry 4 fixes the trend filter at 50 completed daily closes.
EMA_PERIOD = 50


def daily_closes(bars: pd.DataFrame) -> pd.Series:
    """Closing price of each RTH session, indexed by session date.

    The close is the last 1-minute bar of the session, so a half-day closes at
    its own final bar rather than being padded to 16:00.
    """
    rth = bars.between_time(RTH_START, RTH_LAST_BAR)
    if rth.empty:
        return pd.Series(dtype="float64", name="close")
    closes = rth.groupby(rth.index.date)["close"].last()
    closes.index = pd.Index(list(closes.index), name="session_date")
    return closes.astype(float).rename("close")


def ema(values: np.ndarray | pd.Series, period: int = EMA_PERIOD) -> np.ndarray:
    """SMA-seeded exponential moving average.

    Seeded with the simple mean of the first ``period`` observations, then the
    standard recursion with ``alpha = 2 / (period + 1)``. Entries before the
    seed are ``NaN``: there is no defensible value there, and filling one in
    would let the strategy trade on a number that does not yet exist.

    The seeding rule is fixed by entry 4. It is recorded in the spec rather
    than left to a library default because pandas' ``ewm`` offers several and
    they disagree for the first few hundred observations - long enough to cover
    an entire test fold.
    """
    arr = np.asarray(values, dtype=float)
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < period:
        return out
    if not np.isfinite(arr[:period]).all():
        raise ValueError("Cannot seed an EMA on a window containing NaN")

    alpha = 2.0 / (period + 1.0)
    out[period - 1] = arr[:period].mean()
    for i in range(period, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def trend_filter(bars: pd.DataFrame, period: int = EMA_PERIOD) -> pd.Series:
    """EMA value to use *on* each session, indexed by session date.

    The value published for session ``D`` is the EMA through the close of the
    previous session. Sessions before the EMA has seeded carry ``NaN`` and are
    not tradeable - with a 50-day period and data from 2019-05-06 that is the
    first fifty sessions, so the strategy cannot trade until roughly mid-July
    2019.
    """
    closes = daily_closes(bars)
    if closes.empty:
        return pd.Series(dtype="float64", name="trend_ema")
    values = ema(closes.to_numpy(), period)
    # The shift is the whole point: session D reads the EMA as of D-1.
    shifted = pd.Series(values, index=closes.index).shift(1)
    return shifted.rename("trend_ema")


def largest_roll_gap(
    bars: pd.DataFrame, roll_dates: Collection[date_type]
) -> dict[str, float | date_type | None]:
    """Close-to-close steps on roll dates, against ordinary sessions.

    Reported rather than assumed: the EMA runs on an unadjusted continuous
    series, and entry 4 requires the magnitude of that contamination to be on
    record instead of dismissed.

    **This is an upper bound, not the roll artefact itself.** A close-to-close
    step on a roll date contains the day's real market move as well as the
    price difference between the expiring and incoming contracts, and the two
    cannot be separated from a stitched series - doing so needs both contracts
    quoted at the same instant, which this project does not hold. The largest
    step in the MES history is 2020-03-16, a day the index fell about 12%;
    reading that as a roll artefact would be badly wrong.

    ``median_roll`` against ``median_ordinary`` is the honest comparison. If a
    roll date's typical step looks like any other session's, the roll is not
    injecting a step worth worrying about, whatever the maximum says.
    """
    closes = daily_closes(bars)
    empty = {"date": None, "points": 0.0, "pct": 0.0,
             "median_roll": 0.0, "median_ordinary": 0.0, "n_rolls": 0}
    if closes.empty:
        return empty

    # The contaminated step is the first *session close* under a new contract,
    # which is not the same thing as the roll date. Twelve of MES's 29 rolls
    # are detected on a Sunday Globex reopen that has no RTH session at all, so
    # keying on the roll date attributes those steps to nothing and leaves the
    # contaminated Monday counted as ordinary. Reading the contract change off
    # the closes themselves avoids that entirely.
    rth = bars.between_time(RTH_START, RTH_LAST_BAR)
    if "instrument_id" in rth.columns:
        ids = rth.groupby(rth.index.date)["instrument_id"].last()
        ids.index = closes.index
        changed = ids.ne(ids.shift(1)) & ids.shift(1).notna()
    else:
        rolls = {d for d in roll_dates if d in closes.index}
        changed = pd.Series([d in rolls for d in closes.index], index=closes.index)

    prev = closes.shift(1)
    steps = (closes - prev).abs()
    on_roll = steps[changed].dropna()
    ordinary = steps[~changed].dropna()
    if on_roll.empty:
        return empty

    worst = on_roll.idxmax()
    return {
        "date": worst,
        "points": float(on_roll.loc[worst]),
        "pct": float(100.0 * on_roll.loc[worst] / prev.loc[worst]),
        "median_roll": float(on_roll.median()),
        "median_ordinary": float(ordinary.median()) if len(ordinary) else float("nan"),
        "n_rolls": int(len(on_roll)),
    }
