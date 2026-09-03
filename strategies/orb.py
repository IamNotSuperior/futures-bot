"""Opening-range breakout.

The opening range is the high/low of the first ``opening_range_minutes`` of
regular trading hours. A close beyond that range, on the resampled timeframe
and inside the trading window, arms an entry in that direction. Each direction
fires at most once per session.

Lookahead
---------
A 5-minute bar labelled 09:45 covers 09:45-09:49 and is only known once 09:50
arrives, so its breakout is acted on at the *next* bar's open. Every entry
signal is shifted accordingly (see :mod:`strategies.base` for the convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import time
from typing import Collection

import pandas as pd

from base import Strategy, empty_signals, validate_signals
import rules

RTH_START = "09:30"
RTH_LAST_BAR = "15:59"

OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


@dataclass
class ORBParams:
    """Opening-range breakout parameters."""

    #: Length of the opening range in minutes (15 -> 09:30-09:45).
    opening_range_minutes: int = 15
    #: No new entries at or after this ET wall-clock time.
    trade_window_end: time = time(11, 30)
    #: Stop distance as a multiple of the opening-range height.
    stop_multiple: float = 1.0
    #: Target distance as a multiple of the stop distance.
    target_multiple: float = 2.0
    #: Timeframe the signal logic runs on.
    bar_minutes: int = 5

    def __post_init__(self) -> None:
        if self.opening_range_minutes <= 0:
            raise ValueError("opening_range_minutes must be positive")
        if self.bar_minutes <= 0:
            raise ValueError("bar_minutes must be positive")
        if self.opening_range_minutes % self.bar_minutes:
            raise ValueError(
                f"opening_range_minutes ({self.opening_range_minutes}) must be a "
                f"whole number of {self.bar_minutes}-minute bars"
            )
        if self.stop_multiple <= 0:
            raise ValueError("stop_multiple must be positive")
        if self.target_multiple <= 0:
            raise ValueError("target_multiple must be positive")


def resample_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-minute RTH bars to ``minutes``-minute bars, session by session.

    Bars are labelled by their opening minute (``label="left"``,
    ``closed="left"``), and sessions are resampled independently so no bucket
    can straddle the overnight break.
    """
    rth = bars.between_time(RTH_START, RTH_LAST_BAR)
    if rth.empty:
        return rth.copy()

    cols = {k: v for k, v in OHLC_AGG.items() if k in rth.columns}
    out = (
        rth.groupby(rth.index.date, group_keys=False)
        .apply(lambda g: g.resample(f"{minutes}min", label="left", closed="left").agg(cols))
    )
    return out.dropna(subset=["open"]).sort_index()


class OpeningRangeBreakout(Strategy):
    """Opening-range breakout on resampled intraday bars."""

    name = "orb"

    def __init__(
        self,
        params: ORBParams | None = None,
        roll_dates: Collection[date_type] = (),
        early_close_dates: Collection[date_type] = (),
    ) -> None:
        self.params = params or ORBParams()
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)

    # -- internals ---------------------------------------------------------

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame) -> None:
        """Fill ``out`` rows for one session, in place.

        Walks the session in order holding one position at a time. A breakout in
        the opposite direction while a trade is open is ignored - the strategy
        never asks for two positions at once, so the execution layer is never
        handed a long and a short to net against each other. A direction that was
        passed over this way can still trade later in the session if it breaks
        out again once the book is flat, but only once per session either way.
        """
        p = self.params
        opening_end = (
            pd.Timestamp(f"{session.index[0].date()} 09:30")
            + pd.Timedelta(minutes=p.opening_range_minutes)
        ).time()

        opening = session[session.index.time < opening_end]
        if opening.empty:
            return
        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        or_height = or_high - or_low
        if or_height <= 0:
            return

        after = session[session.index.time >= opening_end]
        if after.empty:
            return

        stop_distance = or_height * p.stop_multiple
        target_distance = stop_distance * p.target_multiple

        traded = {"long": False, "short": False}
        n = len(session)
        i = int(session.index.get_indexer([after.index[0]])[0])

        while i < n:
            bar = session.iloc[i]
            direction = None
            if not traded["long"] and bar["close"] > or_high:
                direction = "long"
            elif not traded["short"] and bar["close"] < or_low:
                direction = "short"

            if direction is None:
                i += 1
                continue

            if i + 1 >= n:
                break  # breakout on the final bar; nothing left to act on
            action_ts = session.index[i + 1]

            # The window and the entry cutoff only ever move forward, so once
            # they bite there is nothing later in the session to find.
            if action_ts.time() >= p.trade_window_end:
                break
            if not rules.is_entry_allowed(
                action_ts, self.early_close_dates, self.roll_dates
            ):
                break

            entry_price = float(session.loc[action_ts, "open"])
            if direction == "long":
                stop = entry_price - stop_distance
                target = entry_price + target_distance
                out.loc[action_ts, "entry_long"] = True
            else:
                stop = entry_price + stop_distance
                target = entry_price - target_distance
                out.loc[action_ts, "entry_short"] = True

            out.loc[action_ts, "stop_price"] = stop
            out.loc[action_ts, "target_price"] = target
            out.loc[action_ts, "opening_range_height"] = or_height

            traded[direction] = True
            exit_i = self._mark_exit(session, i + 1, direction, stop, target, out)
            if exit_i is None:
                break
            # Flat again only from the bar after the exit bar.
            i = exit_i + 1

    def _mark_exit(
        self,
        session: pd.DataFrame,
        action_i: int,
        direction: str,
        stop: float,
        target: float,
        out: pd.DataFrame,
    ) -> int | None:
        """Mark where the trade leaves. Returns the exit bar's position."""
        exit_col = "exit_long" if direction == "long" else "exit_short"
        action_ts = session.index[action_i]
        flatten_at = rules.flatten_deadline(action_ts, self.early_close_dates)

        last_holdable = None
        for i in range(action_i + 1, len(session)):
            ts = session.index[i]
            if ts.time() >= flatten_at:
                break
            last_holdable = i
            bar = session.iloc[i]

            if direction == "long":
                hit_stop = bar["low"] <= stop
                hit_target = bar["high"] >= target
            else:
                hit_stop = bar["high"] >= stop
                hit_target = bar["low"] <= target

            if hit_stop or hit_target:
                # Both inside one bar is unresolvable at this resolution; assume
                # the stop filled first rather than booking an optimistic win.
                out.loc[ts, exit_col] = True
                out.loc[ts, "exit_reason"] = "stop" if hit_stop else "target"
                out.loc[ts, "exit_price"] = stop if hit_stop else target
                return i

        if last_holdable is None:
            return None

        ts = session.index[last_holdable]
        out.loc[ts, exit_col] = True
        out.loc[ts, "exit_reason"] = "session_end"
        out.loc[ts, "exit_price"] = float(session.iloc[last_holdable]["close"])
        return last_holdable

    # -- interface ---------------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals_resampled(
            resample_bars(bars, self.params.bar_minutes)
        )

    def generate_signals_resampled(self, resampled: pd.DataFrame) -> pd.DataFrame:
        """Signals from bars already at ``bar_minutes`` resolution.

        Each session is evaluated independently - the opening range, the
        breakout and the exit all come from bars inside that session - so
        generating over a long span and slicing the result by date gives
        exactly the same signals as generating over each slice separately.
        A parameter scan relies on that to resample and generate once.
        """
        if resampled.empty:
            return validate_signals(empty_signals(resampled.index))

        out = empty_signals(resampled.index)
        out["stop_price"] = pd.Series(pd.NA, index=resampled.index, dtype="Float64")
        out["target_price"] = pd.Series(pd.NA, index=resampled.index, dtype="Float64")
        out["exit_price"] = pd.Series(pd.NA, index=resampled.index, dtype="Float64")
        out["opening_range_height"] = pd.Series(
            pd.NA, index=resampled.index, dtype="Float64"
        )
        out["exit_reason"] = pd.Series(pd.NA, index=resampled.index, dtype="string")

        for day, session in resampled.groupby(resampled.index.date):
            if rules.is_roll_day(day, self.roll_dates):
                continue
            self._session_signals(session, out)

        return validate_signals(out)

    def bars_for_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """The resampled bars the signals are indexed on, for the backtest."""
        return resample_bars(bars, self.params.bar_minutes)
