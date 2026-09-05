"""ORB-2: fixed-bracket opening-range break with a daily trend filter.

Implements hypothesis entry 4 in ``research/hypotheses.md``, frozen at commit
``17338c5``. Every constant below is fixed by that entry; none of them is a
parameter to be searched, and there is no grid. The only pre-registered
variation is :attr:`ORB2Params.use_trend_filter`, the ON/OFF arm that isolates
the trend-filter claim.

The shape of a session
----------------------
1. The opening range is the high and low of 09:30-09:45.
2. If that range is wider than 10 points the session is skipped - a fixed
   10-point stop would otherwise sit inside the range, where an ordinary
   retrace stops the trade out for reasons unrelated to the signal.
3. At 09:45 one direction is chosen: long if the 09:45 open is above the daily
   trend EMA, short if below, nothing at all if exactly equal. With the filter
   off, both directions are armed and the first fill wins.
4. A stop order rests 1.0 point beyond the range until 11:30, when an unfilled
   order is cancelled.
5. From the fill: a 10-point stop and an 18-point target, both fixed in points
   rather than scaled to volatility.
6. Anything still open is flattened at 15:55.

Fills
-----
Everything is resolved on 1-minute bars. A stop order fills at its own level,
or at the bar's open when the bar opened through it - gaps fill at the open,
never at the better price. Where a single bar touches both the stop and the
target, the stop is assumed filled, which is entry 1's convention carried over
unchanged.

The entry bar is included in the exit search. The fill happens inside that bar,
so the rest of it can reach the stop, and pretending otherwise would book a
free minute of protection. That choice creates the population entry 4 calls
``entries_and_exits_in_one_bar``: a backtest on 1-minute data cannot see
whether such an exit came 8 seconds or 50 seconds after the fill, so rule 6's
30-second floor is unverifiable there. The live guard enforces it; this module
counts the trades at risk so the gap is measured rather than assumed empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import time
from typing import Collection

import numpy as np
import pandas as pd

from base import Strategy, empty_signals, validate_signals
import rules

RTH_START = "09:30"
RTH_LAST_BAR = "15:59"

#: Session date -> diagnostic columns, for the reporting entry 4 requires.
DIAGNOSTIC_COLUMNS = [
    "range_height",
    "range_high",
    "range_low",
    "trend_ema",
    "open_0945",
    "direction",
    "skipped_reason",
    "entered",
    "ambiguous_both_stops",
    "same_bar_entry_exit",
]

SKIP_ROLL = "roll_day"
SKIP_WIDE_RANGE = "range_over_ceiling"
SKIP_UNSEEDED = "ema_unseeded"
SKIP_ON_EMA = "open_equals_ema"
SKIP_NO_BARS = "incomplete_session"
SKIP_NO_FILL = "entry_never_filled"
SKIP_BLOCKED = "entry_blocked_by_rules"


@dataclass(frozen=True)
class ORB2Params:
    """ORB-2 parameters. Every value is fixed by entry 4, not searched."""

    #: Opening range runs from 09:30 up to (not including) this time.
    opening_range_end: time = time(9, 45)
    #: Stop order rests this far beyond the range, in points.
    entry_offset: float = 1.0
    #: Skip the session when the opening range is wider than this, in points.
    max_range_points: float = 10.0
    #: Fixed stop distance from the fill, in points.
    stop_points: float = 10.0
    #: Fixed target distance from the fill, in points.
    target_points: float = 18.0
    #: An unfilled entry order is cancelled at this time.
    entry_cancel_time: time = time(11, 30)
    #: Any open position is flattened at this time.
    flatten_time: time = time(15, 55)
    #: The pre-registered ON/OFF arm. False arms both directions.
    use_trend_filter: bool = True
    #: Exit the forced flatten at the *open of the flatten bar* rather than the
    #: close of the bar before it. Entry 5 needs this - its Pine closes at the
    #: 10:30 bar open - and entry 4 does not, so it defaults to entry 4's
    #: behaviour and leaves those results reproducible.
    flatten_at_next_open: bool = False

    def __post_init__(self) -> None:
        if self.stop_points <= 0 or self.target_points <= 0:
            raise ValueError("stop_points and target_points must be positive")
        if self.max_range_points <= 0:
            raise ValueError("max_range_points must be positive")
        if self.entry_offset < 0:
            raise ValueError("entry_offset must not be negative")
        if not self.opening_range_end < self.entry_cancel_time:
            raise ValueError("opening_range_end must precede entry_cancel_time")
        if not self.entry_cancel_time <= self.flatten_time:
            raise ValueError("entry_cancel_time must not follow flatten_time")


class ORB2(Strategy):
    """Fixed-bracket opening-range break, optionally filtered by a daily EMA."""

    name = "orb2"

    def __init__(
        self,
        params: ORB2Params | None = None,
        trend_ema: pd.Series | None = None,
        roll_dates: Collection[date_type] = (),
        early_close_dates: Collection[date_type] = (),
    ) -> None:
        self.params = params or ORB2Params()
        if self.params.use_trend_filter and trend_ema is None:
            raise ValueError(
                "use_trend_filter=True needs a trend_ema series; "
                "build one with trend.trend_filter(bars) over the full history"
            )
        # Computed over the whole history and injected, never derived from the
        # window being evaluated: a 50-day EMA rebuilt from a one-year slice
        # would be unseeded for that year's first fifty sessions.
        self.trend_ema = trend_ema
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)
        self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    # -- internals ---------------------------------------------------------

    def _ema_for(self, day: date_type) -> float:
        if self.trend_ema is None:
            return float("nan")
        try:
            value = self.trend_ema.loc[day]
        except KeyError:
            return float("nan")
        if isinstance(value, pd.Series):  # duplicated index entry
            value = value.iloc[0]
        return float(value)

    def _exit_deadline(self, day: date_type) -> time:
        """Earlier of the strategy's 15:55 flatten and the rule 2 deadline.

        On an early-close session the rules deadline moves to 13:00, so it
        binds first. The strategy never gets to hold past what rule 2 allows.
        """
        return min(self.params.flatten_time, rules.flatten_deadline(day, self.early_close_dates))

    def _simulate_exit(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        start: int,
        last: int,
        direction: str,
        stop: float,
        target: float,
        opens: np.ndarray | None = None,
        n_bars: int | None = None,
    ) -> tuple[int, float, str]:
        """Walk bars from the entry bar to the deadline. Returns the exit.

        ``start`` is the entry bar itself, deliberately: the fill happened
        inside it and the rest of that minute can still reach the stop.

        The forced flatten lands on the close of ``last`` by default. With
        ``flatten_at_next_open`` it lands on the *open of the following bar* -
        the flatten bar itself - which is what a script closing "at the 10:30
        bar open" actually does. If there is no following bar the close of
        ``last`` is used, because there is nothing later to fill against.
        """
        for i in range(start, last + 1):
            if direction == "long":
                hit_stop = lows[i] <= stop
                hit_target = highs[i] >= target
            else:
                hit_stop = highs[i] >= stop
                hit_target = lows[i] <= target
            if hit_stop or hit_target:
                # Both inside one bar is unresolvable at this resolution;
                # assume the stop filled first rather than booking an
                # optimistic win.
                if hit_stop:
                    return i, stop, "stop"
                return i, target, "target"

        if (self.params.flatten_at_next_open and opens is not None
                and n_bars is not None and last + 1 < n_bars):
            return last + 1, float(opens[last + 1]), "session_end"
        return last, float(closes[last]), "session_end"

    def _fill_index(
        self, highs: np.ndarray, lows: np.ndarray, opens: np.ndarray,
        first: int, last: int, direction: str, level: float,
    ) -> tuple[int, float] | None:
        """First bar reaching a resting stop order, and the price it fills at."""
        for i in range(first, last + 1):
            if direction == "long":
                if highs[i] >= level:
                    # Gapped through: fill at the open, never the better price.
                    return i, max(level, float(opens[i]))
            else:
                if lows[i] <= level:
                    return i, min(level, float(opens[i]))
        return None

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame, day: date_type) -> dict:
        """Fill ``out`` rows for one session in place. Returns its diagnostics."""
        p = self.params
        diag = {c: None for c in DIAGNOSTIC_COLUMNS}
        diag.update(entered=False, ambiguous_both_stops=False, same_bar_entry_exit=False)

        opening = session[session.index.time < p.opening_range_end]
        after = session[session.index.time >= p.opening_range_end]
        if opening.empty or after.empty:
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag

        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        height = or_high - or_low
        diag.update(range_height=height, range_high=or_high, range_low=or_low)

        if height > p.max_range_points:
            diag["skipped_reason"] = SKIP_WIDE_RANGE
            return diag

        open_0945 = float(after.iloc[0]["open"])
        diag["open_0945"] = open_0945

        directions: list[str]
        if p.use_trend_filter:
            ema_value = self._ema_for(day)
            diag["trend_ema"] = ema_value
            if not np.isfinite(ema_value):
                diag["skipped_reason"] = SKIP_UNSEEDED
                return diag
            if open_0945 > ema_value:
                directions = ["long"]
            elif open_0945 < ema_value:
                directions = ["short"]
            else:
                # Pre-registered: the boundary is a no-trade, decided in
                # advance so it is not settled by whichever way it pays.
                diag["skipped_reason"] = SKIP_ON_EMA
                return diag
        else:
            directions = ["long", "short"]

        # Index arithmetic on numpy views: this loop runs per session over
        # 1-minute bars for seven years, and .iloc per bar dominates otherwise.
        idx = session.index
        opens = session["open"].to_numpy(float)
        highs = session["high"].to_numpy(float)
        lows = session["low"].to_numpy(float)
        closes = session["close"].to_numpy(float)

        first_after = int(idx.get_indexer([after.index[0]])[0])
        times = np.array([t.time() for t in idx])
        live = np.flatnonzero(times < p.entry_cancel_time)
        last_entry = int(live[-1]) if live.size else -1
        deadline = self._exit_deadline(day)
        holdable = np.flatnonzero(times < deadline)
        last_hold = int(holdable[-1]) if holdable.size else -1
        if last_entry < first_after or last_hold < first_after:
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag

        levels = {
            "long": or_high + p.entry_offset,
            "short": or_low - p.entry_offset,
        }
        fills = {
            d: self._fill_index(highs, lows, opens, first_after, last_entry, d, levels[d])
            for d in directions
        }
        filled = {d: f for d, f in fills.items() if f is not None}
        if not filled:
            diag["skipped_reason"] = SKIP_NO_FILL
            return diag

        earliest = min(f[0] for f in filled.values())
        contenders = [d for d, f in filled.items() if f[0] == earliest]

        outcomes = {}
        for d in contenders:
            fill_i, fill_price = filled[d]
            if d == "long":
                stop, target = fill_price - p.stop_points, fill_price + p.target_points
            else:
                stop, target = fill_price + p.stop_points, fill_price - p.target_points
            exit_i, exit_price, reason = self._simulate_exit(
                highs, lows, closes, fill_i, last_hold, d, stop, target,
                opens=opens, n_bars=len(session),
            )
            sign = 1.0 if d == "long" else -1.0
            outcomes[d] = {
                "fill_i": fill_i, "fill_price": fill_price, "stop": stop,
                "target": target, "exit_i": exit_i, "exit_price": exit_price,
                "reason": reason, "points": (exit_price - fill_price) * sign,
            }

        if len(contenders) > 1:
            # Both stops reached inside one bar. Which filled first is
            # unobservable at this resolution, so resolve it the way the
            # stop-first rule resolves its own ambiguity: pessimistically.
            diag["ambiguous_both_stops"] = True
            direction = min(outcomes, key=lambda d: outcomes[d]["points"])
        else:
            direction = contenders[0]

        chosen = outcomes[direction]
        entry_ts = idx[chosen["fill_i"]]

        # Rule 9: the guard is consulted at runtime, not assumed satisfied by
        # the 11:30 cancel. An early-close session moves the deadline.
        if not rules.is_entry_allowed(entry_ts, self.early_close_dates, self.roll_dates):
            diag["skipped_reason"] = SKIP_BLOCKED
            return diag

        exit_ts = idx[chosen["exit_i"]]
        entry_col = "entry_long" if direction == "long" else "entry_short"
        exit_col = "exit_long" if direction == "long" else "exit_short"

        out.loc[entry_ts, entry_col] = True
        out.loc[entry_ts, "entry_price"] = chosen["fill_price"]
        out.loc[entry_ts, "stop_price"] = chosen["stop"]
        out.loc[entry_ts, "target_price"] = chosen["target"]
        out.loc[entry_ts, "opening_range_height"] = height
        out.loc[exit_ts, exit_col] = True
        out.loc[exit_ts, "exit_price"] = chosen["exit_price"]
        out.loc[exit_ts, "exit_reason"] = chosen["reason"]

        diag.update(
            direction=direction,
            entered=True,
            same_bar_entry_exit=chosen["exit_i"] == chosen["fill_i"],
        )
        return diag

    # -- interface ---------------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Signals on the 1-minute index of ``bars``.

        Each session is evaluated using only bars inside it plus the injected
        daily EMA, which is a function of *earlier* sessions only. Generating
        over a long span and slicing by date therefore gives exactly the same
        signals as generating over each slice separately.
        """
        rth = bars.between_time(RTH_START, RTH_LAST_BAR)
        if rth.empty:
            self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
            return validate_signals(empty_signals(rth.index))

        out = empty_signals(rth.index)
        for col in ("entry_price", "stop_price", "target_price", "exit_price",
                    "opening_range_height"):
            out[col] = pd.Series(pd.NA, index=rth.index, dtype="Float64")
        out["exit_reason"] = pd.Series(pd.NA, index=rth.index, dtype="string")

        rows: dict[date_type, dict] = {}
        for day, session in rth.groupby(rth.index.date):
            if rules.is_roll_day(day, self.roll_dates):
                rows[day] = {
                    **{c: None for c in DIAGNOSTIC_COLUMNS},
                    "skipped_reason": SKIP_ROLL,
                    "entered": False,
                    "ambiguous_both_stops": False,
                    "same_bar_entry_exit": False,
                }
                continue
            rows[day] = self._session_signals(session, out, day)

        diagnostics = pd.DataFrame.from_dict(rows, orient="index", columns=DIAGNOSTIC_COLUMNS)
        diagnostics.index.name = "session_date"
        self.diagnostics = diagnostics
        return validate_signals(out)
