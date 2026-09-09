"""15-minute opening-range breakout, flat at 10:30 ET (plumbing reproduction).

Mechanism, as submitted
-----------------------
Range = the 09:30-09:45 candle high/low on MES. At 09:45 a buy stop is placed
``trigger_offset`` above the range high and a sell stop the same distance below
the range low; the first fill wins and the other order is cancelled. The day is
skipped if the range is wider than ``max_range_points``. A fixed
``stop_points`` stop and ``target_points`` target bracket the fill. Unfilled
orders are cancelled and any open position is flattened at 10:30 ET. One trade
per day.

What is *not* here
------------------
Contract count (the submission says 4), dollar risk, the daily loss limit, the
entry cutoff and the forced flatten belong to the execution layer, which reads
:mod:`rules`. This module publishes booleans plus informational price levels
and nothing else. The 09:30-10:30 window and the fixed point brackets are part
of the pattern definition, not risk decisions; ``rules.is_entry_allowed`` and
``rules.flatten_deadline`` are consulted only as backstops that can make the
window narrower, never wider.

Lookahead
---------
A resting stop order fills intrabar, which is not knowable from bars strictly
before ``T``. The no-lookahead rendering used here is: a trigger *touch* is
detected on a completed 1-minute bar, and the entry is marked on the *next*
bar, to be filled at that bar's open (see :mod:`base` for the convention). On
1-minute data the difference between the two is one bar of path, and it is not
systematically favourable - a runaway breakout fills worse than the stop
price, a spike-and-fade fills better.

Two further conventions, stated rather than hidden:

* If a single bar touches both triggers, which one filled first is
  unresolvable at this resolution, so the day is stood aside rather than
  guessed.
* Stop/target scanning starts on the bar *after* the entry bar, matching
  ``strategies/orb.py``. The entry minute itself is not scanned, which is
  mildly optimistic for trades that would have been stopped inside their first
  minute.
"""

import base
import rules
import pandas as pd

RTH_START = "09:30"
RTH_LAST_BAR = "15:59"

OPEN_TIME = pd.Timestamp("2000-01-01 09:30").time()
DEFAULT_FLAT_TIME = pd.Timestamp("2000-01-01 10:30").time()


class ORBFlat1030Params:
    """Parameters for the 09:30-10:30 opening-range breakout."""

    def __init__(
        self,
        opening_range_minutes=15,
        trigger_offset=1.0,
        max_range_points=10.0,
        stop_points=10.0,
        target_points=18.0,
        flat_time=None,
    ):
        #: Length of the opening range in minutes (15 -> 09:30-09:45).
        self.opening_range_minutes = int(opening_range_minutes)
        #: Stop-order distance beyond the range extreme, in index points.
        self.trigger_offset = float(trigger_offset)
        #: Days whose opening range exceeds this many points are skipped.
        self.max_range_points = float(max_range_points)
        #: Fixed stop distance from the fill, in index points.
        self.stop_points = float(stop_points)
        #: Fixed target distance from the fill, in index points.
        self.target_points = float(target_points)
        #: Orders are cancelled and any open position flattened at this time.
        self.flat_time = DEFAULT_FLAT_TIME if flat_time is None else flat_time

        if self.opening_range_minutes <= 0:
            raise ValueError("opening_range_minutes must be positive")
        if self.trigger_offset < 0:
            raise ValueError("trigger_offset must not be negative")
        if self.max_range_points <= 0:
            raise ValueError("max_range_points must be positive")
        if self.stop_points <= 0:
            raise ValueError("stop_points must be positive")
        if self.target_points <= 0:
            raise ValueError("target_points must be positive")

    def __repr__(self):  # pragma: no cover - trivial
        return (
            f"<ORBFlat1030Params opening_range_minutes={self.opening_range_minutes} "
            f"trigger_offset={self.trigger_offset} "
            f"max_range_points={self.max_range_points} "
            f"stop_points={self.stop_points} target_points={self.target_points} "
            f"flat_time={self.flat_time}>"
        )


def rth_minute_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """The regular-hours slice of ``bars``, sorted. Signals are indexed on this."""
    if bars is None or len(bars) == 0:
        return bars.copy() if bars is not None else pd.DataFrame()
    return bars.between_time(RTH_START, RTH_LAST_BAR).sort_index()


class OpeningRangeBreakoutFlat1030(base.Strategy):
    """Opening-range breakout on 1-minute bars, entry window closing at 10:30."""

    name = "orborb_flat_1030"

    def __init__(self, params=None, roll_dates=(), early_close_dates=()):
        self.params = params or ORBFlat1030Params()
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)

    # -- internals ---------------------------------------------------------

    def _opening_end(self):
        return (
            pd.Timestamp("2000-01-01 09:30")
            + pd.Timedelta(minutes=self.params.opening_range_minutes)
        ).time()

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame) -> None:
        """Fill ``out`` rows for one session, in place. At most one trade."""
        p = self.params

        # An incomplete opening range is not the range the rule specifies.
        if session.index[0].time() > OPEN_TIME:
            return

        opening_end = self._opening_end()
        opening = session[session.index.time < opening_end]
        if opening.empty:
            return

        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        height = or_high - or_low
        if height <= 0 or height > p.max_range_points:
            return

        up_trigger = or_high + p.trigger_offset
        dn_trigger = or_low - p.trigger_offset

        after = session[session.index.time >= opening_end]
        if len(after) < 2:
            return

        # The mechanism's own 10:30 cancel, tightened by the rules-layer
        # deadline if that ever falls earlier (early close).
        window_end = min(
            p.flat_time, rules.flatten_deadline(after.index[0], self.early_close_dates)
        )

        entry_i = None
        direction = None
        for i in range(len(after) - 1):
            ts = after.index[i]
            if ts.time() >= window_end:
                return  # orders cancelled unfilled
            bar = after.iloc[i]
            hit_up = float(bar["high"]) >= up_trigger
            hit_dn = float(bar["low"]) <= dn_trigger
            if hit_up and hit_dn:
                return  # which stop filled first is unresolvable; stand aside
            if not (hit_up or hit_dn):
                continue

            action_ts = after.index[i + 1]
            if action_ts.time() >= window_end:
                return  # nothing left to act on before the cancel
            if not rules.is_entry_allowed(
                action_ts, self.early_close_dates, self.roll_dates
            ):
                return

            direction = "long" if hit_up else "short"
            entry_i = i + 1
            break

        if entry_i is None:
            return

        action_ts = after.index[entry_i]
        entry_price = float(after.iloc[entry_i]["open"])
        if direction == "long":
            stop = entry_price - p.stop_points
            target = entry_price + p.target_points
            out.loc[action_ts, "entry_long"] = True
        else:
            stop = entry_price + p.stop_points
            target = entry_price - p.target_points
            out.loc[action_ts, "entry_short"] = True

        out.loc[action_ts, "stop_price"] = stop
        out.loc[action_ts, "target_price"] = target
        out.loc[action_ts, "opening_range_height"] = height

        self._mark_exit(after, entry_i, direction, stop, target, window_end, out)

    def _mark_exit(self, after, entry_i, direction, stop, target, window_end, out):
        """Mark where the trade leaves: stop, target, or the 10:30 flatten."""
        exit_col = "exit_long" if direction == "long" else "exit_short"
        last_holdable = None

        for j in range(entry_i + 1, len(after)):
            ts = after.index[j]
            if ts.time() >= window_end:
                out.loc[ts, exit_col] = True
                out.loc[ts, "exit_reason"] = "time_flat"
                out.loc[ts, "exit_price"] = float(after.iloc[j]["open"])
                return
            bar = after.iloc[j]
            if direction == "long":
                hit_stop = float(bar["low"]) <= stop
                hit_target = float(bar["high"]) >= target
            else:
                hit_stop = float(bar["high"]) >= stop
                hit_target = float(bar["low"]) <= target

            if hit_stop or hit_target:
                # Both inside one bar is unresolvable here; book the stop
                # rather than an optimistic win.
                out.loc[ts, exit_col] = True
                out.loc[ts, "exit_reason"] = "stop" if hit_stop else "target"
                out.loc[ts, "exit_price"] = stop if hit_stop else target
                return
            last_holdable = j

        if last_holdable is None:
            return  # data ends on the entry bar; the execution layer flattens

        ts = after.index[last_holdable]
        out.loc[ts, exit_col] = True
        out.loc[ts, "exit_reason"] = "data_end"
        out.loc[ts, "exit_price"] = float(after.iloc[last_holdable]["close"])

    # -- interface ---------------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = rth_minute_bars(bars)
        if len(frame) == 0:
            return base.validate_signals(base.empty_signals(frame.index))

        out = base.empty_signals(frame.index)
        out["stop_price"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        out["target_price"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        out["exit_price"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        out["opening_range_height"] = pd.Series(
            pd.NA, index=frame.index, dtype="Float64"
        )
        out["exit_reason"] = pd.Series(pd.NA, index=frame.index, dtype="string")

        for day, session in frame.groupby(frame.index.date):
            if rules.is_roll_day(day, self.roll_dates):
                continue
            self._session_signals(session, out)

        return base.validate_signals(out)

    def bars_for_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """The bars the signals are indexed on, for the backtest."""
        return rth_minute_bars(bars)