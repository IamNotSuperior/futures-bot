"""15-minute opening-range breakout, full-day arm - plumbing test.

Reproduces entry 4 of the hypothesis log with the trend filter OFF. The
opening range is the 09:30-09:45 window. A stop order is worked one point
beyond each side of that range from 09:45; the first side to trade wins and
the other side is stood down. Days whose opening range is wider than
``max_range_points`` are skipped. Exits are a fixed 10-point stop and an
18-point target; anything still open at the execution layer's flatten
deadline is marked out at the last holdable bar.

No edge is claimed here. The mechanism is a transcription of an already-logged
idea so the pipeline can be exercised against a known verdict.

Signals only
------------
The strategy publishes booleans and informational price levels. Position
size, the contract cap, the daily loss limit, the minimum hold and the hard
forced flatten are the execution layer's business and are read from
:mod:`rules` there, not here. ``contracts`` is a transcription of the size the
operator wrote on the submission and nothing else.

Lookahead
---------
Decisions run on 1-minute RTH bars. A trigger touched inside bar ``t`` is only
known once bar ``t`` has completed, so the entry is published at ``t+1`` and
filled at that bar's open, per the convention in :mod:`base`. This is the
conservative rendering of a resting stop order: a real stop would have filled
at the trigger price inside bar ``t``, which is not knowable before ``t``
closes. The difference is a known, deliberate divergence from the submitted
mechanism and biases fills against the strategy or in its favour depending on
the bar - it is not modelled as a free improvement.

Two ambiguities are resolved pessimistically rather than by assumption:

* both triggers touched inside the same bar - which stop filled first is
  unresolvable at 1-minute resolution, so the day is stood down;
* stop and target both touched inside the same bar - the stop is assumed to
  have filled first.
"""

import numpy as np
import pandas as pd

import base
import rules

RTH_START = "09:30"
RTH_LAST_BAR = "15:59"
RTH_START_MINUTE = 9 * 60 + 30


def _minute_of_day(ts) -> int:
    """ET wall-clock minutes since midnight for ``ts``."""
    return int(ts.hour) * 60 + int(ts.minute)


class ORBFullDayParams:
    """Parameters for the full-day (no trend filter) opening-range breakout."""

    def __init__(
        self,
        opening_range_minutes: int = 15,
        trigger_offset: float = 1.0,
        max_range_points: float = 10.0,
        stop_points: float = 10.0,
        target_points: float = 18.0,
        entry_window_end_minute: int = 11 * 60 + 30,
    ) -> None:
        if opening_range_minutes <= 0:
            raise ValueError("opening_range_minutes must be positive")
        if trigger_offset < 0:
            raise ValueError("trigger_offset must not be negative")
        if max_range_points <= 0:
            raise ValueError("max_range_points must be positive")
        if stop_points <= 0:
            raise ValueError("stop_points must be positive")
        if target_points <= 0:
            raise ValueError("target_points must be positive")
        if entry_window_end_minute <= RTH_START_MINUTE + opening_range_minutes:
            raise ValueError("entry_window_end_minute must be after the opening range")

        #: Length of the opening range in minutes (15 -> 09:30-09:45).
        self.opening_range_minutes = int(opening_range_minutes)
        #: Points beyond the range where the stop order rests.
        self.trigger_offset = float(trigger_offset)
        #: Days with a range wider than this are skipped entirely.
        self.max_range_points = float(max_range_points)
        #: Fixed stop distance in points.
        self.stop_points = float(stop_points)
        #: Fixed target distance in points.
        self.target_points = float(target_points)
        #: Unfilled entry orders are cancelled at this ET minute-of-day.
        self.entry_window_end_minute = int(entry_window_end_minute)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ORBFullDayParams(opening_range_minutes={self.opening_range_minutes}, "
            f"trigger_offset={self.trigger_offset}, "
            f"max_range_points={self.max_range_points}, "
            f"stop_points={self.stop_points}, "
            f"target_points={self.target_points}, "
            f"entry_window_end_minute={self.entry_window_end_minute})"
        )


class ORBFullDayTest(base.Strategy):
    """15-minute opening-range breakout on 1-minute bars, no trend filter."""

    name = "orb_full_day_test"
    contracts = 4

    def __init__(
        self,
        params: "ORBFullDayParams | None" = None,
        roll_dates=(),
        early_close_dates=(),
    ) -> None:
        self.params = params or ORBFullDayParams()
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)

    # -- internals ---------------------------------------------------------

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame) -> None:
        """Fill ``out`` rows for one session, in place. At most one trade."""
        p = self.params
        idx = session.index
        n = len(idx)
        if n == 0:
            return

        mins = np.fromiter((_minute_of_day(ts) for ts in idx), dtype=int, count=n)
        opening_end = RTH_START_MINUTE + p.opening_range_minutes
        opening_mask = (mins >= RTH_START_MINUTE) & (mins < opening_end)
        if not bool(opening_mask.any()):
            return

        opening = session[opening_mask]
        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        height = or_high - or_low
        if not np.isfinite(height) or height < 0.0:
            return
        if height > p.max_range_points:
            return  # range filter: stand down for the day

        buy_trigger = or_high + p.trigger_offset
        sell_trigger = or_low - p.trigger_offset

        for i in range(n):
            if mins[i] < opening_end:
                continue
            if mins[i] >= p.entry_window_end_minute:
                return  # unfilled entry orders cancelled

            bar = session.iloc[i]
            hit_long = float(bar["high"]) >= buy_trigger
            hit_short = float(bar["low"]) <= sell_trigger

            if hit_long and hit_short:
                # Which side filled first is unknowable at this resolution.
                return
            if not (hit_long or hit_short):
                continue

            if i + 1 >= n:
                return  # trigger on the final bar; nothing left to act on

            action_ts = idx[i + 1]
            if not rules.is_entry_allowed(
                action_ts, self.early_close_dates, self.roll_dates
            ):
                return

            entry_price = float(session.iloc[i + 1]["open"])
            if hit_long:
                direction = "long"
                stop = entry_price - p.stop_points
                target = entry_price + p.target_points
                out.loc[action_ts, "entry_long"] = True
            else:
                direction = "short"
                stop = entry_price + p.stop_points
                target = entry_price - p.target_points
                out.loc[action_ts, "entry_short"] = True

            out.loc[action_ts, "stop_price"] = stop
            out.loc[action_ts, "target_price"] = target
            out.loc[action_ts, "trigger_price"] = (
                buy_trigger if direction == "long" else sell_trigger
            )
            out.loc[action_ts, "opening_range_height"] = height

            self._mark_exit(session, i + 1, direction, stop, target, out)
            return  # one trade per day

    def _mark_exit(
        self,
        session: pd.DataFrame,
        action_i: int,
        direction: str,
        stop: float,
        target: float,
        out: pd.DataFrame,
    ):
        """Mark where the trade leaves. Returns the exit bar's position or None."""
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
                hit_stop = float(bar["low"]) <= stop
                hit_target = float(bar["high"]) >= target
            else:
                hit_stop = float(bar["high"]) >= stop
                hit_target = float(bar["low"]) <= target

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
        rth = self.bars_for_signals(bars)
        if rth.empty:
            return base.validate_signals(base.empty_signals(rth.index))

        out = base.empty_signals(rth.index)
        out["stop_price"] = pd.Series(pd.NA, index=rth.index, dtype="Float64")
        out["target_price"] = pd.Series(pd.NA, index=rth.index, dtype="Float64")
        out["trigger_price"] = pd.Series(pd.NA, index=rth.index, dtype="Float64")
        out["exit_price"] = pd.Series(pd.NA, index=rth.index, dtype="Float64")
        out["opening_range_height"] = pd.Series(pd.NA, index=rth.index, dtype="Float64")
        out["exit_reason"] = pd.Series(pd.NA, index=rth.index, dtype="string")

        for day, session in rth.groupby(rth.index.date):
            if rules.is_roll_day(day, self.roll_dates):
                continue
            self._session_signals(session, out)

        return base.validate_signals(out)

    def bars_for_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """The RTH 1-minute bars the signals are indexed on, for the backtest."""
        if bars.empty:
            return bars.copy()
        return bars.between_time(RTH_START, RTH_LAST_BAR).sort_index()