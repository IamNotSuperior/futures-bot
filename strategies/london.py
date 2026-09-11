"""Entry 6: London open breakout of the overnight range.

Implements hypothesis entry 6 in ``research/hypotheses.md``, frozen at commit
``068927d`` with its open implementation details fixed at ``6fe805f``. Every
constant is fixed by that entry. The only pre-registered variations are the
target multiple (1x and 2x, run as independent arms) and the trend filter
(ON/OFF).

The shape of a session
----------------------
1. The overnight range is the high and low of 19:00-02:55 ET. **That window
   begins on the previous calendar day**, so a Monday trade is measured against
   a range that started Sunday at the Globex reopen.
2. The session is skipped if the range is wider than 40 points, because the
   risk-based size would fall below one contract.
3. Between 03:00 and 05:00 the first 5-minute candle to *close* outside the
   range triggers; the trade is taken at the next candle's open.
4. Stop at the far side of the range, target a multiple of the range height.
5. Anything still open is flattened at the 09:25 bar's open.

The date the range belongs to
-----------------------------
This is the part that is easy to get wrong, and ``tests/test_london.py``
asserts it directly. ``rules.session_date`` is the calendar date of a timestamp
and the range window is not: bars from 19:00 onward belong to the *next* day's
London session. :func:`london_date` is the only place that mapping lives, and
nothing else in this module is allowed to assume the two agree.

Sizing
------
``contracts = floor($200 / (range_pts x $5))`` clamped to [1, 5], computed from
the **range height** rather than the realised stop distance. Because the entry
sits beyond the range edge, the real stop is further away than the range height
and actual risk therefore exceeds $200 by the overshoot. Entry 6 keeps that
deliberately and requires the overshoot and realised risk to be reported, so
the breach is measured rather than assumed small. Size varies per session, so
**P&L is not linear in contract count here** the way it is for entries 4 and 5.

Entry 10 (``size_on="stop"``)
-----------------------------
Entry 6's addendum of 2026-09-05 measured that rule permitting a stop larger
than the daily loss limit, and bound every *new* strategy to size off the
realised stop instead. Entry 10 is the same signal held through the US session
(``flatten_time=15:55``, ``early_close_flatten_time=12:55``) and is a new
strategy, so it carries ``size_on="stop"``: ``floor(min($200, daily loss limit)
/ (stop_distance x point_value))``, and the session is **skipped** when one
contract already risks more than the budget. The defaults remain entry 6's, so
entries 6 and 7 reproduce unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as date_type
from datetime import time, timedelta
from typing import Collection

import numpy as np
import pandas as pd

from base import Strategy, empty_signals, validate_signals
import rules

OHLC_AGG = {"open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"}

DIAGNOSTIC_COLUMNS = [
    "range_high", "range_low", "range_pts", "range_bars", "contracts",
    "ema", "open_0300", "direction", "entry_price", "overshoot",
    "stop_distance", "risk_dollars", "skipped_reason", "entered",
    "same_bar_entry_exit",
]

SKIP_ROLL = "roll_day"
SKIP_WIDE_RANGE = "range_over_cap"
SKIP_THIN_RANGE = "range_coverage_too_thin"
SKIP_NO_BARS = "incomplete_session"
SKIP_NO_BREAK = "no_break_in_window"
SKIP_UNSEEDED = "ema_unseeded"
SKIP_FILTERED = "filter_blocked"
SKIP_BLOCKED = "entry_blocked_by_rules"
#: ``size_on="stop"`` only: one contract at the realised stop already risks
#: more than the budget, so nothing is taken.
SKIP_STOP_OVER_BUDGET = "stop_over_risk_budget"

SIZING_RULES = ("range", "stop")

#: Points per contract. Imported rather than restated where possible; the
#: contract spec lives in the engine, which strategies do not import.
POINT_VALUE = 5.0


def london_date(ts: pd.Timestamp) -> date_type:
    """The London session date a bar belongs to.

    Bars from 19:00 ET onward belong to the **next** calendar day's session.
    This is the single definition of that mapping; ``rules.session_date`` is a
    different thing and the two must not be conflated.
    """
    return (ts + timedelta(days=1)).date() if ts.hour >= 19 else ts.date()


@dataclass(frozen=True)
class LondonParams:
    """Entry 6's parameters. Fixed by the entry, not searched."""

    range_start: time = time(19, 0)
    range_end: time = time(2, 55)        # inclusive
    entry_start: time = time(3, 0)
    entry_end: time = time(5, 0)         # exclusive, governs the trigger
    flatten_time: time = time(9, 25)
    bar_minutes: int = 5
    max_range_points: float = 40.0
    risk_dollars: float = 200.0
    max_contracts: int = 5
    target_multiple: float = 1.0         # the pre-registered arm: 1.0 or 2.0
    use_trend_filter: bool = True
    min_range_bars: int = 60             # of 107 in a complete window
    #: Dollars per point. MES is $5.00, MNQ $2.00. The risk rule is stated in
    #: dollars, so its expression in points follows the contract - entry 7's
    #: MNQ replication keeps the same $200 budget and its skip threshold
    #: becomes 100 points rather than 40.
    point_value: float = 5.0
    #: Entry 10: the flatten on an early-close session, where ``flatten_time``
    #: may not exist. None means the rules deadline binds (rule 2's backstop).
    early_close_flatten_time: time | None = None
    #: ``"range"`` is entry 6's rule; ``"stop"`` is the log's corrected rule
    #: for new strategies (entry 6's addendum, 2026-09-05).
    size_on: str = "range"

    def __post_init__(self) -> None:
        if self.target_multiple <= 0:
            raise ValueError("target_multiple must be positive")
        if self.max_range_points <= 0 or self.risk_dollars <= 0:
            raise ValueError("max_range_points and risk_dollars must be positive")
        if not 1 <= self.max_contracts <= rules.POSITION_CAP:
            raise ValueError(
                f"max_contracts must be within the position cap "
                f"({rules.POSITION_CAP}), got {self.max_contracts}"
            )
        if not self.entry_start < self.entry_end <= self.flatten_time:
            raise ValueError("entry_start < entry_end <= flatten_time required")
        if self.size_on not in SIZING_RULES:
            raise ValueError(f"size_on must be one of {SIZING_RULES}, got {self.size_on!r}")
        if self.early_close_flatten_time is not None:
            if not self.entry_end <= self.early_close_flatten_time < rules.EARLY_SESSION_CLOSE:
                raise ValueError(
                    f"early_close_flatten_time must sit between entry_end and the "
                    f"early close ({rules.EARLY_SESSION_CLOSE}); got "
                    f"{self.early_close_flatten_time}"
                )


def contracts_for(range_pts: float, params: LondonParams) -> int:
    """``floor(risk / (range_pts x point_value))``, clamped to [1, max]."""
    raw = math.floor(params.risk_dollars / (range_pts * params.point_value))
    return int(min(max(raw, 1), params.max_contracts))


def contracts_for_stop(stop_distance: float, params: LondonParams) -> int:
    """The log's corrected rule: size off the realised stop, capped twice.

    ``floor(min(risk, daily loss limit) / (stop_distance x point_value))``,
    clamped to the position cap. **Zero means skip**: one contract at this
    stop already risks more than the budget, and clamping up to one - which
    is what the range-based rule does - would be the breach entry 6's
    addendum measured. Both caps are read from ``rules``; neither is restated.
    """
    per_contract = stop_distance * params.point_value
    if per_contract <= 0:
        return 0
    if per_contract > params.risk_dollars:
        return 0
    budget = min(params.risk_dollars, rules.DAILY_LOSS_LIMIT)
    raw = math.floor(budget / per_contract)
    return int(min(raw, params.max_contracts, rules.POSITION_CAP))


def resample_continuous(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample the whole series, not session by session.

    ORB resamples per session so a bucket cannot straddle the overnight break.
    Here the overnight break is the point, so the series is resampled straight
    through and empty buckets dropped.
    """
    cols = {k: v for k, v in OHLC_AGG.items() if k in bars.columns}
    out = bars.resample(f"{minutes}min", label="left", closed="left").agg(cols)
    return out.dropna(subset=["open"]).sort_index()


class LondonBreakout(Strategy):
    """First break of the overnight range after the European cash open."""

    name = "london"

    def __init__(
        self,
        params: LondonParams | None = None,
        trend_ema: pd.Series | None = None,
        roll_dates: Collection[date_type] = (),
        early_close_dates: Collection[date_type] = (),
    ) -> None:
        self.params = params or LondonParams()
        if self.params.use_trend_filter and trend_ema is None:
            raise ValueError(
                "use_trend_filter=True needs a trend_ema series; build one with "
                "trend.intraday_ema(bars, minutes=5, period=200)"
            )
        self.trend_ema = trend_ema
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)
        self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    # -- internals ---------------------------------------------------------

    def _ema_at(self, ts: pd.Timestamp) -> float:
        if self.trend_ema is None:
            return float("nan")
        try:
            value = self.trend_ema.loc[ts]
        except KeyError:
            return float("nan")
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value)

    def _session_signals(self, group: pd.DataFrame, five: pd.DataFrame,
                         out: pd.DataFrame, day: date_type) -> dict:
        """One London session. ``group`` is 1-minute, ``five`` is its 5-minute."""
        p = self.params
        diag = {c: None for c in DIAGNOSTIC_COLUMNS}
        diag.update(entered=False, same_bar_entry_exit=False)

        times = np.array([t.time() for t in five.index])
        range_mask = (times >= p.range_start) | (times <= p.range_end)
        range_bars = five[range_mask]
        if len(range_bars) < p.min_range_bars:
            diag["skipped_reason"] = (
                SKIP_THIN_RANGE if len(range_bars) else SKIP_NO_BARS
            )
            diag["range_bars"] = int(len(range_bars))
            return diag

        hi = float(range_bars["high"].max())
        lo = float(range_bars["low"].min())
        span = hi - lo
        diag.update(range_high=hi, range_low=lo, range_pts=span,
                    range_bars=int(len(range_bars)))
        if span <= 0:
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag
        if span * p.point_value > p.risk_dollars:
            diag["skipped_reason"] = SKIP_WIDE_RANGE
            return diag

        window = five[(times >= p.entry_start) & (times < p.entry_end)]
        if window.empty:
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag

        first_ts = window.index[0]
        diag["open_0300"] = float(window.iloc[0]["open"])
        if p.use_trend_filter:
            ema_value = self._ema_at(first_ts)
            diag["ema"] = ema_value
            if not np.isfinite(ema_value):
                diag["skipped_reason"] = SKIP_UNSEEDED
                return diag

        # The first close outside the range, in order. First break only.
        five_idx = five.index
        trigger_pos = None
        direction = None
        for ts in window.index:
            close = float(five.loc[ts, "close"])
            if close > hi:
                direction = "long"
            elif close < lo:
                direction = "short"
            else:
                continue
            trigger_pos = int(five_idx.get_indexer([ts])[0])
            break

        if trigger_pos is None:
            diag["skipped_reason"] = SKIP_NO_BREAK
            return diag

        # The trigger is what the window governs; the action bar may sit past
        # 05:00 and the trade is still taken.
        if trigger_pos + 1 >= len(five_idx):
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag
        action_ts = five_idx[trigger_pos + 1]

        if p.use_trend_filter:
            ema_value = diag["ema"]
            if ((direction == "long" and not diag["open_0300"] > ema_value)
                    or (direction == "short" and not diag["open_0300"] < ema_value)):
                diag.update(direction=direction, skipped_reason=SKIP_FILTERED)
                return diag

        if not rules.is_entry_allowed(action_ts, self.early_close_dates,
                                      self.roll_dates):
            diag.update(direction=direction, skipped_reason=SKIP_BLOCKED)
            return diag

        # Fills resolve on 1-minute bars from the action bar onward.
        try:
            start = int(group.index.get_indexer([action_ts])[0])
        except Exception:  # pragma: no cover - action bar always exists
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag
        if start < 0:
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag

        entry_price = float(group.iloc[start]["open"])
        if direction == "long":
            stop, target = lo, entry_price + p.target_multiple * span
            overshoot = entry_price - hi
        else:
            stop, target = hi, entry_price - p.target_multiple * span
            overshoot = lo - entry_price
        stop_distance = abs(entry_price - stop)

        if p.size_on == "stop":
            contracts = contracts_for_stop(stop_distance, p)
            if contracts == 0:
                diag.update(direction=direction, entry_price=entry_price,
                            overshoot=float(overshoot), stop_distance=stop_distance,
                            skipped_reason=SKIP_STOP_OVER_BUDGET)
                return diag
        else:
            contracts = contracts_for(span, p)

        flatten = self._flatten_for(day)
        g_times = np.array([t.time() for t in group.index])
        g_dates = np.array([t.date() for t in group.index])
        holdable = np.flatnonzero((g_times < flatten) & (g_dates == day))
        last_hold = int(holdable[-1]) if holdable.size else -1
        if last_hold < start:
            diag["skipped_reason"] = SKIP_NO_BARS
            return diag

        exit_i, exit_price, reason = self._simulate_exit(
            group, start, last_hold, direction, stop, target,
            f"flatten_{flatten.strftime('%H%M')}",
        )

        exit_ts = group.index[exit_i]
        entry_col = "entry_long" if direction == "long" else "entry_short"
        exit_col = "exit_long" if direction == "long" else "exit_short"
        out.loc[action_ts, entry_col] = True
        out.loc[action_ts, "entry_price"] = entry_price
        out.loc[action_ts, "stop_price"] = stop
        out.loc[action_ts, "target_price"] = target
        out.loc[action_ts, "range_pts"] = span
        out.loc[action_ts, "contracts"] = contracts
        out.loc[exit_ts, exit_col] = True
        out.loc[exit_ts, "exit_price"] = exit_price
        out.loc[exit_ts, "exit_reason"] = reason

        diag.update(
            direction=direction, entered=True, entry_price=entry_price,
            contracts=contracts, overshoot=float(overshoot),
            stop_distance=stop_distance,
            risk_dollars=stop_distance * p.point_value * contracts,
            same_bar_entry_exit=exit_i == start,
        )
        return diag

    def _flatten_for(self, day: date_type) -> time:
        """The flatten that applies on ``day``.

        Entry 6's 09:25 is the same on every day. Entry 10's 15:55 does not
        exist on an early-close session, so those days use
        ``early_close_flatten_time`` (12:55) when it is set; when it is not,
        the rules deadline - the exchange close - is the backstop, so no
        configuration can hold a position past a close (rule 2).
        """
        p = self.params
        if day in self.early_close_dates:
            if p.early_close_flatten_time is not None:
                return p.early_close_flatten_time
            return min(p.flatten_time, rules.flatten_deadline(day, self.early_close_dates))
        return p.flatten_time

    def _simulate_exit(self, group, start, last, direction, stop, target,
                       flatten_reason: str = "flatten_0925"):
        """Walk 1-minute bars from the entry bar to the flatten.

        The entry bar is included: the fill happened inside it and the rest of
        that minute can still reach the stop. The flatten lands on the open of
        the bar *after* the last holdable one - the 09:25 bar for entry 6, the
        15:55 bar for entry 10 - and is labelled with that time.
        """
        highs = group["high"].to_numpy(float)
        lows = group["low"].to_numpy(float)
        opens = group["open"].to_numpy(float)
        closes = group["close"].to_numpy(float)

        for i in range(start, last + 1):
            if direction == "long":
                hit_stop, hit_target = lows[i] <= stop, highs[i] >= target
            else:
                hit_stop, hit_target = highs[i] >= stop, lows[i] <= target
            if hit_stop or hit_target:
                # Both inside one bar is unresolvable here; assume the stop.
                if hit_stop:
                    return i, stop, "stop"
                return i, target, "target"

        if last + 1 < len(opens):
            return last + 1, float(opens[last + 1]), flatten_reason
        return last, float(closes[last]), flatten_reason

    # -- interface ---------------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Signals on the 1-minute index of ``bars``.

        Sessions are grouped by :func:`london_date`, so each group runs from
        19:00 on the previous calendar day to 18:59 on its own.
        """
        if bars.empty:
            self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
            return validate_signals(empty_signals(bars.index))

        out = empty_signals(bars.index)
        for col in ("entry_price", "stop_price", "target_price", "exit_price",
                    "range_pts", "contracts"):
            out[col] = pd.Series(pd.NA, index=bars.index, dtype="Float64")
        out["exit_reason"] = pd.Series(pd.NA, index=bars.index, dtype="string")

        five = resample_continuous(bars, self.params.bar_minutes)
        keys = pd.Index([london_date(ts) for ts in bars.index], name="london_date")
        five_keys = pd.Index([london_date(ts) for ts in five.index])

        rows: dict[date_type, dict] = {}
        for day, group in bars.groupby(keys):
            if rules.is_roll_day(day, self.roll_dates):
                rows[day] = {**{c: None for c in DIAGNOSTIC_COLUMNS},
                             "skipped_reason": SKIP_ROLL, "entered": False,
                             "same_bar_entry_exit": False}
                continue
            rows[day] = self._session_signals(
                group, five[five_keys == day], out, day
            )

        diagnostics = pd.DataFrame.from_dict(rows, orient="index",
                                             columns=DIAGNOSTIC_COLUMNS)
        diagnostics.index.name = "london_date"
        self.diagnostics = diagnostics
        return validate_signals(out)
