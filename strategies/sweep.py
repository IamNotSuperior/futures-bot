"""Entry 16: liquidity sweep of the prior session's high or low, with the
resting stop order as the counterparty. Frozen at the entry 16 freeze commit.

Levels
------
For each RTH session the levels are the previous RTH session's high and low
(09:30 to 15:59 ET bars, the same definition ``bots/chart_read.rth_sessions``
uses). A session is eligible for the real test only when its 09:30 open lies
inside [PDL, PDH]; the placebo level sits a quarter of the prior range
inside the real one on each side, with the same eligibility against its own
band.

The event
---------
A sweep on the high side is the first RTH bar from 09:30 through 14:59 whose
high exceeds the level (the cross bar), followed by the first later bar whose
close is below the level (the rejection bar), the rejection at or before
14:59. The extreme is the highest high from the cross bar to the rejection
bar; the decision bar is the bar after the rejection. Mirror for the low
side. The first rejection of either side is the session's event; a cross
with no rejection is a continuation and is counted, not traded.

The measured quantity is the sign-adjusted reversion ``m``: the decision
open minus the open thirty bars later for a high sweep (the open at 15:55
if thirty bars would pass it), the reverse for a low sweep, so that a move
back inside the range reads positive.

The fade arm
------------
Short at the decision open after a high sweep, long after a low sweep; stop
a fixed buffer beyond the sweep extreme; target the same distance the other
way (1:1); stop-first inside a bar; flat at the 15:55 open. One trade per
session, real events only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import time
from typing import Collection

import numpy as np
import pandas as pd

import rules
from base import Strategy, empty_signals, validate_signals
from engine import MES
from opex import EXIT_FLATTEN, EXIT_STOP, EXIT_TARGET, SKIP_EARLY, SKIP_MISSING, SKIP_ROLL

RTH_START = time(9, 30)
RTH_LAST = time(15, 59)
#: The last bar on which a cross or a rejection may occur.
LAST_EVENT_BAR = time(14, 59)
#: Bar labelled 15:55 - its open is the exit fill.
FLATTEN_BAR = time(15, 55)

HORIZON_BARS = 30
PLACEBO_FRACTION = 0.25
STOP_BUFFER_TICKS = 4
TICK = MES.tick_size

REAL = "real"
PLACEBO = "placebo"
CONTINUATION = "continuation"
SKIP_OUTSIDE = "open_outside_band"
SKIP_NO_EVENT = "no_event"

EVENT_COLUMNS = ["date", "year", "kind", "side", "level", "cross_ts", "reject_ts", "extreme",
                 "depth", "decision_ts", "m30", "m1555"]


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


def prior_session_levels(bars: pd.DataFrame) -> pd.DataFrame:
    """``pdh`` and ``pdl`` per session date: the previous RTH session's high and low."""
    rth = bars.between_time(RTH_START, RTH_LAST)
    if rth.empty:
        return pd.DataFrame(columns=["pdh", "pdl"])
    grouped = rth.groupby(rth.index.map(rules.session_date))
    per = pd.DataFrame({"high": grouped["high"].max(), "low": grouped["low"].min()}).sort_index()
    out = pd.DataFrame({"pdh": per["high"].shift(1), "pdl": per["low"].shift(1)}).dropna()
    out.index.name = "session_date"
    return out


# ---------------------------------------------------------------------------
# The event
# ---------------------------------------------------------------------------


def _side_event(window: pd.DataFrame, level: float, side: str) -> dict | None:
    highs = window["high"].to_numpy(float)
    lows = window["low"].to_numpy(float)
    closes = window["close"].to_numpy(float)
    crossed = highs > level if side == "high" else lows < level
    if not crossed.any():
        return None
    i = int(np.argmax(crossed))
    inside = closes < level if side == "high" else closes > level
    later = np.zeros_like(inside)
    later[i + 1:] = inside[i + 1:]
    if not later.any():
        return {"side": side, "cross_i": i, "reject_i": None}
    j = int(np.argmax(later))
    extreme = float(highs[i:j + 1].max()) if side == "high" else float(lows[i:j + 1].min())
    return {"side": side, "cross_i": i, "reject_i": j, "extreme": extreme}


def find_sweep(session: pd.DataFrame, high_level: float, low_level: float,
               continuations: bool = False) -> dict | None:
    """The session's sweep event, or None. With ``continuations`` on, a
    session that crossed without rejecting returns ``{"side": None,
    "continuation": <side>}`` instead of None."""
    times = session.index.time
    window = session[(times >= RTH_START) & (times <= LAST_EVENT_BAR)]
    events = [e for e in (_side_event(window, high_level, "high"),
                          _side_event(window, low_level, "low")) if e is not None]
    sweeps = [e for e in events if e["reject_i"] is not None]
    if not sweeps:
        if continuations:
            crossed = [e["side"] for e in events]
            return {"side": None, "continuation": crossed[0] if crossed else None}
        return None
    ev = min(sweeps, key=lambda e: (e["reject_i"], e["cross_i"]))
    reject_ts = window.index[ev["reject_i"]]
    after = session.index[session.index > reject_ts]
    if len(after) == 0:
        return None
    level = high_level if ev["side"] == "high" else low_level
    depth = ev["extreme"] - level if ev["side"] == "high" else level - ev["extreme"]
    return {"side": ev["side"], "level": float(level), "cross_ts": window.index[ev["cross_i"]],
            "reject_ts": reject_ts, "extreme": float(ev["extreme"]), "depth": float(depth),
            "decision_ts": after[0]}


def _reversion(session: pd.DataFrame, event: dict) -> tuple[float, float]:
    """Sign-adjusted 30-bar and to-15:55 reversion from the decision open."""
    opens = session["open"]
    idx = session.index
    pos = int(idx.get_loc(event["decision_ts"]))
    flatten_rows = session[idx.time == FLATTEN_BAR]
    flat_pos = int(idx.get_loc(flatten_rows.index[0])) if not flatten_rows.empty else len(idx) - 1
    later = min(pos + HORIZON_BARS, flat_pos)
    sign = -1.0 if event["side"] == "high" else 1.0
    start = float(opens.iloc[pos])
    return (sign * (float(opens.iloc[later]) - start),
            sign * (float(opens.iloc[flat_pos]) - start))


def session_events(bars: pd.DataFrame, roll_dates: Collection[date_type],
                   early_close_dates: Collection[date_type]) -> pd.DataFrame:
    """One row per (session, kind) event: real sweeps, placebo sweeps, and
    continuations at the real level on sessions with no sweep."""
    rolls, early = set(roll_dates), set(early_close_dates)
    levels = prior_session_levels(bars)
    rows: list[dict] = []
    for day, session in bars.groupby(bars.index.map(rules.session_date)):
        if day not in levels.index:
            continue
        times = session.index.time
        open_rows = session[times == RTH_START]
        if open_rows.empty or session[times == FLATTEN_BAR].empty:
            continue
        if rules.is_roll_day(day, rolls) or day in early:
            continue
        pdh, pdl = float(levels.loc[day, "pdh"]), float(levels.loc[day, "pdl"])
        span = pdh - pdl
        open_price = float(open_rows["open"].iloc[0])
        bands = {REAL: (pdh, pdl), PLACEBO: (pdh - PLACEBO_FRACTION * span, pdl + PLACEBO_FRACTION * span)}
        for kind, (hi, lo) in bands.items():
            if not (lo <= open_price <= hi) or span <= 0:
                continue
            ev = find_sweep(session, hi, lo, continuations=(kind == REAL))
            if ev is None:
                continue
            if ev["side"] is None:
                if ev["continuation"] is not None:
                    rows.append({"date": day, "year": day.year, "kind": CONTINUATION,
                                 "side": ev["continuation"], "level": hi if ev["continuation"] == "high" else lo,
                                 "cross_ts": pd.NaT, "reject_ts": pd.NaT, "extreme": np.nan,
                                 "depth": np.nan, "decision_ts": pd.NaT, "m30": np.nan,
                                 "m1555": _reversion_to_flatten_from_open(session, ev["continuation"])})
                continue
            m30, m1555 = _reversion(session, ev)
            rows.append({"date": day, "year": day.year, "kind": kind, "side": ev["side"],
                         "level": ev["level"], "cross_ts": ev["cross_ts"], "reject_ts": ev["reject_ts"],
                         "extreme": ev["extreme"], "depth": ev["depth"],
                         "decision_ts": ev["decision_ts"], "m30": m30, "m1555": m1555})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _reversion_to_flatten_from_open(session: pd.DataFrame, side: str) -> float:
    """For a continuation: the 15:55 open against the 09:30 open, signed so
    that a move back inside the range reads positive, for the report only."""
    times = session.index.time
    o = float(session[times == RTH_START]["open"].iloc[0])
    f = float(session[times == FLATTEN_BAR]["open"].iloc[0])
    return (o - f) if side == "high" else (f - o)


# ---------------------------------------------------------------------------
# The fade arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepParams:
    """The stop buffer beyond the sweep extreme, in ticks. The target mirrors
    the stop distance, so the bracket is 1:1 by construction."""

    stop_buffer_ticks: int = STOP_BUFFER_TICKS

    def __post_init__(self) -> None:
        if self.stop_buffer_ticks <= 0:
            raise ValueError("stop_buffer_ticks must be positive")

    @property
    def buffer_points(self) -> float:
        return self.stop_buffer_ticks * TICK


DIAGNOSTIC_COLUMNS = ["side", "entered", "skipped_reason", "depth", "entry_price",
                      "stop_price", "target_price", "entry_bar_breach", "exit_reason"]


class SweepFade(Strategy):
    """Fade the sweep of the prior session's high or low from the bar after
    the rejection."""

    name = "sweep_fade"

    def __init__(self, params: SweepParams | None = None,
                 roll_dates: Collection[date_type] = (),
                 early_close_dates: Collection[date_type] = ()) -> None:
        self.params = params or SweepParams()
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)
        self.diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = empty_signals(bars.index)
        for col in ("entry_price", "stop_price", "target_price", "exit_price"):
            out[col] = pd.Series(np.nan, index=bars.index, dtype="float64")
        out["exit_reason"] = pd.Series(pd.NA, index=bars.index, dtype="string")

        events = session_events(bars, self.roll_dates, self.early_close_dates)
        real = events[events["kind"] == REAL]
        by_day = {ts.date(): g for ts, g in bars.groupby(bars.index.normalize())}

        diag: list[dict] = []
        for _, row in real.iterrows():
            day = row["date"]
            record = {"date": day, "side": row["side"], "entered": False, "skipped_reason": None,
                      "depth": float(row["depth"]), "entry_price": np.nan, "stop_price": np.nan,
                      "target_price": np.nan, "entry_bar_breach": False, "exit_reason": None}
            diag.append(record)
            self._session_signals(by_day[day], row, out, record)

        self.diagnostics = (pd.DataFrame(diag, columns=["date", *DIAGNOSTIC_COLUMNS])
                            .set_index("date") if diag
                            else pd.DataFrame(columns=DIAGNOSTIC_COLUMNS))
        return validate_signals(out)

    def _session_signals(self, session: pd.DataFrame, event, out: pd.DataFrame,
                         record: dict) -> None:
        entry_ts = event["decision_ts"]
        times = session.index.time
        flatten_rows = session[times == FLATTEN_BAR]
        if flatten_rows.empty or entry_ts not in session.index:
            record["skipped_reason"] = SKIP_MISSING
            return
        flatten_ts = flatten_rows.index[0]
        if not rules.is_entry_allowed(entry_ts, self.early_close_dates, self.roll_dates):
            record["skipped_reason"] = SKIP_ROLL if rules.is_roll_day(
                entry_ts, self.roll_dates) else SKIP_EARLY
            return

        short = event["side"] == "high"
        entry_price = float(session.loc[entry_ts, "open"])
        buffer = self.params.buffer_points
        stop = float(event["extreme"]) + buffer if short else float(event["extreme"]) - buffer
        distance = abs(stop - entry_price)
        target = entry_price - distance if short else entry_price + distance
        entry_col, exit_col = ("entry_short", "exit_short") if short else ("entry_long", "exit_long")

        out.loc[entry_ts, entry_col] = True
        out.loc[entry_ts, "entry_price"] = entry_price
        out.loc[entry_ts, "stop_price"] = stop
        out.loc[entry_ts, "target_price"] = target
        record.update(entered=True, entry_price=entry_price, stop_price=stop, target_price=target)
        entry_high, entry_low = float(session.loc[entry_ts, "high"]), float(session.loc[entry_ts, "low"])
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
