"""Leveraged-ETF end-of-day rebalance drift.

Implements hypothesis 2 in ``research/hypotheses.md`` exactly as frozen at
commit 43aa4e1. Every timing choice below is from that spec and must not be
adjusted in response to results - a change gets a dated addendum in the entry.

Timing, and why it is what it is
--------------------------------
Bars are labelled by opening minute and closed left, so the bar labelled 15:25
covers 15:25-15:29 and closes at 15:29:59. That close is the price "as of
15:30" and is the signal. Entry is the open of the bar labelled 15:35. The bar
labelled 15:30 is deliberately skipped: using its close would measure the signal
through 15:34:59, which is a later and different quantity than the spec's
"return to 3:30".

That leaves a five-minute gap between measurement and entry in which the move
can continue without us. It is the conservative reading and it is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import time
from typing import Collection

import pandas as pd

from base import Strategy, empty_signals, validate_signals
from orb import resample_bars
import rules

#: Bar labelled 09:30 - its open is the RTH open.
OPEN_BAR = time(9, 30)
#: Bar labelled 15:25 - its close is the price as of 15:30:00.
SIGNAL_BAR = time(15, 25)
#: Bar labelled 15:35 - its open is the fill.
ENTRY_BAR = time(15, 35)

EXIT_REASON_STOP = "stop"
EXIT_REASON_CLOSE = "cash_close"


@dataclass
class EODParams:
    """Parameters for the end-of-day rebalance strategy."""

    #: Minimum |open -> 15:30| return required to trade.
    threshold: float = 0.005
    #: Stop distance as a fraction of the entry price. None means time exit only.
    stop_pct: float | None = 0.005
    #: Timeframe the signal logic runs on.
    bar_minutes: int = 5

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.stop_pct is not None and self.stop_pct <= 0:
            raise ValueError("stop_pct must be positive or None")
        if self.bar_minutes <= 0:
            raise ValueError("bar_minutes must be positive")

    @property
    def label(self) -> str:
        stop = "none" if self.stop_pct is None else f"{self.stop_pct:.2%}"
        return f"thr={self.threshold:.2%}/stop={stop}"


THRESHOLDS = [0.005, 0.0075, 0.010, 0.015]
STOP_PCTS = [0.0025, 0.005, None]


def parameter_grid() -> list[EODParams]:
    """The 12 combinations frozen in the hypothesis entry."""
    return [
        EODParams(threshold=t, stop_pct=s) for t in THRESHOLDS for s in STOP_PCTS
    ]


def describe(params: EODParams) -> dict:
    return {
        "threshold": params.threshold,
        "stop_pct": -1.0 if params.stop_pct is None else params.stop_pct,
    }


class EODRebalanceDrift(Strategy):
    """Trade the direction of the day's move into the closing rebalance."""

    name = "eod_rebalance"

    def __init__(
        self,
        params: EODParams | None = None,
        roll_dates: Collection[date_type] = (),
        early_close_dates: Collection[date_type] = (),
    ) -> None:
        self.params = params or EODParams()
        self.roll_dates = set(roll_dates)
        self.early_close_dates = set(early_close_dates)

    # -- internals ---------------------------------------------------------

    def _session_signals(self, session: pd.DataFrame, out: pd.DataFrame) -> None:
        p = self.params
        times = session.index.time

        open_rows = session[times == OPEN_BAR]
        signal_rows = session[times == SIGNAL_BAR]
        entry_rows = session[times == ENTRY_BAR]
        # An early-close session has no 15:25 or 15:35 bar and drops out here.
        if open_rows.empty or signal_rows.empty or entry_rows.empty:
            return

        open_px = float(open_rows["open"].iloc[0])
        signal_px = float(signal_rows["close"].iloc[0])
        if open_px <= 0:
            return

        signal = signal_px / open_px - 1.0
        if abs(signal) < p.threshold:
            return

        entry_ts = entry_rows.index[0]
        if not rules.is_entry_allowed(
            entry_ts, self.early_close_dates, self.roll_dates
        ):
            return

        long = signal > 0
        entry_price = float(entry_rows["open"].iloc[0])
        out.loc[entry_ts, "entry_long" if long else "entry_short"] = True
        out.loc[entry_ts, "signal_return"] = signal

        stop = None
        if p.stop_pct is not None:
            stop = entry_price * (1 - p.stop_pct) if long else entry_price * (1 + p.stop_pct)
            out.loc[entry_ts, "stop_price"] = stop

        self._mark_exit(session, entry_ts, long, stop, out)

    def _mark_exit(self, session, entry_ts, long: bool, stop, out) -> None:
        """Stop if touched, otherwise the RTH close."""
        exit_col = "exit_long" if long else "exit_short"
        flatten_at = rules.flatten_deadline(entry_ts, self.early_close_dates)

        holdable = session[(session.index > entry_ts) & (session.index.time < flatten_at)]
        if holdable.empty:
            # Nothing to exit into; drop the entry rather than invent a fill.
            out.loc[entry_ts, "entry_long"] = False
            out.loc[entry_ts, "entry_short"] = False
            return

        if stop is not None:
            hit = (
                holdable["low"] <= stop if long else holdable["high"] >= stop
            )
            if hit.any():
                ts = holdable.index[hit.to_numpy().argmax()]
                out.loc[ts, exit_col] = True
                out.loc[ts, "exit_reason"] = EXIT_REASON_STOP
                out.loc[ts, "exit_price"] = float(stop)
                return

        ts = holdable.index[-1]
        out.loc[ts, exit_col] = True
        out.loc[ts, "exit_reason"] = EXIT_REASON_CLOSE
        out.loc[ts, "exit_price"] = float(holdable["close"].iloc[-1])

    # -- interface ---------------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals_resampled(
            resample_bars(bars, self.params.bar_minutes)
        )

    def generate_signals_resampled(self, resampled: pd.DataFrame) -> pd.DataFrame:
        """Signals from bars already at ``bar_minutes`` resolution.

        Sessions are independent, so generating over a long span and slicing by
        date matches generating on each slice - the property the walk-forward
        relies on.
        """
        if resampled.empty:
            return validate_signals(empty_signals(resampled.index))

        out = empty_signals(resampled.index)
        for col in ("stop_price", "exit_price", "signal_return"):
            out[col] = pd.Series(pd.NA, index=resampled.index, dtype="Float64")
        out["exit_reason"] = pd.Series(pd.NA, index=resampled.index, dtype="string")

        for day, session in resampled.groupby(resampled.index.date):
            if rules.is_roll_day(day, self.roll_dates):
                continue
            self._session_signals(session, out)

        return validate_signals(out)

    def bars_for_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        return resample_bars(bars, self.params.bar_minutes)
