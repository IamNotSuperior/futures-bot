"""Strategy interface.

A strategy answers one question only: *where would this pattern trade?* It
returns boolean signals and nothing else. Position sizing, the contract cap,
the daily loss limit, the minimum hold and the forced flatten are all decided
by the execution layer against :mod:`strategies.rules`. Keeping the split
strict means a new strategy cannot accidentally opt out of a risk control -
there is nowhere in this interface for it to express one.

Timing convention
-----------------
``True`` at timestamp ``T`` in an entry column means *act at the open of bar
T*, decided using only bars strictly before ``T``. Signals are therefore
already shifted by the strategy and carry no lookahead: a backtest can fill
them at ``bars.loc[T, "open"]`` directly.

Exit columns follow the same convention for time-based exits. Where a strategy
also publishes ``stop_price``/``target_price`` columns, an exit marked at ``T``
was triggered intrabar and should be filled at that level rather than at the
open. Those extra columns are optional and informational; the four boolean
columns below are the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_SIGNAL_COLUMNS = ("entry_long", "entry_short", "exit_long", "exit_short")


class Strategy(ABC):
    """Base class for signal-generating strategies."""

    #: Human-readable name, used in reports.
    name: str = "unnamed"

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a signals frame for ``bars``.

        Args:
            bars: OHLCV bars indexed by ET-localised timestamps, as produced by
                :func:`data.loader.load_bars`.

        Returns:
            A DataFrame containing at least the four boolean columns in
            :data:`REQUIRED_SIGNAL_COLUMNS`. The index is whatever timeframe the
            strategy makes decisions on, which need not match ``bars`` - a
            strategy that resamples returns signals on the resampled index.
        """

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"


def empty_signals(index: pd.Index) -> pd.DataFrame:
    """An all-False signals frame on ``index``."""
    return pd.DataFrame(
        {col: pd.Series(False, index=index, dtype=bool) for col in REQUIRED_SIGNAL_COLUMNS}
    )


def validate_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Check a signals frame satisfies the interface. Returns it unchanged.

    Raises:
        ValueError: if a required column is missing, is not boolean, or if the
            same bar carries a long and a short entry at once.
    """
    missing = [c for c in REQUIRED_SIGNAL_COLUMNS if c not in signals.columns]
    if missing:
        raise ValueError(f"Signals frame is missing required columns: {missing}")

    non_bool = [c for c in REQUIRED_SIGNAL_COLUMNS if signals[c].dtype != bool]
    if non_bool:
        raise ValueError(
            f"Signal columns must be boolean, got non-boolean: "
            f"{ {c: str(signals[c].dtype) for c in non_bool} }"
        )

    both = signals["entry_long"] & signals["entry_short"]
    if both.any():
        raise ValueError(
            f"{int(both.sum())} bar(s) carry a long and a short entry simultaneously, "
            f"first at {signals.index[both][0]}"
        )
    return signals
