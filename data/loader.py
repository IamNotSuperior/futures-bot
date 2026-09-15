"""Credential, configuration and bar loading for the data layer.

Secrets live in a ``.env`` file at the project root, which is gitignored.
See ``.env.example`` for the expected keys.
"""

from __future__ import annotations

from datetime import date as date_type
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

ET = "America/New_York"

# Project root is one level above this file (data/loader.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env once at import time. Existing environment variables win, so a real
# environment (CI, a shell export) can override the local file without editing it.
load_dotenv(ENV_PATH, override=False)


class MissingCredentialError(RuntimeError):
    """Raised when a required credential is not present in the environment."""


def get_databento_api_key() -> str:
    """Return the Databento API key.

    Raises:
        MissingCredentialError: if DATABENTO_API_KEY is unset or empty. The
            error message never includes the key value itself.
    """
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        raise MissingCredentialError(
            "DATABENTO_API_KEY is not set. Add it to "
            f"{ENV_PATH} (see .env.example) or export it in your environment."
        )
    return key


def get_topstepx_credentials() -> tuple[str, str]:
    """Return ``(username, api_key)`` for the ProjectX Gateway.

    Two values rather than one, because the gateway authenticates a key
    against the account that owns it. The upstream script this was ported
    from keeps these in a ``config.json``; they live in ``.env`` here so the
    project has exactly one gitignored place a secret can be.

    Raises:
        MissingCredentialError: if either is unset or blank. The message
            names the variable and never any part of either value.
    """
    values = {}
    for name in ("TOPSTEPX_USERNAME", "TOPSTEPX_API_KEY"):
        value = os.environ.get(name, "").strip()
        if not value:
            raise MissingCredentialError(
                f"{name} is not set. Add it to {ENV_PATH} (see .env.example) "
                "or export it in your environment."
            )
        values[name] = value
    return values["TOPSTEPX_USERNAME"], values["TOPSTEPX_API_KEY"]


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load a cached bar parquet, indexed by ET-localised timestamps.

    Databento writes ``ts_event`` in UTC; bars are returned converted to
    exchange local time so that session and rule logic can work in wall-clock
    terms without repeating the conversion everywhere.
    """
    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(df.index)
    elif "ts_event" in df.columns:
        ts = pd.DatetimeIndex(pd.to_datetime(df["ts_event"]))
    else:
        raise ValueError(
            "No timestamp found: expected a DatetimeIndex or a 'ts_event' column, "
            f"got columns {list(df.columns)}"
        )

    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    df = df.reset_index(drop=True)
    df.index = ts.tz_convert(ET)
    df.index.name = "ts_event_et"
    return df.sort_index()


def detect_roll_dates(bars: pd.DataFrame) -> set[date_type]:
    """Session dates on which the underlying contract changed.

    A continuous series such as ``MES.v.0`` is stitched from a sequence of real
    contracts, and ``instrument_id`` identifies which one each bar came from.
    A roll date is the first session whose instrument differs from the previous
    session's. Prices are not back-adjusted across that boundary, so a position
    held over it sees an artificial jump - hence the trading rule that no new
    position is opened on a roll date.

    The first session in the file is never reported: there is no prior session
    to have rolled from.
    """
    if "instrument_id" not in bars.columns:
        raise ValueError(
            "Cannot detect rolls without an 'instrument_id' column; "
            f"got {list(bars.columns)}"
        )

    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(ET)
    elif str(idx.tz) != ET:
        idx = idx.tz_convert(ET)

    # The dominant instrument per session, so a handful of stray bars from the
    # outgoing contract cannot masquerade as a roll.
    per_session = (
        pd.Series(bars["instrument_id"].to_numpy(), index=idx.date)
        .groupby(level=0)
        .agg(lambda s: s.value_counts().idxmax())
        .sort_index()
    )

    changed = per_session.ne(per_session.shift(1))
    changed.iloc[0] = False
    return set(per_session.index[changed])


def detect_early_close_dates(bars: pd.DataFrame) -> set[date_type]:
    """Sessions that closed early (CME holiday half-sessions).

    Detected from the data rather than a hardcoded calendar: a session whose
    last regular-hours bar lands before 15:59 ET ended early. Sessions with no
    regular-hours bars at all are full closures, not early closes, and are
    excluded.
    """
    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(ET)
    elif str(idx.tz) != ET:
        idx = idx.tz_convert(ET)

    # A Series, not a column-less DataFrame: `DataFrame.empty` is True when any
    # axis is empty, so a zero-column frame reports empty however many rows it
    # has, and the guard below would swallow the whole dataset.
    marker = pd.Series(True, index=idx)
    rth = marker.between_time("09:30", "15:59")
    if rth.empty:
        return set()

    last_bar = pd.Series(rth.index, index=rth.index.date).groupby(level=0).max()
    early = last_bar[last_bar.dt.time < pd.Timestamp("15:59").time()]
    return set(early.index)
