"""Calendar roll for gateway-sourced MES/MNQ bars.

Databento's ``MES.v.0`` rolls on volume. The ProjectX gateway reports only
volume filled on its own platform, so that rule cannot be reproduced from
gateway data - not approximately, but in principle, because the quantity the
rule keys on is not the one being served.

The substitute is the CME equity-index convention: quarterly contracts
(March, June, September, December), expiring the third Friday, rolling on the
Thursday eight days earlier. There is **no back-adjustment**, matching
``.v.0``, which is not back-adjusted either - the price jump at a roll is
real and the project handles it by refusing new entries on a roll date.

See ``docs/superpowers/specs/2026-09-15-projectx-bar-source-design.md`` §5.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from loader import PROJECT_ROOT

ET = "America/New_York"

#: CME equity-index futures list quarterly.
QUARTERLY_MONTHS = (3, 6, 9, 12)

#: CME month codes for the quarterly cycle.
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}

#: Days between the roll and expiry. The Thursday before expiry week.
ROLL_OFFSET_DAYS = 8

FRIDAY = 4


def expiry(year: int, month: int) -> date:
    """Third Friday of a quarterly month."""
    if month not in QUARTERLY_MONTHS:
        raise ValueError(
            f"{month} is not a quarterly month; expected one of {QUARTERLY_MONTHS}."
        )
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(FRIDAY - first.weekday()) % 7)
    return first_friday + timedelta(days=14)


def roll_date(year: int, month: int) -> date:
    """The session the front contract changes: eight days before expiry."""
    return expiry(year, month) - timedelta(days=ROLL_OFFSET_DAYS)


def _quarterlies(year: int) -> list[tuple[int, int]]:
    return [(year, m) for m in QUARTERLY_MONTHS]


def active_contract(day: date) -> tuple[int, int]:
    """The ``(year, month)`` contract that is front month on ``day``.

    A contract is front month until its own roll date, on which the next
    quarterly takes over. Expiry week therefore already belongs to the
    following contract, which is the point of rolling early.
    """
    for year in (day.year, day.year + 1):
        for candidate in _quarterlies(year):
            if day < roll_date(*candidate):
                return candidate
    raise ValueError(f"No quarterly contract found for {day}.")


def contract_id(root: str, year: int, month: int) -> str:
    """Gateway contract id, e.g. ``CON.F.US.MES.U25``."""
    return f"CON.F.US.{root}.{MONTH_CODES[month]}{year % 100:02d}"


def stitch(frames: dict[str, pd.DataFrame], root: str) -> pd.DataFrame:
    """Splice per-contract frames into one continuous series.

    Each bar is kept only if its contract is the one active on that bar's
    **ET session date** - the project reasons in exchange local time
    everywhere else, and a roll that moved with UTC would land mid-session
    for part of the year.

    No back-adjustment is applied. ``instrument_id`` travels with the bars,
    so ``loader.detect_roll_dates`` reports the splice.
    """
    kept = []
    for cid, frame in frames.items():
        if frame.empty:
            continue
        index = frame.index
        if index.tz is None:
            index = index.tz_localize("UTC")
        session_dates = index.tz_convert(ET).date
        wanted = [contract_id(root, *active_contract(d)) == cid
                  for d in session_dates]
        kept.append(frame.loc[wanted])

    if not kept:
        return pd.concat(list(frames.values())) if frames else pd.DataFrame()

    combined = pd.concat(kept).sort_index()
    unique = ~combined.index.duplicated(keep="first")
    return combined.loc[unique]


def output_path(root: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Where a gateway pull is written.

    The name carries the source. Databento files are ``mes_v_0_ohlcv_1m_*``;
    a gateway file must not be mistakable for one, because the two are not
    comparable and a later session will otherwise put them side by side.
    """
    span = f"{pd.Timestamp(start):%Y-%m}_{pd.Timestamp(end):%Y-%m}"
    return PROJECT_ROOT / "data" / f"{root.lower()}_projectx_ohlcv_1m_{span}.parquet"


def pull(client, root: str, start: pd.Timestamp,
         end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, pd.Timestamp | None]]:
    """Fetch every resolved contract, stitch them, and report the depth.

    The second return value maps each contract id to the earliest bar the
    gateway actually served for it, or ``None`` where it served nothing. That
    mapping is the measurement the design refused to guess at: whether the
    gateway holds minute bars for expired contracts at all, and how far back.
    A contract that returns nothing is reported rather than dropped, because
    an absence is the finding.
    """
    frames: dict[str, pd.DataFrame] = {}
    depth: dict[str, pd.Timestamp | None] = {}

    for cid in client.resolve_contracts(root):
        frame = client.retrieve_bars(cid, start, end)
        frames[cid] = frame
        depth[cid] = None if frame.empty else pd.Timestamp(frame.index.min())

    return stitch(frames, root=root), depth
