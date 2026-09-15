"""Data-quality validation for intraday MES/MNQ OHLCV bar data.

Usage:
    python data/validate.py <path-to-parquet>

Reports:
  * date range, bar count, timezone confirmation
  * missing RTH minutes per day
  * days with abnormally few bars
  * OHLC integrity violations (high < low, open/close outside [low, high])
  * zero-volume RTH bars
  * gaps greater than 2% between consecutive bars within a session
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ET = "America/New_York"

# CME equity-index futures regular trading hours, in exchange local time.
# Bars are labelled by their opening minute, so a full session runs 09:30
# through 15:59 inclusive.
RTH_START = "09:30"
RTH_LAST_BAR = "15:59"
RTH_MINUTES = 390

GAP_THRESHOLD_PCT = 2.0

# CLAUDE.md rule 1: MES and MNQ only.
ALLOWED_INSTRUMENTS = {"MES", "MNQ"}

OHLC = ["open", "high", "low", "close"]

# Where a file's volume column came from. Parquet does not carry `.attrs`,
# so provenance rides on the filename - which is also why a gateway pull is
# never written under a Databento name. See
# docs/superpowers/specs/2026-09-15-projectx-bar-source-design.md §6.
EXCHANGE_VOLUME = "cme_exchange"
PLATFORM_VOLUME = "projectx_platform"


def volume_source(path) -> str:
    """Whether a file's volume is CME's tape or a broker's own fills.

    A ProjectX-sourced file counts only trades filled on that platform, so a
    volume-based signal scored on it is measuring the broker's flow rather
    than the market's. The report says which it is holding rather than
    leaving the reader to infer it from the filename.
    """
    return PLATFORM_VOLUME if "_projectx_" in str(path) else EXCHANGE_VOLUME



def load_bars(path: str | Path) -> pd.DataFrame:
    """Load a bar parquet file and return it indexed by ET-localised timestamps.

    Databento stores ``ts_event`` as UTC. The raw timezone is preserved on the
    returned frame's ``.attrs`` so the report can confirm what was on disk.
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

    # Read the timezone off the index itself. Going via .values would strip the
    # tz to naive UTC first and misreport tz-aware data as naive.
    source_tz = str(ts.tz) if ts.tz is not None else None

    if ts.tz is None:
        # Databento timestamps are UTC; an unlocalised index is assumed UTC.
        ts = ts.tz_localize("UTC")

    df = df.reset_index(drop=True)
    df.index = ts.tz_convert(ET)
    df.index.name = "ts_event_et"
    df = df.sort_index()
    df.attrs["source_tz"] = source_tz
    return df


def rth_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return only bars falling inside regular trading hours."""
    return df.between_time(RTH_START, RTH_LAST_BAR)


def check_timezone_and_range(df: pd.DataFrame) -> list[str]:
    lines = []
    source_tz = df.attrs.get("source_tz")
    lines.append(f"Source timezone on disk : {source_tz or 'naive (assumed UTC)'}")
    lines.append(f"Report timezone         : {df.index.tz}")
    lines.append(f"Total bars              : {len(df):,}")
    lines.append(f"First bar               : {df.index.min()}")
    lines.append(f"Last bar                : {df.index.max()}")
    sessions = sorted(set(df.index.date))
    lines.append(f"Distinct session dates  : {len(sessions):,}")
    if "symbol" in df.columns:
        symbols = sorted(df["symbol"].dropna().unique())
        shown = ", ".join(map(str, symbols[:10]))
        more = f" (+{len(symbols) - 10} more)" if len(symbols) > 10 else ""
        lines.append(f"Symbols                 : {len(symbols)} [{shown}{more}]")
        roots = {str(s)[:3].upper() for s in symbols}
        disallowed = roots - ALLOWED_INSTRUMENTS
        if disallowed:
            lines.append(
                f"  !! Non-allowlisted instrument roots present: {sorted(disallowed)} "
                f"(CLAUDE.md rule 1 permits only {sorted(ALLOWED_INSTRUMENTS)})"
            )
    return lines


def check_missing_rth_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing minute bars inside RTH for each session date."""
    rth = rth_slice(df)
    rows = []
    for date, group in rth.groupby(rth.index.date):
        expected = pd.date_range(
            start=pd.Timestamp(f"{date} {RTH_START}", tz=ET),
            end=pd.Timestamp(f"{date} {RTH_LAST_BAR}", tz=ET),
            freq="1min",
        )
        present = pd.DatetimeIndex(group.index.unique())
        missing = expected.difference(present)
        if len(missing):
            rows.append(
                {
                    "date": date,
                    "bars_present": len(present),
                    "missing_minutes": len(missing),
                    "first_missing": missing[0].strftime("%H:%M"),
                    "last_missing": missing[-1].strftime("%H:%M"),
                }
            )
    return pd.DataFrame(rows)


def check_zero_rth_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Dates that have bars but no RTH bars whatsoever.

    These are invisible to the missing-minute and sparse-day checks, because
    both group over the RTH slice and a date with no RTH rows simply never
    appears in the grouping. They matter: on a continuous front-month series a
    date like this is usually a contract-roll artefact, where the expiring
    contract stops trading at the RTH open.

    Sunday is excluded - Globex opens 18:00 ET Sunday and that session has no
    RTH component by design.
    """
    rth_dates = set(rth_slice(df).index.date)
    rows = []
    for date, group in df.groupby(df.index.date):
        if date in rth_dates:
            continue
        ts = pd.Timestamp(date)
        if ts.day_name() == "Sunday":
            continue
        rows.append(
            {
                "date": date,
                "weekday": ts.day_name(),
                "bars": len(group),
                "first_bar": group.index.min().strftime("%H:%M"),
                "last_bar": group.index.max().strftime("%H:%M"),
            }
        )
    return pd.DataFrame(rows)


def check_sparse_days(df: pd.DataFrame, floor_ratio: float = 0.9) -> pd.DataFrame:
    """Flag session dates whose RTH bar count is abnormally low.

    Only covers dates that have at least one RTH bar; dates with none are
    reported by :func:`check_zero_rth_sessions`.
    """
    rth = rth_slice(df)
    counts = rth.groupby(rth.index.date).size().rename("rth_bars")
    if counts.empty:
        return pd.DataFrame()
    flagged = counts[counts < RTH_MINUTES * floor_ratio]
    out = flagged.to_frame()
    out["pct_of_full_session"] = (out["rth_bars"] / RTH_MINUTES * 100).round(1)
    out["severity"] = [
        "severe" if c < RTH_MINUTES * 0.5 else "mild" for c in out["rth_bars"]
    ]
    return out.reset_index(names="date")


def check_missing_weekdays(df: pd.DataFrame) -> list:
    """Weekdays inside the data's range that have no bars at all.

    Exchange holidays legitimately appear here; this is a prompt to eyeball the
    list, not an error on its own.
    """
    present = set(df.index.date)
    all_weekdays = pd.bdate_range(df.index.min().date(), df.index.max().date())
    return [d.date() for d in all_weekdays if d.date() not in present]


def check_ohlc_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the OHLC relationships are internally inconsistent."""
    missing_cols = [c for c in OHLC if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing OHLC columns: {missing_cols}")

    problems = pd.DataFrame(index=df.index)
    problems["high_lt_low"] = df["high"] < df["low"]
    problems["close_above_high"] = df["close"] > df["high"]
    problems["close_below_low"] = df["close"] < df["low"]
    problems["open_above_high"] = df["open"] > df["high"]
    problems["open_below_low"] = df["open"] < df["low"]
    problems["null_price"] = df[OHLC].isna().any(axis=1)
    problems["nonpositive_price"] = (df[OHLC] <= 0).any(axis=1)

    bad = problems[problems.any(axis=1)]
    if bad.empty:
        return pd.DataFrame()
    return df.loc[bad.index, OHLC].join(bad)


def check_zero_volume_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-volume bars inside RTH."""
    if "volume" not in df.columns:
        return pd.DataFrame()
    rth = rth_slice(df)
    zero = rth[rth["volume"] == 0]
    return zero[[c for c in OHLC + ["volume"] if c in zero.columns]]


def check_price_gaps(df: pd.DataFrame, threshold_pct: float = GAP_THRESHOLD_PCT):
    """Close-to-close moves above the threshold between consecutive bars.

    Only compared within a single session date, so the overnight break is not
    reported as a gap.
    """
    out = df[["close"]].copy()
    out["session"] = out.index.date
    out["prev_close"] = out["close"].shift(1)
    out["prev_session"] = pd.Series(out["session"]).shift(1).values
    same_session = out["session"] == out["prev_session"]
    out["pct_change"] = (out["close"] / out["prev_close"] - 1.0) * 100.0
    gaps = out[same_session & (out["pct_change"].abs() > threshold_pct)]
    return gaps[["prev_close", "close", "pct_change"]].assign(
        pct_change=lambda d: d["pct_change"].round(3)
    )


def _section(title: str) -> str:
    return f"\n{'=' * 72}\n{title}\n{'=' * 72}"


def _preview(df: pd.DataFrame, limit: int = 10) -> str:
    with pd.option_context("display.max_columns", None, "display.width", 200):
        if len(df) <= limit:
            return df.to_string()
        return f"{df.head(limit).to_string()}\n... ({len(df) - limit:,} more rows)"


def run_report(path: str | Path) -> int:
    """Print the full validation report. Returns the number of issue categories hit."""
    df = load_bars(path)
    issues = 0

    print(_section("OVERVIEW / TIMEZONE CONFIRMATION"))
    print("\n".join(check_timezone_and_range(df)))

    print(_section("SESSIONS WITH BARS BUT NO RTH BARS AT ALL"))
    zero_rth = check_zero_rth_sessions(df)
    if zero_rth.empty:
        print("OK - every non-Sunday session with data has RTH coverage.")
    else:
        issues += 1
        print(f"{len(zero_rth):,} non-Sunday session(s) with data but zero RTH bars:")
        print("These are lost trading days for any RTH strategy.")
        print(_preview(zero_rth, limit=30))

    print(_section("MISSING RTH MINUTES PER DAY"))
    missing = check_missing_rth_minutes(df)
    if missing.empty:
        print("OK - every session has all 390 RTH minute bars.")
    else:
        issues += 1
        total = int(missing["missing_minutes"].sum())
        print(
            f"{len(missing):,} session(s) with missing RTH minutes "
            f"({total:,} bars missing in total)."
        )
        print("Note: half sessions (early 13:00 close) legitimately appear here.")
        print(_preview(missing.sort_values("missing_minutes", ascending=False)))

    print(_section("DAYS WITH ABNORMALLY FEW BARS"))
    sparse = check_sparse_days(df)
    if sparse.empty:
        print("OK - no session below 90% of a full 390-bar RTH session.")
    else:
        issues += 1
        print(f"{len(sparse):,} session(s) below 90% of a full RTH session:")
        print(_preview(sparse.sort_values("rth_bars")))

    missing_weekdays = check_missing_weekdays(df)
    print(f"\nWeekdays with no bars at all: {len(missing_weekdays)}")
    if missing_weekdays:
        print("(exchange holidays expected here)")
        print(_preview(pd.DataFrame({"date": missing_weekdays}), limit=20))

    print(_section("OHLC INTEGRITY (high<low, open/close outside range)"))
    bad_ohlc = check_ohlc_integrity(df)
    if bad_ohlc.empty:
        print("OK - no OHLC integrity violations.")
    else:
        issues += 1
        print(f"{len(bad_ohlc):,} bar(s) with inconsistent OHLC:")
        print(_preview(bad_ohlc))

    print(_section("ZERO-VOLUME RTH BARS"))
    zero_vol = check_zero_volume_rth(df)
    if zero_vol.empty:
        print("OK - no zero-volume bars inside RTH.")
    else:
        issues += 1
        pct = len(zero_vol) / max(len(rth_slice(df)), 1) * 100
        print(f"{len(zero_vol):,} zero-volume RTH bar(s) ({pct:.3f}% of RTH bars):")
        print(_preview(zero_vol))

    print(_section(f"PRICE GAPS > {GAP_THRESHOLD_PCT}% BETWEEN CONSECUTIVE BARS"))
    gaps = check_price_gaps(df)
    if gaps.empty:
        print(f"OK - no intra-session bar-to-bar move exceeded {GAP_THRESHOLD_PCT}%.")
    else:
        issues += 1
        print(f"{len(gaps):,} bar-to-bar move(s) above {GAP_THRESHOLD_PCT}%:")
        print(_preview(gaps.reindex(gaps["pct_change"].abs().sort_values(ascending=False).index)))

    print(_section("SUMMARY"))
    if issues == 0:
        print("PASS - no issues found across all checks.")
    else:
        print(f"{issues} check(s) reported findings - review the sections above.")
    return issues


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 2
    run_report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
