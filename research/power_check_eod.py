"""Power check for hypothesis 2 (LETF end-of-day rebalance drift).

Counts how many sessions clear each signal threshold. Reads the signal
distribution only - the open->15:30 return - and never touches the
15:35->16:00 outcome, so it is a power check rather than peeking.

    python research/power_check_eod.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from orb import resample_bars  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"

THRESHOLDS = [0.005, 0.0075, 0.010, 0.015]

# Bars are labelled by opening minute and closed left, so the bar labelled
# 15:25 closes at 15:29:59 - the price "as of 15:30" per the frozen spec.
SIGNAL_BAR = time(15, 25)
ENTRY_BAR = time(15, 35)
OPEN_BAR = time(9, 30)

# Walk-forward test years. 2019 is training-only and never a test year, so it
# contributes no out-of-sample trades.
OOS_YEARS = range(2020, 2027)


def session_signals(bars5: pd.DataFrame, roll_dates: set) -> pd.DataFrame:
    """One row per session: the open->15:30 return, where the session supports it."""
    rows = []
    for day, session in bars5.groupby(bars5.index.date):
        times = set(session.index.time)
        if OPEN_BAR not in times or SIGNAL_BAR not in times or ENTRY_BAR not in times:
            # Early closes have no 15:25 bar, so they drop out here.
            continue
        if rules.is_roll_day(day, roll_dates):
            continue
        open_px = float(session[session.index.time == OPEN_BAR]["open"].iloc[0])
        signal_px = float(session[session.index.time == SIGNAL_BAR]["close"].iloc[0])
        rows.append(
            {"date": day, "year": day.year, "signal": signal_px / open_px - 1.0}
        )
    return pd.DataFrame(rows)


def main() -> int:
    bars = loader.load_bars(PARQUET)
    roll_dates = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    bars5 = resample_bars(bars, 5)

    all_sessions = len(set(bars5.index.date))
    sig = session_signals(bars5, roll_dates)

    print(f"Sessions with 5-minute RTH bars      : {all_sessions:,}")
    print(f"Eligible (has 09:30/15:25/15:35, not a roll day): {len(sig):,}")
    print(f"  excluded: {len(early)} early closes, {len(roll_dates)} roll days, "
          f"plus any session missing those bars")
    print()

    header = f"{'year':>6}{'sessions':>10}" + "".join(
        f"{f'>={t:.2%}':>12}" for t in THRESHOLDS
    )
    print(header)
    print("-" * len(header))
    for year, group in sig.groupby("year"):
        counts = [(group["signal"].abs() >= t).sum() for t in THRESHOLDS]
        marker = "" if year in OOS_YEARS else "   (train only)"
        print(f"{year:>6}{len(group):>10}"
              + "".join(f"{c:>12}" for c in counts) + marker)

    oos = sig[sig["year"].isin(OOS_YEARS)]
    print("-" * len(header))
    oos_counts = [(oos["signal"].abs() >= t).sum() for t in THRESHOLDS]
    print(f"{'OOS':>6}{len(oos):>10}" + "".join(f"{c:>12}" for c in oos_counts))
    print()
    print("Pooled out-of-sample trade counts (2020-2026), against the "
          "pre-registered floor of 30:")
    for t, c in zip(THRESHOLDS, oos_counts):
        verdict = "OK" if c >= 30 else "BELOW FLOOR - insufficient evidence"
        print(f"  >= {t:.2%} : {c:>5}   {verdict}")

    print()
    print("Per-fold minimum (smallest single test year), for reference:")
    for t in THRESHOLDS:
        per_year = [
            int((g["signal"].abs() >= t).sum())
            for y, g in oos.groupby("year")
        ]
        print(f"  >= {t:.2%} : min {min(per_year):>4}   median {int(pd.Series(per_year).median()):>4}"
              f"   per-year counts {per_year}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
