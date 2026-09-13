"""Power check for entry 11 (turn-of-month, intraday). Frozen at ``a6d9af1``.

Counts the eligible window sessions per fold year and what each exclusion
removes. Reads the calendar and the bar index only - ``tom.session_calendar``
never touches a price - so this is a power check and not a peek.

The entry's floor: any fold year with fewer than 20 eligible window sessions
stops the entry before any strategy code runs.

    python research/power_check_tom.py                       # MES, entry 11
    python research/power_check_tom.py --parquet data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet   # entry 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
from tom import (  # noqa: E402
    CONTROL_LABEL, WINDOW_LABELS, cash_half_days, session_calendar,
)
from walkforward import build_folds  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"

#: Pre-registered floor, per fold year.
MIN_WINDOW_SESSIONS = 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(PARQUET),
                    help="bar file to count sessions on (default: MES)")
    args = ap.parse_args(argv)
    parquet = Path(args.parquet)

    bars = loader.load_bars(parquet)
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    half = cash_half_days(early)
    cal = session_calendar(bars, rolls, early)
    years = [f.test_year for f in build_folds()]

    print(f"Bar file                              : {parquet.name}")
    print(f"Sessions with a 09:30 bar             : {len(cal):,}")
    print(f"Early-close sessions in the data      : {len(early)}  "
          f"({len(half)} cash half-days kept in the calendar, "
          f"{len(early) - len(half)} cash-closed holiday sessions removed)")
    print(f"Roll days                             : {len(rolls)}")
    print()

    header = (f"{'year':>6}{'window':>8}{'eligible':>10}{'control':>9}"
              + "".join(f"{lab:>7}" for lab in WINDOW_LABELS)
              + f"{'roll':>6}{'early':>7}{'missing':>9}")
    print(header)
    print("-" * len(header))
    failures = []
    for year in years:
        g = cal[cal["year"] == year]
        w = g[g["window"]]
        ok = w[w["eligible"]]
        ctrl = g[(g["label"] == CONTROL_LABEL) & g["eligible"]]
        per_label = [int((ok["label"] == lab).sum()) for lab in WINDOW_LABELS]
        skipped = w[~w["eligible"]]["skipped_reason"].value_counts()
        print(f"{year:>6}{len(w):>8}{len(ok):>10}{len(ctrl):>9}"
              + "".join(f"{n:>7}" for n in per_label)
              + f"{int(skipped.get('roll_day', 0)):>6}"
              f"{int(skipped.get('early_close', 0)):>7}"
              f"{int(skipped.get('missing_bars', 0)):>9}")
        if len(ok) < MIN_WINDOW_SESSIONS:
            failures.append((year, len(ok)))

    pooled = cal[cal["year"].isin(years)]
    ok = pooled[pooled["window"] & pooled["eligible"]]
    ctrl = pooled[(pooled["label"] == CONTROL_LABEL) & pooled["eligible"]]
    print("-" * len(header))
    print(f"{'pooled':>6}{int(pooled['window'].sum()):>8}{len(ok):>10}{len(ctrl):>9}"
          + "".join(f"{int((ok['label'] == lab).sum()):>7}" for lab in WINDOW_LABELS))
    print()

    skipped = pooled[pooled["window"] & ~pooled["eligible"]]
    if len(skipped):
        print("Window sessions skipped, 2020-2026:")
        for _, r in skipped.sort_values("date").iterrows():
            print(f"  {r['date']}  {r['label']:<4} {r['skipped_reason']}")
        print()

    if failures:
        print(f"POWER CHECK FAILED: below the {MIN_WINDOW_SESSIONS}-session floor in "
              + ", ".join(f"{y} ({n})" for y, n in failures))
        return 1
    print(f"POWER CHECK PASSED: every fold year has at least {MIN_WINDOW_SESSIONS} "
          f"eligible window sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
