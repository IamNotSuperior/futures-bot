"""Power check for entry 15 (quarterly futures expiry). Frozen at the entry
15 freeze commit.

Counts the eligible quarterly expiry days per fold year, the Friday control,
and the monthly expiry days excluded from both. Reads the calendar and the
bar index only - ``quarterly.session_calendar`` never touches a price.

The entry's floor: at least 3 eligible quarterly days in every fold year
2020 to 2025 and at least 1 in 2026, else the entry stops.

    python research/power_check_quarterly.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
from opex import CONTROL_FRIDAY  # noqa: E402
from quarterly import MONTHLY_EXPIRY, QUARTERLY, session_calendar  # noqa: E402
from walkforward import build_folds  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"

MIN_QUARTERLY_SESSIONS = 3
MIN_QUARTERLY_SESSIONS_PARTIAL_YEAR = 1
PARTIAL_YEAR = 2026


def floor_for(year: int) -> int:
    return MIN_QUARTERLY_SESSIONS_PARTIAL_YEAR if year == PARTIAL_YEAR else MIN_QUARTERLY_SESSIONS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(PARQUET))
    args = ap.parse_args(argv)
    parquet = Path(args.parquet)

    bars = loader.load_bars(parquet)
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    cal = session_calendar(bars, rolls, early)
    years = [f.test_year for f in build_folds()]

    print(f"Bar file                              : {parquet.name}")
    print(f"Sessions with a 09:30 bar             : {len(cal):,}")
    print()
    header = (f"{'year':>6}{'quarterly':>11}{'eligible':>10}{'fridays':>9}{'monthly':>9}"
              f"{'skipped':>9}{'floor':>7}")
    print(header)
    print("-" * len(header))
    failures = []
    for year in years:
        g = cal[cal["year"] == year]
        q = g[g["label"] == QUARTERLY]
        ok = q[q["eligible"]]
        fri = g[(g["label"] == CONTROL_FRIDAY) & g["eligible"]]
        mon = g[g["label"] == MONTHLY_EXPIRY]
        print(f"{year:>6}{len(q):>11}{len(ok):>10}{len(fri):>9}{len(mon):>9}"
              f"{len(q) - len(ok):>9}{floor_for(year):>7}")
        if len(ok) < floor_for(year):
            failures.append((year, len(ok)))
    pooled = cal[cal["year"].isin(years)]
    q = pooled[pooled["label"] == QUARTERLY]
    ok = q[q["eligible"]]
    fri = pooled[(pooled["label"] == CONTROL_FRIDAY) & pooled["eligible"]]
    mon = pooled[pooled["label"] == MONTHLY_EXPIRY]
    print("-" * len(header))
    print(f"{'pooled':>6}{len(q):>11}{len(ok):>10}{len(fri):>9}{len(mon):>9}{len(q) - len(ok):>9}")
    print()
    thursdays = ok[[d.weekday() == 3 for d in ok["date"]]]
    if len(thursdays):
        print("Quarterly expiry days on a Thursday: " + ", ".join(str(d) for d in thursdays["date"]))
        print()
    skipped = q[~q["eligible"]]
    for _, r in skipped.iterrows():
        print(f"  skipped {r['date']} {r['skipped_reason']}")
    if failures:
        print("POWER CHECK FAILED: below the floor in "
              + ", ".join(f"{y} ({n})" for y, n in failures))
        return 1
    print(f"POWER CHECK PASSED: every fold year has at least {MIN_QUARTERLY_SESSIONS} eligible "
          f"quarterly expiry days ({MIN_QUARTERLY_SESSIONS_PARTIAL_YEAR} in {PARTIAL_YEAR}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
