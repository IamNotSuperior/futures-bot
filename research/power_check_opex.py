"""Power check for entry 14 (monthly option expiry, intraday). Frozen at ``f9f129c``.

Counts the eligible expiry days per fold year, quarterly against
non-quarterly, the two control populations, and what each exclusion removed.
Reads the calendar and the bar index only - ``opex.session_calendar`` never
touches a price - so this is a power check and not a peek.

The entry's floor: at least 8 eligible expiry days in every fold year 2020
to 2025 and at least 5 in 2026 (eight months), else the entry stops before
any strategy code runs.

    python research/power_check_opex.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
from opex import CONTROL_FRIDAY, CONTROL_OTHER, EXPIRY, session_calendar  # noqa: E402
from walkforward import build_folds  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"

#: Pre-registered floors, per fold year.
MIN_EXPIRY_SESSIONS = 8
MIN_EXPIRY_SESSIONS_PARTIAL_YEAR = 5
PARTIAL_YEAR = 2026


def floor_for(year: int) -> int:
    return MIN_EXPIRY_SESSIONS_PARTIAL_YEAR if year == PARTIAL_YEAR else MIN_EXPIRY_SESSIONS


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
    print(f"Early-close sessions in the data      : {len(early)}")
    print(f"Roll days                             : {len(rolls)}")
    print()

    header = (f"{'year':>6}{'expiry':>8}{'eligible':>10}{'quarterly':>11}{'fridays':>9}"
              f"{'others':>8}{'roll':>6}{'early':>7}{'missing':>9}{'floor':>7}")
    print(header)
    print("-" * len(header))
    failures = []
    for year in years:
        g = cal[cal["year"] == year]
        ex = g[g["label"] == EXPIRY]
        ok = ex[ex["eligible"]]
        fri = g[(g["label"] == CONTROL_FRIDAY) & g["eligible"]]
        oth = g[(g["label"] == CONTROL_OTHER) & g["eligible"]]
        skipped = ex[~ex["eligible"]]["skipped_reason"].value_counts()
        print(f"{year:>6}{len(ex):>8}{len(ok):>10}{int(ok['quarterly'].sum()):>11}{len(fri):>9}"
              f"{len(oth):>8}{int(skipped.get('roll_day', 0)):>6}"
              f"{int(skipped.get('early_close', 0)):>7}"
              f"{int(skipped.get('missing_bars', 0)):>9}{floor_for(year):>7}")
        if len(ok) < floor_for(year):
            failures.append((year, len(ok)))

    pooled = cal[cal["year"].isin(years)]
    ex = pooled[pooled["label"] == EXPIRY]
    ok = ex[ex["eligible"]]
    fri = pooled[(pooled["label"] == CONTROL_FRIDAY) & pooled["eligible"]]
    oth = pooled[(pooled["label"] == CONTROL_OTHER) & pooled["eligible"]]
    print("-" * len(header))
    print(f"{'pooled':>6}{len(ex):>8}{len(ok):>10}{int(ok['quarterly'].sum()):>11}{len(fri):>9}"
          f"{len(oth):>8}")
    print()

    thursdays = ok[[d.weekday() == 3 for d in ok["date"]]]
    if len(thursdays):
        print("Expiry days on a Thursday (Good Friday shift):")
        for d in thursdays["date"]:
            print(f"  {d}")
        print()

    skipped = ex[~ex["eligible"]]
    if len(skipped):
        print("Expiry days skipped, 2020-2026:")
        for _, r in skipped.sort_values("date").iterrows():
            print(f"  {r['date']}  {'quarterly' if r['quarterly'] else 'monthly':<9} {r['skipped_reason']}")
        print()

    if failures:
        print("POWER CHECK FAILED: below the floor in "
              + ", ".join(f"{y} ({n})" for y, n in failures))
        return 1
    print(f"POWER CHECK PASSED: every fold year has at least {MIN_EXPIRY_SESSIONS} eligible "
          f"expiry days ({MIN_EXPIRY_SESSIONS_PARTIAL_YEAR} in {PARTIAL_YEAR}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
