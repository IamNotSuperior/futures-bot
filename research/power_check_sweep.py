"""Power check for entry 16 (liquidity sweep). Frozen at the entry 16 freeze
commit.

Counts the real and placebo events per fold year, the continuations, and
the sessions ineligible by reason. It reads crosses and rejections, which
are price events, and never the reversion after them: ``sweep.session_events``
computes ``m30``, so this script drops that column before anything is
printed and prints counts only.

The entry's floor: at least 40 real events and 40 placebo events in every
fold year 2020 to 2025 and 20 of each in 2026, else the entry stops.

    python research/power_check_sweep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
from sweep import CONTINUATION, PLACEBO, REAL, session_events  # noqa: E402
from walkforward import build_folds  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"

MIN_EVENTS = 40
MIN_EVENTS_PARTIAL_YEAR = 20
PARTIAL_YEAR = 2026


def floor_for(year: int) -> int:
    return MIN_EVENTS_PARTIAL_YEAR if year == PARTIAL_YEAR else MIN_EVENTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(PARQUET))
    args = ap.parse_args(argv)
    parquet = Path(args.parquet)

    bars = loader.load_bars(parquet)
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    ev = session_events(bars, rolls, early).drop(columns=["m30", "m1555"])
    years = [f.test_year for f in build_folds()]

    print(f"Bar file                              : {parquet.name}")
    header = (f"{'year':>6}{'real':>7}{'high':>6}{'low':>5}{'placebo':>9}{'contin.':>9}"
              f"{'both':>6}{'floor':>7}")
    print(header)
    print("-" * len(header))
    failures = []
    for year in years:
        g = ev[ev["year"] == year]
        real = g[g["kind"] == REAL]
        plac = g[g["kind"] == PLACEBO]
        cont = g[g["kind"] == CONTINUATION]
        both = len(set(real["date"]) & set(plac["date"]))
        print(f"{year:>6}{len(real):>7}{int((real['side'] == 'high').sum()):>6}"
              f"{int((real['side'] == 'low').sum()):>5}{len(plac):>9}{len(cont):>9}{both:>6}"
              f"{floor_for(year):>7}")
        if len(real) < floor_for(year) or len(plac) < floor_for(year):
            failures.append((year, len(real), len(plac)))
    pooled = ev[ev["year"].isin(years)]
    print("-" * len(header))
    print(f"{'pooled':>6}{int((pooled['kind'] == REAL).sum()):>7}"
          f"{int(((pooled['kind'] == REAL) & (pooled['side'] == 'high')).sum()):>6}"
          f"{int(((pooled['kind'] == REAL) & (pooled['side'] == 'low')).sum()):>5}"
          f"{int((pooled['kind'] == PLACEBO).sum()):>9}"
          f"{int((pooled['kind'] == CONTINUATION).sum()):>9}")
    print()
    if failures:
        print("POWER CHECK FAILED: below the floor in "
              + ", ".join(f"{y} (real {r}, placebo {p})" for y, r, p in failures))
        return 1
    print(f"POWER CHECK PASSED: every fold year has at least {MIN_EVENTS} real and {MIN_EVENTS} "
          f"placebo events ({MIN_EVENTS_PARTIAL_YEAR} in {PARTIAL_YEAR}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
