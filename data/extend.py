"""Pull an earlier span of MES.v.0 bars and merge it into the cached series.

    python data/extend.py --estimate
    python data/extend.py --pull

The merge is by timestamp, keeping the existing rows where the two overlap, so
re-running cannot duplicate bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import databento as db
import pandas as pd

from loader import PROJECT_ROOT, get_databento_api_key, load_bars

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOL = "MES.v.0"
STYPE_IN = "continuous"

# MES listed on 2019-05-06. `end` is exclusive.
NEW_START = "2019-05-06"
NEW_END = "2024-09-01"

MAX_COST_USD = 15.0

EXISTING = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2024-09_2026-08.parquet"
MERGED = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"


def estimate(client: db.Historical) -> float:
    cost = client.metadata.get_cost(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=NEW_START, end=NEW_END, stype_in=STYPE_IN,
    )
    size = client.metadata.get_billable_size(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=NEW_START, end=NEW_END, stype_in=STYPE_IN,
    )
    print(f"{SYMBOL} {SCHEMA}  {NEW_START} -> {NEW_END} (end exclusive)")
    print(f"  estimate  ${cost:.4f}   size {size / 1_048_576:.2f} MB")
    print(f"  cap       ${MAX_COST_USD:.2f}")
    return float(cost)


def merge(new: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Concatenate, dropping any timestamp already present in the existing file."""
    existing = pd.read_parquet(existing_path)
    combined = pd.concat([new, existing])
    before = len(combined)
    # keep="last" keeps the existing file's version of any shared timestamp
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    dropped = before - len(combined)
    print(f"  new rows {len(new):,} + existing {len(existing):,} = "
          f"{len(combined):,} after dropping {dropped:,} duplicate timestamp(s)")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    if not (args.estimate or args.pull):
        ap.error("pass --estimate or --pull")

    client = db.Historical(get_databento_api_key())
    cost = estimate(client)

    if args.estimate:
        return 0
    if cost > MAX_COST_USD:
        print(f"STOP: ${cost:.2f} exceeds the ${MAX_COST_USD:.2f} cap. Nothing pulled.")
        return 1

    print(f"\nProceeding at ${cost:.4f}. Pulling ...")
    data = client.timeseries.get_range(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=NEW_START, end=NEW_END, stype_in=STYPE_IN,
    )
    new = data.to_df()
    print(f"  pulled {len(new):,} bars")

    combined = merge(new, EXISTING)
    combined.to_parquet(MERGED)
    size_mb = MERGED.stat().st_size / 1_048_576
    print(f"\nWrote {MERGED.name} ({size_mb:.2f} MB)")

    span = load_bars(MERGED)
    print(f"  span {span.index.min()} .. {span.index.max()}")
    print(f"  sessions {len(set(span.index.date)):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
