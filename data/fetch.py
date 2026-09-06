"""Databento historical data pull for MES/MNQ intraday bars.

Usage:
    python data/fetch.py --estimate            # cost estimate only, no spend
    python data/fetch.py --pull                # estimate, gate on cost, then pull

The pull is gated on a maximum spend (``MAX_COST_USD``); if the estimate meets
or exceeds it the script stops without downloading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import databento as db
import pandas as pd

from loader import get_databento_api_key, PROJECT_ROOT

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"

# Databento treats `end` as exclusive, so to include the 2026-08-31 session the
# request must run to 2026-09-01.
START = "2024-09-01"
END = "2026-09-01"

MAX_COST_USD = 10.0

# Candidate symbologies for a root on GLBX.MDP3. These resolve to very
# different datasets, so the estimate is run across all of them before
# committing. Use the `.v.0` volume-rolled series: `.c.0` has no RTH bars at
# all on quarterly expiry days, because the expiring contract stops at the open.
def candidates(root: str = "MES") -> list[tuple[str, str, str]]:
    return [
        (f"{root}.c.0", "continuous", "front-month continuous contract"),
        (f"{root}.v.0", "continuous", "highest-volume continuous contract"),
        (f"{root}.FUT", "parent", f"every {root} contract month"),
        (root, "raw_symbol", "literal raw symbol (expected to match nothing)"),
    ]


def cache_path(symbol: str, start: str = START, end: str = END) -> Path:
    """Cache filename, tagged with the span so two pulls cannot collide."""
    slug = symbol.replace(".", "_").lower()
    # The request end is exclusive, so the file is named for the last month it
    # actually contains.
    last = pd.Timestamp(end) - pd.Timedelta(days=1)
    span = f"{pd.Timestamp(start):%Y-%m}_{last:%Y-%m}"
    return PROJECT_ROOT / "data" / f"{slug}_ohlcv_1m_{span}.parquet"


def make_client() -> db.Historical:
    return db.Historical(get_databento_api_key())


def estimate(client: db.Historical, symbol: str, stype_in: str,
             start: str = START, end: str = END) -> dict:
    """Cost and size estimate for one symbology. Metadata calls are not billed."""
    result: dict = {"symbol": symbol, "stype_in": stype_in}
    try:
        result["cost_usd"] = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[symbol],
            schema=SCHEMA,
            start=start,
            end=end,
            stype_in=stype_in,
        )
        result["billable_bytes"] = client.metadata.get_billable_size(
            dataset=DATASET,
            symbols=[symbol],
            schema=SCHEMA,
            start=start,
            end=end,
            stype_in=stype_in,
        )
    except Exception as exc:  # noqa: BLE001 - report and continue to next candidate
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_estimates(client: db.Historical, root: str = "MES",
                  start: str = START, end: str = END) -> list[dict]:
    print(f"Cost estimate | {DATASET} | {SCHEMA} | {start} -> {end} (end exclusive)")
    print("-" * 78)
    results = []
    for symbol, stype_in, note in candidates(root):
        r = estimate(client, symbol, stype_in, start, end)
        r["note"] = note
        results.append(r)
        if "error" in r:
            print(f"{symbol:<10} {stype_in:<12} ERROR  {r['error']}")
        else:
            mb = r["billable_bytes"] / 1_048_576
            print(
                f"{symbol:<10} {stype_in:<12} ${r['cost_usd']:>8.4f}  "
                f"{mb:>9.2f} MB   {note}"
            )
    print("-" * 78)
    return results


def pull(client: db.Historical, symbol: str, stype_in: str, out_path: Path,
         start: str = START, end: str = END) -> pd.DataFrame:
    print(f"\nPulling {symbol} ({stype_in}) {start} -> {end} ...")
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[symbol],
        schema=SCHEMA,
        start=start,
        end=end,
        stype_in=stype_in,
    )
    frame = data.to_df()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Wrote {len(frame):,} bars to {out_path} ({size_mb:.2f} MB)")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", action="store_true", help="estimate only")
    parser.add_argument("--pull", action="store_true", help="estimate then pull")
    parser.add_argument("--symbol", default="MES.v.0")
    parser.add_argument("--stype-in", default="continuous")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END,
                        help="exclusive, per Databento")
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD,
                        help="refuse to pull above this estimate")
    args = parser.parse_args()
    root = args.symbol.split(".")[0]

    if not (args.estimate or args.pull):
        parser.error("pass --estimate or --pull")

    client = make_client()
    results = run_estimates(client, root, args.start, args.end)

    if args.estimate:
        return 0

    chosen = next(
        (r for r in results if r["symbol"] == args.symbol and "cost_usd" in r), None
    )
    if chosen is None:
        print(f"\nNo usable estimate for {args.symbol}; refusing to pull.")
        return 1

    cost = chosen["cost_usd"]
    if cost > args.max_cost:
        print(
            f"\nSTOP: estimate ${cost:.4f} exceeds the ${args.max_cost:.2f} cap. "
            "Nothing was downloaded."
        )
        return 1

    print(f"\nEstimate ${cost:.4f} is within the ${args.max_cost:.2f} cap - proceeding.")
    pull(client, args.symbol, args.stype_in,
         cache_path(args.symbol, args.start, args.end), args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
