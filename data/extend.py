"""Pull forward MES.v.0 bars and accumulate them in a rolling forward file.

    python data/extend.py --estimate                # free: what the next pull would cost
    python data/extend.py --pull [--max-cost X]     # estimate, gate on the cap, pull, merge
    python data/extend.py --merge-file PATH         # fold a local parquet in, no API call
    python data/extend.py --estimate --start 2026-10-01 --end 2026-11-01

Two files, two roles. ``CACHE`` (2019-05 .. 2026-08) is the sample every
verdict up to entry 12 was scored on; this script reads its last bar and
never writes it. ``FORWARD`` holds only sessions after the cache ends -- the
days no verdict has seen -- and is what a forward-data entry scores. Keeping
the two apart is what makes "data nobody has looked at" a property of the
file rather than a date filter a runner has to get right.

The span defaults to the day after the last bar on disk (forward file if it
exists, else the cache) through the date Databento has data to. A start
inside the cache is refused (paying for bars already held, and mixing the
samples); a start that would leave a hole after the last bar is refused
(a runner scoring the forward file would count a silent gap as no sessions).
Re-pulling a span the forward file already holds is fine: the merge keeps
the rows already on disk.

Until 2026-09-14 this script pulled the 2019-05 .. 2024-09 backfill into the
cache; that span is in the cache and that job is done.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import databento as db
import pandas as pd

from loader import PROJECT_ROOT, get_databento_api_key, load_bars

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOL = "MES.v.0"
STYPE_IN = "continuous"

MAX_COST_USD = 15.0

CACHE = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
FORWARD = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_forward.parquet"


def last_bar_date(path: Path) -> date | None:
    """UTC date of the last bar in a parquet, or None if the file is absent."""
    if not path.exists():
        return None
    index = pd.read_parquet(path, columns=[]).index
    return pd.Timestamp(index.max()).date()


def default_span(cache_last: date, forward_last: date | None,
                 available_end: pd.Timestamp) -> tuple[date, date]:
    """Day after the last bar on disk, through the date Databento has data to."""
    start = (forward_last or cache_last) + timedelta(days=1)
    return start, pd.Timestamp(available_end).date()


def validate_span(start: date, end: date, cache_last: date,
                  forward_last: date | None) -> None:
    """Refuse a span that overlaps the cache, leaves a gap, or is empty."""
    if start <= cache_last:
        raise ValueError(
            f"start {start} is inside the cache, which holds bars through "
            f"{cache_last}. The forward file takes only days after that."
        )
    last_on_disk = forward_last or cache_last
    if start > last_on_disk + timedelta(days=1):
        raise ValueError(
            f"start {start} would leave a gap after {last_on_disk}, the last "
            f"bar on disk. Start no later than {last_on_disk + timedelta(days=1)}."
        )
    if end <= start:
        raise ValueError(f"end {end} must be after start {start} (end is exclusive).")


def merge(new: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """Concatenate, keeping the existing row wherever a timestamp is shared."""
    if existing is None:
        return new.sort_index()
    combined = pd.concat([new, existing])
    before = len(combined)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    dropped = before - len(combined)
    print(f"  new rows {len(new):,} + existing {len(existing):,} = "
          f"{len(combined):,} after dropping {dropped:,} duplicate timestamp(s)")
    return combined


def estimate(client: db.Historical, start: date, end: date) -> float:
    kwargs = dict(dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
                  start=str(start), end=str(end), stype_in=STYPE_IN)
    cost = float(client.metadata.get_cost(**kwargs))
    size = client.metadata.get_billable_size(**kwargs)
    print(f"{SYMBOL} {SCHEMA}  {start} -> {end} (end exclusive)")
    print(f"  estimate  ${cost:.4f}   size {size / 1_048_576:.2f} MB")
    return cost


def available_end(client: db.Historical) -> pd.Timestamp:
    return pd.Timestamp(client.metadata.get_dataset_range(DATASET)["end"])


def write_forward(new: pd.DataFrame) -> None:
    existing = pd.read_parquet(FORWARD) if FORWARD.exists() else None
    combined = merge(new, existing)
    FORWARD.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(FORWARD)
    size_mb = FORWARD.stat().st_size / 1_048_576
    print(f"\nWrote {FORWARD.name} ({size_mb:.2f} MB)")
    span = load_bars(FORWARD)
    print(f"  span {span.index.min()} .. {span.index.max()}")
    print(f"  sessions {len(set(span.index.date)):,}")


def run(argv: Sequence[str], client: db.Historical) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimate", action="store_true", help="cost only, no spend")
    ap.add_argument("--pull", action="store_true", help="estimate, gate, pull, merge")
    ap.add_argument("--merge-file", type=Path, default=None,
                    help="merge a local parquet into the forward file; no API call")
    ap.add_argument("--start", type=date.fromisoformat, default=None,
                    help="first day to pull (default: day after the last bar on disk)")
    ap.add_argument("--end", type=date.fromisoformat, default=None,
                    help="exclusive (default: the date Databento has data to)")
    ap.add_argument("--max-cost", type=float, default=MAX_COST_USD,
                    help="refuse to pull above this estimate")
    args = ap.parse_args(list(argv))
    if not (args.estimate or args.pull or args.merge_file):
        ap.error("pass --estimate, --pull or --merge-file")

    cache_last = last_bar_date(CACHE)
    if cache_last is None:
        print(f"STOP: {CACHE} is missing; nothing to define the forward boundary.")
        return 1
    forward_last = last_bar_date(FORWARD)

    if args.merge_file:
        new = pd.read_parquet(args.merge_file)
        start = pd.Timestamp(new.index.min()).date()
        end = pd.Timestamp(new.index.max()).date() + timedelta(days=1)
        try:
            validate_span(start, end, cache_last, forward_last)
        except ValueError as exc:
            print(f"STOP: {exc}")
            return 1
        print(f"Merging {args.merge_file.name}: {len(new):,} bars, {start} -> {end}")
        write_forward(new)
        return 0

    if args.start is None or args.end is None:
        start_default, end_default = default_span(cache_last, forward_last,
                                                  available_end(client))
        start = args.start or start_default
        end = args.end or end_default
        if args.start is None and args.end is None and end <= start:
            print(f"Forward file is current: last bar {forward_last or cache_last}, "
                  f"Databento's last complete day {end - timedelta(days=1)}. "
                  f"Nothing to pull.")
            return 0
    else:
        start, end = args.start, args.end
    try:
        validate_span(start, end, cache_last, forward_last)
    except ValueError as exc:
        print(f"STOP: {exc}")
        return 1

    cost = estimate(client, start, end)
    print(f"  cap       ${args.max_cost:.2f}")
    if args.estimate:
        return 0
    if cost > args.max_cost:
        print(f"STOP: ${cost:.4f} exceeds the ${args.max_cost:.2f} cap. Nothing pulled.")
        return 1

    print(f"\nProceeding at ${cost:.4f}. Pulling ...")
    data = client.timeseries.get_range(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=str(start), end=str(end), stype_in=STYPE_IN,
    )
    new = data.to_df()
    print(f"  pulled {len(new):,} bars")
    write_forward(new)
    return 0


def main() -> int:
    import sys
    return run(sys.argv[1:], db.Historical(get_databento_api_key()))


if __name__ == "__main__":
    raise SystemExit(main())
