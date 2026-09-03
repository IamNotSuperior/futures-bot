"""Eyeball check: run the ORB strategy over one week and print what it did.

Not a pytest test (the filename keeps it out of collection) - it is a script
for looking at signals before any backtest engine exists.

    python tests/smoke_orb.py
    python tests/smoke_orb.py --start 2026-06-15 --end 2026-06-19
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from orb import ORBParams, OpeningRangeBreakout  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2024-09_2026-08.parquet"

DEFAULT_START = "2026-08-24"
DEFAULT_END = "2026-08-28"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--parquet", default=str(PARQUET))
    args = ap.parse_args()

    all_bars = loader.load_bars(args.parquet)
    roll_dates = loader.detect_roll_dates(all_bars)
    early_closes = loader.detect_early_close_dates(all_bars)

    start = pd.Timestamp(args.start, tz=rules.ET)
    end = pd.Timestamp(args.end, tz=rules.ET) + pd.Timedelta(days=1)
    bars = all_bars.loc[(all_bars.index >= start) & (all_bars.index < end)]
    if bars.empty:
        print(f"No bars in {args.start}..{args.end}")
        return 1

    strat = OpeningRangeBreakout(
        ORBParams(), roll_dates=roll_dates, early_close_dates=early_closes
    )
    bars5 = strat.bars_for_signals(bars)
    signals = strat.generate_signals(bars)

    p = strat.params
    print(f"ORB smoke test  {args.start} .. {args.end}")
    print(
        f"params: opening_range={p.opening_range_minutes}m  window_end={p.trade_window_end}  "
        f"stop={p.stop_multiple}xOR  target={p.target_multiple}xstop  bars={p.bar_minutes}m"
    )
    print(f"1m bars={len(bars):,}   {p.bar_minutes}m bars={len(bars5):,}")
    print(f"roll dates in range: {sorted(d for d in roll_dates if start.date() <= d <= end.date()) or 'none'}")
    print(f"early closes in range: {sorted(d for d in early_closes if start.date() <= d <= end.date()) or 'none'}")

    entries = int(signals["entry_long"].sum() + signals["entry_short"].sum())
    exits = int(signals["exit_long"].sum() + signals["exit_short"].sum())
    print(f"\ntotal entries={entries}  exits={exits}")

    for day, session in bars5.groupby(bars5.index.date):
        print("\n" + "=" * 70)
        label = f"{day}  {pd.Timestamp(day).day_name()}"
        if day in roll_dates:
            label += "   [ROLL DAY - entries blocked]"
        if day in early_closes:
            label += "   [EARLY CLOSE]"
        print(label)

        opening_end = (
            pd.Timestamp(f"{day} 09:30") + pd.Timedelta(minutes=p.opening_range_minutes)
        ).time()
        opening = session[session.index.time < opening_end]
        if opening.empty:
            print("  no opening-range bars")
            continue
        or_high = opening["high"].max()
        or_low = opening["low"].min()
        print(
            f"  opening range 09:30-{opening_end.strftime('%H:%M')}: "
            f"high={or_high:.2f} low={or_low:.2f} height={or_high - or_low:.2f} "
            f"({len(opening)} bars)"
        )
        print(f"  session {p.bar_minutes}m bars: {len(session)}  "
              f"{session.index.min().strftime('%H:%M')}-{session.index.max().strftime('%H:%M')}")

        day_sig = signals.loc[signals.index.date == day]
        fired = day_sig[
            day_sig["entry_long"] | day_sig["entry_short"]
            | day_sig["exit_long"] | day_sig["exit_short"]
        ]
        if fired.empty:
            print("  no signals")
            continue

        for ts, row in fired.iterrows():
            bar = session.loc[ts]
            if row["entry_long"] or row["entry_short"]:
                side = "LONG " if row["entry_long"] else "SHORT"
                print(
                    f"  {ts.strftime('%H:%M')}  ENTRY {side} @ open {bar['open']:.2f}   "
                    f"stop={float(row['stop_price']):.2f} target={float(row['target_price']):.2f}"
                )
            if row["exit_long"] or row["exit_short"]:
                side = "LONG " if row["exit_long"] else "SHORT"
                print(
                    f"  {ts.strftime('%H:%M')}  EXIT  {side} reason={row['exit_reason']:<12} "
                    f"price={float(row['exit_price']):.2f}  "
                    f"(bar h={bar['high']:.2f} l={bar['low']:.2f})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
