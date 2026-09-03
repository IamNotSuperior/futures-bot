"""Run the ORB strategy over the cached data and print the report.

    python backtests/run_orb.py
    python backtests/run_orb.py --start 2025-01-01 --end 2025-12-31
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
import rules  # noqa: E402
from engine import MES, CostModel, run_backtest  # noqa: E402
from metrics import compute_metrics, format_report  # noqa: E402
from orb import ORBParams, OpeningRangeBreakout  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2024-09_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--commission", type=float, default=1.25, help="per side, per contract")
    ap.add_argument("--slippage-ticks", type=float, default=1.0, help="per side")
    ap.add_argument("--opening-range", type=int, default=15)
    ap.add_argument("--stop-multiple", type=float, default=1.0)
    ap.add_argument("--target-multiple", type=float, default=2.0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    bars = loader.load_bars(args.parquet)
    if args.start:
        bars = bars[bars.index >= pd.Timestamp(args.start, tz=rules.ET)]
    if args.end:
        end = pd.Timestamp(args.end, tz=rules.ET) + pd.Timedelta(days=1)
        bars = bars[bars.index < end]
    if bars.empty:
        print("No bars in range.")
        return 1

    roll_dates = loader.detect_roll_dates(bars)
    early_closes = loader.detect_early_close_dates(bars)

    params = ORBParams(
        opening_range_minutes=args.opening_range,
        stop_multiple=args.stop_multiple,
        target_multiple=args.target_multiple,
    )
    strat = OpeningRangeBreakout(params, roll_dates=roll_dates, early_close_dates=early_closes)

    signals = strat.generate_signals(bars)
    bars5 = strat.bars_for_signals(bars)
    costs = CostModel(
        commission_per_side=args.commission, slippage_ticks=args.slippage_ticks
    )
    trades = run_backtest(signals, bars5, spec=MES, costs=costs, contracts=1)

    print("=" * 72)
    print("SETUP")
    print("=" * 72)
    print(f"  Instrument         {MES.symbol}  "
          f"(tick {MES.tick_size}, ${MES.tick_value}/tick, ${MES.point_value}/point)")
    print(f"  Data               {Path(args.parquet).name}")
    print(f"  Range              {bars.index.min().date()} .. {bars.index.max().date()}")
    print(f"  Sessions           {len(set(bars5.index.date)):,}  "
          f"({len(roll_dates)} roll days excluded, {len(early_closes)} early closes)")
    print(f"  Size               1 contract (fixed)")
    print(f"  Commission         ${costs.commission_per_side:.2f}/side  "
          f"(${costs.commission_round_turn():.2f} round turn)")
    print(f"  Slippage           {costs.slippage_ticks:g} tick/side  "
          f"(${2 * costs.slippage_ticks * MES.tick_value:.2f} round turn)")
    print(f"  ORB params         open_range={params.opening_range_minutes}m  "
          f"window_end={params.trade_window_end}  "
          f"stop={params.stop_multiple}xOR  target={params.target_multiple}xstop")
    print()

    metrics = compute_metrics(trades)
    print(format_report(metrics, "ORB BACKTEST"))

    if not args.no_save and not trades.empty:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = f"{bars.index.min().date()}_{bars.index.max().date()}"
        out = RESULTS_DIR / f"orb_trades_{stamp}.csv"
        trades.to_csv(out, index=False)
        print(f"\nTrade list -> {out.relative_to(PROJECT_ROOT)}  ({len(trades):,} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
