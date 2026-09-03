"""ORB parameter scan with a time-based in-sample / out-of-sample split.

The whole point of the split is that ranking happens on in-sample only. The
out-of-sample column is never used to choose anything - it is there to answer
one question: does an in-sample ranking predict anything at all? The rank
correlation between the two Sharpe columns is that answer in a single number.

    python backtests/scan.py
    python backtests/scan.py --slippage-ticks 2
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from datetime import date, time as time_type
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from engine import MES, CostModel, build_trades, enforce_daily_loss_limit, price_trades  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from orb import ORBParams, OpeningRangeBreakout, resample_bars  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2024-09_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

# --- the split -------------------------------------------------------------
# Chosen once, before any results were looked at, and never moved.
IS_START = date(2024, 9, 1)
IS_END = date(2025, 12, 31)
OOS_START = date(2026, 1, 1)
OOS_END = date(2026, 8, 31)

# --- the grid --------------------------------------------------------------
OPENING_RANGE_MINUTES = [5, 10, 15, 30]
TRADE_WINDOW_ENDS = [time_type(10, 30), time_type(11, 30), time_type(12, 30)]
STOP_MULTIPLES = [0.5, 0.75, 1.0, 1.5]
TARGET_MULTIPLES = [1.0, 1.5, 2.0, 3.0]

MIN_RELIABLE_TRADES = 100


def parameter_grid() -> list[ORBParams]:
    return [
        ORBParams(
            opening_range_minutes=orm,
            trade_window_end=twe,
            stop_multiple=sm,
            target_multiple=tm,
        )
        for orm, twe, sm, tm in itertools.product(
            OPENING_RANGE_MINUTES, TRADE_WINDOW_ENDS, STOP_MULTIPLES, TARGET_MULTIPLES
        )
    ]


def slice_by_date(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows whose session date falls in ``[start, end]`` inclusive."""
    days = pd.Series(frame.index.date, index=frame.index)
    return frame[(days >= start) & (days <= end)]


def summarise(trades: pd.DataFrame, halts: pd.DataFrame, prefix: str) -> dict:
    """Flatten the metrics we rank and report on, prefixed for one window."""
    m = compute_metrics(trades)
    if not m.get("trade_count"):
        return {
            f"{prefix}_trades": 0,
            f"{prefix}_net_pnl": 0.0,
            f"{prefix}_profit_factor": np.nan,
            f"{prefix}_sharpe": np.nan,
            f"{prefix}_max_drawdown": 0.0,
            f"{prefix}_win_rate": np.nan,
            f"{prefix}_loss_limit_halts": 0,
        }
    return {
        f"{prefix}_trades": m["trade_count"],
        f"{prefix}_net_pnl": m["net_pnl"],
        f"{prefix}_profit_factor": m["profit_factor"],
        f"{prefix}_sharpe": m["sharpe"],
        f"{prefix}_max_drawdown": m["max_drawdown"],
        f"{prefix}_win_rate": m["win_rate_pct"],
        f"{prefix}_loss_limit_halts": len(halts),
    }


def evaluate(
    signals: pd.DataFrame,
    bars5: pd.DataFrame,
    costs: CostModel,
    start: date,
    end: date,
    prefix: str,
) -> dict:
    """Price one window of an already-generated signal set."""
    win_signals = slice_by_date(signals, start, end)
    win_bars = slice_by_date(bars5, start, end)
    trades = price_trades(build_trades(win_signals, win_bars), MES, costs, 1)
    trades, halts = enforce_daily_loss_limit(
        trades, win_bars, MES, costs, 1, rules.DAILY_LOSS_LIMIT
    )
    return summarise(trades, halts, prefix)


def run_scan(bars5: pd.DataFrame, roll_dates, early_closes, costs: CostModel,
             verbose: bool = True) -> pd.DataFrame:
    """Every grid combination, evaluated on both windows."""
    grid = parameter_grid()
    rows = []
    started = time.time()

    for i, params in enumerate(grid, 1):
        strat = OpeningRangeBreakout(params, roll_dates=roll_dates,
                                     early_close_dates=early_closes)
        # Sessions are independent, so one pass over the full span is
        # equivalent to generating each window separately (see
        # OpeningRangeBreakout.generate_signals_resampled).
        signals = strat.generate_signals_resampled(bars5)

        row = {
            "opening_range_minutes": params.opening_range_minutes,
            "trade_window_end": params.trade_window_end.strftime("%H:%M"),
            "stop_multiple": params.stop_multiple,
            "target_multiple": params.target_multiple,
        }
        row.update(evaluate(signals, bars5, costs, IS_START, IS_END, "is"))
        row.update(evaluate(signals, bars5, costs, OOS_START, OOS_END, "oos"))
        row["reliable"] = row["is_trades"] >= MIN_RELIABLE_TRADES
        rows.append(row)

        if verbose and (i % 24 == 0 or i == len(grid)):
            elapsed = time.time() - started
            rate = elapsed / i
            print(f"  {i:>3}/{len(grid)}  {elapsed:6.1f}s elapsed, "
                  f"~{rate * (len(grid) - i):5.1f}s remaining", flush=True)

    return pd.DataFrame(rows)


def rank_correlation(results: pd.DataFrame) -> float:
    """Spearman correlation between in-sample and out-of-sample Sharpe."""
    pair = results[["is_sharpe", "oos_sharpe"]].dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair["is_sharpe"].corr(pair["oos_sharpe"], method="spearman"))


def format_scan_report(results: pd.DataFrame, costs: CostModel, top_n: int = 15) -> str:
    out = ["=" * 108]
    out.append(
        f"ORB PARAMETER SCAN  |  {len(results)} combinations  |  "
        f"slippage {costs.slippage_ticks:g} tick/side, commission "
        f"${costs.commission_per_side:.2f}/side"
    )
    out.append(
        f"in-sample {IS_START} .. {IS_END}   |   out-of-sample {OOS_START} .. {OOS_END}"
    )
    out.append("=" * 108)

    unreliable = (~results["reliable"]).sum()
    out.append(
        f"\n{unreliable} of {len(results)} combinations have fewer than "
        f"{MIN_RELIABLE_TRADES} in-sample trades and are flagged unreliable."
    )

    ranked = results.sort_values("is_sharpe", ascending=False).head(top_n)
    out.append(f"\nTop {top_n} by IN-SAMPLE Sharpe (out-of-sample shown alongside, "
               f"never used for ranking):\n")
    header = (
        f"{'OR':>4}{'window':>8}{'stop':>6}{'tgt':>6} | "
        f"{'IS PnL':>10}{'IS Shp':>8}{'IS PF':>7}{'IS n':>6}{'IS DD':>9} | "
        f"{'OOS PnL':>10}{'OOS Shp':>9}{'OOS PF':>8}{'OOS n':>7} | flags"
    )
    out.append(header)
    out.append("-" * len(header))
    for _, r in ranked.iterrows():
        flags = []
        if not r["reliable"]:
            flags.append("LOW-N")
        if r["oos_net_pnl"] < 0:
            flags.append("OOS-LOSS")
        if r["is_loss_limit_halts"] or r["oos_loss_limit_halts"]:
            flags.append(f"halts={int(r['is_loss_limit_halts'])}/"
                         f"{int(r['oos_loss_limit_halts'])}")
        out.append(
            f"{int(r['opening_range_minutes']):>4}{r['trade_window_end']:>8}"
            f"{r['stop_multiple']:>6.2f}{r['target_multiple']:>6.2f} | "
            f"{r['is_net_pnl']:>10,.0f}{r['is_sharpe']:>8.2f}"
            f"{r['is_profit_factor']:>7.2f}{int(r['is_trades']):>6}"
            f"{r['is_max_drawdown']:>9,.0f} | "
            f"{r['oos_net_pnl']:>10,.0f}{r['oos_sharpe']:>9.2f}"
            f"{r['oos_profit_factor']:>8.2f}{int(r['oos_trades']):>7} | "
            f"{' '.join(flags)}"
        )

    rho = rank_correlation(results)
    rho_reliable = rank_correlation(results[results["reliable"]])
    out.append("\n" + "-" * 108)
    out.append("DOES IN-SAMPLE RANKING PREDICT OUT-OF-SAMPLE?")
    out.append(
        f"  Spearman rank correlation, IS Sharpe vs OOS Sharpe, all "
        f"{len(results)} combinations : {rho:+.3f}"
    )
    out.append(
        f"  Same, restricted to the {int(results['reliable'].sum())} reliable "
        f"combinations                : {rho_reliable:+.3f}"
    )

    top = ranked
    out.append(
        f"\n  Of the top {top_n} in-sample, {int((top['oos_net_pnl'] > 0).sum())} "
        f"are profitable out-of-sample and "
        f"{int((top['oos_sharpe'] > 0).sum())} have positive out-of-sample Sharpe."
    )
    out.append(
        f"  Median OOS Sharpe among the top {top_n}: "
        f"{top['oos_sharpe'].median():+.3f}   |   "
        f"across all combinations: {results['oos_sharpe'].median():+.3f}"
    )
    out.append("=" * 108)
    return "\n".join(out)


def plot_heatmaps(results: pd.DataFrame, out_path: Path, costs: CostModel) -> None:
    """Stop multiple vs target multiple, Sharpe averaged over the other axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    grids = []
    for col in ("is_sharpe", "oos_sharpe"):
        grids.append(
            results.pivot_table(
                index="stop_multiple", columns="target_multiple",
                values=col, aggfunc="mean",
            )
        )
    vmax = max(abs(np.nanmin([g.to_numpy().min() for g in grids])),
               abs(np.nanmax([g.to_numpy().max() for g in grids])))

    for ax, grid, title in zip(
        axes, grids,
        (f"In-sample Sharpe\n{IS_START} .. {IS_END}",
         f"Out-of-sample Sharpe\n{OOS_START} .. {OOS_END}"),
    ):
        im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(grid.columns)),
                      [f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)), [f"{i:g}" for i in grid.index])
        ax.set_xlabel("target multiple (x stop)")
        ax.set_ylabel("stop multiple (x opening range)")
        ax.set_title(title)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                value = grid.to_numpy()[y, x]
                if not np.isnan(value):
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center",
                            fontsize=9, color="black")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"ORB Sharpe by stop and target multiple "
        f"(averaged over opening range and trading window, "
        f"{costs.slippage_ticks:g} tick slippage/side)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=1.25)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    bars = loader.load_bars(args.parquet)
    roll_dates = loader.detect_roll_dates(bars)
    early_closes = loader.detect_early_close_dates(bars)
    bars5 = resample_bars(bars, 5)

    costs = CostModel(commission_per_side=args.commission,
                      slippage_ticks=args.slippage_ticks)

    print(f"Scanning {len(parameter_grid())} combinations "
          f"at {costs.slippage_ticks:g} tick slippage ...", flush=True)
    results = run_scan(bars5, roll_dates, early_closes, costs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"slip{args.slippage_ticks:g}"
    csv_path = RESULTS_DIR / f"orb_scan_{tag}.csv"
    results.to_csv(csv_path, index=False)

    print()
    print(format_scan_report(results, costs, args.top))
    print(f"\nAll {len(results)} rows -> {csv_path.relative_to(PROJECT_ROOT)}")

    if not args.no_plot:
        png = RESULTS_DIR / f"orb_sharpe_heatmap_{tag}.png"
        plot_heatmaps(results, png, costs)
        print(f"Heatmaps -> {png.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
