"""Walk-forward evaluation: yearly folds, expanding training window.

Each fold picks parameters using only the years *before* the test year, then
runs those parameters once on the test year. No result from a test year ever
influences a selection, and no selection is revisited after seeing its result.

This is the honest version of the earlier single-split scan. A single split can
get lucky; a strategy that survives seven consecutive folds chosen this way has
been asked the question seven times.

    python backtests/walkforward.py
    python backtests/walkforward.py --slippage-ticks 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date
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
from scan import MIN_RELIABLE_TRADES, parameter_grid, slice_by_date  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

DATA_START = date(2019, 5, 6)
DATA_END = date(2026, 8, 31)

#: A fold needs at least this much training history to select on.
MIN_TRAIN_DAYS = 200


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def label(self) -> str:
        return f"{self.test_year}"


def build_folds(data_start: date = DATA_START, data_end: date = DATA_END) -> list[Fold]:
    """One fold per calendar year that has at least a year of prior history."""
    folds = []
    for year in range(data_start.year + 1, data_end.year + 1):
        train_end = date(year - 1, 12, 31)
        if (train_end - data_start).days < MIN_TRAIN_DAYS:
            continue
        test_end = min(date(year, 12, 31), data_end)
        folds.append(
            Fold(
                test_year=year,
                train_start=data_start,
                train_end=train_end,
                test_start=date(year, 1, 1),
                test_end=test_end,
            )
        )
    return folds


def window_metrics(signals, bars5, costs: CostModel, start: date, end: date) -> dict:
    """Evaluate one parameter set on one date window."""
    win_signals = slice_by_date(signals, start, end)
    win_bars = slice_by_date(bars5, start, end)
    trades = price_trades(build_trades(win_signals, win_bars), MES, costs, 1)
    trades, halts = enforce_daily_loss_limit(
        trades, win_bars, MES, costs, 1, rules.DAILY_LOSS_LIMIT
    )
    m = compute_metrics(trades)
    if not m.get("trade_count"):
        return {"trades": 0, "net_pnl": 0.0, "profit_factor": np.nan,
                "sharpe": np.nan, "max_drawdown": 0.0, "win_rate": np.nan,
                "halts": 0}
    return {
        "trades": m["trade_count"],
        "net_pnl": m["net_pnl"],
        "profit_factor": m["profit_factor"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "win_rate": m["win_rate_pct"],
        "halts": len(halts),
    }


def run_walkforward(bars5, roll_dates, early_closes, costs: CostModel,
                    verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns ``(fold_summary, all_pairs)``.

    ``all_pairs`` has one row per (fold, parameter set) with both the training
    and test metrics, which is what the pooled rank correlation is computed on.
    """
    folds = build_folds()
    grid = parameter_grid()
    pairs: list[dict] = []
    started = time.time()

    # Signals depend only on the parameters, so generate once per combination
    # over the whole history and slice per fold.
    for i, params in enumerate(grid, 1):
        strat = OpeningRangeBreakout(params, roll_dates=roll_dates,
                                     early_close_dates=early_closes)
        signals = strat.generate_signals_resampled(bars5)

        for fold in folds:
            train = window_metrics(signals, bars5, costs,
                                   fold.train_start, fold.train_end)
            test = window_metrics(signals, bars5, costs,
                                  fold.test_start, fold.test_end)
            pairs.append(
                {
                    "test_year": fold.test_year,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "opening_range_minutes": params.opening_range_minutes,
                    "trade_window_end": params.trade_window_end.strftime("%H:%M"),
                    "stop_multiple": params.stop_multiple,
                    "target_multiple": params.target_multiple,
                    **{f"train_{k}": v for k, v in train.items()},
                    **{f"test_{k}": v for k, v in test.items()},
                }
            )

        if verbose and (i % 16 == 0 or i == len(grid)):
            elapsed = time.time() - started
            print(f"  {i:>3}/{len(grid)} combos  {elapsed:6.1f}s elapsed, "
                  f"~{elapsed / i * (len(grid) - i):5.1f}s remaining", flush=True)

    all_pairs = pd.DataFrame(pairs)

    # Selection: best training Sharpe among sets with enough training trades.
    summary = []
    for fold in folds:
        block = all_pairs[all_pairs["test_year"] == fold.test_year]
        eligible = block[block["train_trades"] >= MIN_RELIABLE_TRADES]
        pool = eligible if len(eligible) else block
        chosen = pool.loc[pool["train_sharpe"].idxmax()]
        summary.append(
            {
                "test_year": fold.test_year,
                "train_window": f"{fold.train_start} .. {fold.train_end}",
                "test_window": f"{fold.test_start} .. {fold.test_end}",
                "candidates": len(pool),
                "opening_range_minutes": int(chosen["opening_range_minutes"]),
                "trade_window_end": chosen["trade_window_end"],
                "stop_multiple": chosen["stop_multiple"],
                "target_multiple": chosen["target_multiple"],
                "train_sharpe": chosen["train_sharpe"],
                "train_net_pnl": chosen["train_net_pnl"],
                "train_trades": int(chosen["train_trades"]),
                "test_sharpe": chosen["test_sharpe"],
                "test_net_pnl": chosen["test_net_pnl"],
                "test_profit_factor": chosen["test_profit_factor"],
                "test_trades": int(chosen["test_trades"]),
                "test_max_drawdown": chosen["test_max_drawdown"],
                "test_halts": int(chosen["test_halts"]),
                "fold_rank_corr": _corr(block),
            }
        )
    return pd.DataFrame(summary), all_pairs


def _corr(block: pd.DataFrame) -> float:
    pair = block[["train_sharpe", "test_sharpe"]].dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair["train_sharpe"].corr(pair["test_sharpe"], method="spearman"))


def pooled_rank_correlation(all_pairs: pd.DataFrame) -> float:
    """Spearman of training vs test Sharpe across every (fold, parameter) pair."""
    pair = all_pairs[["train_sharpe", "test_sharpe"]].dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair["train_sharpe"].corr(pair["test_sharpe"], method="spearman"))


def format_report(summary: pd.DataFrame, all_pairs: pd.DataFrame,
                  costs: CostModel) -> str:
    line = "=" * 104
    out = [line]
    out.append(
        f"ORB WALK-FORWARD  |  {len(summary)} yearly folds  |  "
        f"{all_pairs['opening_range_minutes'].nunique() * 4 * 4 * 3} parameter sets per fold  |  "
        f"slippage {costs.slippage_ticks:g} tick/side"
    )
    out.append("Parameters chosen on prior years only; each test year seen once.")
    out.append(line)

    header = (
        f"\n{'year':>5} {'train window':>26} {'params (OR/win/stop/tgt)':>26} "
        f"{'trainShp':>9} | {'OOS PnL':>10}{'OOS Shp':>9}{'OOS PF':>8}{'OOS n':>7}{'OOS DD':>9}"
    )
    out.append(header)
    out.append("-" * (len(header) - 1))
    for _, r in summary.iterrows():
        params = (f"{int(r['opening_range_minutes'])}m/{r['trade_window_end']}/"
                  f"{r['stop_multiple']:g}/{r['target_multiple']:g}")
        out.append(
            f"{int(r['test_year']):>5} {r['train_window']:>26} {params:>26} "
            f"{r['train_sharpe']:>9.2f} | {r['test_net_pnl']:>10,.0f}"
            f"{r['test_sharpe']:>9.2f}{r['test_profit_factor']:>8.2f}"
            f"{int(r['test_trades']):>7}{r['test_max_drawdown']:>9,.0f}"
        )

    total = summary["test_net_pnl"].sum()
    wins = int((summary["test_net_pnl"] > 0).sum())
    out.append("-" * (len(header) - 1))
    out.append(
        f"{'TOTAL':>5} {'':>26} {'':>26} {'':>9} | {total:>10,.0f}"
        f"{'':>9}{'':>8}{int(summary['test_trades'].sum()):>7}"
    )

    out.append("\n" + "-" * 104)
    out.append("VERDICT")
    out.append(
        f"  Profitable folds                : {wins} of {len(summary)}"
    )
    out.append(
        f"  Total walk-forward P&L          : ${total:,.2f} "
        f"over {int(summary['test_trades'].sum()):,} trades"
    )
    out.append(
        f"  Median fold OOS Sharpe          : {summary['test_sharpe'].median():+.3f}"
    )
    pooled = pooled_rank_correlation(all_pairs)
    out.append(
        f"  Pooled rank correlation         : {pooled:+.3f}   "
        f"(train Sharpe vs test Sharpe, {len(all_pairs):,} fold-parameter pairs)"
    )
    out.append("  Per-fold rank correlation       : " + "  ".join(
        f"{int(r['test_year'])}:{r['fold_rank_corr']:+.2f}" for _, r in summary.iterrows()
    ))

    # How a chosen set ranks among all sets on its own test year - 0.5 would be
    # exactly the median, i.e. selection adding nothing.
    percentiles = []
    for _, r in summary.iterrows():
        block = all_pairs[all_pairs["test_year"] == r["test_year"]].dropna(
            subset=["test_sharpe"]
        )
        if not len(block):
            continue
        percentiles.append((block["test_sharpe"] < r["test_sharpe"]).mean())
    if percentiles:
        out.append(
            f"  Chosen set's percentile on its own test year: "
            f"mean {np.mean(percentiles) * 100:.0f}th "
            f"(50th = selection added nothing)"
        )
    out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=1.25)
    args = ap.parse_args()

    bars = loader.load_bars(args.parquet)
    roll_dates = loader.detect_roll_dates(bars)
    early_closes = loader.detect_early_close_dates(bars)
    bars5 = resample_bars(bars, 5)
    costs = CostModel(commission_per_side=args.commission,
                      slippage_ticks=args.slippage_ticks)

    folds = build_folds()
    print(f"Walk-forward: {len(folds)} folds x {len(parameter_grid())} parameter sets "
          f"at {costs.slippage_ticks:g} tick slippage", flush=True)
    summary, all_pairs = run_walkforward(bars5, roll_dates, early_closes, costs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"slip{args.slippage_ticks:g}"
    summary.to_csv(RESULTS_DIR / f"orb_walkforward_{tag}.csv", index=False)
    all_pairs.to_csv(RESULTS_DIR / f"orb_walkforward_pairs_{tag}.csv", index=False)

    print()
    print(format_report(summary, all_pairs, costs))
    print(f"\nFold summary -> backtests/results/orb_walkforward_{tag}.csv")
    print(f"All pairs    -> backtests/results/orb_walkforward_pairs_{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
