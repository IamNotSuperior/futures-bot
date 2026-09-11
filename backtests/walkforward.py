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
from datetime import date, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from engine import (  # noqa: E402
    MES, CostModel, build_trades, count_evaluation_blowups, enforce_daily_loss_limit,
    equity_curve_by_day, price_trades, trailing_drawdown_summary,
)
import eval_sim  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from orb import ORBParams, OpeningRangeBreakout, resample_bars  # noqa: E402
from scan import MIN_RELIABLE_TRADES, parameter_grid, slice_by_date  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

DATA_START = date(2019, 5, 6)
DATA_END = date(2026, 8, 31)

#: A fold needs at least this much training history to select on.
MIN_TRAIN_DAYS = 200

#: Monte Carlo paths used when *selecting* on pass probability. Reporting runs
#: the full 20,000; selection only needs to rank, and pays this per candidate.
SELECTION_PATHS = 4_000


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


def window_metrics(signals, bars, costs: CostModel, start: date, end: date,
                   contracts: int = 1, with_pass_probability: bool = False,
                   sim_paths: int = SELECTION_PATHS) -> dict:
    """Evaluate one parameter set on one date window.

    ``with_pass_probability`` adds the evaluation Monte Carlo, which is what a
    pass-probability selection ranks on. It is opt-in because it costs a full
    simulation per (fold, parameter set) - negligible for a single fixed
    configuration, material for a 192-cell grid.
    """
    win_signals = slice_by_date(signals, start, end)
    win_bars = slice_by_date(bars, start, end)
    trades = price_trades(build_trades(win_signals, win_bars), MES, costs, contracts)
    trades, halts = enforce_daily_loss_limit(
        trades, win_bars, MES, costs, contracts, rules.DAILY_LOSS_LIMIT
    )
    m = compute_metrics(trades)
    if not m.get("trade_count"):
        empty = {"trades": 0, "net_pnl": 0.0, "profit_factor": np.nan,
                 "sharpe": np.nan, "max_drawdown": 0.0, "win_rate": np.nan,
                 "halts": 0}
        if with_pass_probability:
            empty["pass_probability"] = 0.0
        return empty
    out = {
        "trades": m["trade_count"],
        "net_pnl": m["net_pnl"],
        "profit_factor": m["profit_factor"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "win_rate": m["win_rate_pct"],
        "halts": len(halts),
    }
    if with_pass_probability:
        out["pass_probability"] = eval_sim.simulate(
            eval_sim.daily_pnl_from_trades(trades), paths=sim_paths
        ).pass_probability
    return out


# ---------------------------------------------------------------------------
# Selection
#
# A selector turns one fold's candidate block into the chosen row. The default
# is the training-Sharpe rule entries 1 and 2 were evaluated under, and it must
# stay exactly as it was so those results remain reproducible. Entry 4 adds a
# pass-probability rule alongside it.
# ---------------------------------------------------------------------------


def _pool(block: pd.DataFrame, min_train_trades: int):
    """Candidates with enough training history, or all of them if none has.

    The fallback is deliberate but dangerous: an eligibility floor set higher
    than any candidate can reach silently selects from the unfiltered pool. The
    counts are returned so a report can show when that happened.
    """
    eligible = block[block["train_trades"] >= min_train_trades]
    return (eligible if len(eligible) else block), len(eligible)


def select_by_train_sharpe(block: pd.DataFrame, min_train_trades: int):
    """Highest training Sharpe. The rule entries 1 and 2 were run under."""
    pool, n_eligible = _pool(block, min_train_trades)
    return pool.loc[pool["train_sharpe"].idxmax()], len(pool), n_eligible


def select_by_pass_probability(block: pd.DataFrame, min_train_trades: int):
    """Highest training evaluation pass probability.

    Ties are broken in the order entry 4 pre-registers: highest training net
    P&L, then lowest training maximum drawdown, then the first parameter set in
    grid order. Many candidates will tie at a pass probability of exactly zero,
    so the tie-break is not a formality - without one, the choice would fall to
    whatever ordering the frame happened to have.
    """
    pool, n_eligible = _pool(block, min_train_trades)
    ordered = pool.sort_values(
        ["train_pass_probability", "train_net_pnl", "train_max_drawdown"],
        ascending=[False, False, True],
        kind="mergesort",  # stable, so grid order is the final tie-break
    )
    return ordered.iloc[0], len(pool), n_eligible


#: name -> (selector, whether it needs the pass-probability simulation)
SELECTORS = {
    "train_sharpe": (select_by_train_sharpe, False),
    "pass_probability": (select_by_pass_probability, True),
}


def _orb_factory(params, roll_dates, early_closes):
    return OpeningRangeBreakout(params, roll_dates=roll_dates,
                                early_close_dates=early_closes)


def _orb_describe(params) -> dict:
    return {
        "opening_range_minutes": params.opening_range_minutes,
        "trade_window_end": params.trade_window_end.strftime("%H:%M"),
        "stop_multiple": params.stop_multiple,
        "target_multiple": params.target_multiple,
    }


def _generate_resampled(strat, bars):
    """Default signal generation: the resampled-bar path ORB uses."""
    return strat.generate_signals_resampled(bars)


def run_walkforward(bars, roll_dates, early_closes, costs: CostModel,
                    grid=None, factory=None, describe=None,
                    min_train_trades: int = MIN_RELIABLE_TRADES,
                    verbose: bool = True, contracts: int = 1,
                    selection: str = "train_sharpe",
                    generate=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns ``(fold_summary, all_pairs)``.

    ``all_pairs`` has one row per (fold, parameter set) with both the training
    and test metrics, which is what the pooled rank correlation is computed on.

    ``grid``/``factory``/``describe`` default to ORB, so the ORB path is
    unchanged by construction rather than by re-running it. ``selection``
    likewise defaults to the training-Sharpe rule entries 1 and 2 used.

    ``generate`` exists because not every strategy decides on resampled bars:
    ORB-2 reads 1-minute bars directly, so it overrides this with
    ``lambda strat, bars: strat.generate_signals(bars)``.

    A strategy with no grid passes a one-element ``grid``. Selection then has a
    single candidate and cannot express a preference, which is the point - entry
    4 has nothing to select, and each fold is simply its out-of-sample year.
    """
    if selection not in SELECTORS:
        raise ValueError(
            f"Unknown selection {selection!r}; expected one of {sorted(SELECTORS)}"
        )
    selector, needs_pass_prob = SELECTORS[selection]

    folds = build_folds()
    grid = parameter_grid() if grid is None else grid
    factory = _orb_factory if factory is None else factory
    describe = _orb_describe if describe is None else describe
    generate = _generate_resampled if generate is None else generate
    pairs: list[dict] = []
    started = time.time()

    # Signals depend only on the parameters, so generate once per combination
    # over the whole history and slice per fold.
    for i, params in enumerate(grid, 1):
        strat = factory(params, roll_dates, early_closes)
        signals = generate(strat, bars)

        for fold in folds:
            train = window_metrics(signals, bars, costs,
                                   fold.train_start, fold.train_end,
                                   contracts, needs_pass_prob)
            test = window_metrics(signals, bars, costs,
                                  fold.test_start, fold.test_end,
                                  contracts, needs_pass_prob)
            pairs.append(
                {
                    "test_year": fold.test_year,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    **describe(params),
                    **{f"train_{k}": v for k, v in train.items()},
                    **{f"test_{k}": v for k, v in test.items()},
                }
            )

        if verbose and (i % 16 == 0 or i == len(grid)):
            elapsed = time.time() - started
            print(f"  {i:>3}/{len(grid)} combos  {elapsed:6.1f}s elapsed, "
                  f"~{elapsed / i * (len(grid) - i):5.1f}s remaining", flush=True)

    all_pairs = pd.DataFrame(pairs)

    summary = []
    for fold in folds:
        block = all_pairs[all_pairs["test_year"] == fold.test_year]
        chosen, n_pool, n_eligible = selector(block, min_train_trades)
        param_cols = list(describe(grid[0]).keys())
        extra = (
            {"train_pass_probability": chosen["train_pass_probability"],
             "test_pass_probability": chosen["test_pass_probability"]}
            if needs_pass_prob else {}
        )
        summary.append(
            {
                "test_year": fold.test_year,
                "train_window": f"{fold.train_start} .. {fold.train_end}",
                "test_window": f"{fold.test_start} .. {fold.test_end}",
                "candidates": n_pool,
                "eligible": n_eligible,
                **{c: chosen[c] for c in param_cols},
                "train_sharpe": chosen["train_sharpe"],
                "train_net_pnl": chosen["train_net_pnl"],
                "train_trades": int(chosen["train_trades"]),
                "test_sharpe": chosen["test_sharpe"],
                "test_net_pnl": chosen["test_net_pnl"],
                "test_profit_factor": chosen["test_profit_factor"],
                "test_trades": int(chosen["test_trades"]),
                "test_max_drawdown": chosen["test_max_drawdown"],
                "test_halts": int(chosen["test_halts"]),
                "low_confidence": bool(int(chosen["test_trades"]) < 20),
                "fold_rank_corr": _corr(block),
                **extra,
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
                  costs: CostModel, param_cols=None, title: str = "ORB") -> str:
    line = "=" * 104
    out = [line]
    per_fold = len(all_pairs) // max(len(summary), 1)
    out.append(
        f"{title} WALK-FORWARD  |  {len(summary)} yearly folds  |  "
        f"{per_fold} parameter sets per fold  |  "
        f"slippage {costs.slippage_ticks:g} tick/side"
    )
    out.append("Parameters chosen on prior years only; each test year seen once.")
    out.append(line)

    if param_cols is None:
        param_cols = ["opening_range_minutes", "trade_window_end",
                      "stop_multiple", "target_multiple"]
    header = (
        f"\n{'year':>5} {'train window':>26} {'chosen params':>26} "
        f"{'trainShp':>9} | {'OOS PnL':>10}{'OOS Shp':>9}{'OOS PF':>8}{'OOS n':>7}{'OOS DD':>9}"
    )
    out.append(header)
    out.append("-" * (len(header) - 1))
    for _, r in summary.iterrows():
        params = "/".join(
            f"{r[c]:g}" if isinstance(r[c], (int, float, np.floating)) else str(r[c])
            for c in param_cols
        )
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
    ap.add_argument("--commission", type=float, default=rules.COMMISSION_PER_SIDE)
    args = ap.parse_args()

    bars = loader.load_bars(args.parquet)
    roll_dates = loader.detect_roll_dates(bars)
    early_closes = loader.detect_early_close_dates(bars)
    bars = resample_bars(bars, 5)
    costs = CostModel(commission_per_side=args.commission,
                      slippage_ticks=args.slippage_ticks)

    folds = build_folds()
    print(f"Walk-forward: {len(folds)} folds x {len(parameter_grid())} parameter sets "
          f"at {costs.slippage_ticks:g} tick slippage", flush=True)
    summary, all_pairs = run_walkforward(bars, roll_dates, early_closes, costs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"slip{args.slippage_ticks:g}"
    summary.to_csv(RESULTS_DIR / f"orb_walkforward_{tag}.csv", index=False)
    all_pairs.to_csv(RESULTS_DIR / f"orb_walkforward_pairs_{tag}.csv", index=False)

    print()
    print(format_report(summary, all_pairs, costs))

    stream = oos_trade_stream(bars, roll_dates, early_closes, costs, summary)
    print()
    print(format_drawdown_report(stream, "ORB, stitched out-of-sample stream"))
    if not stream.empty:
        stream.to_csv(RESULTS_DIR / f"orb_oos_stream_{tag}.csv", index=False)
        equity_curve_by_day(stream).to_csv(
            RESULTS_DIR / f"orb_oos_equity_{tag}.csv", index=False)
    print(f"\nFold summary -> backtests/results/orb_walkforward_{tag}.csv")
    print(f"All pairs    -> backtests/results/orb_walkforward_pairs_{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _orb_rebuild(row) -> ORBParams:
    hh, mm = str(row["trade_window_end"]).split(":")
    return ORBParams(
        opening_range_minutes=int(row["opening_range_minutes"]),
        trade_window_end=dt_time(int(hh), int(mm)),
        stop_multiple=float(row["stop_multiple"]),
        target_multiple=float(row["target_multiple"]),
    )


def oos_trade_stream(bars, roll_dates, early_closes, costs, summary,
                     factory=None, rebuild=None, contracts: int = 1,
                     generate=None) -> pd.DataFrame:
    """The trades actually taken out-of-sample, stitched across folds.

    Each fold contributes its test year traded with the parameters that fold
    chose. That concatenation is the real out-of-sample experience: what an
    account following this process would have held, in order. Anything measured
    on a single fold's parameters over all seven years would be hindsight.
    """
    factory = _orb_factory if factory is None else factory
    rebuild = _orb_rebuild if rebuild is None else rebuild
    generate = _generate_resampled if generate is None else generate
    folds = {f.test_year: f for f in build_folds()}

    frames = []
    for _, row in summary.iterrows():
        fold = folds[int(row["test_year"])]
        strat = factory(rebuild(row), roll_dates, early_closes)
        signals = generate(strat, bars)
        win_signals = slice_by_date(signals, fold.test_start, fold.test_end)
        win_bars = slice_by_date(bars, fold.test_start, fold.test_end)
        trades = price_trades(build_trades(win_signals, win_bars), MES, costs, contracts)
        trades, _ = enforce_daily_loss_limit(
            trades, win_bars, MES, costs, contracts, rules.DAILY_LOSS_LIMIT
        )
        if not trades.empty:
            frames.append(trades)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        "entry_time").reset_index(drop=True)


def format_drawdown_report(stream: pd.DataFrame, label: str = "") -> str:
    """EOD trailing drawdown over the stitched out-of-sample stream."""
    equity = equity_curve_by_day(stream)
    summary = trailing_drawdown_summary(equity)
    blow = count_evaluation_blowups(stream)

    out = ["=" * 104,
           f"EOD TRAILING DRAWDOWN{(' - ' + label) if label else ''}",
           "=" * 104]
    if not len(equity):
        out.append("  No trades.")
        return "\n".join(out)

    out += [
        f"  Starting balance            ${rules.ACCOUNT_SIZE:>12,.2f}",
        f"  Final balance               ${summary['final_balance']:>12,.2f}",
        f"  Peak end-of-day balance     ${summary['peak_balance']:>12,.2f}",
        f"  Worst drawdown from peak    ${summary['max_drawdown_from_peak']:>12,.2f}"
        f"   (firm line ${rules.FIRM.max_trailing_drawdown:,.0f})",
        f"  Closest approach to the firm line  "
        f"${summary['min_headroom_to_firm']:>7,.2f} of headroom left",
        "",
        f"  Days in internal WARN state (>= ${rules.TRAILING_DD_WARN:,.0f}) : "
        f"{summary['warn_days']:>5}",
        f"  Days past the internal STOP (>= ${rules.TRAILING_DD_STOP:,.0f}) : "
        f"{summary['internal_stop_days']:>5}",
        f"  Trading days                                     : {summary['days']:>5}",
        "",
        f"  $50K evaluations BLOWN       {blow['blowups']:>5}",
        f"  $50K evaluations PASSED      {blow['passes']:>5}   "
        f"(+${rules.PROFIT_TARGET:,.0f} reached before the trailing line)",
    ]
    if blow["blowups"]:
        out.append("\n  Each blow-up restarts a fresh $50K account the next day:")
        for d, n in zip(blow["dates"], blow["days_survived"]):
            out.append(f"    account died {d} after {n} trading day(s)")
    return "\n".join(out)
