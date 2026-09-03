"""Walk-forward and pre-registered analysis for hypothesis 2.

Runs the frozen test from research/hypotheses.md (commit 43aa4e1):
walk-forward over yearly folds, the pooled out-of-sample effect-size table,
each of the three kill criteria marked, and the volatility-normalised
early-versus-late check.

    python backtests/run_eod.py
    python backtests/run_eod.py --slippage-ticks 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from eod_rebalance import (  # noqa: E402
    ENTRY_BAR, OPEN_BAR, SIGNAL_BAR, THRESHOLDS, EODParams, EODRebalanceDrift,
    describe, parameter_grid,
)
from engine import equity_curve_by_day  # noqa: E402
from engine import CostModel  # noqa: E402
from orb import resample_bars  # noqa: E402
from walkforward import (  # noqa: E402
    build_folds, format_drawdown_report, format_report, oos_trade_stream,
    pooled_rank_correlation, run_walkforward,
)

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

# Frozen in the hypothesis entry.
MIN_TRAIN_TRADES = 30
MIN_POOLED_OOS = 30
LOW_CONFIDENCE_TRADES = 20
OOS_YEARS = range(2020, 2027)

PARAM_COLS = ["threshold", "stop_pct"]


def _factory(params, roll_dates, early_closes):
    return EODRebalanceDrift(params, roll_dates=roll_dates,
                             early_close_dates=early_closes)


def _rebuild(row) -> EODParams:
    """Reconstruct params from a summary row. -1 encodes "no stop"."""
    stop = float(row["stop_pct"])
    return EODParams(threshold=float(row["threshold"]),
                     stop_pct=None if stop < 0 else stop)


def session_table(bars5, roll_dates) -> pd.DataFrame:
    """Per-session signal and raw outcome, for the effect-size measurement.

    This is the statistical question - does the drift exist - measured on raw
    returns before costs, deliberately separate from whether it is tradeable.
    """
    rows = []
    for day, session in bars5.groupby(bars5.index.date):
        times = session.index.time
        o = session[times == OPEN_BAR]
        s = session[times == SIGNAL_BAR]
        e = session[times == ENTRY_BAR]
        if o.empty or s.empty or e.empty:
            continue
        if rules.is_roll_day(day, roll_dates):
            continue
        open_px = float(o["open"].iloc[0])
        signal_px = float(s["close"].iloc[0])
        entry_px = float(e["open"].iloc[0])
        close_px = float(session["close"].iloc[-1])
        if open_px <= 0 or entry_px <= 0:
            continue
        signal = signal_px / open_px - 1.0
        rows.append(
            {
                "date": day,
                "year": day.year,
                "signal": signal,
                "continuation": np.sign(signal) * (close_px / entry_px - 1.0),
                "abs_move": abs(signal),
            }
        )
    return pd.DataFrame(rows)


def effect_size_table(sessions: pd.DataFrame) -> pd.DataFrame:
    """E(theta) and H(theta) on pooled out-of-sample sessions."""
    oos = sessions[sessions["year"].isin(OOS_YEARS)]
    rows = []
    for t in THRESHOLDS:
        sub = oos[oos["abs_move"] >= t]
        n = len(sub)
        if n == 0:
            rows.append({"threshold": t, "n": 0, "E_bps": np.nan,
                         "H_pct": np.nan, "t_stat": np.nan, "sufficient": False})
            continue
        e = sub["continuation"].mean() * 10_000
        h = (sub["continuation"] > 0).mean() * 100
        sd = sub["continuation"].std(ddof=1) * 10_000
        t_stat = e / (sd / np.sqrt(n)) if sd > 0 else np.nan
        rows.append({"threshold": t, "n": n, "E_bps": e, "H_pct": h,
                     "t_stat": t_stat, "sufficient": n >= MIN_POOLED_OOS})
    return pd.DataFrame(rows)


def criterion_3(table: pd.DataFrame) -> tuple[str, str]:
    """Effect size must increase with threshold. Returns (verdict, detail)."""
    usable = table[table["sufficient"] & table["E_bps"].notna()]
    if len(usable) < 2:
        return "INSUFFICIENT", "fewer than two thresholds cleared the 30-trade floor"
    rho = usable["threshold"].corr(usable["E_bps"], method="spearman")
    lo = usable.iloc[0]["E_bps"]
    hi = usable.iloc[-1]["E_bps"]
    monotone = rho > 0
    endpoints = hi > lo
    detail = (f"Spearman(threshold, E) = {rho:+.3f}; "
              f"E({usable.iloc[-1]['threshold']:.2%}) = {hi:+.2f} bps vs "
              f"E({usable.iloc[0]['threshold']:.2%}) = {lo:+.2f} bps")
    return ("PASS" if (monotone and endpoints) else "FAIL"), detail


def volatility_check(sessions: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Early-vs-late, normalised by how volatile each year actually was.

    Recorded before results: early folds were expected to be stronger because
    the effect is thought to have decayed. But the earliest fold contains March
    2020, and a high-volatility year produces both more qualifying days and
    larger rebalance notionals. So raw early-vs-late cannot separate decay from
    volatility; the per-year effect is divided by that year's realised
    volatility of the continuation window to make the comparison fair.
    """
    lines = []
    oos = sessions[sessions["year"].isin(OOS_YEARS)]
    per_year = []
    for year, group in oos.groupby("year"):
        qualifying = group[group["abs_move"] >= THRESHOLDS[0]]
        vol = group["continuation"].std(ddof=1) * 10_000
        e = qualifying["continuation"].mean() * 10_000 if len(qualifying) else np.nan
        per_year.append(
            {"year": year, "n": len(qualifying), "E_bps": e, "vol_bps": vol,
             "E_over_vol": e / vol if vol and not np.isnan(e) else np.nan}
        )
    tbl = pd.DataFrame(per_year)
    lines.append(f"  {'year':>6}{'n':>7}{'E (bps)':>11}{'vol (bps)':>12}"
                 f"{'E/vol':>10}{'fold OOS PnL':>15}")
    pnl = dict(zip(summary["test_year"], summary["test_net_pnl"]))
    for _, r in tbl.iterrows():
        lines.append(
            f"  {int(r['year']):>6}{int(r['n']):>7}{r['E_bps']:>11.2f}"
            f"{r['vol_bps']:>12.2f}{r['E_over_vol']:>10.3f}"
            f"{pnl.get(r['year'], float('nan')):>15,.0f}"
        )
    early = tbl[tbl["year"] <= 2022]["E_over_vol"].mean()
    late = tbl[tbl["year"] >= 2023]["E_over_vol"].mean()
    lines.append("")
    lines.append(f"  Mean E/vol, early folds (2020-2022) : {early:+.4f}")
    lines.append(f"  Mean E/vol, late folds  (2023-2026) : {late:+.4f}")
    if np.isnan(early) or np.isnan(late):
        lines.append("  Cannot compare - insufficient data in one half.")
    elif early > late:
        lines.append("  Early > late, consistent with the recorded decay expectation.")
    else:
        lines.append("  Late >= early. Recorded expectation was the opposite: this is "
                     "a FLAG to investigate, not a result to celebrate.")
    return "\n".join(lines)


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

    grid = parameter_grid()
    print(f"EOD rebalance walk-forward: {len(build_folds())} folds x {len(grid)} "
          f"parameter sets at {costs.slippage_ticks:g} tick slippage", flush=True)

    summary, all_pairs = run_walkforward(
        bars5, roll_dates, early_closes, costs,
        grid=grid, factory=_factory, describe=describe,
        min_train_trades=MIN_TRAIN_TRADES,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"slip{args.slippage_ticks:g}"
    summary.to_csv(RESULTS_DIR / f"eod_walkforward_{tag}.csv", index=False)
    all_pairs.to_csv(RESULTS_DIR / f"eod_walkforward_pairs_{tag}.csv", index=False)

    print()
    print(format_report(summary, all_pairs, costs, PARAM_COLS, "EOD REBALANCE"))

    stream = oos_trade_stream(bars5, roll_dates, early_closes, costs, summary,
                              factory=_factory, rebuild=_rebuild)
    print()
    print(format_drawdown_report(stream, "EOD rebalance, stitched OOS stream"))
    if not stream.empty:
        stream.to_csv(RESULTS_DIR / f"eod_oos_stream_{tag}.csv", index=False)
        equity_curve_by_day(stream).to_csv(
            RESULTS_DIR / f"eod_oos_equity_{tag}.csv", index=False)

    sessions = session_table(bars5, roll_dates)
    table = effect_size_table(sessions)

    print("\n" + "=" * 104)
    print("POOLED OUT-OF-SAMPLE EFFECT SIZE  (raw returns, before costs)")
    print("=" * 104)
    print(f"  E = mean[sign(open->15:30) x (15:35->16:00)], "
          f"H = share continuing, over 2020-2026")
    print(f"\n  {'threshold':>10}{'n':>8}{'E (bps)':>11}{'H (%)':>9}"
          f"{'t-stat':>9}   evidence")
    for _, r in table.iterrows():
        ev = "ok" if r["sufficient"] else f"INSUFFICIENT (<{MIN_POOLED_OOS})"
        print(f"  {r['threshold']:>10.2%}{int(r['n']):>8}{r['E_bps']:>11.2f}"
              f"{r['H_pct']:>9.1f}{r['t_stat']:>9.2f}   {ev}")

    # --- kill criteria -----------------------------------------------------
    folds_profitable = int((summary["test_net_pnl"] > 0).sum())
    median_sharpe = summary["test_sharpe"].median()
    c1 = "PASS" if folds_profitable >= 4 else "FAIL"
    c2 = "PASS" if median_sharpe >= 0.3 else "FAIL"
    c3, c3_detail = criterion_3(table)

    print("\n" + "=" * 104)
    print("KILL CRITERIA  (fixed before any code; any one failure kills it)")
    print("=" * 104)
    print(f"  [{c1}] 1. At least 4 of 7 folds profitable out-of-sample")
    print(f"          -> {folds_profitable} of {len(summary)} profitable")
    print(f"  [{c2}] 2. Median fold out-of-sample Sharpe at least 0.30")
    print(f"          -> median {median_sharpe:+.3f}")
    print(f"  [{c3}] 3. Effect size increases with threshold")
    print(f"          -> {c3_detail}")

    low_conf = summary[summary["low_confidence"]]
    if len(low_conf):
        print(f"\n  Low-confidence folds (<{LOW_CONFIDENCE_TRADES} test trades): "
              f"{', '.join(str(int(y)) for y in low_conf['test_year'])}")

    verdict = "REJECTED" if "FAIL" in (c1, c2, c3) else (
        "INCONCLUSIVE" if c3 == "INSUFFICIENT" else "SURVIVES"
    )
    print(f"\n  VERDICT: {verdict}")

    print("\n" + "=" * 104)
    print("VOLATILITY-NORMALISED EARLY VS LATE")
    print("=" * 104)
    print(volatility_check(sessions, summary))

    print(f"\n  Pooled rank correlation (train vs test Sharpe): "
          f"{pooled_rank_correlation(all_pairs):+.3f}")
    print(f"\nFold summary -> backtests/results/eod_walkforward_{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
