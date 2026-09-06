"""Entry 7: replication of entry 6's one non-null finding, on MNQ.

Frozen at ``bff2bdc``. One pre-registered test: the 1x arm with the filter ON at
2 ticks per side, target share against the de-meaned-bootstrap benchmark,
one-sided z at alpha = 0.05.

MES is recomputed under the same corrected benchmark, because comparing a
corrected MNQ number against entry 6's uncorrected one would measure nothing.

    python backtests/run_entry7.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
import eval_sim  # noqa: E402
import bootstrap_benchmark as bb  # noqa: E402
from engine import (  # noqa: E402
    MES, MNQ, CostModel, build_trades, count_evaluation_blowups,
    equity_curve_by_day, price_trades,
)
from metrics import compute_metrics  # noqa: E402
from london import LondonBreakout, LondonParams  # noqa: E402
from run_london import apply_daily_limit  # noqa: E402
from scan import slice_by_date  # noqa: E402
from trend import intraday_ema  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"
START, END = date(2020, 1, 1), date(2026, 8, 31)
YEARS = list(range(2020, 2027))

INSTRUMENTS = {
    "MES": dict(parquet="mes_v_0_ohlcv_1m_2019-05_2026-08.parquet", spec=MES),
    "MNQ": dict(parquet="mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet", spec=MNQ),
}

#: Entry 7's kill criterion.
MIN_DEPARTURE_POINTS = 2.0
CRITICAL_Z = 1.65

#: Entry 6's criteria, reported for completeness.
MIN_PASS_PROBABILITY = 0.25
MAX_BLOWUPS = 1
MIN_PROFITABLE_FOLDS = 4

LINE = "=" * 100


def build(name: str, costs: CostModel):
    """Entry 6's 1x arm, filter ON, on one instrument."""
    cfg = INSTRUMENTS[name]
    spec = cfg["spec"]
    bars = loader.load_bars(str(PROJECT_ROOT / "data" / cfg["parquet"]))
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    ema = intraday_ema(bars, 5, 200)

    params = LondonParams(target_multiple=1.0, use_trend_filter=True,
                          point_value=spec.point_value)
    strat = LondonBreakout(params, trend_ema=ema, roll_dates=rolls,
                           early_close_dates=early)
    signals = strat.generate_signals(bars)

    win_signals = slice_by_date(signals, START, END)
    win_bars = slice_by_date(bars, START, END)
    raw = build_trades(win_signals, win_bars)
    priced = price_trades(raw, spec, costs, 1)

    sizes = win_signals["contracts"].dropna()
    n = priced["entry_time"].map(sizes)
    if n.isna().any():
        raise RuntimeError("a trade has no size; signals and trades disagree")
    n = n.astype(int)
    out = priced.copy()
    out["contracts"] = n
    for col in ("gross_pnl", "commission", "slippage_cost", "net_pnl"):
        out[col] = out[col] * n

    out, halts = apply_daily_limit(out, win_bars, costs, spec)

    # The bootstrap needs the barrier levels, which build_trades does not carry.
    for col in ("stop_price", "target_price"):
        out[col] = out["entry_time"].map(win_signals[col].dropna()).astype(float)
    if out[["stop_price", "target_price"]].isna().any().any():
        raise RuntimeError("a trade is missing its barrier levels")

    return bars, win_bars, out, strat.diagnostics, halts, params


def observed_share(trades: pd.DataFrame) -> tuple[float, int, int, int]:
    reasons = trades["exit_reason"].value_counts()
    t = int(reasons.get("target", 0))
    s = int(reasons.get("stop", 0))
    resolved = t + s
    return ((t / resolved) if resolved else float("nan"), t, s,
            int(reasons.get("flatten_0925", 0)))


def pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slippage-ticks", type=float, default=2.0)
    ap.add_argument("--replications", type=int, default=bb.REPLICATIONS)
    ap.add_argument("--paths", type=int, default=20_000)
    args = ap.parse_args()

    costs = CostModel(commission_per_side=1.25, slippage_ticks=args.slippage_ticks)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(LINE)
    print("ENTRY 7 - replication of entry 6's non-null finding, on MNQ")
    print(f"  entry 6 spec, 1x arm, filter ON, {args.slippage_ticks:g} ticks/side")
    print(f"  benchmark: de-meaned bootstrap, {args.replications:,} replications, "
          f"seed {bb.SEED}")
    print(f"  test: one-sided z, alpha 0.05, critical z = {CRITICAL_Z}")
    print(LINE, flush=True)

    results = {}
    for name in ("MES", "MNQ"):
        print(f"\nBuilding {name} ...", flush=True)
        bars, win_bars, trades, diag, halts, params = build(name, costs)
        print(f"  {len(trades):,} trades", flush=True)

        obs, targets, stops, flattens = observed_share(trades)
        print(f"  bootstrapping {name} ...", flush=True)
        bench = bb.benchmark(trades, win_bars, time(9, 25),
                             replications=args.replications)
        z = bench.z_against(obs, targets + stops)
        naive = bb.infinite_horizon(trades)

        m = compute_metrics(trades)
        daily = eval_sim.daily_pnl_from_trades(trades)
        sim = eval_sim.simulate(daily, paths=args.paths)
        blow = count_evaluation_blowups(trades)
        t = trades.assign(year=pd.to_datetime(trades["entry_time"]).dt.year)
        profitable = sum(1 for y in YEARS
                         if not t[t["year"] == y].empty
                         and t[t["year"] == y]["net_pnl"].sum() > 0)

        results[name] = dict(
            trades=trades, diag=diag, observed=obs, targets=targets, stops=stops,
            flattens=flattens, bench=bench, z=z, naive=naive, metrics=m,
            sim=sim, blowups=int(blow["blowups"]), profitable=profitable,
            halts=halts, point_value=params.point_value,
            departure=100 * (obs - bench.target_share),
            naive_departure=100 * (obs - naive),
        )
        trades.to_csv(RESULTS / f"entry7_{name.lower()}_1x_on.csv", index=False)

    print()
    print(LINE)
    print("THE TEST - target share against the corrected benchmark")
    print(LINE)
    print(f"{'':<38}{'MES (recomputed)':>22}{'MNQ (replication)':>22}")
    print("-" * 100)

    def row(label, fmt):
        print(f"  {label:<36}" + "".join(f"{fmt(results[k]):>22}" for k in ("MES", "MNQ")))

    row("Trades", lambda r: f"{len(r['trades']):,}")
    row("Targets / stops", lambda r: f"{r['targets']} / {r['stops']}")
    row("09:25 flattens", lambda r: f"{r['flattens']}")
    row("Observed target share", lambda r: pct(r["observed"]))
    row("Bootstrap benchmark", lambda r: pct(r["bench"].target_share))
    row("Departure (points)", lambda r: f"{r['departure']:+.2f}")
    row("z (one-sided)", lambda r: f"{r['z']:+.2f}")
    row("Clears z = 1.65?", lambda r: "YES" if r["z"] >= CRITICAL_Z else "no")
    row("Clears +2.0 points?",
        lambda r: "YES" if r["departure"] >= MIN_DEPARTURE_POINTS else "no")
    print("-" * 100)
    print("  For comparison only - the invalid benchmark entry 6 used:")
    row("  a/(a+b), infinite horizon", lambda r: pct(r["naive"]))
    row("  departure under it (points)", lambda r: f"{r['naive_departure']:+.2f}")

    print()
    print(LINE)
    print("BOOTSTRAP DIAGNOSTICS")
    print(LINE)
    row("Trades bootstrapped", lambda r: f"{r['bench'].n_trades:,}")
    row("Replications each", lambda r: f"{r['bench'].replications:,}")
    row("Mean resolved fraction", lambda r: pct(r["bench"].resolved_fraction))
    row("Per-trade benchmark sd", lambda r: f"{r['bench'].per_trade_dispersion:.4f}")
    row("Pooled targets / stops",
        lambda r: f"{r['bench'].pooled_targets:,} / {r['bench'].pooled_stops:,}")

    print()
    print(LINE)
    print("ENTRY 6'S KILL CRITERIA ON MNQ - for completeness, not the test")
    print(LINE)
    row("Net P&L", lambda r: f"${r['metrics']['net_pnl']:,.0f}")
    row("Sharpe", lambda r: f"{r['metrics']['sharpe']:.2f}")
    row("Profit factor", lambda r: f"{r['metrics']['profit_factor']:.3f}")
    row("Max drawdown", lambda r: f"${r['metrics']['max_drawdown']:,.0f}")
    row("Folds profitable", lambda r: f"{r['profitable']} of 7")
    row("Evaluations blown", lambda r: f"{r['blowups']}")
    row("Pass probability", lambda r: pct(r["sim"].pass_probability))
    row("Daily-loss halts", lambda r: f"{r['halts']}")

    print()
    print(LINE)
    print("PER-FOLD NET P&L")
    print(LINE)
    print(f"{'year':>6}" + "".join(f"{k:>16}" for k in ("MES", "MNQ")))
    for year in YEARS:
        cells = []
        for k in ("MES", "MNQ"):
            g = results[k]["trades"]
            g = g[pd.to_datetime(g["entry_time"]).dt.year == year]
            cells.append(f"{g['net_pnl'].sum():>16,.0f}" if len(g) else f"{'-':>16}")
        print(f"{year:>6}" + "".join(cells))

    print()
    print(LINE)
    print("EXIT REASONS")
    print(LINE)
    for k in ("MES", "MNQ"):
        r = results[k]["trades"].groupby("exit_reason")["net_pnl"].agg(
            n="count", total="sum", mean="mean")
        print(f"\n  {k}")
        print("    " + r.round(2).to_string().replace("\n", "\n    "))

    mnq = results["MNQ"]
    replicated = (mnq["departure"] >= MIN_DEPARTURE_POINTS
                  and mnq["z"] >= CRITICAL_Z)
    print()
    print(LINE)
    print("ENTRY 7 KILL CRITERION")
    print(LINE)
    print(f"  MNQ departure {mnq['departure']:+.2f} points "
          f"(needs >= +{MIN_DEPARTURE_POINTS:.1f})")
    print(f"  MNQ z         {mnq['z']:+.2f} (needs >= {CRITICAL_Z})")
    print(f"  VERDICT: {'REPLICATED' if replicated else 'NOT REPLICATED'}"
          f"   -> the London family {'stays open' if replicated else 'closes'}")
    mes = results["MES"]
    print()
    print(f"  MES under the corrected benchmark: {mes['departure']:+.2f} points, "
          f"z = {mes['z']:+.2f}")
    print(f"  MES under entry 6's invalid benchmark: "
          f"{mes['naive_departure']:+.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
