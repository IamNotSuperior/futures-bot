"""Entry 6: London open breakout. Every pre-registered test, in one run.

Entry 6 is frozen at ``068927d``; its open implementation details at ``6fe805f``.
Two arms (target 1x and 2x the range) are judged independently against the same
kill criteria, each with the trend filter ON and OFF.

    python backtests/run_london.py                  # base case, 2 ticks
    python backtests/run_london.py --slippage-ticks 1   # optimistic sensitivity

Per-trade sizing
----------------
Size varies per session, and ``engine.price_trades`` takes a scalar. P&L is
exactly linear in contracts *within a trade* - ``net_pnl = contracts x
(net_points x 5 - 2.5)`` - so trades are priced at one contract and each row is
then scaled by its own size. That uses the tested engine path rather than a
second implementation of it.

The daily loss limit is applied per contract-size group. That is exact here and
only here: entry 6 takes at most one trade a day, so no day mixes sizes, and
grouping by size cannot split a day.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
import eval_sim  # noqa: E402
from engine import (  # noqa: E402
    MES, CostModel, build_trades, count_evaluation_blowups,
    enforce_daily_loss_limit, equity_curve_by_day, price_trades,
    trailing_drawdown_summary,
)
from metrics import compute_metrics, sharpe_ratio  # noqa: E402
from london import (  # noqa: E402
    SKIP_ROLL, SKIP_WIDE_RANGE, LondonBreakout, LondonParams,
)
from scan import slice_by_date  # noqa: E402
from trend import intraday_ema  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS = PROJECT_ROOT / "backtests" / "results"

YEARS = list(range(2020, 2027))
START, END = date(2020, 1, 1), date(2026, 8, 31)
ARMS = {"1x": 1.0, "2x": 2.0}
FILTERS = {"ON": True, "OFF": False}

MIN_POOLED_PASS_PROBABILITY = 0.25
MAX_BLOWUPS = 1
MIN_PROFITABLE_FOLDS = 4

LINE = "=" * 104


def size_trades(trades: pd.DataFrame, signals: pd.DataFrame,
                costs: CostModel) -> pd.DataFrame:
    """Price at one contract, then scale each row by its own size."""
    if trades.empty:
        return trades.assign(contracts=pd.Series(dtype=int))
    priced = price_trades(trades, MES, costs, 1)
    sizes = signals["contracts"].dropna()
    mapped = priced["entry_time"].map(sizes)
    if mapped.isna().any():
        raise RuntimeError("a trade has no contract size; signals and trades "
                           "disagree, which is a bug rather than a finding")
    n = mapped.astype(int)
    out = priced.copy()
    out["contracts"] = n
    for col in ("gross_pnl", "commission", "slippage_cost", "net_pnl"):
        out[col] = out[col] * n
    return out


def apply_daily_limit(trades: pd.DataFrame, bars: pd.DataFrame,
                      costs: CostModel, spec=MES) -> tuple[pd.DataFrame, int]:
    """Rule 5, applied per size group. Exact given one trade a day.

    ``spec`` must be the instrument actually being traded. It used to be
    hardcoded to MES, which silently marked MNQ positions at $5.00 a point
    instead of $2.00 - inflating open P&L by two and a half times and firing
    spurious daily-loss exits that removed would-be stops from the bracket.
    """
    if trades.empty:
        return trades, 0
    kept, halts = [], 0
    for size, group in trades.groupby("contracts"):
        k, h = enforce_daily_loss_limit(
            group.drop(columns=["contracts"]), bars, spec, costs, int(size),
            rules.DAILY_LOSS_LIMIT,
        )
        halts += len(h)
        if not k.empty:
            kept.append(k.assign(contracts=int(size)))
    if not kept:
        return trades.iloc[0:0], halts
    out = pd.concat(kept, ignore_index=True).sort_values("entry_time")
    return out.reset_index(drop=True), halts


def build_arm(bars, ema, rolls, early, target_multiple: float,
              use_filter: bool, costs: CostModel):
    strat = LondonBreakout(
        LondonParams(target_multiple=target_multiple, use_trend_filter=use_filter),
        trend_ema=ema if use_filter else None,
        roll_dates=rolls, early_close_dates=early,
    )
    signals = strat.generate_signals(bars)
    win_signals = slice_by_date(signals, START, END)
    win_bars = slice_by_date(bars, START, END)
    trades = size_trades(build_trades(win_signals, win_bars), win_signals, costs)
    trades, halts = apply_daily_limit(trades, win_bars, costs)
    return strat, trades, halts


def pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def arm_stats(trades: pd.DataFrame, diag: pd.DataFrame, paths: int) -> dict:
    if trades.empty:
        return {"trades": 0}
    m = compute_metrics(trades)
    daily = eval_sim.daily_pnl_from_trades(trades)
    sim = eval_sim.simulate(daily, paths=paths)
    blow = count_evaluation_blowups(trades)
    dd = trailing_drawdown_summary(equity_curve_by_day(trades))

    reasons = trades["exit_reason"].value_counts()
    targets = int(reasons.get("target", 0))
    stops = int(reasons.get("stop", 0))
    bracket = targets + stops

    wins = trades[trades["exit_reason"] == "target"]["net_pnl"]
    losses = trades[trades["exit_reason"] == "stop"]["net_pnl"]
    mean_win = float(wins.mean()) if len(wins) else float("nan")
    mean_loss = float(-losses.mean()) if len(losses) else float("nan")
    breakeven = (mean_loss / (mean_win + mean_loss)
                 if np.isfinite(mean_win) and np.isfinite(mean_loss) else float("nan"))

    entered = diag[diag["entered"] == True]  # noqa: E712
    stop_d = entered["stop_distance"].astype(float)
    tgt_d = stop_d * np.nan  # filled below from the trades' own levels
    # Barrier ratio from realised distances: a / (a + b) is the driftless value.
    rng = entered["range_pts"].astype(float)
    mult = float(trades.attrs.get("target_multiple", 1.0))
    tgt_d = rng * mult
    rw = (stop_d.mean() / (stop_d.mean() + tgt_d.mean())
          if len(stop_d) else float("nan"))

    return {
        "trades": len(trades),
        "net_pnl": m["net_pnl"],
        "sharpe": m["sharpe"],
        "profit_factor": m["profit_factor"],
        "max_drawdown": m["max_drawdown"],
        "eod_drawdown": float(dd["max_drawdown_from_peak"]),
        "win_rate": m["win_rate_pct"] / 100.0,
        "targets": targets, "stops": stops,
        "flattens": int(reasons.get("flatten_0925", 0)),
        "target_share": (targets / bracket) if bracket else float("nan"),
        "breakeven": breakeven,
        "random_walk": rw,
        "mean_win": mean_win, "mean_loss": mean_loss,
        "mean_day": float(daily.mean()), "sd_day": float(daily.std(ddof=1)),
        "pass_probability": sim.pass_probability,
        "ci_low": sim.ci_low, "ci_high": sim.ci_high,
        "blowup_probability": sim.blowup_probability,
        "expected_attempts": sim.expected_attempts,
        "blowups": int(blow["blowups"]), "passes": int(blow["passes"]),
        "reasons": trades.groupby("exit_reason")["net_pnl"].agg(
            n="count", total="sum", mean="mean"),
    }


def fold_frame(trades: pd.DataFrame, paths: int) -> pd.DataFrame:
    """Per-year fold statistics as a frame, one row per year in :data:`YEARS`.

    Column names deliberately match ``backtests/walkforward.py``'s output
    (``test_year``, ``test_net_pnl``, ...) so that ``bots/runners.py`` can
    render a London fold table with the same code it uses for ORB. A second
    column vocabulary would mean a second renderer, and the two would drift.

    A year with no trades is kept as a zero row rather than dropped: "2021 had
    no trades" and "2021 is missing from the table" read identically once
    rendered, and only one of them is true.

    This is the single source of the fold numbers - :func:`fold_table` renders
    it rather than recomputing, so the printed table and the saved CSV cannot
    disagree.
    """
    # utc=True, then back to ET. Two reasons, both load-bearing:
    #
    # A saved stream read back from CSV carries mixed offsets - -04:00 in EDT,
    # -05:00 in EST - and pandas refuses to parse that into one column without
    # `utc=True`. The live path passes tz-aware timestamps and never hit it.
    #
    # And the fold year must be the *ET* year. This strategy trades 19:00 to
    # 09:25, so a 31 December evening entry is 1 January in UTC and would be
    # counted in the following year's fold.
    t = trades.assign(
        year=pd.to_datetime(trades["entry_time"], utc=True)
              .dt.tz_convert(rules.ET).dt.year
    )
    rows = []
    for year in YEARS:
        g = t[t["year"] == year]
        if g.empty:
            rows.append({
                "test_year": year, "test_trades": 0, "test_net_pnl": 0.0,
                "test_sharpe": 0.0, "test_profit_factor": 0.0,
                "test_win_rate_pct": 0.0, "test_max_drawdown": 0.0,
                "test_pass_probability": 0.0, "low_confidence": True,
            })
            continue
        m = compute_metrics(g)
        daily = eval_sim.daily_pnl_from_trades(g)
        p = eval_sim.simulate(daily, paths=max(2000, paths // 5)).pass_probability
        rows.append({
            "test_year": year,
            "test_trades": len(g),
            "test_net_pnl": float(m["net_pnl"]),
            "test_sharpe": float(m["sharpe"]),
            "test_profit_factor": float(m["profit_factor"]),
            "test_win_rate_pct": float(m["win_rate_pct"]),
            "test_max_drawdown": float(m["max_drawdown"]),
            "test_pass_probability": float(p),
            "low_confidence": len(g) < 20,
        })
    return pd.DataFrame(rows)


def fold_table(trades: pd.DataFrame, label: str, paths: int,
               frame: pd.DataFrame | None = None) -> tuple[str, int]:
    """The printed per-fold table. Rendered from :func:`fold_frame`."""
    folds = fold_frame(trades, paths) if frame is None else frame
    out = [LINE, f"PER-FOLD - {label}", LINE,
           f"{'year':>6}{'trades':>8}{'net P&L':>12}{'Sharpe':>9}{'PF':>8}"
           f"{'hit%':>8}{'maxDD':>11}{'pass p':>9}  flag"]
    for _, r in folds.iterrows():
        if int(r["test_trades"]) == 0:
            out.append(f"{int(r['test_year']):>6}{0:>8}{'-':>12}")
            continue
        flag = "low-confidence" if r["low_confidence"] else ""
        out.append(
            f"{int(r['test_year']):>6}{int(r['test_trades']):>8}"
            f"{r['test_net_pnl']:>12,.0f}{r['test_sharpe']:>9.2f}"
            f"{r['test_profit_factor']:>8.2f}{r['test_win_rate_pct']:>8.1f}"
            f"{r['test_max_drawdown']:>11,.0f}"
            f"{pct(r['test_pass_probability']):>9}  {flag}"
        )
    profitable = int((folds["test_net_pnl"] > 0).sum())
    out += ["-" * 104,
            f"  total ${trades['net_pnl'].sum():>+,.2f}   "
            f"folds profitable {profitable} of {len(YEARS)}   "
            f"trades {len(trades):,}"]
    return "\n".join(out), profitable


def kill_block(stats: dict, profitable: int, label: str) -> str:
    c1 = stats.get("pass_probability", 0.0) >= MIN_POOLED_PASS_PROBABILITY
    c2 = stats.get("blowups", 99) <= MAX_BLOWUPS
    c3 = profitable >= MIN_PROFITABLE_FOLDS and stats.get("net_pnl", -1) > 0
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    return "\n".join([
        f"KILL CRITERIA - {label}",
        f"  1  pass probability >= 25%          "
        f"{pct(stats.get('pass_probability')):>10}   {mark(c1)}",
        f"  2  evaluations blown <= 1           {stats.get('blowups','-'):>10}"
        f"   {mark(c2)}",
        f"  3  >=4 of 7 folds AND P&L > 0       "
        f"{profitable} of 7, ${stats.get('net_pnl', 0):,.0f}".ljust(48)
        + f"   {mark(c3)}",
        f"  VERDICT: {'SURVIVES' if (c1 and c2 and c3) else 'REJECTED'}",
    ])


def rebuild_folds(tag: str, paths: int) -> int:
    """Regenerate the per-fold CSVs from already-saved trade streams.

    The fold table depends only on the trades, so it does not need the signal
    generation the full run does. This exists because the streams were saved
    before the fold CSVs were, and re-running the strategy for twenty minutes
    to recover a table already implied by the data on disk is waste.
    """
    written = 0
    for arm in ARMS:
        for filt in FILTERS:
            stream = RESULTS / f"london_{arm}_{filt.lower()}_{tag}.csv"
            if not stream.exists():
                print(f"  skip {stream.name} - not on disk")
                continue
            trades = pd.read_csv(stream)
            out = RESULTS / f"london_{arm}_{filt.lower()}_folds_{tag}.csv"
            fold_frame(trades, paths).to_csv(out, index=False)
            print(f"  wrote {out.name} from {len(trades):,} trades")
            written += 1
    if not written:
        print(f"No saved streams for {tag}; run without --folds-only first.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slippage-ticks", type=float, default=2.0,
                    help="2 is entry 6's base case; 1 is the optimistic run")
    ap.add_argument("--commission", type=float, default=rules.COMMISSION_PER_SIDE)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--folds-only", action="store_true",
                    help="rebuild the per-fold CSVs from the saved trade "
                         "streams and exit; seconds rather than the ~20 "
                         "minutes a full run takes, because the fold table is "
                         "a function of the trades alone")
    args = ap.parse_args()

    if args.folds_only:
        return rebuild_folds(f"slip{args.slippage_ticks:g}", args.paths)

    costs = CostModel(commission_per_side=args.commission,
                      slippage_ticks=args.slippage_ticks)
    tag = f"slip{args.slippage_ticks:g}"
    base = " (BASE CASE)" if args.slippage_ticks == 2 else " (optimistic sensitivity)"

    print("Loading bars ...", flush=True)
    bars = loader.load_bars(str(PARQUET))
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    ema = intraday_ema(bars, 5, 200)

    print(LINE)
    print(f"ENTRY 6 - London breakout of the overnight range{base}")
    print(f"  range 19:00-02:55 ET, break 03:00-05:00, flat 09:25")
    print(f"  {costs.slippage_ticks:g} tick slippage/side, "
          f"${costs.commission_per_side:.2f} commission/side")
    print(f"  risk-based sizing, $400 daily loss limit, no drawdown halt")
    print(LINE, flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    results, folds, streams, diags = {}, {}, {}, {}

    for arm, mult in ARMS.items():
        for filt, use in FILTERS.items():
            key = (arm, filt)
            print(f"Running {arm} target, filter {filt} ...", flush=True)
            strat, trades, halts = build_arm(bars, ema, rolls, early, mult,
                                             use, costs)
            trades.attrs["target_multiple"] = mult
            streams[key] = trades
            diags[key] = strat.diagnostics
            if not trades.empty:
                trades.to_csv(
                    RESULTS / f"london_{arm}_{filt.lower()}_{tag}.csv", index=False)
            results[key] = arm_stats(trades, strat.diagnostics, args.paths)
            results[key]["halts"] = halts

    for arm in ARMS:
        for filt in FILTERS:
            key = (arm, filt)
            label = f"target {arm}, filter {filt}, {tag}"
            print()
            frame = fold_frame(streams[key], args.paths)
            frame.to_csv(
                RESULTS / f"london_{arm}_{filt.lower()}_folds_{tag}.csv",
                index=False)
            table, profitable = fold_table(streams[key], label, args.paths,
                                           frame=frame)
            folds[key] = profitable
            print(table)

    print()
    print(LINE)
    print("POOLED OUT-OF-SAMPLE 2020-2026")
    print(LINE)
    hdr = f"{'':<32}" + "".join(f"{a+'/'+f:>17}" for a in ARMS for f in FILTERS)
    print(hdr)
    print("-" * 104)
    keys = [(a, f) for a in ARMS for f in FILTERS]

    def row(name, fmt):
        print(f"  {name:<30}" + "".join(f"{fmt(results[k]):>17}" for k in keys))

    row("Trades", lambda r: f"{r.get('trades',0):,}")
    row("Net P&L", lambda r: f"${r.get('net_pnl',0):,.0f}")
    row("Sharpe", lambda r: f"{r.get('sharpe',float('nan')):.2f}")
    row("Profit factor", lambda r: f"{r.get('profit_factor',float('nan')):.3f}")
    row("Max drawdown", lambda r: f"${r.get('max_drawdown',0):,.0f}")
    row("Win rate (all exits)", lambda r: pct(r.get("win_rate")))
    row("Targets / stops", lambda r: f"{r.get('targets',0)} / {r.get('stops',0)}")
    row("09:25 flattens", lambda r: f"{r.get('flattens',0)}")
    print("-" * 104)
    row("Target share of bracket", lambda r: pct(r.get("target_share")))
    row("  break-even (realised)", lambda r: pct(r.get("breakeven")))
    row("  random-walk value", lambda r: pct(r.get("random_walk")))
    print("-" * 104)
    row("Evaluations blown", lambda r: f"{r.get('blowups','-')}")
    row("Pass probability", lambda r: pct(r.get("pass_probability")))
    row("Expected attempts", lambda r: f"{r.get('expected_attempts',0):.1f}")
    row("Mean day", lambda r: f"${r.get('mean_day',0):,.2f}")
    row("Daily-loss halts", lambda r: f"{r.get('halts',0)}")

    print()
    print(LINE)
    print("P&L BY EXIT REASON")
    print(LINE)
    for k in keys:
        r = results[k]
        if not r.get("trades"):
            continue
        print(f"\n  target {k[0]}, filter {k[1]}")
        print("    " + r["reasons"].round(2).to_string().replace("\n", "\n    "))

    print()
    print(LINE)
    print("SIZING, OVERSHOOT AND REALISED RISK  (target 1x, filter OFF)")
    print(LINE)
    d = diags[("1x", "OFF")]
    entered = d[d["entered"] == True]  # noqa: E712
    entered = entered[[pd.Timestamp(i).year in YEARS for i in entered.index]]
    sizes = entered["contracts"].astype(int).value_counts().sort_index()
    for n, c in sizes.items():
        print(f"  {n} contract(s): {c:>5,}  ({100*c/len(entered):.1f}%)")
    over = entered["overshoot"].astype(float)
    rng = entered["range_pts"].astype(float)
    risk = entered["risk_dollars"].astype(float)
    print(f"  overshoot (points):  median {over.median():.2f}  mean {over.mean():.2f}"
          f"  p90 {over.quantile(0.9):.2f}  max {over.max():.2f}")
    print(f"  overshoot as % of range: median "
          f"{100*(over/rng).median():.1f}%  mean {100*(over/rng).mean():.1f}%")
    print(f"  realised risk ($):   median {risk.median():,.0f}  mean {risk.mean():,.0f}"
          f"  max {risk.max():,.0f}   (rule sizes for $200)")
    print(f"  trades risking over $200: {int((risk > 200).sum()):,} of {len(risk):,}"
          f"  ({100*(risk>200).mean():.1f}%)")
    print(f"  trades risking over $400 (the daily limit): "
          f"{int((risk > rules.DAILY_LOSS_LIMIT).sum()):,}")

    print()
    print(LINE)
    print("SKIPPED SESSIONS  (target 1x, filter OFF, 2020-2026)")
    print(LINE)
    oos = d[[pd.Timestamp(i).year in YEARS for i in d.index]]
    counts = oos["skipped_reason"].value_counts(dropna=False)
    for reason, c in counts.items():
        name = "traded" if pd.isna(reason) else reason
        print(f"  {name:<28}{c:>6,}")
    wide = oos[oos["skipped_reason"] == SKIP_WIDE_RANGE]
    if len(wide):
        h = wide["range_pts"].astype(float)
        print(f"  range-cap skips: median {h.median():.2f} pts, max {h.max():.2f}")

    print()
    print(LINE)
    print("TREND FILTER - ON vs OFF, and the long-only diagnostic")
    print(LINE)
    for arm in ARMS:
        on, off = streams[(arm, "ON")], streams[(arm, "OFF")]
        if on.empty or off.empty:
            continue
        e_on, e_off = on["net_pnl"].mean(), off["net_pnl"].mean()
        d_e = e_on - e_off
        se = float(np.sqrt(on["net_pnl"].var(ddof=1) / len(on)
                           + off["net_pnl"].var(ddof=1) / len(off)))
        t = d_e / se if se else float("nan")
        print(f"\n  target {arm}")
        print(f"    ON  {len(on):>5,} trades, mean ${e_on:>8,.2f}, "
              f"{100*(on['direction']=='long').mean():.1f}% long")
        print(f"    OFF {len(off):>5,} trades, mean ${e_off:>8,.2f}, "
              f"{100*(off['direction']=='long').mean():.1f}% long")
        print(f"    dE = ${d_e:,.2f} per trade   (SE ${se:,.2f}, t = {t:+.2f})")
        onl = on[on["direction"] == "long"]["net_pnl"]
        offl = off[off["direction"] == "long"]["net_pnl"]
        if len(onl) and len(offl):
            dl = onl.mean() - offl.mean()
            sel = float(np.sqrt(onl.var(ddof=1)/len(onl) + offl.var(ddof=1)/len(offl)))
            print(f"    long-only: ON ${onl.mean():,.2f} (n {len(onl):,})  vs  "
                  f"OFF ${offl.mean():,.2f} (n {len(offl):,})")
            print(f"    long-only dE = ${dl:,.2f}   (SE ${sel:,.2f})")

    print()
    print(LINE)
    print("THE 2026 FOLD ON ITS OWN (partial year, ends 2026-08-31)")
    print(LINE)
    for k in keys:
        s = streams[k]
        if s.empty:
            continue
        g = s[pd.to_datetime(s["entry_time"]).dt.year == 2026]
        if g.empty:
            print(f"  {k[0]}/{k[1]}: no trades")
            continue
        m = compute_metrics(g)
        print(f"  {k[0]}/{k[1]}: {len(g)} trades, ${m['net_pnl']:,.0f}, "
              f"Sharpe {m['sharpe']:.2f}, PF {m['profit_factor']:.2f}"
              f"{'  [low-confidence]' if len(g) < 20 else ''}")

    print()
    print(LINE)
    print("KILL CRITERIA - each arm judged independently, filter OFF is the arm")
    print(LINE)
    for arm in ARMS:
        print()
        print(kill_block(results[(arm, "OFF")], folds[(arm, "OFF")],
                         f"target {arm}, filter OFF"))
        print()
        print(kill_block(results[(arm, "ON")], folds[(arm, "ON")],
                         f"target {arm}, filter ON"))

    print()
    print(LINE)
    print("RULE 6 MEASURABILITY")
    print(LINE)
    for k in keys:
        dd_ = diags[k]
        same = int(dd_["same_bar_entry_exit"].fillna(False).sum())
        print(f"  {k[0]}/{k[1]}: {same} trades entered and exited inside one "
              f"1-minute bar")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
