"""ORB-2 evaluation: every test entry 4 pre-registers, in one run.

Entry 4 in ``research/hypotheses.md`` is frozen at commit ``17338c5``. Nothing
here chooses a parameter - the configuration is fixed, so each fold is simply
its out-of-sample year and the walk-forward machinery is used for its fold
bookkeeping rather than for selection. The single pre-registered variation is
the filter ON/OFF arm.

    python backtests/run_orb2.py
    python backtests/run_orb2.py --slippage-ticks 2

Outputs land in ``backtests/results/orb2_*``.
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
import eval_sim  # noqa: E402
import walkforward as wf  # noqa: E402
from engine import (  # noqa: E402
    MES, CostModel, build_trades, count_evaluation_blowups,
    enforce_daily_loss_limit, equity_curve_by_day, price_trades,
    trailing_drawdown_summary,
)
from orb2 import ORB2, ORB2Params, SKIP_NO_FILL, SKIP_WIDE_RANGE  # noqa: E402
from scan import slice_by_date  # noqa: E402
from trend import largest_roll_gap, trend_filter  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

ARMS = {"ON": True, "OFF": False}
SIZES = (4, 1)
OOS_YEARS = list(range(2020, 2027))

#: Entry 4's kill criteria, keyed on the 4-contract run.
KILL_SIZE = 4
MIN_POOLED_PASS_PROBABILITY = 0.25
MAX_BLOWUPS = 1
MIN_PROFITABLE_FOLDS = 4
MIN_FILTER_EDGE_PER_CONTRACT = 5.00

LINE = "=" * 96


# ---------------------------------------------------------------------------
# Wiring ORB-2 into the walk-forward machinery
# ---------------------------------------------------------------------------


def make_hooks(trend_ema: pd.Series):
    """factory / describe / rebuild / generate for ORB-2."""

    def factory(params, roll_dates, early_closes):
        return ORB2(params, trend_ema=trend_ema if params.use_trend_filter else None,
                    roll_dates=roll_dates, early_close_dates=early_closes)

    def describe(params) -> dict:
        return {"use_trend_filter": params.use_trend_filter}

    def rebuild(row) -> ORB2Params:
        return ORB2Params(use_trend_filter=bool(row["use_trend_filter"]))

    def generate(strat, bars):
        return strat.generate_signals(bars)

    return factory, describe, rebuild, generate


def run_arm(bars, roll_dates, early_closes, costs, trend_ema,
            use_filter: bool, contracts: int):
    """Fold table and stitched out-of-sample stream for one arm at one size."""
    factory, describe, rebuild, generate = make_hooks(trend_ema)
    grid = [ORB2Params(use_trend_filter=use_filter)]

    summary, pairs = wf.run_walkforward(
        bars, roll_dates, early_closes, costs,
        grid=grid, factory=factory, describe=describe,
        min_train_trades=30, verbose=False, contracts=contracts,
        selection="pass_probability", generate=generate,
    )
    stream = wf.oos_trade_stream(
        bars, roll_dates, early_closes, costs, summary,
        factory=factory, rebuild=rebuild, contracts=contracts, generate=generate,
    )
    return summary, pairs, stream


def diagnostics_for(bars, roll_dates, early_closes, trend_ema, use_filter: bool):
    """Per-session diagnostics over the whole history, for the skip reports."""
    strat = ORB2(
        ORB2Params(use_trend_filter=use_filter),
        trend_ema=trend_ema if use_filter else None,
        roll_dates=roll_dates, early_close_dates=early_closes,
    )
    strat.generate_signals(bars)
    diag = strat.diagnostics.copy()
    diag["year"] = [d.year for d in diag.index]
    return diag


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_pct(x) -> str:
    return "  n/a " if x is None or not np.isfinite(x) else f"{100 * x:5.2f}%"


def fold_table(summary: pd.DataFrame, stream: pd.DataFrame, label: str) -> str:
    """Per-fold results. The hit rate comes from the out-of-sample trades
    themselves rather than the walk-forward summary, which does not carry one -
    and entry 4 turns on the hit rate against its 39.29% break-even."""
    hits = {}
    if not stream.empty:
        s = stream.assign(year=pd.to_datetime(stream["entry_time"]).dt.year)
        hits = (s["net_pnl"] > 0).groupby(s["year"]).mean().to_dict()

    out = [LINE, f"FOLD TABLE - {label}", LINE,
           f"{'year':>6}{'trades':>8}{'net P&L':>12}{'Sharpe':>9}{'PF':>7}"
           f"{'hit%':>8}{'maxDD':>10}{'pass p':>9}  flag"]
    for _, r in summary.iterrows():
        flag = "low-confidence" if r["low_confidence"] else ""
        hit = hits.get(int(r["test_year"]))
        hit_s = f"{100 * hit:>8.1f}" if hit is not None else f"{'-':>8}"
        out.append(
            f"{int(r['test_year']):>6}{int(r['test_trades']):>8}"
            f"{r['test_net_pnl']:>12,.0f}{r['test_sharpe']:>9.2f}"
            f"{r['test_profit_factor']:>7.2f}{hit_s}"
            f"{r['test_max_drawdown']:>10,.0f}"
            f"{fmt_pct(r.get('test_pass_probability')):>9}  {flag}"
        )
    total = summary["test_net_pnl"].sum()
    profitable = int((summary["test_net_pnl"] > 0).sum())
    out += [
        "-" * 96,
        f"  Total net P&L                   ${total:>12,.2f}",
        f"  Folds profitable                {profitable} of {len(summary)}",
        f"  Total trades                    {int(summary['test_trades'].sum()):,}",
        f"  Median fold Sharpe              {summary['test_sharpe'].median():+.3f}",
    ]
    return "\n".join(out)


def pooled_block(stream: pd.DataFrame, contracts: int, label: str,
                 paths: int = 20_000) -> tuple[str, dict]:
    """Pooled out-of-sample evaluation figures for one arm at one size."""
    out = [LINE, f"POOLED OUT-OF-SAMPLE 2020-2026 - {label}", LINE]
    if stream.empty:
        return "\n".join(out + ["  No trades."]), {}

    daily = eval_sim.daily_pnl_from_trades(stream)
    sim = eval_sim.simulate(daily, paths=paths)
    equity = equity_curve_by_day(stream)
    dd = trailing_drawdown_summary(equity)
    blowups = count_evaluation_blowups(stream)

    per_contract = stream["net_pnl"].mean() / contracts
    wins = stream[stream["net_pnl"] > 0]
    stats = {
        "trades": len(stream),
        "net_pnl": float(stream["net_pnl"].sum()),
        "mean_per_trade_per_contract": float(per_contract),
        "hit_rate": float(len(wins) / len(stream)),
        "mean_day": float(daily.mean()),
        "sd_day": float(daily.std(ddof=1)),
        "pass_probability": sim.pass_probability,
        "payout_probability": sim.payout_probability,
        "blowup_probability": sim.blowup_probability,
        "timeout_probability": sim.timeout_probability,
        "expected_attempts": sim.expected_attempts,
        "blowups": int(blowups["blowups"]),
        "passes": int(blowups["passes"]),
        "worst_drawdown": float(dd["max_drawdown_from_peak"]),
        "sd_error": float(stream["net_pnl"].std(ddof=1) / contracts / np.sqrt(len(stream))),
    }
    out += [
        f"  Trades                          {stats['trades']:,}",
        f"  Net P&L                         ${stats['net_pnl']:>12,.2f}",
        f"  Mean per trade, per contract    ${stats['mean_per_trade_per_contract']:>12,.2f}",
        f"  Hit rate                        {100 * stats['hit_rate']:.2f}%"
        f"   (break-even 39.29%)",
        f"  Mean day / sd day               ${stats['mean_day']:,.2f} / "
        f"${stats['sd_day']:,.2f}",
        f"  Worst drawdown from peak        ${stats['worst_drawdown']:>12,.2f}",
        f"  Evaluations blown / passed      {stats['blowups']} / {stats['passes']}",
        f"  Pass probability                {fmt_pct(sim.pass_probability)}"
        f"   (95% CI {fmt_pct(sim.ci_low)} - {fmt_pct(sim.ci_high)})",
        f"  Payout probability              {fmt_pct(sim.payout_probability)}"
        f"   (touched ${rules.PAYOUT_BALANCE:,.0f} before the trail)",
        f"  Blow-up / timeout               {fmt_pct(sim.blowup_probability)} / "
        f"{fmt_pct(sim.timeout_probability)}",
        f"  Expected attempts to pass       {sim.expected_attempts:.2f}",
    ]
    return "\n".join(out), stats


def exit_reason_table(streams: dict, label: str) -> str:
    out = [LINE, f"EXIT REASONS BY YEAR - {label}", LINE]
    for arm, stream in streams.items():
        if stream.empty:
            continue
        s = stream.assign(year=pd.to_datetime(stream["entry_time"]).dt.year)
        pivot = s.pivot_table(index="year", columns="exit_reason",
                              values="net_pnl", aggfunc="count").fillna(0).astype(int)
        out.append(f"\n  Filter {arm}")
        out.append("    " + pivot.to_string().replace("\n", "\n    "))
        shares = pivot.sum() / pivot.sum().sum()
        out.append("    share: " + "  ".join(
            f"{k} {100 * v:.1f}%" for k, v in shares.items()))
    return "\n".join(out)


def skipped_day_table(diags: dict) -> str:
    out = [LINE, "SKIPPED SESSIONS BY YEAR (out-of-sample 2020-2026)", LINE]
    for arm, diag in diags.items():
        oos = diag[diag["year"].isin(OOS_YEARS)]
        counts = (
            oos.groupby(["year", "skipped_reason"]).size().unstack(fill_value=0)
        )
        entered = oos.groupby("year")["entered"].sum().astype(int)
        counts.insert(0, "entered", entered)
        out.append(f"\n  Filter {arm}")
        out.append("    " + counts.to_string().replace("\n", "\n    "))

        wide = oos[oos["skipped_reason"] == SKIP_WIDE_RANGE]
        if len(wide):
            heights = wide["range_height"].astype(float)
            out.append(
                f"    range-ceiling skips: {len(wide):,} sessions, "
                f"opening range median {heights.median():.2f} pts, "
                f"mean {heights.mean():.2f}, max {heights.max():.2f}"
            )
        traded = oos[oos["entered"] == True]  # noqa: E712
        if len(traded):
            h = traded["range_height"].astype(float)
            out.append(
                f"    traded sessions:     {len(traded):,}, "
                f"opening range median {h.median():.2f} pts, mean {h.mean():.2f}"
            )
    return "\n".join(out)


def filter_diagnostic(streams: dict, contracts: int) -> str:
    """The A/B that entry 4 exists to run, plus the long-only confound check."""
    on, off = streams["ON"], streams["OFF"]
    out = [LINE, "TREND-FILTER DIAGNOSTIC (pooled out-of-sample, per contract)", LINE]
    if on.empty or off.empty:
        return "\n".join(out + ["  One arm has no trades; nothing to compare."])

    def per_contract(s):
        return s["net_pnl"] / contracts

    e_on, e_off = per_contract(on).mean(), per_contract(off).mean()
    d = e_on - e_off
    se = float(np.sqrt(
        per_contract(on).var(ddof=1) / len(on) + per_contract(off).var(ddof=1) / len(off)
    ))
    t = d / se if se > 0 else float("nan")

    hit_on = float((on["net_pnl"] > 0).mean())
    hit_off = float((off["net_pnl"] > 0).mean())

    out += [
        f"{'':<28}{'ON':>14}{'OFF':>14}",
        f"  {'trades':<26}{len(on):>14,}{len(off):>14,}",
        f"  {'mean per trade ($/contract)':<26}{e_on:>14,.2f}{e_off:>14,.2f}",
        f"  {'hit rate':<26}{100 * hit_on:>13.2f}%{100 * hit_off:>13.2f}%",
        "-" * 96,
        f"  dE = E_on - E_off               ${d:>10,.2f} per contract"
        f"   (SE ${se:,.2f}, t = {t:+.2f})",
        f"  Hit-rate difference             {100 * (hit_on - hit_off):+.2f} points",
        f"  Threshold to survive            ${MIN_FILTER_EDGE_PER_CONTRACT:,.2f}"
        f"  (one round turn)",
        f"  FILTER CLAIM                    "
        f"{'SURVIVES' if d > MIN_FILTER_EDGE_PER_CONTRACT else 'REJECTED'}",
    ]

    # The confound: MES rose over the period, so the filter is mostly a
    # long-only switch. Comparing longs with longs removes the drift.
    out += ["", "  Long/short split and the long-only diagnostic:"]
    for arm, stream in (("ON", on), ("OFF", off)):
        longs = int((stream["direction"] == "long").sum())
        out.append(
            f"    {arm:<4} long {longs:,} / short {len(stream) - longs:,}"
            f"   ({100 * longs / len(stream):.1f}% long)"
        )
    on_long = per_contract(on[on["direction"] == "long"])
    off_long = per_contract(off[off["direction"] == "long"])
    if len(on_long) and len(off_long):
        d_long = on_long.mean() - off_long.mean()
        se_long = float(np.sqrt(
            on_long.var(ddof=1) / len(on_long) + off_long.var(ddof=1) / len(off_long)
        ))
        out += [
            f"    long-only  E_on ${on_long.mean():,.2f}  vs  E_off "
            f"${off_long.mean():,.2f}   (n {len(on_long):,} / {len(off_long):,})",
            f"    long-only dE                  ${d_long:>10,.2f} per contract"
            f"   (SE ${se_long:,.2f})",
            f"    Reading: {'advantage survives like-for-like' if d_long > MIN_FILTER_EDGE_PER_CONTRACT else 'advantage does NOT survive like-for-like - the filter is acting as a long-only switch'}",
        ]
    return "\n".join(out)


def same_bar_report(streams: dict, diags: dict) -> str:
    out = [LINE, "RULE 6 MEASURABILITY - trades entering and exiting in one bar", LINE]
    for arm, stream in streams.items():
        if stream.empty:
            continue
        same = int((stream["entry_time"] == stream["exit_time"]).sum())
        out.append(
            f"  Filter {arm:<4} {same:,} of {len(stream):,} trades "
            f"({100 * same / len(stream):.2f}%) cannot be checked against the "
            f"30-second floor at 1-minute resolution"
        )
    for arm, diag in diags.items():
        oos = diag[diag["year"].isin(OOS_YEARS)]
        amb = int(oos["ambiguous_both_stops"].fillna(False).sum())
        out.append(f"  Filter {arm:<4} {amb:,} sessions had both entry stops "
                   f"reached inside one bar (resolved pessimistically)")
    return "\n".join(out)


def kill_criteria(summary: pd.DataFrame, stats: dict, filter_survives: bool) -> str:
    total = summary["test_net_pnl"].sum()
    profitable = int((summary["test_net_pnl"] > 0).sum())

    c1 = stats.get("pass_probability", 0.0) >= MIN_POOLED_PASS_PROBABILITY
    c2 = stats.get("blowups", 99) <= MAX_BLOWUPS
    c3 = profitable >= MIN_PROFITABLE_FOLDS and total > 0

    def mark(ok):
        return "PASS" if ok else "FAIL"

    out = [LINE, f"KILL CRITERIA (entry 4, measured at {KILL_SIZE} contracts)", LINE,
           f"  1  Pooled OOS pass probability >= 25%     "
           f"{fmt_pct(stats.get('pass_probability')):>8}   {mark(c1)}",
           f"  2  Evaluations blown <= 1                 "
           f"{stats.get('blowups', 'n/a'):>8}   {mark(c2)}",
           f"  3  >= 4 folds profitable AND P&L > 0      "
           f"{profitable} of {len(summary)}, ${total:,.0f}".ljust(52) + f"   {mark(c3)}",
           "-" * 96,
           f"  VERDICT: {'SURVIVES' if (c1 and c2 and c3) else 'REJECTED'}"
           f"   (any one failure kills the hypothesis)",
           "",
           f"  4  Filter claim (non-fatal)               "
           f"{'SURVIVES' if filter_survives else 'REJECTED':>8}",
           ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=1.25)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild the report from saved CSVs, no backtest")
    args = ap.parse_args()

    costs = CostModel(commission_per_side=args.commission,
                      slippage_ticks=args.slippage_ticks)
    tag = f"slip{args.slippage_ticks:g}"

    print("Loading bars ...", flush=True)
    bars = loader.load_bars(args.parquet)
    roll_dates = loader.detect_roll_dates(bars)
    early_closes = loader.detect_early_close_dates(bars)
    trend_ema = trend_filter(bars)

    gap = largest_roll_gap(bars, roll_dates)
    print(LINE)
    print("ORB-2  |  fixed 10/18 bracket, 09:30-09:45 range, 50-day EMA filter")
    print(f"        {costs.slippage_ticks:g} tick slippage/side, "
          f"${costs.commission_per_side:.2f} commission/side")
    print(f"        {len(roll_dates)} roll dates, {len(early_closes)} early closes")
    print(f"        contract-change steps: {gap['n_rolls']}, median "
          f"{gap['median_roll']:.2f} pts vs {gap['median_ordinary']:.2f} pts on an "
          f"ordinary session")
    print(f"        largest such step {gap['points']:.2f} pts on {gap['date']} "
          f"- an upper bound containing that day's real move, not the roll alone")
    print(LINE, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def paths_for(arm, contracts):
        base = f"{arm.lower()}_{contracts}c_{tag}"
        return (RESULTS_DIR / f"orb2_folds_{base}.csv",
                RESULTS_DIR / f"orb2_oos_{base}.csv")

    diag_path = RESULTS_DIR / f"orb2_diagnostics_{tag}.csv"
    if args.from_cache and diag_path.exists():
        cached = pd.read_csv(diag_path)
        cached["session_date"] = pd.to_datetime(cached["session_date"])
        diags = {arm: g.set_index("session_date").assign(
                     year=lambda d: d.index.year)
                 for arm, g in cached.groupby("arm")}
    else:
        diags = {arm: diagnostics_for(bars, roll_dates, early_closes, trend_ema, flag)
                 for arm, flag in ARMS.items()}
        pd.concat([d.assign(arm=a) for a, d in diags.items()]).to_csv(
            diag_path, index_label="session_date")

    results: dict[tuple[str, int], dict] = {}
    streams: dict[int, dict[str, pd.DataFrame]] = {c: {} for c in SIZES}
    summaries: dict[tuple[str, int], pd.DataFrame] = {}

    for contracts in SIZES:
        for arm, flag in ARMS.items():
            fold_csv, oos_csv = paths_for(arm, contracts)
            if args.from_cache and fold_csv.exists():
                print(f"Reusing cached filter {arm} at {contracts} contract(s)",
                      flush=True)
                summary = pd.read_csv(fold_csv)
                stream = pd.read_csv(oos_csv) if oos_csv.exists() else pd.DataFrame()
                # The stored timestamps span EST and EDT, so pandas refuses to
                # infer one offset. Parse as UTC, then put them back in ET.
                for col in ("entry_time", "exit_time"):
                    if col in stream.columns:
                        stream[col] = pd.to_datetime(
                            stream[col], utc=True
                        ).dt.tz_convert("America/New_York")
            else:
                print(f"Running filter {arm} at {contracts} contract(s) ...", flush=True)
                summary, pairs, stream = run_arm(
                    bars, roll_dates, early_closes, costs, trend_ema, flag, contracts
                )
                summary.to_csv(fold_csv, index=False)
                if not stream.empty:
                    stream.to_csv(oos_csv, index=False)
            summaries[(arm, contracts)] = summary
            streams[contracts][arm] = stream

    print()
    for contracts in SIZES:
        for arm in ARMS:
            label = f"filter {arm}, {contracts} contract(s), {tag}"
            print(fold_table(summaries[(arm, contracts)],
                             streams[contracts][arm], label))
            print()
            block, stats = pooled_block(streams[contracts][arm], contracts, label,
                                        paths=args.paths)
            results[(arm, contracts)] = stats
            print(block)
            print()

    # Entry 4 asks for 2026 separately as well as inside the seven.
    print(LINE)
    print("THE 2026 FOLD ON ITS OWN (partial year, ends 2026-08-31)")
    print(LINE)
    for (arm, contracts), summary in summaries.items():
        row = summary[summary["test_year"] == 2026]
        if len(row):
            r = row.iloc[0]
            print(f"  filter {arm}, {contracts}c: {int(r['test_trades'])} trades, "
                  f"${r['test_net_pnl']:,.0f}, Sharpe {r['test_sharpe']:.2f}, "
                  f"PF {r['test_profit_factor']:.2f}"
                  f"{'  [low-confidence]' if r['low_confidence'] else ''}")
    print()

    print(skipped_day_table(diags))
    print()
    print(exit_reason_table(streams[KILL_SIZE], f"{KILL_SIZE} contracts"))
    print()
    print(same_bar_report(streams[KILL_SIZE], diags))
    print()

    filter_text = filter_diagnostic(streams[KILL_SIZE], KILL_SIZE)
    print(filter_text)
    print()

    on, off = streams[KILL_SIZE]["ON"], streams[KILL_SIZE]["OFF"]
    survives = False
    if not on.empty and not off.empty:
        d = (on["net_pnl"].mean() - off["net_pnl"].mean()) / KILL_SIZE
        survives = d > MIN_FILTER_EDGE_PER_CONTRACT

    print(kill_criteria(summaries[("ON", KILL_SIZE)],
                        results.get(("ON", KILL_SIZE), {}), survives))
    print()

    # Ruling 3's assertion, checked on the real streams rather than only in a
    # unit test: the 1-contract series must be the 4-contract series over four.
    print(LINE)
    print("SIZE LINEARITY CHECK")
    print(LINE)
    for arm in ARMS:
        a, b = streams[4][arm], streams[1][arm]
        if a.empty or b.empty:
            continue
        ok = np.allclose(a["net_pnl"].to_numpy(), b["net_pnl"].to_numpy() * 4)
        print(f"  filter {arm:<4} 4c == 1c x 4, trade for trade: "
              f"{'OK' if ok else 'MISMATCH - this is a bug, not a finding'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
