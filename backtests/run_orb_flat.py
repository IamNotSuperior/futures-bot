"""Entry 5: ORB flat by 10:30. Every test the entry pre-registers, in one run.

Entry 5 in ``research/hypotheses.md`` is frozen at commit ``c7c6f01``. Nothing
here is chosen after the fact - there is one fixed configuration, transcribed
from ``orb_flat_1030.pine``, and two reporting windows.

The verdict turns on the seven-year figures only. The two-year window is
reported alongside as context and is pre-registered as *not* evidence: entry 1's
ORB returned +$2,250 over that same span and -$6,073 across seven folds.

    python backtests/run_orb_flat.py
    python backtests/run_orb_flat.py --slippage-ticks 2
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
from engine import (  # noqa: E402
    MES, CostModel, apply_internal_guards, build_trades,
    count_evaluation_blowups, equity_curve_by_day, price_trades,
    trailing_drawdown_summary,
)
from metrics import compute_metrics  # noqa: E402
from orb2 import ORB2, ORB2Params, SKIP_WIDE_RANGE  # noqa: E402
from scan import slice_by_date  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

CONTRACTS = 4
DATA_END = date(2026, 8, 31)
SEVEN_YEAR = (date(2020, 1, 1), DATA_END)
TWO_YEAR = (date(2024, 9, 1), DATA_END)
YEARS = list(range(2020, 2027))

#: Entry 5's kill criteria, all on the seven-year window at 4 contracts.
MIN_POOLED_PASS_PROBABILITY = 0.25
MAX_BLOWUPS = 1
MIN_PROFITABLE_YEARS = 4

BREAK_EVEN = 55.0 / 140.0  # target/stop arithmetic, 39.29%
LINE = "=" * 98


def entry5_params() -> ORB2Params:
    """The frozen configuration. Every value is fixed by entry 5."""
    return ORB2Params(
        opening_range_end=time(9, 45),
        entry_offset=1.0,
        max_range_points=10.0,
        stop_points=10.0,
        target_points=18.0,
        entry_cancel_time=time(10, 30),
        flatten_time=time(10, 30),
        use_trend_filter=False,
        flatten_at_next_open=True,
    )


# ---------------------------------------------------------------------------
# Running
#
# Both internal guards - rule 5's daily loss limit and rule 5b's $1,500
# trailing halt - are `engine.apply_internal_guards`, the same call the
# generated-strategy runner makes. The halt lived in this file until the
# second caller arrived; `tests/test_guard_parity.py` keeps the two runners
# on one basis.
# ---------------------------------------------------------------------------


def build_stream(bars, signals, start: date, end: date, costs: CostModel,
                 dd_halt: bool = True):
    """Trades for one window, with the internal guards applied in order.

    ``dd_halt=False`` leaves the trailing-drawdown halt off. That is not the
    frozen specification - entry 5 requires the halt - but it is the basis
    entries 1 and 4 were measured on, and without it their blow-up counts and
    year-by-year P&L are not comparable with anything here. Both are reported.
    """
    win_signals = slice_by_date(signals, start, end)
    win_bars = slice_by_date(bars, start, end)
    trades = price_trades(build_trades(win_signals, win_bars), MES, costs, CONTRACTS)
    return apply_internal_guards(
        trades, win_bars, MES, costs, CONTRACTS, trailing_halt=dd_halt
    )


def window_stats(trades: pd.DataFrame, label: str, paths: int) -> dict:
    if trades.empty:
        return {"label": label, "trades": 0}
    m = compute_metrics(trades)
    daily = eval_sim.daily_pnl_from_trades(trades)
    sim = eval_sim.simulate(daily, paths=paths)
    dd = trailing_drawdown_summary(equity_curve_by_day(trades))
    blow = count_evaluation_blowups(trades)

    n = len(trades)
    p = m["win_rate_pct"] / 100.0
    reasons = trades.groupby("exit_reason")["net_pnl"].agg(
        n="count", total="sum", mean="mean")
    targets = int(reasons["n"].get("target", 0))
    stops = int(reasons["n"].get("stop", 0))
    bracket = targets + stops

    return {
        "label": label,
        "trades": n,
        "net_pnl": m["net_pnl"],
        "win_rate": p,
        "win_se": float(np.sqrt(p * (1 - p) / n)),
        "profit_factor": m["profit_factor"],
        "max_drawdown": m["max_drawdown"],
        "eod_drawdown": float(dd["max_drawdown_from_peak"]),
        "mean_trade": float(trades["net_pnl"].mean()),
        "mean_day": float(daily.mean()),
        "sd_day": float(daily.std(ddof=1)),
        "trading_days": int(len(daily)),
        "pass_probability": sim.pass_probability,
        "payout_probability": sim.payout_probability,
        "ci_low": sim.ci_low,
        "ci_high": sim.ci_high,
        "blowup_probability": sim.blowup_probability,
        "timeout_probability": sim.timeout_probability,
        "expected_attempts": sim.expected_attempts,
        "blowups": int(blow["blowups"]),
        "passes": int(blow["passes"]),
        "targets": targets,
        "stops": stops,
        "target_share": (targets / bracket) if bracket else float("nan"),
        "reasons": reasons,
        "same_bar": int((trades["entry_time"] == trades["exit_time"]).sum()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def side_by_side(a: dict, b: dict) -> str:
    out = [LINE, "SIDE BY SIDE", LINE,
           f"{'':<34}{a['label']:>28}{b['label']:>28}"]
    out.append("-" * 98)

    def row(name, fmt):
        out.append(f"  {name:<32}{fmt(a):>28}{fmt(b):>28}")

    row("Trades", lambda r: f"{r['trades']:,}")
    row("Net P&L", lambda r: f"${r['net_pnl']:,.2f}")
    row("Mean per trade", lambda r: f"${r['mean_trade']:,.2f}")
    row("Win rate", lambda r: f"{pct(r['win_rate'])}")
    row("  standard error", lambda r: f"{100 * r['win_se']:.2f} pts")
    row("  +/- 1 SE", lambda r: f"{pct(r['win_rate'] - r['win_se'])} - "
                                f"{pct(r['win_rate'] + r['win_se'])}")
    row("Profit factor", lambda r: f"{r['profit_factor']:.3f}")
    row("Max drawdown (trade equity)", lambda r: f"${r['max_drawdown']:,.2f}")
    row("Max drawdown (EOD trail)", lambda r: f"${r['eod_drawdown']:,.2f}")
    row("Target share of bracket", lambda r: pct(r["target_share"]))
    row("  break-even", lambda r: pct(BREAK_EVEN))
    row("Trading days", lambda r: f"{r['trading_days']:,}")
    row("Mean day", lambda r: f"${r['mean_day']:,.2f}")
    row("Sd day", lambda r: f"${r['sd_day']:,.2f}")
    row("Evaluations blown / passed", lambda r: f"{r['blowups']} / {r['passes']}")
    out.append("-" * 98)
    out.append("  eval_sim, run on each window's own daily distribution")
    row("Pass probability", lambda r: pct(r["pass_probability"]))
    row("  95% CI", lambda r: f"{pct(r['ci_low'])} - {pct(r['ci_high'])}")
    row(f"Payout probability (${rules.PAYOUT_BALANCE:,.0f})",
        lambda r: pct(r["payout_probability"]))
    row("Blow-up probability", lambda r: pct(r["blowup_probability"]))
    row("Timeout probability", lambda r: pct(r["timeout_probability"]))
    row("Expected attempts to pass", lambda r: f"{r['expected_attempts']:.2f}")
    return "\n".join(out)


def exit_reason_block(stats: dict) -> str:
    out = [f"\n  {stats['label']}"]
    r = stats["reasons"].copy()
    r["mean"] = r["mean"].round(2)
    r["total"] = r["total"].round(2)
    out.append("    " + r.to_string().replace("\n", "\n    "))
    out.append(f"    target share of bracket outcomes: "
               f"{pct(stats['target_share'])}  (break-even {pct(BREAK_EVEN)})")
    return "\n".join(out)


def yearly_table(trades: pd.DataFrame, label: str) -> str:
    out = [LINE, f"YEARLY BREAKDOWN - {label}", LINE,
           f"{'year':>6}{'trades':>8}{'net P&L':>12}{'win%':>8}{'PF':>8}"
           f"{'maxDD':>11}{'stops':>7}{'targets':>9}{'flatten':>9}"]
    t = trades.assign(year=pd.to_datetime(trades["entry_time"]).dt.year)
    profitable = 0
    for year in YEARS:
        g = t[t["year"] == year]
        if g.empty:
            out.append(f"{year:>6}{0:>8}{'-':>12}{'-':>8}{'-':>8}{'-':>11}")
            continue
        m = compute_metrics(g)
        counts = g["exit_reason"].value_counts()
        if m["net_pnl"] > 0:
            profitable += 1
        out.append(
            f"{year:>6}{len(g):>8}{m['net_pnl']:>12,.0f}"
            f"{m['win_rate_pct']:>8.1f}{m['profit_factor']:>8.2f}"
            f"{m['max_drawdown']:>11,.0f}"
            f"{int(counts.get('stop', 0)):>7}{int(counts.get('target', 0)):>9}"
            f"{int(counts.get('session_end', 0)):>9}"
        )
    out.append("-" * 98)
    out.append(f"  Years profitable: {profitable} of {len(YEARS)}")
    return "\n".join(out)


def kill_criteria(seven: dict, profitable_years: int) -> str:
    c1 = seven["pass_probability"] >= MIN_POOLED_PASS_PROBABILITY
    c2 = seven["blowups"] <= MAX_BLOWUPS
    c3 = profitable_years >= MIN_PROFITABLE_YEARS and seven["net_pnl"] > 0

    def mark(ok):
        return "PASS" if ok else "FAIL"

    return "\n".join([
        LINE, "KILL CRITERIA (entry 5, seven years, 4 contracts)", LINE,
        f"  1  Pooled pass probability >= 25%      "
        f"{pct(seven['pass_probability']):>10}   {mark(c1)}",
        f"  2  Evaluations blown <= 1              "
        f"{seven['blowups']:>10}   {mark(c2)}",
        f"  3  >= 4 of 7 years profitable AND P&L > 0  "
        f"{profitable_years} of 7, ${seven['net_pnl']:,.0f}".ljust(56)
        + f"   {mark(c3)}",
        "-" * 98,
        f"  VERDICT: {'SURVIVES' if (c1 and c2 and c3) else 'REJECTED'}"
        f"   (any one failure kills the hypothesis)",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=1.25)
    ap.add_argument("--paths", type=int, default=20_000)
    args = ap.parse_args()

    costs = CostModel(commission_per_side=args.commission,
                      slippage_ticks=args.slippage_ticks)
    tag = f"slip{args.slippage_ticks:g}"

    print("Loading bars ...", flush=True)
    bars = loader.load_bars(args.parquet)
    roll_dates = loader.detect_roll_dates(bars)
    early_closes = loader.detect_early_close_dates(bars)

    strat = ORB2(entry5_params(), roll_dates=roll_dates,
                 early_close_dates=early_closes)
    signals = strat.generate_signals(bars)
    diag = strat.diagnostics

    print(LINE)
    print("ENTRY 5 - ORB flat by 10:30, filter OFF, 4 contracts")
    print(f"  09:30-09:45 range, both stops 1.0pt beyond, 10pt stop / 18pt target")
    print(f"  cancel and flatten at 10:30 (exit at the 10:30 bar open)")
    print(f"  {costs.slippage_ticks:g} tick slippage/side, "
          f"${costs.commission_per_side:.2f} commission/side")
    print(f"  guards: ${rules.DAILY_LOSS_LIMIT:,.0f} daily loss, "
          f"${rules.TRAILING_DD_STOP:,.0f} trailing drawdown halt")
    print(LINE, flush=True)

    seven_trades, seven_loss, seven_dd = build_stream(bars, signals, *SEVEN_YEAR, costs)
    two_trades, two_loss, two_dd = build_stream(bars, signals, *TWO_YEAR, costs)
    # Same configuration with the drawdown halt off: the basis entries 1 and 4
    # used, without which their blow-up counts cannot be compared with these.
    seven_nh, _, _ = build_stream(bars, signals, *SEVEN_YEAR, costs, dd_halt=False)
    two_nh, _, _ = build_stream(bars, signals, *TWO_YEAR, costs, dd_halt=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seven_trades.to_csv(RESULTS_DIR / f"orb_flat_7yr_{tag}.csv", index=False)
    two_trades.to_csv(RESULTS_DIR / f"orb_flat_2yr_{tag}.csv", index=False)

    seven = window_stats(seven_trades, "All seven years", args.paths)
    two = window_stats(two_trades, "Last 2 years", args.paths)
    seven_nh_s = window_stats(seven_nh, "All seven years", args.paths)
    two_nh_s = window_stats(two_nh, "Last 2 years", args.paths)

    print()
    print("SPECIFICATION AS FROZEN - both internal guards active")
    print(side_by_side(two, seven))
    print()
    print(LINE)
    print("THE DRAWDOWN HALT ENDS THE SEVEN-YEAR RUN IN 2020")
    print(LINE)
    print(f"  The $1,500 trailing halt fires and never releases: a halted account")
    print(f"  takes no trades, so its balance never recovers and the halt is")
    print(f"  permanent. {len(seven_dd):,} sessions are blocked and the run stops.")
    print(f"  That is faithful to the Pine, but it makes kill criteria 2 and 3")
    print(f"  degenerate - no trades cannot blow an evaluation, and six years with")
    print(f"  no trades cannot be profitable. The same configuration without the")
    print(f"  halt is therefore reported below, on the basis entries 1 and 4 used.")
    print()
    print("SAME CONFIGURATION, DRAWDOWN HALT OFF - comparable to entries 1 and 4")
    print(side_by_side(two_nh_s, seven_nh_s))

    print()
    print(LINE)
    print("P&L BY EXIT REASON")
    print(LINE)
    print(exit_reason_block(two))
    print(exit_reason_block(seven))

    print()
    print(exit_reason_block(seven_nh_s))

    print()
    print(yearly_table(seven_trades, "seven years, halt ON (as frozen)"))
    print()
    print(yearly_table(seven_nh, "seven years, halt OFF (comparable basis)"))

    def count_profitable(trades_df):
        if trades_df.empty:
            return 0
        t = trades_df.assign(year=pd.to_datetime(trades_df["entry_time"]).dt.year)
        return sum(
            1 for y in YEARS
            if not t[t["year"] == y].empty and t[t["year"] == y]["net_pnl"].sum() > 0
        )

    profitable_years = count_profitable(seven_trades)
    profitable_nh = count_profitable(seven_nh)

    print()
    print(LINE)
    print("GUARDS AND MEASURABILITY")
    print(LINE)
    print(f"  Daily-loss halts     7yr {len(seven_loss):>4}   2yr {len(two_loss):>4}")
    print(f"  Drawdown halts       7yr {len(seven_dd):>4}   2yr {len(two_dd):>4}")
    print(f"  Same-bar entry+exit  7yr {seven['same_bar']:>4}   2yr {two['same_bar']:>4}"
          f"   (rule 6 unverifiable at 1-minute resolution)")
    oos = diag[[d.year in YEARS for d in diag.index]]
    amb = int(oos["ambiguous_both_stops"].fillna(False).sum())
    print(f"  Both stops in one bar, seven years: {amb}"
          f"   (resolved pessimistically)")
    skipped = oos[oos["skipped_reason"] == SKIP_WIDE_RANGE]
    entered = oos[oos["entered"] == True]  # noqa: E712
    print(f"  Sessions skipped by the 10pt ceiling: {len(skipped):,} of {len(oos):,}")
    if len(skipped):
        h = skipped["range_height"].astype(float)
        print(f"    their opening range: median {h.median():.2f} pts, "
              f"mean {h.mean():.2f}, max {h.max():.2f}")
    if len(entered):
        h = entered["range_height"].astype(float)
        print(f"    traded sessions:     median {h.median():.2f} pts, "
              f"mean {h.mean():.2f}")

    print()
    print("AS FROZEN (halt ON)")
    print(kill_criteria(seven, profitable_years))
    print()
    print("ON THE COMPARABLE BASIS (halt OFF)")
    print(kill_criteria(seven_nh_s, profitable_nh))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
