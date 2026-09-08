"""End-of-day and rolling review of the manual journal.

    python journal/review.py                 # today plus the rolling view
    python journal/review.py --day 2026-09-02
    python journal/review.py --rolling-only

Everything numeric here is borrowed, not rebuilt:

  * ``metrics.compute_metrics`` / ``format_report`` - the same performance and
    duration statistics the backtests report
  * ``backtests.review.format_limits`` / ``format_drawdown`` - the same
    trailing-drawdown view, so a paper account and a backtest are read the
    same way
  * ``eval_sim.simulate`` - the same pass-probability model

A second implementation of any of these would eventually disagree with the
first, and the disagreement would be discovered at the worst possible moment.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as date_type
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("strategies", "backtests", "journal"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import rules  # noqa: E402
import store  # noqa: E402
from engine import equity_curve_by_day  # noqa: E402
from eval_sim import EvalConfig, daily_pnl_from_trades, simulate  # noqa: E402
from metrics import compute_metrics, format_report, sharpe_ratio  # noqa: E402
from review import format_drawdown, format_limits  # noqa: E402  (backtests/review.py)

#: Pre-registered in research/hypotheses.md entry 3.
MIN_TRADES_FOR_SIM = 30
CLEAN_TRADES_REQUIRED = 60
REQUIRED_PASS_PROBABILITY = 0.50


def late_entries(closed: pd.DataFrame, early_close_dates=()) -> pd.DataFrame:
    """Tickets opened after the entry cutoff for their session."""
    if closed.empty:
        return closed
    flags = closed.apply(
        lambda r: not rules.is_entry_allowed(r["entry_time"], early_close_dates),
        axis=1,
    )
    return closed[flags]


def hold_violations(closed: pd.DataFrame) -> pd.DataFrame:
    if closed.empty or "duration_seconds" not in closed.columns:
        return closed.iloc[0:0] if not closed.empty else closed
    return closed[closed["duration_seconds"] < rules.MIN_HOLD_SECONDS]


def rule_violations(closed: pd.DataFrame, early_close_dates=()) -> dict:
    """Every rule breach the log can detect after the fact."""
    late = late_entries(closed, early_close_dates)
    short = hold_violations(closed)
    oversized = (
        closed[closed["contracts"].astype(int) > rules.POSITION_CAP]
        if not closed.empty else closed
    )
    return {"late_entries": late, "short_holds": short, "oversized": oversized}


def format_day(closed: pd.DataFrame, day: date_type, early_close_dates=()) -> str:
    line = "=" * 88
    out = [line, f"DAY REVIEW - {day}", line]

    today = closed[closed["session_date"] == day] if not closed.empty else closed
    if today.empty:
        out.append("  No closed trades on this date.")
        return "\n".join(out)

    net = float(today["net_pnl"].sum())
    budget_left = rules.DAILY_LOSS_LIMIT + net
    out += [
        f"  Trades              {len(today)}",
        f"  Net P&L             ${net:>10,.2f}",
        f"  Wins / losses       {int((today['net_pnl'] > 0).sum())} / "
        f"{int((today['net_pnl'] < 0).sum())}",
        f"  Daily loss budget   ${rules.DAILY_LOSS_LIMIT:,.0f} limit, "
        f"${max(budget_left, 0):,.2f} would have been left",
    ]
    if rules.is_daily_loss_breached(net):
        out.append("  *** DAILY LOSS LIMIT BREACHED ***")

    out.append("")
    out.append("  Trades:")
    for _, r in today.iterrows():
        held = int(r.get("duration_seconds") or 0)
        out.append(
            f"    {pd.Timestamp(r['entry_time']).strftime('%H:%M')}  "
            f"{r['direction']:<5} {int(r['contracts'])} {r['instrument']}  "
            f"{r['entry_price']:>9.2f} -> {r['exit_price']:>9.2f}  "
            f"{r['exit_reason']:<12} ${r['net_pnl']:>9,.2f}  "
            f"{held // 60}m{held % 60:02d}s"
        )
        out.append(f"        thesis: {r['thesis']}")

    violations = rule_violations(today, early_close_dates)
    out.append("")
    out.append("  Rule checks:")
    for label, frame in (
        ("late entries (after cutoff)", violations["late_entries"]),
        (f"holds under {rules.MIN_HOLD_SECONDS}s", violations["short_holds"]),
        (f"size over {rules.POSITION_CAP} contracts", violations["oversized"]),
    ):
        mark = "ok  " if frame.empty else "FLAG"
        out.append(f"    [{mark}] {label}: {len(frame)}")

    return "\n".join(out)


def format_rolling(closed: pd.DataFrame) -> str:
    line = "=" * 88
    out = [line, "ROLLING REVIEW", line]
    if closed.empty:
        out.append("  No closed trades yet.")
        return "\n".join(out)

    equity = equity_curve_by_day(closed)
    daily = closed.groupby("session_date")["net_pnl"].sum().sort_index()

    out += [
        f"  Closed trades       {len(closed)}",
        f"  Trading days        {len(daily)}",
        f"  Net P&L             ${closed['net_pnl'].sum():>10,.2f}",
        f"  Sharpe (annualised) {sharpe_ratio(daily):>10.3f}",
        f"  Win rate            {(closed['net_pnl'] > 0).mean() * 100:>10.1f}%",
        "",
        "  Equity by day (most recent 15):",
    ]
    for _, r in equity.tail(15).iterrows():
        out.append(
            f"    {r['session_date']}  day ${r['day_pnl']:>9,.2f}  "
            f"balance ${r['balance']:>12,.2f}  "
            f"dd ${r['drawdown_from_peak']:>8,.2f}  {r['state']}"
        )
    return "\n".join(out)


def format_thesis_review(closed: pd.DataFrame) -> str:
    """Which kinds of reasoning actually pay.

    Groups by the first word of the thesis as a crude tag. It is crude on
    purpose: any richer scheme would need tags chosen in advance, and a tag
    invented after the outcome is known is worth nothing.
    """
    line = "=" * 88
    out = [line, "THESIS vs OUTCOME", line]
    if closed.empty or "thesis" not in closed.columns:
        out.append("  No theses logged yet.")
        return "\n".join(out)

    work = closed.copy()
    work["tag"] = (
        work["thesis"].astype(str).str.strip().str.split().str[0].str.lower()
    )
    grouped = work.groupby("tag").agg(
        trades=("net_pnl", "size"),
        net=("net_pnl", "sum"),
        avg=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda s: (s > 0).mean() * 100),
    ).sort_values("net", ascending=False)

    out.append(f"  {'tag':<22}{'trades':>8}{'net':>12}{'avg':>10}{'win%':>8}")
    for tag, r in grouped.iterrows():
        out.append(
            f"  {str(tag)[:22]:<22}{int(r['trades']):>8}{r['net']:>12,.2f}"
            f"{r['avg']:>10,.2f}{r['win_rate']:>8.1f}"
        )
    out.append("")
    out.append("  Tags with fewer than 5 trades are noise, not evidence.")
    return "\n".join(out)


def readiness(closed: pd.DataFrame) -> dict:
    """The entry-3 gate, computed once so nothing has to restate it.

    Returns the three gate results alongside the numbers behind them.
    ``strategies/registry.py`` reads this rather than recomputing the
    thresholds: a second implementation would eventually disagree with this
    one, and the disagreement would surface at the worst possible moment.
    """
    n = len(closed)
    violations = rule_violations(closed)
    dirty = sum(len(v) for v in violations.values())
    # A violation resets the count to zero rather than deducting from it.
    clean = n if dirty == 0 else 0

    net = float(closed["net_pnl"].sum()) if n else 0.0
    expectancy = net / n if n else 0.0

    prob = None
    blowup = None
    attempts = None
    if n >= MIN_TRADES_FOR_SIM:
        result = simulate(daily_pnl_from_trades(closed), EvalConfig(),
                          paths=20_000, seed=0)
        prob = result.pass_probability
        blowup = result.blowup_probability
        attempts = result.expected_attempts

    gates = {
        "clean_trades": clean >= CLEAN_TRADES_REQUIRED,
        "positive_expectancy": expectancy > 0,
        "pass_probability": prob is not None and prob > REQUIRED_PASS_PROBABILITY,
    }
    return {
        "closed_trades": n,
        "clean_trades": clean,
        "violations": dirty,
        "net_pnl": net,
        "expectancy": expectancy,
        "pass_probability": prob,
        "blowup_probability": blowup,
        "expected_attempts": attempts,
        "gates": gates,
        "ready": all(gates.values()),
    }


def format_readiness(closed: pd.DataFrame) -> str:
    """Progress against the pre-registered gate for buying a Lucid eval."""
    line = "=" * 88
    out = [line, "EVALUATION READINESS (research/hypotheses.md entry 3)", line]

    r = readiness(closed)
    n = r["closed_trades"]
    dirty = r["violations"]
    clean = r["clean_trades"]
    expectancy = r["expectancy"]

    out += [
        f"  Rule-clean trades   {clean} / {CLEAN_TRADES_REQUIRED}"
        + ("" if dirty == 0 else
           f"   *** RESET: {dirty} rule violation(s) logged ***"),
        f"  Expectancy/trade    ${expectancy:>10,.2f}   (must be positive)",
    ]

    prob = r["pass_probability"]
    if prob is None:
        out.append(
            f"  Pass probability    not run - needs {MIN_TRADES_FOR_SIM} closed "
            f"trades, have {n}"
        )
    else:
        out += [
            f"  Pass probability    {prob:>10.2%}   "
            f"(must exceed {REQUIRED_PASS_PROBABILITY:.0%})",
            f"  Blow-up probability {r['blowup_probability']:>10.2%}",
            f"  Expected attempts   {r['expected_attempts']:>10.2f}",
        ]

    labels = [
        ("60+ rule-clean trades", "clean_trades"),
        ("positive expectancy", "positive_expectancy"),
        ("pass probability > 50%", "pass_probability"),
    ]
    out.append("")
    for label, key in labels:
        out.append(f"    [{'PASS' if r['gates'][key] else 'no  '}] {label}")
    out.append("")
    out.append(
        f"  VERDICT: {'READY to buy an evaluation' if r['ready'] else 'NOT READY'}"
    )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default=None, help="session date, default today")
    ap.add_argument("--rolling-only", action="store_true")
    ap.add_argument("--journal", default=str(store.TRADES_PATH))
    args = ap.parse_args(argv)

    journal_path = Path(args.journal)
    closed = store.load_closed(journal_path)

    if closed.empty:
        print("No closed trades in the journal yet.")
        print(f"  {journal_path}")
        # Decisions can exist before any manual trade does - the desk posts
        # tickets whether or not the operator has traded by hand.
        import decisions  # noqa: PLC0415

        print()
        print(decisions.format_decisions())
        return 0

    day = (pd.Timestamp(args.day).date() if args.day
           else pd.Timestamp.now(tz=rules.ET).date())

    if not args.rolling_only:
        print(format_day(closed, day))
        print()

    print(format_limits())
    print()
    print(format_rolling(closed))
    print()
    print(format_drawdown(closed))
    print()
    print(format_thesis_review(closed))
    print()
    print(format_readiness(closed))
    print()
    # Operator button presses on desk tickets. Reported, deliberately not
    # counted: see journal/decisions.py for why.
    import decisions  # noqa: PLC0415 - journal/ is on sys.path above

    print(decisions.format_decisions())
    print()
    print(format_report(compute_metrics(closed), "PERFORMANCE (all logged trades)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
