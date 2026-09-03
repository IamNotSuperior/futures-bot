"""Review a trade stream against the account's limits.

Takes any trade list - a backtest CSV, or the journal's `trades.jsonl` once
that exists - and reports the end-of-day trailing drawdown alongside the firm
and internal lines, plus the standard performance report.

    python backtests/review.py backtests/results/orb_oos_stream_slip1.csv
    python backtests/review.py journal/trades.jsonl

The distinction this report exists to make: a strategy's P&L curve says how
much it made, and the trailing drawdown says whether the account survived long
enough to collect it. Those are different questions and a good answer to the
first is routinely a failing answer to the second.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import rules  # noqa: E402
from engine import (  # noqa: E402
    count_evaluation_blowups, equity_curve_by_day, trailing_drawdown_summary,
)
from metrics import compute_metrics, format_report  # noqa: E402

REQUIRED = {"entry_time", "exit_time", "net_pnl"}


def load_trades(path: Path) -> pd.DataFrame:
    """Load a trade list from CSV or JSON Lines.

    Only the columns the review needs are required; anything else the producer
    wrote is carried through untouched.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Point this at a backtest trade CSV "
            f"(backtests/results/) or a journal trades.jsonl."
        )

    if path.suffix == ".jsonl":
        trades = pd.read_json(path, lines=True)
    elif path.suffix == ".csv":
        trades = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type {path.suffix!r}; expected .csv or .jsonl")

    missing = REQUIRED - set(trades.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    for col in ("entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True).dt.tz_convert(rules.ET)
    if "session_date" not in trades.columns:
        trades["session_date"] = trades["entry_time"].dt.date
    else:
        trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.date
    if "duration_seconds" not in trades.columns:
        trades["duration_seconds"] = (
            trades["exit_time"] - trades["entry_time"]
        ).dt.total_seconds()
    return trades.sort_values("entry_time").reset_index(drop=True)


def format_limits() -> str:
    out = ["=" * 96, "ACCOUNT LIMITS", "=" * 96,
           f"  {rules.FIRM.name}, ${rules.FIRM.account_size:,.0f} nominal, "
           f"${rules.FIRM.profit_target:,.0f} target", "",
           f"  {'limit':<28}{'internal':>12}{'firm':>12}   headroom"]
    for label, internal, firm, note in rules.buffer_report():
        out.append(f"  {label:<28}{internal:>12,.0f}{firm:>12,.0f}   {note}")
    out.append("")
    out.append("  Guards enforce the internal column. A stop at a firm number "
               "means a guard failed.")
    return "\n".join(out)


def format_drawdown(trades: pd.DataFrame) -> str:
    equity = equity_curve_by_day(trades)
    summary = trailing_drawdown_summary(equity)
    blow = count_evaluation_blowups(trades)

    out = ["=" * 96, "END-OF-DAY TRAILING DRAWDOWN", "=" * 96]
    if not len(equity):
        out.append("  No trades.")
        return "\n".join(out)

    out += [
        f"  Starting balance                  ${rules.ACCOUNT_SIZE:>12,.2f}",
        f"  Peak end-of-day balance           ${summary['peak_balance']:>12,.2f}",
        f"  Final balance                     ${summary['final_balance']:>12,.2f}",
        f"  Worst drawdown from peak          ${summary['max_drawdown_from_peak']:>12,.2f}",
        f"  Least headroom to the firm line   ${summary['min_headroom_to_firm']:>12,.2f}",
        "",
        f"  Trading days                       {summary['days']:>12,}",
        f"  Days at internal WARN (>=${rules.TRAILING_DD_WARN:,.0f})  "
        f"{summary['warn_days']:>12,}",
        f"  Days past internal STOP (>=${rules.TRAILING_DD_STOP:,.0f}) "
        f"{summary['internal_stop_days']:>12,}",
        "",
        f"  $50K evaluations BLOWN             {blow['blowups']:>12,}",
        f"  $50K evaluations PASSED            {blow['passes']:>12,}",
    ]
    if blow["blowups"]:
        out.append("")
        out.append("  Each blow-up restarts a fresh account the following day:")
        for d, n in zip(blow["dates"], blow["days_survived"]):
            out.append(f"    terminated {d} after {n} trading day(s)")

    worst = equity.loc[equity["drawdown_from_peak"].idxmax()]
    out += [
        "",
        f"  Deepest day: {worst['session_date']}  balance "
        f"${worst['balance']:,.2f}  peak ${worst['peak_eod_balance']:,.2f}  "
        f"down ${worst['drawdown_from_peak']:,.2f}  state={worst['state']}",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trades", help="path to a trade CSV or journal .jsonl")
    ap.add_argument("--no-performance", action="store_true",
                    help="skip the standard performance report")
    args = ap.parse_args()

    trades = load_trades(Path(args.trades))
    print(format_limits())
    print()
    print(format_drawdown(trades))
    if not args.no_performance:
        print()
        print(format_report(compute_metrics(trades), "PERFORMANCE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
