"""Re-price a saved trade stream at a different commission, and re-score it.

Every verdict before 2026-09-11 was scored at $1.25 a side. Lucid support
then confirmed the MES rate on a 50K Pro evaluation is $0.50 a side
(``rules.COMMISSION_PER_SIDE``; the $1.25 lives on as
``rules.ASSUMED_COMMISSION_PER_SIDE``). Commission never touches a fill, so a
saved out-of-sample stream can be re-priced *exactly*: each trade moves by
twice the per-side difference times its size, and the size is recovered from
the commission the trade carried - which is what lets entry 6's per-session
sizing come through row by row.

What this cannot do, and says so in every addendum that uses it: re-decide
anything that depended on P&L while the stream was being built. A daily-loss
flatten that fired at $1.25 stays fired; a walk-forward's parameter choice
stays what $1.25 selected. The saved streams carry one such exit in 5,700
trades, so the first is moot; the second is stated where it applies.

    python backtests/reprice.py backtests/results/orb_oos_stream_slip1.csv
    python backtests/reprice.py backtests/results/orb2_oos_on_4c_slip1.csv --old 1.25
    python backtests/reprice.py backtests/results/orb_flat_7yr_slip1.csv --halt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("strategies", "backtests"):
    _p = str(PROJECT_ROOT / _folder)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_sim  # noqa: E402
import rules  # noqa: E402
from engine import count_evaluation_blowups, enforce_trailing_drawdown_halt  # noqa: E402
from metrics import compute_metrics  # noqa: E402


def reprice_stream(
    trades: pd.DataFrame,
    old_commission_per_side: float,
    new_commission_per_side: float,
) -> pd.DataFrame:
    """The same trades priced at ``new`` instead of ``old`` commission.

    Size is recovered from the ``commission`` column, which ``price_trades``
    writes as ``2 * per_side * contracts``. Fills, points, slippage and exits
    are untouched; only ``commission`` and ``net_pnl`` change. The input is
    not modified.
    """
    if "commission" not in trades.columns:
        raise ValueError(
            "stream has no 'commission' column, so contracts cannot be "
            "recovered; re-pricing needs a stream written by engine.price_trades"
        )
    out = trades.copy()
    sizes = out["commission"].astype(float) / (2.0 * old_commission_per_side)
    rounded = sizes.round()
    if not np.allclose(sizes, rounded, atol=1e-6) or (rounded < 1).any():
        raise ValueError(
            f"stream was not priced at ${old_commission_per_side:.2f} a side: "
            f"its commission column does not resolve to whole contracts"
        )
    contracts = rounded.astype(int)
    new_commission = (2.0 * new_commission_per_side * contracts).astype(float)
    # Delta first, then add: at old == new the delta is exactly 0.0 and every
    # net_pnl comes back bit-identical.
    out["net_pnl"] = out["net_pnl"] + (out["commission"] - new_commission)
    out["commission"] = new_commission
    return out


def _normalise(trades: pd.DataFrame) -> pd.DataFrame:
    """Timestamps in ET and a ``session_date`` column, whatever the source.

    A stream read back from CSV carries strings spanning EST and EDT, which
    pandas will not infer as one offset; parse as UTC and convert.
    """
    t = trades.copy()
    for col in ("entry_time", "exit_time"):
        if col in t.columns:
            t[col] = pd.to_datetime(t[col], utc=True).dt.tz_convert(rules.ET)
    t["session_date"] = t["entry_time"].dt.date
    return t


def score_stream(
    trades: pd.DataFrame,
    paths: int = 20_000,
    trailing_halt: bool = False,
    seed: int = 0,
) -> dict:
    """Headline figures for one stream, as the runners report them.

    ``trailing_halt=True`` re-applies the engine's end-of-day halt to the
    stream first, for entries whose basis had it: a halt that fired at $1.25
    may fire later, or not at all, at $0.50, and re-pricing the *halted*
    stream would miss that. Pass the unhalted stream and let this decide.
    """
    t = _normalise(trades)
    dd_halts = 0
    if trailing_halt:
        t, halts = enforce_trailing_drawdown_halt(t)
        dd_halts = len(halts)
    if t.empty:
        raise ValueError("no trades to score")

    m = compute_metrics(t)
    years = t["entry_time"].dt.year
    by_year = {int(y): g for y, g in t.groupby(years)}
    # A year with one trading day has no Sharpe; leave it out of the median
    # rather than let NaN decide it.
    year_sharpe = [s for s in (compute_metrics(g)["sharpe"] for g in by_year.values())
                   if np.isfinite(s)]
    daily = eval_sim.daily_pnl_from_trades(t)
    sim = eval_sim.simulate(daily, paths=paths, seed=seed)
    blow = count_evaluation_blowups(t)

    return {
        "trades": int(len(t)),
        "net_pnl": float(t["net_pnl"].sum()),
        "mean_per_trade": float(t["net_pnl"].mean()),
        "years": sorted(by_year),
        "profitable_years": int(sum(g["net_pnl"].sum() > 0 for g in by_year.values())),
        "year_pnl": {y: float(g["net_pnl"].sum()) for y, g in by_year.items()},
        "median_year_sharpe": float(np.median(year_sharpe)) if year_sharpe else float("nan"),
        "sharpe": float(m["sharpe"]),
        "profit_factor": float(m["profit_factor"]),
        "max_drawdown": float(m["max_drawdown"]),
        "max_daily_loss": float(m["max_daily_loss"]),
        "pass_probability": sim.pass_probability,
        "payout_probability": sim.payout_probability,
        "blowups": int(blow["blowups"]),
        "dd_halts": dd_halts,
    }


def compare(path: Path, old: float, new: float, paths: int,
            trailing_halt: bool) -> tuple[dict, dict]:
    trades = pd.read_csv(path)
    before = score_stream(trades, paths, trailing_halt)
    after = score_stream(reprice_stream(trades, old, new), paths, trailing_halt)
    return before, after


def format_comparison(before: dict, after: dict, label: str,
                      old: float, new: float) -> str:
    line = "=" * 78
    out = [line, f"RE-PRICED  {label}", line,
           f"  commission per side          ${old:.2f}  ->  ${new:.2f}"
           f"   (${2 * (old - new):.2f} per contract per round turn)",
           f"{'':<30}{'as scored':>22}{'re-priced':>22}", "-" * 78]

    def row(name, key, fmt):
        out.append(f"  {name:<28}{fmt(before[key]):>22}{fmt(after[key]):>22}")

    money = lambda x: f"${x:,.2f}"  # noqa: E731
    pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    row("Trades", "trades", lambda x: f"{x:,}")
    row("Net P&L", "net_pnl", money)
    row("Mean per trade", "mean_per_trade", money)
    row("Years profitable", "profitable_years",
        lambda x: f"{x} of {len(before['years'])}")
    row("Median year Sharpe", "median_year_sharpe", lambda x: f"{x:+.3f}")
    row("Sharpe", "sharpe", lambda x: f"{x:.2f}")
    row("Profit factor", "profit_factor", lambda x: f"{x:.3f}")
    row("Max drawdown", "max_drawdown", money)
    row("Worst day", "max_daily_loss", money)
    row("Pass probability", "pass_probability", pct)
    row("Payout probability", "payout_probability", pct)
    row("Evaluations blown", "blowups", str)
    row("Sessions blocked by the halt", "dd_halts", str)
    out.append("-" * 78)
    out.append("  per year (as scored -> re-priced):")
    for y in before["years"]:
        out.append(f"    {y}  ${before['year_pnl'][y]:>10,.2f}  ->  "
                   f"${after['year_pnl'].get(y, 0.0):>10,.2f}")
    out.append(line)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", help="a trade CSV written by a runner")
    ap.add_argument("--old", type=float, default=rules.ASSUMED_COMMISSION_PER_SIDE,
                    help="per side, what the stream was priced at")
    ap.add_argument("--new", type=float, default=rules.COMMISSION_PER_SIDE,
                    help="per side, what to re-price it at")
    ap.add_argument("--halt", action="store_true",
                    help="re-apply the end-of-day trailing halt to both streams")
    ap.add_argument("--paths", type=int, default=20_000)
    args = ap.parse_args(argv)

    path = Path(args.stream)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    before, after = compare(path, args.old, args.new, args.paths, args.halt)
    print(format_comparison(before, after, path.name, args.old, args.new))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
