"""Performance metrics and the CLAUDE.md-mandated backtest report.

Thresholds are imported from :mod:`strategies.rules` rather than restated here,
so the report and the live guards cannot drift apart (CLAUDE.md rule 9).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategies"))

import rules  # noqa: E402

TRADING_DAYS_PER_YEAR = 252


def daily_pnl(trades: pd.DataFrame) -> pd.Series:
    """Net P&L per session date, indexed by date."""
    if trades.empty:
        return pd.Series(dtype="float64")
    return trades.groupby("session_date")["net_pnl"].sum().sort_index()


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    """Cumulative net P&L after each trade."""
    if trades.empty:
        return pd.Series(dtype="float64")
    return trades["net_pnl"].cumsum()


def max_drawdown(curve: pd.Series) -> float:
    """Largest peak-to-trough decline in dollars (returned as a negative number).

    The running peak starts at zero, so a strategy that never gets above water
    still reports its full decline from the starting balance.
    """
    if curve.empty:
        return 0.0
    peak = curve.cummax().clip(lower=0.0)
    return float((curve - peak).min())


def sharpe_ratio(daily: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe of the daily P&L series.

    Computed on dollar P&L, not returns: there is no account equity in this
    model, so this measures consistency of daily dollars rather than a
    risk-adjusted return on capital. Comparable across parameter sets, not
    against a published fund Sharpe.
    """
    if len(daily) < 2:
        return float("nan")
    std = daily.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(daily.mean() / std * math.sqrt(periods))


def daily_loss_breaches(trades: pd.DataFrame) -> pd.DataFrame:
    """Sessions where running intraday P&L would have hit the daily loss limit.

    Checked on the running total after each trade, not the day's closing figure:
    the limit is a stop-trading trigger the moment it is touched, so a day that
    dipped past it and recovered still counts as a breach.
    """
    if trades.empty:
        return pd.DataFrame(columns=["session_date", "worst_running_pnl", "closing_pnl"])

    rows = []
    for day, group in trades.groupby("session_date"):
        running = group.sort_values("entry_time")["net_pnl"].cumsum()
        worst = float(running.min())
        if rules.is_daily_loss_breached(worst):
            rows.append(
                {
                    "session_date": day,
                    "worst_running_pnl": worst,
                    "closing_pnl": float(running.iloc[-1]),
                }
            )
    return pd.DataFrame(rows, columns=["session_date", "worst_running_pnl", "closing_pnl"])


def compute_metrics(trades: pd.DataFrame) -> dict:
    """All reported metrics for a priced trade list."""
    m: dict = {"trade_count": int(len(trades))}
    if trades.empty:
        return m

    net = trades["net_pnl"]
    wins = net[net > 0]
    losses = net[net < 0]

    m["net_pnl"] = float(net.sum())
    m["gross_pnl"] = float(trades["gross_pnl"].sum())
    m["total_commission"] = float(trades["commission"].sum())
    m["total_slippage"] = float(trades["slippage_cost"].sum())

    m["win_count"] = int(len(wins))
    m["loss_count"] = int(len(losses))
    m["win_rate_pct"] = float(len(wins) / len(net) * 100)
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    m["gross_win"] = gross_win
    m["gross_loss"] = gross_loss
    m["profit_factor"] = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    curve = equity_curve(trades)
    m["max_drawdown"] = max_drawdown(curve)

    daily = daily_pnl(trades)
    m["trading_days"] = int(len(daily))
    m["sharpe"] = sharpe_ratio(daily)
    m["max_daily_loss"] = float(daily.min())
    m["max_daily_loss_date"] = daily.idxmin()
    m["best_day"] = float(daily.max())
    m["best_day_date"] = daily.idxmax()

    # Percentages of total profit are only meaningful against a profitable
    # result; reporting them off a negative denominator would invert the sign
    # and read as a pass.
    total_profit = m["net_pnl"]
    if total_profit > 0:
        m["worst_day_pct_of_total_profit"] = m["max_daily_loss"] / total_profit * 100
        m["best_day_pct_of_total_profit"] = m["best_day"] / total_profit * 100
        m["consistency_breach"] = (
            m["best_day_pct_of_total_profit"] > rules.WORST_DAY_FLAG_PCT
        )
    else:
        m["worst_day_pct_of_total_profit"] = None
        m["best_day_pct_of_total_profit"] = None
        m["consistency_breach"] = None

    dur = trades["duration_seconds"]
    m["avg_duration_seconds"] = float(dur.mean())
    m["median_duration_seconds"] = float(dur.median())
    m["min_duration_seconds"] = float(dur.min())
    m["shortest_hold_violates_rule"] = bool(dur.min() < rules.MIN_HOLD_SECONDS)
    # Rule 6 is enforced upstream, so a single hold under the floor in a
    # priced stream means the enforcement failed, whatever share of profit
    # it carried. Counted here so every surface can name it as a regression
    # rather than wait for the 30% warning line.
    m["min_hold_violation_count"] = int((dur < rules.MIN_HOLD_SECONDS).sum())

    # Rule 7: share of winning profit earned by trades held 5 seconds or less.
    scalps = trades[dur <= rules.MICROSCALP_SECONDS]
    scalp_profit = float(scalps.loc[scalps["net_pnl"] > 0, "net_pnl"].sum())
    m["microscalp_trade_count"] = int(len(scalps))
    m["microscalp_profit"] = scalp_profit
    if gross_win > 0:
        m["microscalp_profit_pct"] = scalp_profit / gross_win * 100
        m["microscalp_breach"] = (
            m["microscalp_profit_pct"] > rules.MICROSCALP_PROFIT_FLAG_PCT
        )
    else:
        m["microscalp_profit_pct"] = None
        m["microscalp_breach"] = None
    m["hold_regression"] = bool(
        m["min_hold_violation_count"] > 0 or m["microscalp_trade_count"] > 0
    )

    by_reason = trades.groupby("exit_reason").agg(
        trades=("net_pnl", "size"),
        net_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda s: (s > 0).mean() * 100),
    )
    m["by_exit_reason"] = by_reason.sort_values("net_pnl", ascending=False)

    m["daily_loss_breaches"] = daily_loss_breaches(trades)
    return m


def _fmt_duration(seconds: float) -> str:
    if seconds != seconds:  # NaN
        return "n/a"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def format_report(m: dict, title: str = "BACKTEST REPORT") -> str:
    """Render the report, including every CLAUDE.md-required metric."""
    line = "=" * 72
    out = [line, title, line]

    if not m.get("trade_count"):
        out.append("No trades.")
        return "\n".join(out)

    out += [
        "",
        "-- Performance " + "-" * 57,
        f"  Net P&L                    ${m['net_pnl']:>12,.2f}",
        f"  Gross P&L (pre-cost)       ${m['gross_pnl']:>12,.2f}",
        f"  Commission                 ${-m['total_commission']:>12,.2f}",
        f"  Slippage                   ${-m['total_slippage']:>12,.2f}",
        f"  Trades                      {m['trade_count']:>12,}",
        f"  Trading days                {m['trading_days']:>12,}",
        "",
        f"  Win rate                    {m['win_rate_pct']:>11.1f}%  "
        f"({m['win_count']}W / {m['loss_count']}L)",
        f"  Profit factor               {m['profit_factor']:>12.3f}",
        f"  Average win                ${m['avg_win']:>12,.2f}",
        f"  Average loss               ${m['avg_loss']:>12,.2f}",
        f"  Max drawdown               ${m['max_drawdown']:>12,.2f}",
        f"  Sharpe (annualised)         {m['sharpe']:>12.3f}",
    ]

    out += [
        "",
        "-- CLAUDE.md required metrics " + "-" * 42,
        f"  Max daily loss             ${m['max_daily_loss']:>12,.2f}   "
        f"on {m['max_daily_loss_date']}",
        f"  Worst day as % of profit    {_pct(m['worst_day_pct_of_total_profit']):>12}",
        f"  Average trade duration      {_fmt_duration(m['avg_duration_seconds']):>12}",
        f"  Profit from trades <= {rules.MICROSCALP_SECONDS}s   "
        f"{_pct(m['microscalp_profit_pct']):>12}   "
        f"({m['microscalp_trade_count']} trades)",
    ]

    out += ["", "-- Rule checks " + "-" * 57]

    best_pct = m["best_day_pct_of_total_profit"]
    if m["consistency_breach"] is None:
        out.append(
            f"  [n/a ] Consistency: net P&L is not positive, so best day as a "
            f"share of profit is undefined"
        )
    else:
        flag = "FLAG" if m["consistency_breach"] else " ok "
        out.append(
            f"  [{flag}] Consistency: best day {best_pct:.1f}% of total profit "
            f"(limit {rules.WORST_DAY_FLAG_PCT:.0f}%), {m['best_day_date']} "
            f"${m['best_day']:,.2f}"
        )

    if m["microscalp_breach"] is None:
        out.append("  [n/a ] Microscalping: no winning trades to measure against")
    else:
        flag = "FLAG" if m["microscalp_breach"] else " ok "
        out.append(
            f"  [{flag}] Microscalping: {m['microscalp_profit_pct']:.1f}% of profit from "
            f"trades <= {rules.MICROSCALP_SECONDS}s "
            f"(flag above {rules.MICROSCALP_PROFIT_FLAG_PCT:.0f}%, "
            f"firm limit {rules.MICROSCALP_FIRM_LIMIT_PCT:.0f}%)"
        )

    flag = "FLAG" if m["shortest_hold_violates_rule"] else " ok "
    out.append(
        f"  [{flag}] Minimum hold: shortest trade "
        f"{_fmt_duration(m['min_duration_seconds'])} "
        f"(floor {rules.MIN_HOLD_SECONDS}s)"
    )
    if m["hold_regression"]:
        out.append(
            f"  [REGRESSION] Hold regression: {m['min_hold_violation_count']} trade(s) "
            f"held under {rules.MIN_HOLD_SECONDS}s, {m['microscalp_trade_count']} held "
            f"<= {rules.MICROSCALP_SECONDS}s. Rule 6 enforcement is not working; "
            f"investigate before reading any other figure."
        )
    else:
        out.append(
            f"  [ ok ] Hold regression: none - no trade under the "
            f"{rules.MIN_HOLD_SECONDS}s floor"
        )

    breaches = m["daily_loss_breaches"]
    if len(breaches) == 0:
        out.append(
            f"  [ ok ] Daily loss limit: no session reached "
            f"-${rules.DAILY_LOSS_LIMIT:,.0f}"
        )
    else:
        out.append(
            f"  [FLAG] Daily loss limit: {len(breaches)} session(s) would have hit "
            f"-${rules.DAILY_LOSS_LIMIT:,.0f} and stopped trading:"
        )
        for _, r in breaches.iterrows():
            out.append(
                f"           {r['session_date']}  worst running "
                f"${r['worst_running_pnl']:,.2f}  closed ${r['closing_pnl']:,.2f}"
            )

    out += ["", "-- P&L by exit reason " + "-" * 50]
    by = m["by_exit_reason"]
    out.append(f"  {'reason':<14}{'trades':>8}{'net P&L':>14}{'avg':>12}{'win rate':>11}")
    for reason, r in by.iterrows():
        out.append(
            f"  {reason:<14}{int(r['trades']):>8}{r['net_pnl']:>14,.2f}"
            f"{r['avg_pnl']:>12,.2f}{r['win_rate']:>10.1f}%"
        )

    out.append(line)
    return "\n".join(out)
