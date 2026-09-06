"""The work the research bot does, with no Discord in it.

Everything here is a plain function that takes strings and returns data. The
Discord layer in ``bots/research.py`` is a thin shell over these, which is what
makes them testable without a gateway connection and what lets the bot run them
on a worker thread.

Nothing here re-derives a threshold or a verdict. Walk-forward and evaluation
figures are read from the saved run outputs in ``backtests/results/`` rather
than recomputed: the ORB walk-forward alone takes about 55 minutes, and a chat
command that silently kicks off an hour of work is a worse answer than one that
says when the cached run was produced.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on this box, and none wanted
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("journal", "backtests", "data", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import loader  # noqa: E402
import rules  # noqa: E402
import eval_sim  # noqa: E402
from engine import (  # noqa: E402
    MES, CostModel, build_trades, count_evaluation_blowups,
    enforce_daily_loss_limit, equity_curve_by_day, price_trades,
)
from metrics import compute_metrics  # noqa: E402
from registry import Registry, parse_hypotheses  # noqa: E402
from scan import slice_by_date  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS = PROJECT_ROOT / "backtests" / "results"

# matplotlib reads paired dollar signs as mathtext and silently italicises the
# label. Every chart in this repo turns that off; this one is no exception.
plt.rcParams["text.parse_math"] = False

_BARS_CACHE: dict[str, object] = {}


class WorkError(RuntimeError):
    """Something the user should see as a message rather than a traceback."""


@dataclass(frozen=True)
class Runner:
    """How to build and run one registry strategy."""
    bar_minutes: int      # 1 means the raw 1-minute bars
    contracts: int
    build: object         # (bars, roll_dates, early_closes) -> Strategy
    walkforward_csv: str | None
    oos_csv: str | None
    note: str = ""


def _orb(bars, roll_dates, early_closes):
    from orb import ORBParams, OpeningRangeBreakout
    return OpeningRangeBreakout(ORBParams(), roll_dates=roll_dates,
                                early_close_dates=early_closes)


def _eod(bars, roll_dates, early_closes):
    from eod_rebalance import EODParams, EODRebalanceDrift
    return EODRebalanceDrift(EODParams(), roll_dates=roll_dates,
                             early_close_dates=early_closes)


def _orb2(bars, roll_dates, early_closes):
    from orb2 import ORB2, ORB2Params
    from trend import trend_filter
    return ORB2(ORB2Params(use_trend_filter=True), trend_ema=trend_filter(bars),
                roll_dates=roll_dates, early_close_dates=early_closes)


def _orb_flat(bars, roll_dates, early_closes):
    from orb2 import ORB2
    from run_orb_flat import entry5_params
    return ORB2(entry5_params(), roll_dates=roll_dates,
                early_close_dates=early_closes)


RUNNERS: dict[str, Runner] = {
    "orb": Runner(5, 1, _orb, "orb_walkforward_slip1.csv",
                  "orb_oos_stream_slip1.csv"),
    "eod_rebalance": Runner(5, 1, _eod, "eod_walkforward_slip1.csv",
                            "eod_oos_stream_slip1.csv"),
    "orb2": Runner(1, 4, _orb2, "orb2_folds_on_4c_slip1.csv",
                   "orb2_oos_on_4c_slip1.csv",
                   note="entry 4's filter-ON arm at 4 contracts"),
    "orb_flat_1030": Runner(1, 4, _orb_flat, None, "orb_flat_7yr_slip1.csv",
                            note="entry 5, guards active; no fold table - "
                                 "entry 5 has no walk-forward, only OOS years"),
}


def registry() -> Registry:
    return Registry.load()


def known_strategies() -> list[str]:
    return [n for n in registry().names() if n in RUNNERS]


def _runner(name: str) -> Runner:
    if name not in RUNNERS:
        known = ", ".join(known_strategies())
        raise WorkError(
            f"`{name}` has no runnable configuration. Runnable: {known}."
            + (" `manual_discretionary` is a log entry, not code."
               if name == "manual_discretionary" else "")
        )
    return RUNNERS[name]


def _bars():
    if "bars" not in _BARS_CACHE:
        if not PARQUET.exists():
            raise WorkError(f"bar data not found at {PARQUET.name}")
        bars = loader.load_bars(str(PARQUET))
        _BARS_CACHE["bars"] = bars
        _BARS_CACHE["rolls"] = loader.detect_roll_dates(bars)
        _BARS_CACHE["early"] = loader.detect_early_close_dates(bars)
    return _BARS_CACHE["bars"], _BARS_CACHE["rolls"], _BARS_CACHE["early"]


def parse_day(text: str, label: str) -> date:
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise WorkError(f"{label} must look like 2024-09-01, got `{text}`") from None


# ---------------------------------------------------------------------------
# /backtest
# ---------------------------------------------------------------------------


def run_backtest(name: str, start_text: str, end_text: str) -> tuple[dict, bytes]:
    """Metrics and an equity-curve PNG for one strategy over one window."""
    run = _runner(name)
    start, end = parse_day(start_text, "start"), parse_day(end_text, "end")
    if start > end:
        raise WorkError(f"start {start} is after end {end}")

    bars, rolls, early = _bars()
    strat = run.build(bars, rolls, early)

    if run.bar_minutes == 1:
        signal_bars = bars.between_time("09:30", "15:59")
        signals = strat.generate_signals(bars)
    else:
        from orb import resample_bars
        signal_bars = resample_bars(bars, run.bar_minutes)
        signals = strat.generate_signals_resampled(signal_bars)

    win_signals = slice_by_date(signals, start, end)
    win_bars = slice_by_date(signal_bars, start, end)
    costs = CostModel()
    trades = price_trades(build_trades(win_signals, win_bars), MES, costs,
                          run.contracts)
    trades, halts = enforce_daily_loss_limit(
        trades, win_bars, MES, costs, run.contracts, rules.DAILY_LOSS_LIMIT
    )
    if trades.empty:
        raise WorkError(
            f"`{name}` took no trades between {start} and {end}."
        )

    m = compute_metrics(trades)
    blow = count_evaluation_blowups(trades)
    fields = {
        "Window": f"{start} to {end}",
        "Contracts": f"{run.contracts}",
        "Trades": f"{m['trade_count']:,}",
        "Net P&L": f"${m['net_pnl']:,.2f}",
        "Gross P&L": f"${m['gross_pnl']:,.2f}",
        "Commission + slippage":
            f"${m['total_commission'] + m['total_slippage']:,.2f}",
        "Win rate": f"{m['win_rate_pct']:.2f}%",
        "Profit factor": f"{m['profit_factor']:.3f}",
        "Sharpe": f"{m['sharpe']:.2f}",
        "Max drawdown": f"${m['max_drawdown']:,.2f}",
        "Worst day": f"${m['max_daily_loss']:,.2f} on {m['max_daily_loss_date']}",
        "Best day": f"${m['best_day']:,.2f} on {m['best_day_date']}",
        "Avg duration": f"{m['avg_duration_seconds'] / 60:.1f} min",
        "Profit from <=5s holds": f"{m.get('microscalp_profit_pct', 0.0):.2f}%",
        "Daily-loss halts": f"{len(halts)}",
        "Evaluations blown": f"{blow['blowups']}",
    }
    return fields, equity_png(trades, f"{name}  {start} to {end}")


def equity_png(trades: pd.DataFrame, title: str) -> bytes:
    """Cumulative net P&L by session date, as PNG bytes."""
    equity = equity_curve_by_day(trades)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=140)
    if "session_date" in equity.columns and "balance" in equity.columns:
        x = pd.to_datetime(equity["session_date"])
        y = equity["balance"] - equity["balance"].iloc[0]
    else:  # fall back to a trade-ordered curve
        x = pd.to_datetime(trades["exit_time"])
        y = trades["net_pnl"].cumsum()
    ax.plot(x, y, linewidth=1.4, color="#1f77b4")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.fill_between(x, y, 0, where=(y >= 0), color="#1f77b4", alpha=0.15)
    ax.fill_between(x, y, 0, where=(y < 0), color="#d62728", alpha=0.15)
    ax.set_title(title)
    ax.set_ylabel("cumulative net P&L (USD)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /walkforward
# ---------------------------------------------------------------------------


def run_walkforward(name: str) -> str:
    """The saved per-fold table. Never recomputed - see the module docstring."""
    run = _runner(name)
    if not run.walkforward_csv:
        raise WorkError(
            f"`{name}` has no walk-forward run. {run.note or ''}".strip()
        )
    path = RESULTS / run.walkforward_csv
    if not path.exists():
        raise WorkError(
            f"no saved walk-forward for `{name}` - expected "
            f"`backtests/results/{run.walkforward_csv}`. Run it from the CLI; "
            f"the ORB one takes about 55 minutes, which is why this command "
            f"reads the saved output rather than starting one."
        )
    df = pd.read_csv(path)
    produced = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    lines = [f"{'year':>6}{'trades':>8}{'net P&L':>12}{'Sharpe':>9}"
             f"{'PF':>7}{'maxDD':>11}  flag"]
    lines.append("-" * 60)
    for _, r in df.iterrows():
        flag = "low-confidence" if r.get("low_confidence") else ""
        lines.append(
            f"{int(r['test_year']):>6}{int(r['test_trades']):>8}"
            f"{r['test_net_pnl']:>12,.0f}{r['test_sharpe']:>9.2f}"
            f"{r['test_profit_factor']:>7.2f}{r['test_max_drawdown']:>11,.0f}"
            f"  {flag}"
        )
    total = df["test_net_pnl"].sum()
    good = int((df["test_net_pnl"] > 0).sum())
    lines += [
        "-" * 60,
        f"  total {total:>+,.2f}   folds profitable {good} of {len(df)}"
        f"   median fold Sharpe {df['test_sharpe'].median():+.3f}",
    ]
    header = f"**{name}** walk-forward{(' - ' + run.note) if run.note else ''}\n"
    footer = f"\nfrom `{run.walkforward_csv}`, produced {produced}"
    return header + "```\n" + "\n".join(lines) + "\n```" + footer


# ---------------------------------------------------------------------------
# /evalsim
# ---------------------------------------------------------------------------


def run_evalsim(name: str, paths: int = 20_000) -> dict:
    """Pass probability and expected attempts from the saved OOS trade stream."""
    run = _runner(name)
    if not run.oos_csv:
        raise WorkError(f"`{name}` has no saved out-of-sample stream.")
    path = RESULTS / run.oos_csv
    if not path.exists():
        raise WorkError(
            f"no saved out-of-sample stream for `{name}` - expected "
            f"`backtests/results/{run.oos_csv}`."
        )
    trades = pd.read_csv(path)
    if "net_pnl" not in trades.columns:
        raise WorkError(f"`{run.oos_csv}` has no net_pnl column")

    daily = eval_sim.daily_pnl_from_trades(trades)
    result = eval_sim.simulate(daily, paths=paths)
    blow = count_evaluation_blowups(trades)
    produced = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {
        "Source": f"{run.oos_csv} ({len(trades):,} trades, produced {produced})",
        "Trading days": f"{len(daily):,}",
        "Mean day": f"${daily.mean():,.2f}",
        "Day sd": f"${daily.std(ddof=1):,.2f}",
        "Pass probability": f"{result.pass_probability:.2%}",
        "95% CI": f"{result.ci_low:.2%} - {result.ci_high:.2%}",
        "Blow-up probability": f"{result.blowup_probability:.2%}",
        "Ran out of time": f"{result.timeout_probability:.2%}",
        "Expected attempts": f"{result.expected_attempts:.2f}",
        "Expected cost to pass": f"${result.expected_cost_to_pass:,.0f}",
        "Evaluations blown (historical)": f"{blow['blowups']}",
    }


# ---------------------------------------------------------------------------
# /hypotheses and /status
# ---------------------------------------------------------------------------


def hypothesis_text(number: int | None = None) -> str:
    entries = parse_hypotheses()
    if number is None:
        lines = ["**Hypothesis log**", "```"]
        for n in sorted(entries):
            e = entries[n]
            lines.append(f"{n}. {e.title[:52]:<52} {e.status}")
        lines.append("```")
        return "\n".join(lines)

    entry = entries.get(number)
    if entry is None:
        raise WorkError(
            f"no entry {number}; the log has {sorted(entries)}"
        )
    out = [f"**Entry {entry.number}. {entry.title} — {entry.status}**"]
    if entry.verdict_commit:
        out.append(f"verdict commit `{entry.verdict_commit}`")
    out.append("")
    out.append(entry.summary())
    verdict = _verdict_paragraph(entry.body)
    if verdict:
        out += ["", "**Verdict**", verdict]
    return "\n".join(out)[:1900]


def _verdict_paragraph(body: str) -> str:
    """The first paragraph under a verdict heading, if the entry has one."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("###") and "verdict" in line.lower():
            for block in "\n".join(lines[i + 1:]).split("\n\n"):
                text = block.strip()
                if text and not text.startswith(("**Date", "**Spec", "**Code",
                                                 "**Note", "**Verdict commit",
                                                 "**Reports")):
                    return text[:600] + ("..." if len(text) > 600 else "")
    return ""


def status_text() -> str:
    reg = registry()
    lines = ["**Strategy registry**", "```", reg.status_table(), "```"]
    return "\n".join(lines)[:1900]
