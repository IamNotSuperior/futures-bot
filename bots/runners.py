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
for _folder in ("journal", "backtests", "data", "strategies",
                "strategies/generated", "bots"):
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
    """How to build and run one registry strategy.

    ``build`` is optional. A strategy that sizes per session - entry 6 sets
    contracts from the overnight range height - cannot be replayed through
    ``engine.price_trades``, which takes one scalar ``contracts`` for the whole
    trade list. Those entries carry ``build=None``, which makes ``/backtest``
    refuse with a reason rather than quietly reporting a uniform-size run that
    was never the strategy. ``/walkforward`` and ``/evalsim`` read saved
    outputs and work regardless.
    """
    bar_minutes: int      # 1 means the raw 1-minute bars
    contracts: int
    build: object | None  # (bars, roll_dates, early_closes) -> Strategy
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


def _tom(bars, roll_dates, early_closes):
    from run_entry11 import ARMS
    from tom import TurnOfMonth
    return TurnOfMonth(ARMS["stop"], roll_dates=roll_dates,
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
    # Entry 6 and 7. Read-only: build is None because these size per session
    # from the overnight range, which the scalar-contracts engine cannot
    # replay. See the Runner docstring.
    #
    # All three are the 2-tick arm. That is entry 6's pre-registered BASE
    # CASE, not the optimistic sensitivity, and it is the arm each verdict
    # quotes - london_1x reproduces -$7,702 with 1 of 7 folds, london_2x
    # -$20,781 with 2 of 7, entry 7's MNQ +$116 with 4 of 7. Pointing these at
    # the slip1 files would show numbers that appear in no verdict.
    "london_1x": Runner(1, 0, None, "london_1x_on_folds_slip2.csv",
                        "london_1x_on_slip2.csv",
                        note="entry 6, 1x target, filter ON, 2 ticks/side "
                             "(base case); risk-based per-session sizing"),
    "london_2x": Runner(1, 0, None, "london_2x_off_folds_slip2.csv",
                        "london_2x_off_slip2.csv",
                        note="entry 6, 2x target, filter OFF, 2 ticks/side - "
                             "the unfiltered arm the verdict quotes"),
    "london_mnq_replication": Runner(1, 0, None, "entry7_mnq_folds.csv",
                                     "entry7_mnq_1x_on.csv",
                                     note="entry 7, MNQ, entry 6's 1x/ON spec "
                                          "reproduced unchanged, 2 ticks/side"),
    # Entry 10. Same per-session sizing, so build is None. The MES files are
    # the verdict's base case - 2 ticks, $0.50, the guarded stream; MNQ's
    # equivalents sit beside them as entry10_mnq_*.
    "london_full_day": Runner(1, 0, None, "entry10_mes_folds.csv",
                              "entry10_mes_slip2.csv",
                              note="entry 10, entry 6's 1x/ON held to 15:55, "
                                   "MES base case (2 ticks/side, $0.50, "
                                   "trailing halt ON); MNQ in entry10_mnq_*"),
    # Entry 11. Scalar 4 contracts, so build works. The saved files are the
    # verdict's stop arm at the base case - 1 tick, $0.50, the guarded stream,
    # which the halt ends after 11 trades; the halt-OFF stream and the signal
    # arm sit beside them as entry11_*_nohalt and entry11_signal_*.
    "tom_intraday": Runner(1, 4, _tom, "entry11_folds.csv", "entry11_stop_slip1.csv",
                           note="entry 11, turn-of-month long 09:30-15:55, "
                                "15-point stop arm, 4 contracts, 1 tick/side "
                                "(base case), trailing halt ON"),
    # Entry 12, Part A. Read-only: build is None because this runs on MNQ
    # bars with the MNQ spec and a stop derived from MNQ's control sd, and
    # /backtest replays on the MES parquet at the MES spec. The saved files
    # are the verdict's stop arm at the base case, guarded stream.
    "tom_intraday_mnq": Runner(1, 1, None, "entry12_a_folds.csv", "entry12_a_stop_slip1.csv",
                               note="entry 12 Part A, entry 11's test on MNQ, one "
                                    "contract, 69.5-point stop (0.366 x control sd), "
                                    "1 tick/side, trailing halt ON; Part B never run"),
}


def registry() -> Registry:
    return Registry.load()


def known_strategies() -> list[str]:
    import submissions  # noqa: PLC0415

    return [n for n in registry().names()
            if n in RUNNERS or submissions.is_generated(n)]


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
    if run.build is None:
        raise WorkError(
            f"`{name}` cannot be re-run here: it sizes per session, and the "
            f"backtest engine takes one contract count for the whole trade "
            f"list. Reporting it at a uniform size would be a different "
            f"strategy from the one that was tested. "
            f"Use `/walkforward {name}` or `/evalsim {name}`, which read the "
            f"saved run.{(' ' + run.note) if run.note else ''}"
        )
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


def run_walkforward_generated(name: str, progress=None):
    """Actually run the folds for a generated strategy. Returns the result.

    Generated strategies have no saved output to read, so this is the one path
    that computes. It is deliberately separate from :func:`run_walkforward` -
    the existing entries' verdicts are frozen in the log, and a command that
    recomputed them could report a number no verdict ever said.
    """
    import submissions  # noqa: PLC0415

    if not submissions.is_generated(name):
        raise WorkError(f"`{name}` is not a generated strategy")
    reg = registry()
    if name not in reg.names():
        # A module left behind by a submission that never reached review.
        # `is_generated` only asks whether the file exists, so this is
        # reachable by typing the name; without the guard it is a KeyError
        # traceback rather than an answer.
        raise WorkError(
            f"`{name}` has a generated module on disk but no registry record - "
            f"its submission never reached review. Re-submit it with "
            f"`/submit {name}`; the retry reuses the existing hypothesis entry."
        )
    record = reg.get(name)
    if record.status not in ("testing", "paper", "live"):
        raise WorkError(
            f"`{name}` is at `{record.status}` status. A walk-forward runs "
            f"after approval moves it to `testing`."
        )
    sys.path.insert(0, str(PROJECT_ROOT / "strategies" / "generated"))
    import run_generated  # noqa: PLC0415

    try:
        return run_generated.run(name, record.class_path, progress=progress)
    except (ValueError, FileNotFoundError) as exc:
        raise WorkError(str(exc)) from None


def run_walkforward(name: str) -> str:
    """The saved per-fold table. Never recomputed - see the module docstring."""
    import submissions  # noqa: PLC0415

    if submissions.is_generated(name):
        raise WorkError(
            f"`{name}` is a generated strategy - its walk-forward runs rather "
            f"than being read from a cached table. Use the /walkforward "
            f"command, which dispatches it as a background job."
        )
    run = _runner(name)
    if not run.walkforward_csv:
        raise WorkError(
            f"`{name}` has no walk-forward run. {run.note or ''}".strip()
        )
    path = RESULTS / run.walkforward_csv
    if not path.exists():
        how = {
            "london_1x": "backtests\\run_london.py --folds-only",
            "london_2x": "backtests\\run_london.py --folds-only",
            "london_mnq_replication": "backtests\\run_entry7.py --folds-only",
        }.get(name, "backtests\\walkforward.py")
        raise WorkError(
            f"no saved walk-forward for `{name}` - expected "
            f"`backtests/results/{run.walkforward_csv}`. Regenerate with "
            f"`venv\\Scripts\\python.exe {how}`. This command reads the saved "
            f"output rather than starting a run: the ORB walk-forward alone "
            f"takes about 55 minutes."
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
        "Payout probability": f"{result.payout_probability:.2%}",
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


# ---------------------------------------------------------------------------
# /read
# ---------------------------------------------------------------------------


def run_read(symbol: str = "MES", timeframe: str = "5m", now=None):
    """A chart read. Returns ``(ChartRead, embed fields)``.

    The calendar comes from ``pretrade.calendar_dates`` - the same three
    sources the manual pre-trade check uses - so a roll day or an early close
    reads identically here and there.
    """
    import chart_read  # noqa: PLC0415
    import pretrade  # noqa: PLC0415

    symbol = (symbol or "MES").strip().upper()
    if not rules.is_allowed_instrument(symbol):
        raise WorkError(
            f"`{symbol}` is not permitted; rule 1 allows "
            f"{', '.join(sorted(rules.ALLOWED_INSTRUMENTS))}"
        )
    now = now or pd.Timestamp.now(tz=rules.ET)
    early, rolls, closed, _ = pretrade.calendar_dates(False, rules.session_date(now))
    try:
        result = chart_read.read(symbol, timeframe, now=now,
                                 early_close_dates=early, roll_dates=rolls,
                                 closed_dates=closed)
    except chart_read.ReadError as exc:
        raise WorkError(str(exc)) from None
    return result, chart_read.format_read(result)


def status_text() -> str:
    reg = registry()
    lines = ["**Strategy registry**", "```", reg.status_table(), "```"]
    return "\n".join(lines)[:1900]
