"""Entry 11: turn-of-month institutional flows, intraday. Frozen at ``a6d9af1``.

Long MES at the 09:30 open on T-1, T+1, T+2 and T+3 by the cash trading
calendar, flat at the 15:55 open, 4 contracts. Two arms with identical
entries: the signal arm has no stop and measures the drift; the stop arm has
a 15-point stop and is the Lucid-safe form. The control is the open-to-15:55
return of every eligible non-window session.

Build order, fixed by the entry
-------------------------------
1. ``--reproduce`` - pre-registered test 1, run first and gating. The signal
   arm's unguarded one-contract stream at zero cost must have exactly the
   eligible window sessions as its trade dates and reproduce the bar-derived
   open-to-15:55 return on each; the 4-contract stream must be exactly four
   times the 1-contract one; the stop arm's entries must be identical to the
   signal arm's.
2. ``--run`` - the mechanism test on pre-cost returns, both arms under both
   guard bases at 1 tick (base) and 2 ticks (sensitivity), the five kill
   criteria, and a verdict block written into ``results/`` before anyone
   reads a number.

Guards
------
``engine.apply_internal_guards`` at a scalar 4 contracts, in its order: the
daily loss limit marked to market, then the end-of-day trailing halt on the
loss-limited stream. ``trailing_halt=False`` is the comparable basis, reported
alongside; evaluations blown are read there.

    python backtests/run_entry11.py --reproduce
    python backtests/run_entry11.py --run
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import eval_sim  # noqa: E402
import loader  # noqa: E402
import rules  # noqa: E402
from engine import (  # noqa: E402
    MES, CostModel, apply_internal_guards, build_trades, count_evaluation_blowups,
    price_trades,
)
from metrics import compute_metrics  # noqa: E402
from run_london import fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402
from tom import (  # noqa: E402
    CONTROL_LABEL, WINDOW_LABELS, TOMParams, TurnOfMonth, session_returns,
)
from walkforward import build_folds  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS = PROJECT_ROOT / "backtests" / "results"

#: The scored span and the fold years, imported from the walk-forward so a
#: date is never restated.
FOLDS = build_folds()
START, END = FOLDS[0].test_start, FOLDS[-1].test_end
YEARS = [f.test_year for f in FOLDS]
TOTAL_FOLDS = len(YEARS)

#: The operator's specification, as frozen.
CONTRACTS = 4
STOP_POINTS = 15.0
BASE_SLIPPAGE_TICKS = 1.0
SENSITIVITY_TICKS = (2.0,)

ARMS = {"signal": TOMParams(), "stop": TOMParams(stop_points=STOP_POINTS)}

#: Entry 11's kill criteria, as frozen. Any one failure kills the entry.
MIN_T = 2.0
MIN_YEARS_WINDOW_BEATS_CONTROL = 4
MIN_PASS_PROBABILITY = 0.25
MAX_BLOWUPS = 1
MIN_PROFITABLE_FOLDS = 4          # rule 13, with a positive total, at 1 and 2 ticks

LINE = "=" * 96


def round_turn_points(costs: CostModel, spec=MES) -> float:
    """One round turn per contract, in points of the instrument. Computed
    from the cost model the run is given - 0.70 MES points at $0.50 and one
    tick - never written down."""
    dollars = costs.commission_round_turn(1) + costs.slippage_round_turn(spec, 1)
    return dollars / spec.point_value


# ---------------------------------------------------------------------------
# Loading and generating
# ---------------------------------------------------------------------------


@dataclass
class Loaded:
    bars: pd.DataFrame
    win_bars: pd.DataFrame
    rolls: set
    early: set


def load(parquet: Path = PARQUET) -> Loaded:
    bars = loader.load_bars(str(parquet))
    return Loaded(bars=bars, win_bars=slice_by_date(bars, START, END),
                  rolls=set(loader.detect_roll_dates(bars)),
                  early=set(loader.detect_early_close_dates(bars)))


def generate(loaded: Loaded, params: TOMParams):
    """Signals over the full history, sliced to the scored span, plus the
    strategy's diagnostics."""
    strat = TurnOfMonth(params, roll_dates=loaded.rolls, early_close_dates=loaded.early)
    signals = strat.generate_signals(loaded.bars)
    return slice_by_date(signals, START, END), strat.diagnostics


def scored_returns(loaded: Loaded) -> pd.DataFrame:
    """The mechanism test's population: ``tom.session_returns`` over the
    whole file, sliced to the scored span - the same order as the signals.

    Built on the sliced bars instead, the December 2019 boundary is invisible
    and the first three sessions of 2020 lose their labels while the strategy
    trades them. The first reproduction run caught exactly that.
    """
    returns = session_returns(loaded.bars, loaded.rolls, loaded.early)
    keep = (returns["date"] >= START) & (returns["date"] <= END)
    return returns[keep].reset_index(drop=True)


def guarded_streams(win_signals: pd.DataFrame, loaded: Loaded, costs: CostModel):
    """Standard (both guards) and comparable (halt OFF) streams at 4 contracts."""
    raw = price_trades(build_trades(win_signals, loaded.win_bars), MES, costs, CONTRACTS)
    standard, loss_halts, dd_halts = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=True)
    comparable, _, _ = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=False)
    return standard, comparable, len(loss_halts), len(dd_halts)


# ---------------------------------------------------------------------------
# The mechanism test: window against control, pre-cost
# ---------------------------------------------------------------------------


def welch(a, b) -> dict:
    """Welch's unequal-variance t for mean(a) - mean(b), one-sided p for
    a > b. Plain arithmetic so the number on the verdict can be checked by
    hand; scipy supplies only the t distribution."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a = float(a.var(ddof=1)) if n_a > 1 else float("nan")
    var_b = float(b.var(ddof=1)) if n_b > 1 else float("nan")
    se2 = var_a / n_a + var_b / n_b
    diff = mean_a - mean_b
    if se2 > 0:
        t = diff / math.sqrt(se2)
        df = se2 ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        p = float(sps.t.sf(t, df))
    else:
        t, df, p = 0.0, float("nan"), 0.5
    return {"n_a": n_a, "n_b": n_b, "mean_a": mean_a, "mean_b": mean_b,
            "sd_a": math.sqrt(var_a) if np.isfinite(var_a) else float("nan"),
            "sd_b": math.sqrt(var_b) if np.isfinite(var_b) else float("nan"),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "diff": diff, "t": float(t), "df": float(df), "p_one_sided": p}


def mechanism_test(returns: pd.DataFrame) -> dict:
    """Pre-registered tests 2, 3 and 4 on the session-return population.

    ``returns`` is ``tom.session_returns``: one eligible session per row with
    its label and its open-to-15:55 points. The pooled Welch test decides
    criterion 1; the yearly comparison decides criterion 2; the per-label
    table is reported and decides nothing.
    """
    window = returns[returns["window"]]
    control = returns[returns["label"] == CONTROL_LABEL]
    pooled = welch(window["points"], control["points"])

    by_year = []
    for year in sorted(returns["year"].unique()):
        w = window[window["year"] == year]["points"]
        c = control[control["year"] == year]["points"]
        if len(w) < 2 or len(c) < 2:
            continue
        r = welch(w, c)
        by_year.append({"year": int(year), "n_window": r["n_a"], "mean_window": r["mean_a"],
                        "n_control": r["n_b"], "mean_control": r["mean_b"],
                        "diff": r["diff"], "t": r["t"],
                        "window_beats_control": r["diff"] > 0})
    by_year = pd.DataFrame(by_year)

    by_label = []
    for lab in WINDOW_LABELS:
        w = window[window["label"] == lab]["points"]
        if len(w) < 2:
            continue
        r = welch(w, control["points"])
        by_label.append({"label": lab, "n": r["n_a"], "mean": r["mean_a"],
                         "sd": r["sd_a"], "diff": r["diff"], "t": r["t"]})
    by_label = pd.DataFrame(by_label)

    return {"pooled": pooled, "by_year": by_year, "by_label": by_label,
            "years_window_beats_control": int(by_year["window_beats_control"].sum())
            if len(by_year) else 0}


# ---------------------------------------------------------------------------
# The reproduction check (pre-registered test 1)
# ---------------------------------------------------------------------------


FREE = CostModel(commission_per_side=0.0, slippage_ticks=0.0)


def reproduction_differences(loaded: Loaded) -> list[str]:
    """Empty when the signal arm reproduces the session returns exactly."""
    problems: list[str] = []
    returns = scored_returns(loaded)
    window = returns[returns["window"]].sort_values("date")

    sig_a, _ = generate(loaded, ARMS["signal"])
    one = price_trades(build_trades(sig_a, loaded.win_bars), MES, FREE, 1)
    four = price_trades(build_trades(sig_a, loaded.win_bars), MES, FREE, 4)

    ours = list(one["session_date"])
    theirs = list(window["date"])
    if ours != theirs:
        missing = sorted(set(theirs) - set(ours))
        extra = sorted(set(ours) - set(theirs))
        problems.append(f"trade dates differ: {len(ours)} trades against "
                        f"{len(theirs)} window sessions; missing {missing[:5]}, "
                        f"extra {extra[:5]}")
    else:
        gap = np.abs(one["gross_points"].to_numpy() - window["points"].to_numpy())
        if gap.max() > 1e-9:
            where = int(np.argmax(gap))
            problems.append(f"gross points differ on {ours[where]}: trade "
                            f"{one['gross_points'].iloc[where]} against bar return "
                            f"{window['points'].iloc[where]}")
        if not np.allclose(four["gross_pnl"].to_numpy(), 4 * one["gross_pnl"].to_numpy()):
            problems.append("the 4-contract stream is not four times the 1-contract stream")

    sig_b, _ = generate(loaded, ARMS["stop"])
    ea = sig_a[sig_a["entry_long"]]
    eb = sig_b[sig_b["entry_long"]]
    if not ea.index.equals(eb.index):
        problems.append(f"stop arm entries differ from the signal arm's: "
                        f"{len(ea)} against {len(eb)}")
    elif not np.allclose(ea["entry_price"].to_numpy(float), eb["entry_price"].to_numpy(float)):
        problems.append("stop arm entry prices differ from the signal arm's")
    return problems


def reproduce(loaded: Loaded | None = None) -> bool:
    print(LINE)
    print("ENTRY 11 - reproduction check (pre-registered test 1)")
    print("  signal arm, 1 contract, zero cost, unguarded, against tom.session_returns")
    print(LINE, flush=True)
    loaded = loaded or load()
    problems = reproduction_differences(loaded)
    for p in problems:
        print(f"  DIVERGES - {p}")
    if not problems:
        n = int(scored_returns(loaded)["window"].sum())
        print(f"  IDENTICAL: {n} window sessions reproduced session for session; "
              f"4 contracts = 4 x 1 contract; stop-arm entries identical")
    print(LINE)
    print("REPRODUCTION " + ("PASSED" if not problems else "FAILED"))
    print(LINE)
    return not problems


# ---------------------------------------------------------------------------
# Stream statistics, criteria, verdict
# ---------------------------------------------------------------------------


def stream_stats(trades: pd.DataFrame, paths: int, blowup_stream: pd.DataFrame | None,
                 dd_sessions_blocked: int, loss_halts: int) -> dict:
    """Everything pre-registered test 5 reports for one stream, including
    every rule 11 figure."""
    if trades.empty:
        return {"trades": 0, "net_pnl": 0.0, "mean_per_trade": float("nan"),
                "profitable_years": 0, "sharpe": float("nan"),
                "profit_factor": float("nan"), "max_drawdown": 0.0,
                "max_daily_loss": 0.0, "worst_day_pct": None, "best_day_pct": None,
                "best_day": 0.0, "pass_probability": float("nan"),
                "payout_probability": float("nan"), "blowups": 0,
                "dd_sessions_blocked": dd_sessions_blocked, "avg_duration_min": float("nan"),
                "microscalp_pct": 0.0, "trading_days": 0,
                "year_pnl": {y: 0.0 for y in YEARS}, "by_reason": {},
                "loss_limit_exits": 0, "loss_halts": loss_halts}
    m = compute_metrics(trades)
    entry_et = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(rules.ET)
    year_pnl = {int(y): float(g.sum()) for y, g in trades["net_pnl"].groupby(entry_et.dt.year)}
    daily = eval_sim.daily_pnl_from_trades(trades)
    sim = eval_sim.simulate(daily, paths=paths)
    blow = count_evaluation_blowups(blowup_stream if blowup_stream is not None else trades)
    by_reason = trades.groupby(trades["exit_reason"].astype(str))["net_pnl"].agg(
        ["count", "mean", "sum"])
    return {
        "trades": int(len(trades)),
        "net_pnl": float(m["net_pnl"]),
        "mean_per_trade": float(trades["net_pnl"].mean()),
        "profitable_years": int(sum(1 for y in YEARS if year_pnl.get(y, 0.0) > 0)),
        "sharpe": float(m["sharpe"]),
        "profit_factor": float(m["profit_factor"]),
        "max_drawdown": float(m["max_drawdown"]),
        "max_daily_loss": float(m["max_daily_loss"]),
        "worst_day_pct": m["worst_day_pct_of_total_profit"],
        "best_day_pct": m["best_day_pct_of_total_profit"],
        "best_day": float(m["best_day"]),
        "pass_probability": sim.pass_probability,
        "payout_probability": sim.payout_probability,
        "blowups": int(blow["blowups"]),
        "dd_sessions_blocked": dd_sessions_blocked,
        "avg_duration_min": float(m["avg_duration_seconds"]) / 60.0,
        "microscalp_pct": float(m.get("microscalp_profit_pct") or 0.0),
        "trading_days": int(len(daily)),
        "year_pnl": {y: year_pnl.get(y, 0.0) for y in YEARS},
        "by_reason": {k: (int(r["count"]), float(r["mean"]), float(r["sum"]))
                      for k, r in by_reason.iterrows()},
        "loss_limit_exits": int((trades["exit_reason"] == "loss_limit_flatten").sum()),
        "loss_halts": loss_halts,
    }


@dataclass(frozen=True)
class Criterion:
    name: str
    label: str
    shown: str
    threshold: str
    passed: bool


def evaluate_criteria(mechanism: dict, stop_std_1t: dict, stop_cmp_1t: dict,
                      stop_std_2t: dict, round_turn: float) -> list[Criterion]:
    """The five kill criteria, in the entry's order and on the streams it names."""
    pooled = mechanism["pooled"]
    diff, t = float(pooled["diff"]), float(pooled["t"])
    years = int(mechanism["years_window_beats_control"])
    p = float(stop_std_1t["pass_probability"])
    blown = int(stop_cmp_1t["blowups"])

    def rule13(s: dict) -> bool:
        return int(s["profitable_years"]) >= MIN_PROFITABLE_FOLDS and float(s["net_pnl"]) > 0

    return [
        Criterion("excess", "Pooled window excess over control, pre-cost, and its t",
                  f"{diff:+.2f} pts, t = {t:+.2f}",
                  f"> {round_turn:.2f} pts and t >= {MIN_T:.1f}",
                  bool(np.isfinite(t) and diff > round_turn and t >= MIN_T)),
        Criterion("years", "Years in which the window mean beats the control mean",
                  f"{years} of {TOTAL_FOLDS}",
                  f">= {MIN_YEARS_WINDOW_BEATS_CONTROL} of {TOTAL_FOLDS}",
                  years >= MIN_YEARS_WINDOW_BEATS_CONTROL),
        Criterion("pass_probability", "Stop arm, standard stream, 1 tick: eval_sim pass probability",
                  f"{100 * p:.2f}% ({stop_std_1t['trading_days']} trading days)",
                  f">= {100 * MIN_PASS_PROBABILITY:.0f}%",
                  bool(np.isfinite(p) and p >= MIN_PASS_PROBABILITY)),
        Criterion("blowups", "Stop arm, comparable stream, 1 tick: evaluations blown",
                  f"{blown}", f"<= {MAX_BLOWUPS}", blown <= MAX_BLOWUPS),
        Criterion("rule_13", f"Rule 13 on the stop arm, standard stream: >= {MIN_PROFITABLE_FOLDS} "
                  f"of {TOTAL_FOLDS} folds profitable and P&L > 0 at 1 and 2 ticks",
                  f"1 tick {stop_std_1t['profitable_years']} of {TOTAL_FOLDS}, "
                  f"${float(stop_std_1t['net_pnl']):,.0f}; 2 ticks "
                  f"{stop_std_2t['profitable_years']} of {TOTAL_FOLDS}, "
                  f"${float(stop_std_2t['net_pnl']):,.0f}",
                  f">= {MIN_PROFITABLE_FOLDS} of {TOTAL_FOLDS} and > 0, both",
                  rule13(stop_std_1t) and rule13(stop_std_2t)),
    ]


def verdict_status(criteria: list[Criterion]) -> str:
    return "ACCEPTED" if all(c.passed for c in criteria) else "REJECTED"


def _money(x) -> str:
    return f"${x:,.2f}"


def _pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def _pct_plain(x) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


def _basis_table(label: str, std: dict, cmp_: dict) -> list[str]:
    def row(name, key, fmt):
        return f"| {name} | {fmt(std[key])} | {fmt(cmp_[key])} |"
    return [
        f"| {label} | Halt ON (standard) | Halt OFF (comparable) |",
        "|---|---|---|",
        row("Trades", "trades", lambda x: f"{x:,}"),
        row("Trading days", "trading_days", lambda x: f"{x:,}"),
        row("Net P&L", "net_pnl", _money),
        row("Mean per trade", "mean_per_trade", _money),
        row("Folds profitable", "profitable_years", lambda x: f"{x} of {TOTAL_FOLDS}"),
        row("Sharpe", "sharpe", lambda x: f"{x:.2f}"),
        row("Profit factor", "profit_factor", lambda x: f"{x:.3f}"),
        row("Max drawdown", "max_drawdown", _money),
        row("Max daily loss (worst day)", "max_daily_loss", _money),
        row("Worst day as % of total profit", "worst_day_pct", _pct_plain),
        row("Best day as % of total profit (rule 8)", "best_day_pct", _pct_plain),
        row("Pass probability", "pass_probability", _pct),
        row("Payout probability ($52,100)", "payout_probability", _pct),
        f"| Evaluations blown (read on the comparable basis) | {cmp_['blowups']} | {cmp_['blowups']} |",
        f"| Sessions blocked by the trailing halt | {std['dd_sessions_blocked']} | — |",
        row("Daily-loss flattens", "loss_limit_exits", lambda x: f"{x}"),
        row("Avg trade duration", "avg_duration_min", lambda x: f"{x:.1f} min"),
        row("Profit from <=5s holds", "microscalp_pct", lambda x: f"{x:.2f}%"),
    ]


def verdict_block(results: dict, date_text: str) -> str:
    status = results["status"]
    mech = results["mechanism"]
    pooled = mech["pooled"]
    rt1 = results.get("round_turn_1t", round_turn_points(CostModel(slippage_ticks=BASE_SLIPPAGE_TICKS)))
    lines = [f"### Verdict: {status}", ""]
    if status == "ACCEPTED":
        lines.append(
            "**The effect was measured on the intraday 09:30-to-15:55 slice only, "
            "and its cost basis rests on a slippage assumption no backtest can "
            "measure.** Both were pre-registered above; an ACCEPTED here is "
            "conditional on both and says so on its face.")
    else:
        lines.append(
            "**Rejected on the pre-registered criteria.** The intraday slice of "
            "the turn-of-month effect, on MES over 2020–2026, did not clear the "
            "lines the operator set before the run.")
    lines += [
        "",
        f"**Date:** {date_text}",
        "**Code:** `strategies/tom.py`, `backtests/run_entry11.py`, "
        "`research/power_check_tom.py`",
        "**Reports:** `backtests/results/entry11_report.txt`; `entry11_returns.csv` "
        "(every eligible session with its label and open-to-15:55 points); per arm "
        "`entry11_<signal|stop>_slip<1|2>.csv` (standard) and `_nohalt.csv` "
        "(comparable); `entry11_folds.csv` (stop arm, 1 tick, standard)",
        "",
        f"Long MES at the 09:30 open on T-1, T+1, T+2, T+3 by the cash trading "
        f"calendar, flat at the 15:55 open, {CONTRACTS} contracts, "
        f"${rules.COMMISSION_PER_SIDE:.2f} a side, **{BASE_SLIPPAGE_TICKS:g} tick a side "
        f"as the base case** ({rt1:.2f} points a round turn), {START} .. {END}, "
        f"nothing selected. Guards: `engine.apply_internal_guards` at {CONTRACTS} "
        f"contracts, the daily loss limit then the trailing halt; halt OFF alongside.",
        "",
        "#### Kill criteria — any one failure kills the entry",
        "",
        "| # | Criterion | Result | Line | |",
        "|---|---|---|---|---|",
    ]
    for i, c in enumerate(results["criteria"], start=1):
        lines.append(f"| {i} | {c.label} | {c.shown} | {c.threshold} | "
                     f"**{'PASS' if c.passed else 'FAIL'}** |")
    failed = [c.label for c in results["criteria"] if not c.passed]
    lines += ["", f"**{status}.** " + ("Every criterion passed." if not failed else
                                       f"Failed on: {'; '.join(failed)}."), ""]

    lines += [
        "#### The mechanism test, pre-cost, in MES points",
        "",
        f"**Pooled:** window n = {pooled['n_a']:,}, mean {pooled['mean_a']:+.2f}, "
        f"sd {pooled['sd_a']:.2f}, median {pooled['median_a']:+.2f}; control n = "
        f"{pooled['n_b']:,}, mean {pooled['mean_b']:+.2f}, sd {pooled['sd_b']:.2f}, "
        f"median {pooled['median_b']:+.2f}. **Difference {pooled['diff']:+.2f} points, "
        f"Welch t = {pooled['t']:+.2f}** (df {pooled['df']:.0f}), one-sided p = "
        f"{pooled['p_one_sided']:.3f}. One round turn at the base case is {rt1:.2f} points.",
        "",
        "| Year | Window n | Window mean | Control n | Control mean | Difference | t | Window beats control |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in mech["by_year"].iterrows():
        lines.append(f"| {int(r['year'])} | {int(r['n_window'])} | {r['mean_window']:+.2f} | "
                     f"{int(r['n_control'])} | {r['mean_control']:+.2f} | {r['diff']:+.2f} | "
                     f"{r['t']:+.2f} | {'yes' if r['window_beats_control'] else 'no'} |")
    lines += [
        "",
        f"Window beats control in **{mech['years_window_beats_control']} of "
        f"{TOTAL_FOLDS}** years.",
        "",
        "**Per window day, reported and not selected on:**",
        "",
        "| Day | n | Mean | sd | Difference vs control | t |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in mech["by_label"].iterrows():
        lines.append(f"| {r['label']} | {int(r['n'])} | {r['mean']:+.2f} | {r['sd']:.2f} | "
                     f"{r['diff']:+.2f} | {r['t']:+.2f} |")

    for arm, title in (("stop", f"Stop arm ({STOP_POINTS:g}-point stop), the Lucid-safe form"),
                       ("signal", "Signal arm (no stop), the mechanism's trade stream")):
        for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
            r = results["arms"][arm][ticks]
            tag = "base case" if ticks == BASE_SLIPPAGE_TICKS else "sensitivity"
            lines += ["", f"#### {title} — {ticks:g} tick{'s' if ticks != 1 else ''} ({tag})", ""]
            lines += _basis_table(f"2020–2026, {CONTRACTS} contracts, {ticks:g} "
                                  f"tick{'s' if ticks != 1 else ''}, "
                                  f"${rules.COMMISSION_PER_SIDE:.2f}",
                                  r["standard"], r["comparable"])
            std = r["standard"]
            reasons = "; ".join(f"{k}: {n} trades, mean {_money(mu)}, total {_money(tot)}"
                                for k, (n, mu, tot) in sorted(std["by_reason"].items()))
            lines += ["", f"Exits, standard stream: {reasons or 'none'}. Daily-loss halts: "
                          f"{std['loss_halts']}.",
                      "Per fold, standard: " + "; ".join(
                          f"{y} {_money(std['year_pnl'].get(y, 0.0))}" for y in YEARS) + "."]

    diag = results["diagnostics"]
    skipped = diag["skipped"]
    lines += ["", "#### Diagnostics", "",
              f"Window sessions skipped in the scored span: {len(skipped)}"
              + (" (" + ", ".join(f"{r['date']} {r['label']} {r['skipped_reason']}"
                                  for _, r in skipped.iterrows()) + ")" if len(skipped) else "")
              + f". Entry-bar stop breaches on the stop arm: {diag['entry_bar_breaches']}.",
              "",
              "In-sample results are never evidence, and this is the out-of-sample "
              "answer on seven yearly folds with nothing selected." if status == "REJECTED"
              else "Every criterion is satisfied. The next step is `testing`, not "
                   "`paper`; the registry gate is unchanged.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(paths: int = 20_000, commission: float | None = None, progress=print):
    RESULTS.mkdir(parents=True, exist_ok=True)
    commission = rules.COMMISSION_PER_SIDE if commission is None else commission
    progress("[progress] loading bars")
    loaded = load()

    progress("[progress] session returns and the mechanism test")
    returns = scored_returns(loaded)
    returns.to_csv(RESULTS / "entry11_returns.csv", index=False)
    mechanism = mechanism_test(returns)

    arms: dict = {}
    diagnostics: dict = {}
    for arm, params in ARMS.items():
        progress(f"[progress] {arm} arm: generating signals")
        win_signals, diag = generate(loaded, params)
        if arm == "stop":
            oos = diag[[d.year in YEARS for d in diag.index]]
            diagnostics["entry_bar_breaches"] = int(oos["entry_bar_breach"].fillna(False).sum())
            sk = oos[oos["entered"] != True]  # noqa: E712
            diagnostics["skipped"] = pd.DataFrame(
                {"date": sk.index, "label": sk["label"].to_numpy(),
                 "skipped_reason": sk["skipped_reason"].to_numpy()})
        arms[arm] = {}
        for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
            progress(f"[progress] {arm} arm: pricing and guarding at {ticks:g} tick(s)")
            costs = CostModel(commission_per_side=commission, slippage_ticks=ticks)
            standard, comparable, loss_halts, dd_halts = guarded_streams(
                win_signals, loaded, costs)
            tag = f"entry11_{arm}_slip{ticks:g}"
            standard.to_csv(RESULTS / f"{tag}.csv", index=False)
            comparable.to_csv(RESULTS / f"{tag}_nohalt.csv", index=False)
            arms[arm][ticks] = {
                "standard": stream_stats(standard, paths, comparable, dd_halts, loss_halts),
                "comparable": stream_stats(comparable, paths, None, 0, loss_halts),
            }
            if arm == "stop" and ticks == BASE_SLIPPAGE_TICKS:
                fold_frame(standard, paths).to_csv(RESULTS / "entry11_folds.csv", index=False)

    rt1 = round_turn_points(CostModel(commission_per_side=commission,
                                      slippage_ticks=BASE_SLIPPAGE_TICKS))
    criteria = evaluate_criteria(
        mechanism,
        arms["stop"][BASE_SLIPPAGE_TICKS]["standard"],
        arms["stop"][BASE_SLIPPAGE_TICKS]["comparable"],
        arms["stop"][SENSITIVITY_TICKS[0]]["standard"],
        rt1,
    )
    results = {"criteria": criteria, "status": verdict_status(criteria),
               "mechanism": mechanism, "arms": arms, "diagnostics": diagnostics,
               "round_turn_1t": rt1}
    block = verdict_block(results, str(pd.Timestamp.now(tz=rules.ET).date()))
    (RESULTS / "entry11_verdict.md").write_text(block, encoding="utf-8")
    (RESULTS / "entry11_report.txt").write_text(
        "\n".join([LINE, "ENTRY 11 - turn-of-month institutional flows, intraday",
                   f"  ${commission:.2f}/side; base {BASE_SLIPPAGE_TICKS:g} tick/side; "
                   f"sensitivity {', '.join(f'{t:g}' for t in SENSITIVITY_TICKS)}",
                   LINE, block]), encoding="utf-8")
    progress("[progress] verdict written to backtests/results/entry11_verdict.md")
    return results, results["status"], block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", action="store_true",
                    help="pre-registered test 1, run first")
    ap.add_argument("--run", action="store_true",
                    help="the mechanism test, both arms, the verdict")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--commission", type=float, default=rules.COMMISSION_PER_SIDE)
    args = ap.parse_args(argv)
    if args.reproduce:
        return 0 if reproduce() else 1
    if args.run:
        _, status, _ = run(args.paths, args.commission,
                           progress=lambda m: print(m, flush=True))
        print(f"[done] {status}", flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
