"""Entry 10: London 1x/ON held through the US session. Frozen at ``fd0c5c8``.

The entry's specification is entry 6's 1x/ON arm with two changes: the flatten
moves from 09:25 to 15:55 ET (12:55 on early-close sessions) and sizing follows
the log's corrected realised-stop rule. Both instruments, MES and MNQ, run the
identical configuration and both must pass every kill criterion.

Build order, fixed by the entry and by the operator
---------------------------------------------------
1. ``--reproduce`` - the entry's ninth pre-registered test, run first. The new
   code, with the flatten at 09:25, costs at $1.25 and entry 6's range-based
   sizing switched on for this check only, must reproduce entry 6's 1x/ON
   stream on MES (686 trades, -$7,702.50) and entry 7's on MNQ (438, +$116.00)
   trade for trade. Passed on both instruments on 2026-09-11.
2. ``--run`` - the 15:55 test, authorised by the operator after (1). Base case
   2 ticks per side at the confirmed $0.50 commission, 1 and 3 ticks as
   sensitivities, the de-meaned bootstrap at the 15:55 horizon, the four kill
   criteria on both instruments, and a verdict block that is written into the
   entry and committed before anyone reads a number.

Guards
------
Size varies per session, so ``engine.apply_internal_guards`` (a scalar
``contracts``) is not called directly. The two engine functions it composes
are applied in its order: rule 5's daily loss limit per size group
(``run_london.apply_daily_limit``, exact given one trade a day), then rule
5b's trailing halt on the combined stream (``engine.enforce_trailing_
drawdown_halt``). The unhalted stream is the comparable basis, reported
alongside; evaluation blow-ups are read there, as entry 5 established.

    python backtests/run_entry10.py --reproduce
    python backtests/run_entry10.py --run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import bootstrap_benchmark as bb  # noqa: E402
import eval_sim  # noqa: E402
import loader  # noqa: E402
import rules  # noqa: E402
from engine import (  # noqa: E402
    MES, MNQ, CostModel, build_trades, count_evaluation_blowups,
    enforce_trailing_drawdown_halt, price_trades,
)
from london import LondonBreakout, LondonParams  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from run_london import apply_daily_limit, fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402
from trend import intraday_ema  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"
START, END = date(2020, 1, 1), date(2026, 8, 31)
YEARS = list(range(2020, 2027))
TOTAL_FOLDS = len(YEARS)

INSTRUMENTS = {
    "MES": dict(parquet="mes_v_0_ohlcv_1m_2019-05_2026-08.parquet", spec=MES,
                saved="entry7_mes_1x_on.csv"),
    "MNQ": dict(parquet="mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet", spec=MNQ,
                saved="entry7_mnq_1x_on.csv"),
}

#: Entry 6's base case and rule 13's survival bar; the sensitivities the
#: operator asked for around it.
BASE_SLIPPAGE_TICKS = 2.0
SENSITIVITY_TICKS = (1.0, 3.0)
FLATTEN = time(15, 55)
EARLY_CLOSE_FLATTEN = time(12, 55)

#: Entry 10's kill criteria, as frozen. Every line must pass on both
#: instruments; a failure on either kills the entry as a whole.
MIN_PROFITABLE_FOLDS = 4          # rule 13, with a positive total
MIN_PASS_PROBABILITY = 0.35       # entry 6 asked 25%
CRITICAL_Z = 2.5                  # entry 7 asked 1.65
MIN_DEPARTURE_POINTS = 2.0        # carried from entry 7

#: What entries 6 and 7 were scored at. The reproduction must run at exactly
#: this, or the saved streams cannot match to the cent.
REPRODUCTION_COSTS = CostModel(commission_per_side=rules.ASSUMED_COMMISSION_PER_SIDE,
                               slippage_ticks=BASE_SLIPPAGE_TICKS)

LINE = "=" * 96


def ENTRY6_PARAMS(point_value: float) -> LondonParams:  # noqa: N802
    """Entry 6's 1x/ON arm: the module defaults, on the given contract."""
    return LondonParams(target_multiple=1.0, use_trend_filter=True,
                        point_value=point_value)


def ENTRY10_PARAMS(point_value: float) -> LondonParams:  # noqa: N802
    """Entry 10 as frozen: entry 6 with the flatten at 15:55 (12:55 on an
    early close) and realised-stop sizing. Nothing else differs."""
    return LondonParams(target_multiple=1.0, use_trend_filter=True,
                        point_value=point_value,
                        flatten_time=FLATTEN,
                        early_close_flatten_time=EARLY_CLOSE_FLATTEN,
                        size_on="stop")


# ---------------------------------------------------------------------------
# Pipeline: load once, generate per parameter set, price per cost level
# ---------------------------------------------------------------------------


@dataclass
class Loaded:
    name: str
    spec: object
    bars: pd.DataFrame
    win_bars: pd.DataFrame
    rolls: set
    early: set
    ema: pd.Series


def load(name: str) -> Loaded:
    cfg = INSTRUMENTS[name]
    bars = loader.load_bars(str(PROJECT_ROOT / "data" / cfg["parquet"]))
    return Loaded(
        name=name, spec=cfg["spec"], bars=bars,
        win_bars=slice_by_date(bars, START, END),
        rolls=set(loader.detect_roll_dates(bars)),
        early=set(loader.detect_early_close_dates(bars)),
        ema=intraday_ema(bars, 5, 200),
    )


def generate(loaded: Loaded, params: LondonParams):
    """Signals on the scored window plus the strategy's diagnostics."""
    if params.point_value != loaded.spec.point_value:
        raise ValueError(f"{loaded.name}: params carry point_value "
                         f"{params.point_value}, the contract is "
                         f"{loaded.spec.point_value}")
    strat = LondonBreakout(params, trend_ema=loaded.ema, roll_dates=loaded.rolls,
                           early_close_dates=loaded.early)
    signals = strat.generate_signals(loaded.bars)
    return slice_by_date(signals, START, END), strat.diagnostics


def price_sized(win_signals: pd.DataFrame, loaded: Loaded, costs: CostModel):
    """Entry 6's pricing pipeline: price at one contract, scale each row by its
    own size, apply the daily loss limit per size group, attach the barriers.

    Returns the **comparable** (halt-OFF) stream and the daily-loss halt count.
    """
    spec = loaded.spec
    priced = price_trades(build_trades(win_signals, loaded.win_bars), spec, costs, 1)
    sizes = win_signals["contracts"].dropna()
    n = priced["entry_time"].map(sizes)
    if n.isna().any():
        raise RuntimeError("a trade has no size; signals and trades disagree")
    n = n.astype(int)
    out = priced.copy()
    out["contracts"] = n
    for col in ("gross_pnl", "commission", "slippage_cost", "net_pnl"):
        out[col] = out[col] * n
    out, halts = apply_daily_limit(out, loaded.win_bars, costs, spec)
    for col in ("stop_price", "target_price"):
        out[col] = out["entry_time"].map(win_signals[col].dropna()).astype(float)
    if out[["stop_price", "target_price"]].isna().any().any():
        raise RuntimeError("a trade is missing its barrier levels")
    return out, halts


def guard(comparable: pd.DataFrame):
    """The standard basis: the trailing halt on the loss-limited stream."""
    standard, dd = enforce_trailing_drawdown_halt(comparable)
    return standard, len(dd)


def build(name: str, params: LondonParams, costs: CostModel):
    """One instrument, one parameter set: the comparable stream (no halt)."""
    loaded = load(name)
    win_signals, diag = generate(loaded, params)
    out, halts = price_sized(win_signals, loaded, costs)
    return loaded.bars, loaded.win_bars, out, diag, halts


# ---------------------------------------------------------------------------
# The reproduction check
# ---------------------------------------------------------------------------

COMPARED = ["entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "exit_reason", "contracts", "net_pnl"]


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Timestamps in ET whatever the source; a saved CSV carries strings that
    span EST and EDT, which pandas will not infer as one offset."""
    out = frame.copy().reset_index(drop=True)
    for col in ("entry_time", "exit_time"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.tz_convert(rules.ET)
    out["contracts"] = out["contracts"].astype(int)
    return out[COMPARED]


def compare_streams(ours: pd.DataFrame, saved: pd.DataFrame) -> list[str]:
    """Trade-for-trade differences between a fresh stream and a saved one.

    Empty means identical on every compared column. A count mismatch is
    reported alone, because row-by-row comparison is meaningless after one.
    """
    a, b = _normalise(ours), _normalise(saved)
    if len(a) != len(b):
        return [f"trade count: ours {len(a)} trades, saved {len(b)} trades"]
    diffs: list[str] = []
    for col in COMPARED:
        if col in ("entry_price", "exit_price", "net_pnl"):
            mismatch = ~np.isclose(a[col].astype(float).to_numpy(),
                                   b[col].astype(float).to_numpy(), atol=1e-6)
        elif col in ("entry_time", "exit_time"):
            mismatch = (a[col] != b[col]).to_numpy()
        else:
            mismatch = (a[col].astype(str) != b[col].astype(str)).to_numpy()
        for i in np.flatnonzero(mismatch):
            diffs.append(f"row {i} ({a['entry_time'].iloc[i].date()}): {col} "
                         f"ours={a[col].iloc[i]!r} saved={b[col].iloc[i]!r}")
    return diffs


def reproduce() -> bool:
    """Run the new code at entry 6's settings and diff against the saved streams."""
    print(LINE)
    print("ENTRY 10 - reproduction check (pre-registered test 9)")
    print("  entry 6 params: flatten 09:25, range-based sizing, filter ON, 1x")
    print(f"  costs: ${REPRODUCTION_COSTS.commission_per_side:.2f}/side, "
          f"{REPRODUCTION_COSTS.slippage_ticks:g} ticks/side")
    print(LINE, flush=True)

    all_ok = True
    for name, cfg in INSTRUMENTS.items():
        saved_path = RESULTS / cfg["saved"]
        if not saved_path.exists():
            print(f"\n{name}: saved stream {saved_path.name} is not on disk; "
                  f"cannot compare.")
            all_ok = False
            continue
        print(f"\n{name}: building ...", flush=True)
        _, _, ours, _, halts = build(name, ENTRY6_PARAMS(cfg["spec"].point_value),
                                      REPRODUCTION_COSTS)
        saved = pd.read_csv(saved_path)
        out = RESULTS / f"entry10_reproduction_{name.lower()}.csv"
        ours.to_csv(out, index=False)
        diffs = compare_streams(ours, saved)
        print(f"  ours:  {len(ours):,} trades, net ${ours['net_pnl'].sum():,.2f}, "
              f"{halts} daily-loss halts")
        print(f"  saved: {len(saved):,} trades, net ${saved['net_pnl'].sum():,.2f}  "
              f"({saved_path.name})")
        if diffs:
            all_ok = False
            print(f"  DIVERGES - {len(diffs)} difference(s); first ten:")
            for d in diffs[:10]:
                print(f"    {d}")
        else:
            print(f"  IDENTICAL trade for trade on {', '.join(COMPARED)}")
        print(f"  written: {out.name}")

    print()
    print(LINE)
    print("REPRODUCTION " + ("PASSED on both instruments" if all_ok else "FAILED"))
    print(LINE)
    return all_ok


# ---------------------------------------------------------------------------
# The 15:55 run: pieces that decide something, kept pure and tested
# ---------------------------------------------------------------------------


def truncate_for_benchmark(bars: pd.DataFrame, early_close_dates,
                           early_flatten: time) -> pd.DataFrame:
    """Bars the bootstrap may resample: early-close sessions stop at the
    early flatten, inclusive, because that is where the real trade's horizon
    ends. Every other session is untouched."""
    early = set(early_close_dates)
    if not early:
        return bars
    dates = bars.index.date
    times = bars.index.time
    is_early = np.array([d in early for d in dates])
    keep = ~is_early | (times <= early_flatten)
    return bars[keep]


@dataclass(frozen=True)
class Criterion:
    name: str
    label: str
    value: float
    shown: str
    threshold: str
    passed: bool


def evaluate_criteria(stats: dict, observed: float, benchmark: float,
                      n_resolved: int) -> list[Criterion]:
    """The four kill criteria on one instrument's base-case guarded stream."""
    se = (np.sqrt(benchmark * (1 - benchmark) / n_resolved)
          if 0 < benchmark < 1 and n_resolved > 0 else float("nan"))
    z = float((observed - benchmark) / se) if se and np.isfinite(se) and se > 0 else float("nan")
    departure = 100.0 * (observed - benchmark)
    years, net = int(stats["profitable_years"]), float(stats["net_pnl"])
    p = float(stats["pass_probability"])
    return [
        Criterion("rule_13", f"Rule 13: >= {MIN_PROFITABLE_FOLDS} of {TOTAL_FOLDS} "
                  f"folds profitable and P&L > 0", net,
                  f"{years} of {TOTAL_FOLDS}, ${net:,.0f}",
                  f">= {MIN_PROFITABLE_FOLDS} of {TOTAL_FOLDS} and > 0",
                  years >= MIN_PROFITABLE_FOLDS and net > 0),
        Criterion("pass_probability", "Pooled pass probability", p, f"{100 * p:.2f}%",
                  f">= {100 * MIN_PASS_PROBABILITY:.0f}%", p >= MIN_PASS_PROBABILITY),
        Criterion("z", "z against the 15:55-horizon benchmark, one-sided", z,
                  f"{z:+.2f}", f">= {CRITICAL_Z}", bool(np.isfinite(z) and z >= CRITICAL_Z)),
        Criterion("departure", "Departure from the benchmark, points", departure,
                  f"{departure:+.2f}", f">= +{MIN_DEPARTURE_POINTS:.0f}",
                  bool(np.isfinite(departure) and departure >= MIN_DEPARTURE_POINTS)),
    ]


def verdict_status(criteria_by_instrument: dict) -> str:
    """ACCEPTED only if every criterion passes on both instruments."""
    missing = set(INSTRUMENTS) - set(criteria_by_instrument)
    if missing:
        raise ValueError(f"both instruments are required; missing {sorted(missing)}")
    ok = all(c.passed for name in INSTRUMENTS for c in criteria_by_instrument[name])
    return "ACCEPTED" if ok else "REJECTED"


def attribution(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> dict:
    """Criterion 5. A = 15:55 with stop sizing (the test), B = 09:25 with stop
    sizing, C = 09:25 with range sizing (entry 6 at these costs). The extended
    hold is A - B; the sizing change is B - C; together they are A - C."""
    def net(t):
        return float(t["net_pnl"].sum()) if len(t) else 0.0
    return {
        "a_net": net(a), "b_net": net(b), "c_net": net(c),
        "a_trades": int(len(a)), "b_trades": int(len(b)), "c_trades": int(len(c)),
        "extended_hold": net(a) - net(b),
        "sizing_change": net(b) - net(c),
    }


def stream_stats(trades: pd.DataFrame, paths: int, spec, diag: pd.DataFrame | None,
                 blowup_stream: pd.DataFrame | None = None) -> dict:
    """Everything the entry's pre-registered tests 1-8 report, for one stream."""
    t = trades
    if t.empty:
        return {"trades": 0}
    m = compute_metrics(t)
    entry_et = pd.to_datetime(t["entry_time"], utc=True).dt.tz_convert(rules.ET)
    years = entry_et.dt.year
    year_pnl = {int(y): float(g.sum()) for y, g in t["net_pnl"].groupby(years)}
    daily = eval_sim.daily_pnl_from_trades(t)
    sim = eval_sim.simulate(daily, paths=paths)
    blow = count_evaluation_blowups(blowup_stream if blowup_stream is not None else t)

    reasons = t["exit_reason"].astype(str).value_counts()
    targets = int(reasons.get("target", 0))
    stops = int(reasons.get("stop", 0))
    flattens = int(sum(v for k, v in reasons.items() if k.startswith("flatten_")))
    wins = t.loc[t["exit_reason"] == "target", "net_pnl"]
    losses = t.loc[t["exit_reason"] == "stop", "net_pnl"]
    mean_win = float(wins.mean()) if len(wins) else float("nan")
    mean_loss = float(-losses.mean()) if len(losses) else float("nan")
    be = (mean_loss / (mean_win + mean_loss)
          if np.isfinite(mean_win) and np.isfinite(mean_loss) else float("nan"))
    resolved = targets + stops
    risk = ((t["entry_price"] - t["stop_price"]).abs() * spec.point_value
            * t["contracts"].astype(int))

    overshoot = pd.Series(dtype=float)
    if diag is not None and "overshoot" in diag.columns:
        entered = diag[diag["entered"] == True]  # noqa: E712
        overshoot = entry_et.dt.date.map(entered["overshoot"].astype(float)).dropna()

    by_reason = t.groupby(t["exit_reason"].astype(str))["net_pnl"].agg(
        ["count", "mean", "sum"])

    return {
        "trades": int(len(t)),
        "net_pnl": float(t["net_pnl"].sum()),
        "mean_per_trade": float(t["net_pnl"].mean()),
        "years": [y for y in YEARS if y in year_pnl],
        "year_pnl": year_pnl,
        "profitable_years": int(sum(1 for y in YEARS if year_pnl.get(y, 0.0) > 0)),
        "sharpe": float(m["sharpe"]),
        "profit_factor": float(m["profit_factor"]),
        "max_drawdown": float(m["max_drawdown"]),
        "max_daily_loss": float(m["max_daily_loss"]),
        "pass_probability": sim.pass_probability,
        "payout_probability": sim.payout_probability,
        "blowups": int(blow["blowups"]),
        "avg_duration_min": float(m["avg_duration_seconds"]) / 60.0,
        "microscalp_pct": float(m.get("microscalp_profit_pct", 0.0)),
        "targets": targets, "stops": stops, "flattens": flattens,
        "loss_limit_exits": int(reasons.get("loss_limit_flatten", 0)),
        "observed_share": (targets / resolved) if resolved else float("nan"),
        "n_resolved": resolved,
        "mean_win": mean_win, "mean_loss": mean_loss, "realised_break_even": be,
        "sizing": {int(k): int(v) for k, v in
                   t["contracts"].astype(int).value_counts().sort_index().items()},
        "risk_median": float(risk.median()), "risk_mean": float(risk.mean()),
        "risk_max": float(risk.max()),
        "overshoot_median": float(overshoot.median()) if len(overshoot) else float("nan"),
        "overshoot_mean": float(overshoot.mean()) if len(overshoot) else float("nan"),
        "overshoot_max": float(overshoot.max()) if len(overshoot) else float("nan"),
        "same_bar": int((t["entry_time"] == t["exit_time"]).sum()),
        "by_reason": {k: (int(r["count"]), float(r["mean"]), float(r["sum"]))
                      for k, r in by_reason.iterrows()},
    }


# ---------------------------------------------------------------------------
# The verdict block, written into the entry before anyone reads it
# ---------------------------------------------------------------------------


def _money(x) -> str:
    return f"${x:,.2f}"


def _pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def _basis_table(label: str, std: dict, cmp_: dict, dd_halts: int) -> list[str]:
    def row(name, key, fmt):
        return f"| {name} | {fmt(std[key])} | {fmt(cmp_[key])} |"
    return [
        f"| {label} | Halt ON (standard) | Halt OFF (comparable) |",
        "|---|---|---|",
        row("Trades", "trades", lambda x: f"{x:,}"),
        row("Net P&L", "net_pnl", _money),
        row("Mean per trade", "mean_per_trade", _money),
        row("Folds profitable", "profitable_years", lambda x: f"{x} of {TOTAL_FOLDS}"),
        row("Sharpe", "sharpe", lambda x: f"{x:.2f}"),
        row("Profit factor", "profit_factor", lambda x: f"{x:.3f}"),
        row("Max drawdown", "max_drawdown", _money),
        row("Worst day", "max_daily_loss", _money),
        row("Pass probability", "pass_probability", _pct),
        row("Payout probability ($52,100)", "payout_probability", _pct),
        f"| Evaluations blown (read on the comparable basis) | {cmp_['blowups']} | {cmp_['blowups']} |",
        f"| Sessions blocked by the trailing halt | {dd_halts} | — |",
        row("Avg hold", "avg_duration_min", lambda x: f"{x:.1f} min"),
        row("Profit from <=5s holds", "microscalp_pct", lambda x: f"{x:.2f}%"),
    ]


def verdict_block(results: dict, status: str, date_text: str) -> str:
    lines = [f"### Verdict: {status}", ""]
    if status == "ACCEPTED":
        lines.append(
            "**This idea was chosen after seeing the exit-reason table of the "
            "data it was tested on, and its cost basis rests on an unmeasured "
            "03:00 slippage assumption.** Both were pre-registered above; an "
            "ACCEPTED here is conditional on both and says so on its face.")
    else:
        lines.append(
            "**Rejected on the pre-registered criteria, on the data the idea was "
            "drawn from.** The upward bias of the prior recorded above did not "
            "carry the entry over a bar set above the effect that motivated it.")
    lines += [
        "",
        f"**Date:** {date_text}",
        "**Code:** `strategies/london.py` (`flatten_time`, "
        "`early_close_flatten_time`, `size_on`), `backtests/run_entry10.py`",
        "**Reports:** `backtests/results/entry10_report.txt`; per instrument "
        "`entry10_<mes|mnq>_slip2.csv` (standard) and `_slip2_nohalt.csv` "
        "(comparable), `_slip1`/`_slip3` sensitivities, `_folds.csv`, and the "
        "attribution streams `_0925_stop.csv` and `_0925_range.csv`",
        "",
        f"Entry 6's 1×/ON specification held to {FLATTEN.strftime('%H:%M')} ET "
        f"({EARLY_CLOSE_FLATTEN.strftime('%H:%M')} on early-close sessions), "
        f"realised-stop sizing, ${rules.COMMISSION_PER_SIDE:.2f} a side "
        f"commission, **{BASE_SLIPPAGE_TICKS:g} ticks of slippage per side as "
        f"the base case**, 2020-01-01 .. 2026-08-31, nothing selected. Guards: "
        f"the ${rules.DAILY_LOSS_LIMIT:,.0f} daily loss limit per size group, "
        f"then the ${rules.TRAILING_DD_STOP:,.0f} end-of-day trailing halt on "
        f"the combined stream — the two engine functions "
        f"`apply_internal_guards` composes, in its order, composed per size "
        f"group because size varies per session (exact given one trade a "
        f"day). The benchmark is entry 7's de-meaned bootstrap at the "
        f"{FLATTEN.strftime('%H:%M')} horizon, early-close days truncated at "
        f"{EARLY_CLOSE_FLATTEN.strftime('%H:%M')}, seed 0.",
        "",
        "#### Kill criteria — every line must pass on both instruments",
        "",
        "| # | Criterion | MES | MNQ | Line | |",
        "|---|---|---|---|---|---|",
    ]
    mes, mnq = results["MES"]["criteria"], results["MNQ"]["criteria"]
    for i, (a, b) in enumerate(zip(mes, mnq), start=1):
        ok = a.passed and b.passed
        lines.append(f"| {i} | {a.label} | {a.shown} | {b.shown} | {a.threshold} | "
                     f"**{'PASS' if ok else 'FAIL'}** |")
    failed = [f"{n}: {c.label}" for n in ("MES", "MNQ")
              for c in results[n]["criteria"] if not c.passed]
    lines += ["", f"**{status}.** " + (
        "Every criterion passed on both instruments." if not failed else
        f"Failed on: {'; '.join(failed)}. Any one failure on either instrument "
        f"kills the entry as a whole, as pre-registered."), ""]

    for name in ("MES", "MNQ"):
        r = results[name]
        std, cmp_ = r["base"]["standard"], r["base"]["comparable"]
        bench, obs = r["bench"], r["observed"]
        lines += [f"#### {name}", ""]
        lines += _basis_table(f"2020–2026, {BASE_SLIPPAGE_TICKS:g} ticks, "
                              f"${rules.COMMISSION_PER_SIDE:.2f}", std, cmp_,
                              r["base"]["dd_halts"])
        lines += [
            "",
            f"**Exits, standard stream:** {obs['targets']} targets, {obs['stops']} "
            f"stops, {obs['flattens']} flattens "
            f"({100 * obs['flattens'] / max(std['trades'], 1):.1f}% of trades), "
            f"{std.get('loss_limit_exits', 0)} daily-loss flattens, "
            f"{r['base'].get('loss_halts', 0)} daily-loss halts. "
            f"**Observed target share {_pct(obs['share'])}** on "
            f"{obs['targets'] + obs['stops']:,} resolved trades against a "
            f"**benchmark of {_pct(bench['target_share'])}** "
            f"({bench['n_trades']:,} trades bootstrapped, "
            f"{bench['replications']:,} replications, resolved fraction "
            f"{_pct(bench['resolved_fraction'])}): **departure {r['departure']:+.2f} "
            f"points, z = {r['z']:+.2f}**. Realised break-even from mean stop and "
            f"mean target: {_pct(std.get('realised_break_even'))}.",
            "",
            f"**Sizing and risk:** contracts {std.get('sizing')}; realised risk median "
            f"{_money(std.get('risk_median', float('nan')))}, mean "
            f"{_money(std.get('risk_mean', float('nan')))}, max "
            f"{_money(std.get('risk_max', float('nan')))}; overshoot median "
            f"{std.get('overshoot_median', float('nan')):.2f} pts, mean "
            f"{std.get('overshoot_mean', float('nan')):.2f}, max "
            f"{std.get('overshoot_max', float('nan')):.2f}.",
            "",
            "Per fold, standard: " + "; ".join(
                f"{y} {_money(std['year_pnl'].get(y, 0.0))}" for y in YEARS) + ".",
            "",
        ]
        sens = r.get("sens", {})
        for ticks in SENSITIVITY_TICKS:
            s = sens.get(ticks)
            if not s:
                continue
            ss, sc = s["standard"], s["comparable"]
            lines.append(
                f"**{ticks:g} tick{'s' if ticks != 1 else ''} sensitivity:** standard "
                f"{ss['trades']:,} trades, {_money(ss['net_pnl'])}, "
                f"{ss['profitable_years']} of {TOTAL_FOLDS} folds, pass "
                f"{_pct(ss['pass_probability'])}, payout {_pct(ss['payout_probability'])}; "
                f"comparable {_money(sc['net_pnl'])}, {sc['blowups']} blown.")
        att = r["attribution"]
        lines += [
            "",
            f"**Attribution (criterion 5), standard basis at the base case:** the "
            f"15:55/stop-sized stream nets {_money(att['a_net'])} over "
            f"{att['a_trades']:,} trades; the same signals flattened at 09:25 with "
            f"stop sizing net {_money(att['b_net'])} over {att['b_trades']:,}; entry "
            f"6's rule (09:25, range sizing) at these costs nets "
            f"{_money(att['c_net'])} over {att['c_trades']:,}. **The extended hold "
            f"is worth {_money(att['extended_hold'])}; the sizing change is worth "
            f"{_money(att['sizing_change'])}.**",
            "",
        ]
    lines += ["In-sample results are never evidence, and this is the out-of-sample "
              "answer on both instruments." if status == "REJECTED" else
              "Rule 13 and the raised bar are satisfied on both instruments. The "
              "next step is `testing`, not `paper`; the registry gate is unchanged.",
              ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _write(trades: pd.DataFrame, name: str) -> None:
    trades.to_csv(RESULTS / name, index=False)


def run(paths: int = 20_000, replications: int = bb.REPLICATIONS,
        progress=print) -> tuple[dict, str, str]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    report: list[str] = [LINE, "ENTRY 10 - London 1x/ON held to 15:55, MES and MNQ",
                         f"  ${rules.COMMISSION_PER_SIDE:.2f}/side commission; base "
                         f"{BASE_SLIPPAGE_TICKS:g} ticks/side; sensitivities "
                         f"{', '.join(f'{t:g}' for t in SENSITIVITY_TICKS)}",
                         f"  bootstrap: {replications:,} replications at the "
                         f"{FLATTEN} horizon, seed {bb.SEED}", LINE]

    for name, cfg in INSTRUMENTS.items():
        pv = cfg["spec"].point_value
        progress(f"[progress] {name}: loading bars")
        loaded = load(name)
        progress(f"[progress] {name}: generating the 15:55 signals")
        sig_a, diag_a = generate(loaded, ENTRY10_PARAMS(pv))
        progress(f"[progress] {name}: generating the 09:25 attribution signals")
        sig_b, _ = generate(loaded, replace(ENTRY10_PARAMS(pv), flatten_time=time(9, 25),
                                            early_close_flatten_time=None))
        sig_c, _ = generate(loaded, ENTRY6_PARAMS(pv))

        per_ticks: dict = {}
        for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
            progress(f"[progress] {name}: pricing and guarding at {ticks:g} ticks")
            costs = CostModel(commission_per_side=rules.COMMISSION_PER_SIDE,
                              slippage_ticks=ticks)
            comparable, loss_halts = price_sized(sig_a, loaded, costs)
            standard, dd_halts = guard(comparable)
            tag = f"slip{ticks:g}"
            _write(standard, f"entry10_{name.lower()}_{tag}.csv")
            _write(comparable, f"entry10_{name.lower()}_{tag}_nohalt.csv")
            per_ticks[ticks] = {
                "standard": stream_stats(standard, paths, loaded.spec, diag_a, comparable),
                "comparable": stream_stats(comparable, paths, loaded.spec, diag_a),
                "dd_halts": dd_halts, "loss_halts": loss_halts,
                "standard_frame": standard,
            }
            if ticks == BASE_SLIPPAGE_TICKS:
                fold_frame(standard, paths).to_csv(
                    RESULTS / f"entry10_{name.lower()}_folds.csv", index=False)
                b_cmp, _ = price_sized(sig_b, loaded, costs)
                b_std, _ = guard(b_cmp)
                c_cmp, _ = price_sized(sig_c, loaded, costs)
                c_std, _ = guard(c_cmp)
                _write(b_std, f"entry10_{name.lower()}_0925_stop.csv")
                _write(c_std, f"entry10_{name.lower()}_0925_range.csv")
                per_ticks[ticks]["attribution"] = attribution(standard, b_std, c_std)

        base = per_ticks[BASE_SLIPPAGE_TICKS]
        standard = base["standard_frame"]
        progress(f"[progress] {name}: bootstrapping the benchmark at the "
                 f"{FLATTEN} horizon ({replications:,} replications)")
        bars_bench = truncate_for_benchmark(loaded.win_bars, loaded.early,
                                            EARLY_CLOSE_FLATTEN)
        bench = bb.benchmark(standard, bars_bench, FLATTEN, replications=replications)
        std = base["standard"]
        obs_share, n_res = std["observed_share"], std["n_resolved"]
        z = bench.z_against(obs_share, n_res)
        criteria = evaluate_criteria(std, obs_share, bench.target_share, n_res)
        progress(f"[progress] {name}: done")

        results[name] = {
            "criteria": criteria,
            "base": {k: v for k, v in base.items() if k != "standard_frame"},
            "sens": {t: {k: v for k, v in per_ticks[t].items() if k != "standard_frame"}
                     for t in SENSITIVITY_TICKS},
            "bench": {"target_share": bench.target_share, "n_trades": bench.n_trades,
                      "resolved_fraction": bench.resolved_fraction,
                      "replications": bench.replications,
                      "per_trade_dispersion": bench.per_trade_dispersion},
            "observed": {"share": obs_share, "targets": std["targets"],
                         "stops": std["stops"], "flattens": std["flattens"]},
            "z": float(z), "departure": 100.0 * (obs_share - bench.target_share),
            "attribution": base["attribution"],
        }

    status = verdict_status({n: r["criteria"] for n, r in results.items()})
    block = verdict_block(results, status, str(pd.Timestamp.now(tz=rules.ET).date()))
    (RESULTS / "entry10_verdict.md").write_text(block, encoding="utf-8")
    report.append(block)
    (RESULTS / "entry10_report.txt").write_text("\n".join(report), encoding="utf-8")
    progress("[progress] verdict written to backtests/results/entry10_verdict.md")
    return results, status, block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", action="store_true",
                    help="run the pre-registered reproduction check")
    ap.add_argument("--run", action="store_true",
                    help="the 15:55 test; authorised by the operator after the "
                         "reproduction passed")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--replications", type=int, default=bb.REPLICATIONS)
    args = ap.parse_args(argv)
    if args.reproduce:
        return 0 if reproduce() else 1
    if args.run:
        _, status, _ = run(args.paths, args.replications,
                           progress=lambda m: print(m, flush=True))
        print(f"[done] {status}", flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
