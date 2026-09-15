"""Entry 16: liquidity sweep of the prior session's high or low, with the
resting stop order as the counterparty. Frozen at the entry 16 freeze commit.

    python backtests/run_entry16.py --reproduce          # pre-registered test 1, run first
    python backtests/run_entry16.py --run [--live]       # the mechanism tests, the fade arm, the verdict

Mechanism tests on the sign-adjusted 30-bar reversion after a rejection:
real-level sweeps against placebo-level sweeps (Welch, one-sided real
larger) and real-level sweeps against zero (one-sample t, and the count of
years positive). The fade arm shorts the bar after a high-side rejection
and buys after a low-side one, stop a fixed buffer beyond the sweep extreme,
target the same distance the other way, one contract, flat at 15:55.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from engine import (  # noqa: E402
    MES, CostModel, apply_internal_guards, build_trades, price_trades,
)
from live import LiveRun, NullLive  # noqa: E402
from metrics import compute_metrics  # noqa: E402
import run_entry11 as e11  # noqa: E402
from run_entry11 import (  # noqa: E402
    BASE_SLIPPAGE_TICKS, END, LINE, MAX_BLOWUPS, MIN_PASS_PROBABILITY,
    MIN_PROFITABLE_FOLDS, SENSITIVITY_TICKS, START, TOTAL_FOLDS, YEARS, Criterion,
    Loaded, round_turn_points, stream_stats, welch,
)
from run_generated import hold_regression_line  # noqa: E402
from run_london import fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402
from sweep import (  # noqa: E402
    CONTINUATION, HORIZON_BARS, PLACEBO, PLACEBO_FRACTION, REAL, STOP_BUFFER_TICKS,
    SweepFade, SweepParams, session_events,
)

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS = PROJECT_ROOT / "backtests" / "results"

#: The operator's specification, as frozen.
CONTRACTS = 1

#: Entry 16's kill criteria, as frozen. Any one failure kills the entry.
MIN_T = 2.0
MIN_YEARS_POSITIVE = 4


# ---------------------------------------------------------------------------
# Loading and populations
# ---------------------------------------------------------------------------


def load(parquet: Path = PARQUET) -> Loaded:
    bars = loader.load_bars(str(parquet))
    return Loaded(bars=bars, win_bars=slice_by_date(bars, START, END),
                  rolls=set(loader.detect_roll_dates(bars)),
                  early=set(loader.detect_early_close_dates(bars)))


def scored_events(loaded: Loaded) -> pd.DataFrame:
    """Events over the whole file, sliced to the scored span."""
    ev = session_events(loaded.bars, loaded.rolls, loaded.early)
    keep = (ev["date"] >= START) & (ev["date"] <= END)
    return ev[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The mechanism tests
# ---------------------------------------------------------------------------


def one_sample_t(x) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "sd": float("nan"),
                "t": float("nan"), "p_one_sided": float("nan")}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd < 1e-12:
        return {"n": n, "mean": mean, "sd": sd, "t": 0.0, "p_one_sided": 0.5}
    t = mean / (sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "t": float(t), "p_one_sided": float(sps.t.sf(t, n - 1))}


def _welch_larger(a, b) -> dict:
    """Welch t for mean(a) - mean(b), one-sided p for a > b, degenerate-safe."""
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "mean_a": float("nan"), "mean_b": float("nan"),
                "sd_a": float("nan"), "sd_b": float("nan"), "median_a": float("nan"),
                "median_b": float("nan"), "diff": float("nan"), "t": float("nan"),
                "df": float("nan"), "p_one_sided": float("nan")}
    w = welch(np.asarray(a, float), np.asarray(b, float))
    if max(w["sd_a"], w["sd_b"]) < 1e-9 or abs(w["diff"]) < 1e-9:
        w.update(t=0.0 if abs(w["diff"]) < 1e-9 else w["t"])
        if abs(w["diff"]) < 1e-9:
            w.update(df=float("nan"), p_one_sided=0.5)
    return w


def mechanism_test(ev: pd.DataFrame) -> dict:
    real = ev[ev["kind"] == REAL]
    placebo = ev[ev["kind"] == PLACEBO]
    cont = ev[ev["kind"] == CONTINUATION]

    per_year = []
    for year in YEARS:
        r = real[real["year"] == year]["m30"].astype(float)
        p = placebo[placebo["year"] == year]["m30"].astype(float)
        rt = one_sample_t(r)
        per_year.append({"year": year, "n_real": int(len(r)), "n_placebo": int(len(p)),
                         "real_mean": rt["mean"], "real_t": rt["t"],
                         "placebo_mean": float(p.mean()) if len(p) else float("nan"),
                         "positive": bool(np.isfinite(rt["mean"]) and rt["mean"] > 0)})

    by_side = {}
    for side in ("high", "low"):
        r = real[real["side"] == side]["m30"].astype(float)
        p = placebo[placebo["side"] == side]["m30"].astype(float)
        by_side[side] = {"real": one_sample_t(r), "vs_placebo": _welch_larger(r, p)}

    terciles = []
    if len(real) >= 3:
        depth = real["depth"].astype(float)
        cuts = np.quantile(depth, [1 / 3, 2 / 3])
        labels = np.where(depth <= cuts[0], "shallow", np.where(depth <= cuts[1], "middle", "deep"))
        for name in ("shallow", "middle", "deep"):
            sub = real[labels == name]["m30"].astype(float)
            terciles.append({"tercile": name, **one_sample_t(sub),
                             "depth_range": [float(real["depth"][labels == name].min()) if len(sub) else float("nan"),
                                             float(real["depth"][labels == name].max()) if len(sub) else float("nan")]})

    return {
        "real_vs_placebo": _welch_larger(real["m30"].astype(float), placebo["m30"].astype(float)),
        "real_vs_zero": one_sample_t(real["m30"].astype(float)),
        "real_vs_placebo_1555": _welch_larger(real["m1555"].astype(float), placebo["m1555"].astype(float)),
        "per_year": per_year,
        "years_real_positive": int(sum(1 for r in per_year if r["positive"])),
        "by_side": by_side,
        "by_depth_tercile": terciles,
        "continuations": {"n": int(len(cont)), "high": int((cont["side"] == "high").sum()),
                          "low": int((cont["side"] == "low").sum()),
                          "mean_m1555": float(cont["m1555"].astype(float).mean()) if len(cont) else float("nan")},
        "both_events_sessions": int(len(set(real["date"]) & set(placebo["date"]))),
    }


# ---------------------------------------------------------------------------
# The fade arm
# ---------------------------------------------------------------------------


def generate(loaded: Loaded, params: SweepParams):
    strat = SweepFade(params, roll_dates=loaded.rolls, early_close_dates=loaded.early)
    signals = strat.generate_signals(loaded.bars)
    return slice_by_date(signals, START, END), strat.diagnostics


def guarded_streams(signals: pd.DataFrame, loaded: Loaded, costs: CostModel):
    raw = price_trades(build_trades(signals, loaded.win_bars), MES, costs, CONTRACTS)
    standard, loss_halts, dd_halts = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=True)
    comparable, _, _ = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=False)
    return standard, comparable, len(loss_halts), len(dd_halts)


def reproduction_differences(real_events: pd.DataFrame, diagnostics: pd.DataFrame) -> list[str]:
    """The strategy's diagnostics against the population builder's real events."""
    problems: list[str] = []
    expected = {d: float(x) for d, x in zip(real_events["date"], real_events["depth"])}
    diag_dates = set(diagnostics.index)
    missing = sorted(set(expected) - diag_dates)
    if missing:
        problems.append(f"{len(missing)} real event(s) absent from the diagnostics: "
                        f"{[str(d) for d in missing[:5]]}")
    entered = diagnostics[diagnostics["entered"].astype(bool)]
    extra = sorted(set(entered.index) - set(expected))
    if extra:
        problems.append(f"entries on {len(extra)} day(s) with no real event: {[str(d) for d in extra[:5]]}")
    for d in sorted(set(expected) & set(entered.index)):
        if abs(float(entered.loc[d, "depth"]) - expected[d]) > 1e-9:
            problems.append(f"depth differs on {d}: strategy {float(entered.loc[d, 'depth']):.2f}, "
                            f"population {expected[d]:.2f}")
            break
    return problems


def reproduce(live: NullLive | None = None) -> bool:
    live = NullLive() if live is None else live
    with live:
        say = live.progress(lambda m: print(m, flush=True))
        print(LINE)
        say("ENTRY 16 reproduction check (pre-registered test 1)")
        print(LINE, flush=True)
        say("[progress] loading bars and building the event population")
        loaded = load()
        ev = scored_events(loaded)
        real = ev[ev["kind"] == REAL]
        say("[progress] generating the fade arm's signals")
        _, diag = generate(loaded, SweepParams(STOP_BUFFER_TICKS))
        diag = diag[[d >= START for d in diag.index]]
        problems = reproduction_differences(real, diag)
        for p in problems:
            say(f"  DIVERGES - {p}")
        if not problems:
            say(f"  IDENTICAL: {len(real)} real events reproduced day for day with matching "
                f"depths; {int(diag['entered'].astype(bool).sum())} entries at the decision bar")
        print(LINE)
        print("REPRODUCTION " + ("PASSED" if not problems else "FAILED"))
        print(LINE)
        live.finish("REPRODUCED" if not problems else "DIVERGED")
        return not problems


# ---------------------------------------------------------------------------
# Criteria and the verdict
# ---------------------------------------------------------------------------


def evaluate_criteria(mechanism: dict, std_1t: dict, cmp_1t: dict, std_2t: dict) -> list[Criterion]:
    t_vs = float(mechanism["real_vs_placebo"]["t"])
    t0 = float(mechanism["real_vs_zero"]["t"])
    years = int(mechanism["years_real_positive"])
    p = float(std_1t["pass_probability"])
    blown = int(cmp_1t["blowups"])

    def rule13(s: dict) -> bool:
        return int(s["profitable_years"]) >= MIN_PROFITABLE_FOLDS and float(s["net_pnl"]) > 0

    return [
        Criterion("real_vs_placebo", "Real-level reversion above placebo-level reversion, Welch t",
                  f"t = {t_vs:+.2f}", f"t >= {MIN_T:.1f}", bool(np.isfinite(t_vs) and t_vs >= MIN_T)),
        Criterion("real_vs_zero", "Real-level reversion above zero: one-sample t, and years positive",
                  f"t = {t0:+.2f}, {years} of {TOTAL_FOLDS} years",
                  f"t >= {MIN_T:.1f} and >= {MIN_YEARS_POSITIVE} of {TOTAL_FOLDS}",
                  bool(np.isfinite(t0) and t0 >= MIN_T and years >= MIN_YEARS_POSITIVE)),
        Criterion("pass_probability", "Fade arm, standard stream, 1 tick: eval_sim pass probability",
                  f"{100 * p:.2f}% ({std_1t['trading_days']} trading days)",
                  f">= {100 * MIN_PASS_PROBABILITY:.0f}%",
                  bool(np.isfinite(p) and p >= MIN_PASS_PROBABILITY)),
        Criterion("blowups", "Fade arm, comparable stream, 1 tick: evaluations blown",
                  f"{blown}", f"<= {MAX_BLOWUPS}", blown <= MAX_BLOWUPS),
        Criterion("rule_13", f"Rule 13 on the fade arm, standard stream: >= {MIN_PROFITABLE_FOLDS} "
                  f"of {TOTAL_FOLDS} folds profitable and P&L > 0 at 1 and 2 ticks",
                  f"1 tick {std_1t['profitable_years']} of {TOTAL_FOLDS}, "
                  f"${float(std_1t['net_pnl']):,.0f}; 2 ticks {std_2t['profitable_years']} of "
                  f"{TOTAL_FOLDS}, ${float(std_2t['net_pnl']):,.0f}",
                  f">= {MIN_PROFITABLE_FOLDS} of {TOTAL_FOLDS} and > 0, both",
                  rule13(std_1t) and rule13(std_2t)),
    ]


def verdict_status(criteria: list[Criterion]) -> str:
    return "ACCEPTED" if all(c.passed for c in criteria) else "REJECTED"


def _fmt(x, spec: str = "+.2f") -> str:
    return "n/a" if x is None or not np.isfinite(x) else format(x, spec)


def verdict_block(results: dict, date_text: str) -> str:
    status = results["status"]
    mech = results["mechanism"]
    vp, vz, v15 = mech["real_vs_placebo"], mech["real_vs_zero"], mech["real_vs_placebo_1555"]
    rt1 = round_turn_points(CostModel(slippage_ticks=BASE_SLIPPAGE_TICKS))
    lines = [f"### Verdict: {status}", ""]
    if status == "ACCEPTED":
        lines.append(
            "**Measured on the 30-bar reversion after a prior-session-level sweep, and the fade "
            "arm's cost basis rests on a slippage assumption no backtest can measure.** Both were "
            "pre-registered; an ACCEPTED here is conditional on both and says so on its face.")
    else:
        lines.append(
            "**Rejected on the pre-registered criteria.** The sweep-and-reverse reading of the "
            "prior session's high and low, tested against a placebo level and against zero on "
            "MES over 2020–2026, and a fixed 1:1 fade rule, did not clear the lines set before "
            "the run.")
    lines += [
        "",
        f"**Date:** {date_text}",
        "**Code:** `strategies/sweep.py`, `backtests/run_entry16.py`, `research/power_check_sweep.py`",
        "**Reports:** `backtests/results/entry16_report.txt`; `entry16_events.csv` (every real, "
        "placebo and continuation event with its bars, depth and reversion); "
        "`entry16_fade_slip<1|2>.csv` (standard) and `_nohalt.csv` (comparable); "
        "`entry16_folds.csv` (fade arm, 1 tick, standard)",
        "",
        f"Levels: the previous RTH session's high and low; placebo {PLACEBO_FRACTION:g} of the "
        f"prior range inside. Event: cross then close back inside by 14:59, first of either "
        f"side, one per session; reversion measured over {HORIZON_BARS} bars from the decision "
        f"open. Fade arm: stop **{STOP_BUFFER_TICKS} ticks beyond the sweep extreme**, target the "
        f"same distance the other way, **{CONTRACTS} contract**, flat at 15:55, "
        f"${rules.COMMISSION_PER_SIDE:.2f} a side, **{BASE_SLIPPAGE_TICKS:g} tick a side as the "
        f"base case** ({rt1:.2f} points a round turn), {START} .. {END}, nothing selected. "
        f"Guards: `engine.apply_internal_guards` at {CONTRACTS} contract; halt OFF alongside.",
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
    lines.append(hold_regression_line(results.get("metrics_standard_1t", {})))
    lines += [
        "",
        "#### The mechanism tests: 30-bar reversion after a rejection, in MES points",
        "",
        f"**Real against placebo:** real n = {vp['n_a']}, mean {_fmt(vp['mean_a'])}, sd "
        f"{_fmt(vp['sd_a'], '.2f')}, median {_fmt(vp['median_a'])}; placebo n = {vp['n_b']}, mean "
        f"{_fmt(vp['mean_b'])}, sd {_fmt(vp['sd_b'], '.2f')}, median {_fmt(vp['median_b'])}. "
        f"**Difference {_fmt(vp['diff'])} points, Welch t = {_fmt(vp['t'])}**, one-sided p (real "
        f"larger) = {_fmt(vp['p_one_sided'], '.3f')}.",
        "",
        f"**Real against zero:** n = {vz['n']}, mean {_fmt(vz['mean'])} points, sd "
        f"{_fmt(vz['sd'], '.2f')}, **t = {_fmt(vz['t'])}**, one-sided p = "
        f"{_fmt(vz['p_one_sided'], '.3f')}. Positive in **{mech['years_real_positive']} of "
        f"{TOTAL_FOLDS}** years.",
        "",
        f"**To 15:55, reported:** real mean {_fmt(v15['mean_a'])}, placebo mean "
        f"{_fmt(v15['mean_b'])}, difference {_fmt(v15['diff'])}, t = {_fmt(v15['t'])}.",
        "",
        "| Year | Real n | Real mean | Real t | Placebo n | Placebo mean | Real positive |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in mech["per_year"]:
        lines.append(f"| {r['year']} | {r['n_real']} | {_fmt(r['real_mean'])} | {_fmt(r['real_t'])} | "
                     f"{r['n_placebo']} | {_fmt(r['placebo_mean'])} | {'yes' if r['positive'] else 'no'} |")
    lines += ["", "**By side, reported and not selected on:**", "",
              "| Side | Real n | Real mean | Real t | vs placebo diff | vs placebo t |",
              "|---|---|---|---|---|---|"]
    for side, s in mech["by_side"].items():
        lines.append(f"| {side} | {s['real']['n']} | {_fmt(s['real']['mean'])} | {_fmt(s['real']['t'])} | "
                     f"{_fmt(s['vs_placebo']['diff'])} | {_fmt(s['vs_placebo']['t'])} |")
    lines += ["", "**By sweep depth tercile, reported and not selected on:**", "",
              "| Tercile | Depth range (pts) | n | Mean reversion | t |", "|---|---|---|---|---|"]
    for t in mech["by_depth_tercile"]:
        lo, hi = t["depth_range"]
        lines.append(f"| {t['tercile']} | {_fmt(lo, '.2f')} – {_fmt(hi, '.2f')} | {t['n']} | "
                     f"{_fmt(t['mean'])} | {_fmt(t['t'])} |")
    c = mech["continuations"]
    lines += ["", f"**Continuations** (crossed the real level, never closed back inside by 14:59): "
                  f"{c['n']} ({c['high']} high, {c['low']} low); their 09:30-to-15:55 move, signed so "
                  f"that a return inside reads positive, averaged {_fmt(c['mean_m1555'])} points. "
                  f"Sessions with both a real and a placebo event: {mech['both_events_sessions']}."]
    for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
        r = results["arms"][ticks]
        tag = "base case" if ticks == BASE_SLIPPAGE_TICKS else "sensitivity"
        lines += ["", f"#### Fade arm — {ticks:g} tick{'s' if ticks != 1 else ''} ({tag})", ""]
        lines += e11._basis_table(f"2020–2026, {CONTRACTS} contract, {ticks:g} "
                                  f"tick{'s' if ticks != 1 else ''}, ${rules.COMMISSION_PER_SIDE:.2f}",
                                  r["standard"], r["comparable"])
        std = r["standard"]
        reasons = "; ".join(f"{k}: {n} trades, mean {e11._money(mu)}, total {e11._money(tot)}"
                            for k, (n, mu, tot) in sorted(std["by_reason"].items()))
        lines += ["", f"Exits, standard stream: {reasons or 'none'}. Daily-loss halts: "
                      f"{std['loss_halts']}.",
                  "Per fold, standard: " + "; ".join(
                      f"{y} {e11._money(std['year_pnl'].get(y, 0.0))}" for y in YEARS) + "."]
    diag = results["diagnostics"]
    lines += ["", "#### Diagnostics", "",
              f"Real events in the scored span: {diag['real_events']}; entered {diag['entered']}; "
              f"skipped by the rules: {diag['skipped']}. Stop distance in points: median "
              f"{_fmt(diag['stop_median'], '.2f')}, max {_fmt(diag['stop_max'], '.2f')}; trades whose "
              f"stop would have breached the daily limit: {diag['stop_over_daily_limit']}. Entry-bar "
              f"stop breaches: {diag['entry_bar_breaches']}.",
              "",
              "In-sample results are never evidence, and this is the out-of-sample answer on "
              "seven yearly folds with nothing selected." if status == "REJECTED"
              else "Every criterion is satisfied. The next step is `testing`, not `paper`; "
                   "the registry gate is unchanged.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(paths: int = 20_000, commission: float | None = None, progress=print,
        live: NullLive | None = None):
    live = NullLive() if live is None else live
    with live:
        return _run(paths, commission, live.progress(progress), live)


def _run(paths: int, commission: float | None, progress, live: NullLive):
    RESULTS.mkdir(parents=True, exist_ok=True)
    commission = rules.COMMISSION_PER_SIDE if commission is None else commission
    progress("[progress] loading bars")
    loaded = load()

    progress("[progress] building the event population and the mechanism tests")
    ev = scored_events(loaded)
    ev.to_csv(RESULTS / "entry16_events.csv", index=False)
    mechanism = mechanism_test(ev)

    params = SweepParams(STOP_BUFFER_TICKS)
    progress("[progress] fade arm: generating signals")
    signals, diag = generate(loaded, params)
    diag = diag[[d >= START for d in diag.index]]
    entered = diag[diag["entered"].astype(bool)]
    stop_dist = (entered["stop_price"] - entered["entry_price"]).abs().astype(float)
    diagnostics = {
        "real_events": int((ev["kind"] == REAL).sum()),
        "entered": int(len(entered)),
        "skipped": int((~diag["entered"].astype(bool)).sum()),
        "stop_median": float(stop_dist.median()) if len(stop_dist) else float("nan"),
        "stop_max": float(stop_dist.max()) if len(stop_dist) else float("nan"),
        "stop_over_daily_limit": int((stop_dist * MES.point_value * CONTRACTS > rules.DAILY_LOSS_LIMIT).sum()),
        "entry_bar_breaches": int(diag["entry_bar_breach"].fillna(False).astype(bool).sum()),
    }

    arms: dict = {}
    metrics_standard_1t: dict = {}
    for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
        progress(f"[progress] fade arm: pricing and guarding at {ticks:g} tick(s)")
        costs = CostModel(commission_per_side=commission, slippage_ticks=ticks)
        standard, comparable, loss_halts, dd_halts = guarded_streams(signals, loaded, costs)
        standard.to_csv(RESULTS / f"entry16_fade_slip{ticks:g}.csv", index=False)
        comparable.to_csv(RESULTS / f"entry16_fade_slip{ticks:g}_nohalt.csv", index=False)
        arms[ticks] = {
            "standard": stream_stats(standard, paths, comparable, dd_halts, loss_halts),
            "comparable": stream_stats(comparable, paths, None, 0, loss_halts),
        }
        if ticks == BASE_SLIPPAGE_TICKS:
            live.trades(standard, "standard")
            live.trades(comparable, "comparable")
            folds = fold_frame(standard, paths)
            folds.to_csv(RESULTS / "entry16_folds.csv", index=False)
            live.folds(folds)
            metrics_standard_1t = compute_metrics(standard) if len(standard) else {}

    criteria = evaluate_criteria(mechanism, arms[BASE_SLIPPAGE_TICKS]["standard"],
                                 arms[BASE_SLIPPAGE_TICKS]["comparable"],
                                 arms[SENSITIVITY_TICKS[0]]["standard"])
    status = verdict_status(criteria)
    if metrics_standard_1t.get("hold_regression"):
        status = "REJECTED"
    results = {"status": status, "criteria": criteria, "mechanism": mechanism, "arms": arms,
               "diagnostics": diagnostics, "metrics_standard_1t": metrics_standard_1t}
    block = verdict_block(results, str(pd.Timestamp.now(tz=rules.ET).date()))
    (RESULTS / "entry16_verdict.md").write_text(block, encoding="utf-8")
    (RESULTS / "entry16_report.txt").write_text(
        "\n".join([LINE, "ENTRY 16 - liquidity sweep of the prior session's high or low",
                   f"  ${commission:.2f}/side; base {BASE_SLIPPAGE_TICKS:g} tick/side; "
                   f"sensitivity {', '.join(f'{t:g}' for t in SENSITIVITY_TICKS)}", LINE, block]),
        encoding="utf-8")
    progress("[progress] verdict written to backtests/results/entry16_verdict.md")
    live.finish(status)
    return results, status, block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--commission", type=float, default=rules.COMMISSION_PER_SIDE)
    ap.add_argument("--live", action="store_true",
                    help="write a live event stream for bots/liveview.py")
    args = ap.parse_args(argv)
    if args.reproduce:
        live_run = LiveRun("entry16", runner="run_entry16 --reproduce") if args.live else NullLive()
        return 0 if reproduce(live=live_run) else 1
    if args.run:
        live_run = LiveRun("entry16", runner="run_entry16 --run") if args.live else NullLive()
        _, status, _ = run(args.paths, args.commission,
                           progress=lambda m: print(m, flush=True), live=live_run)
        print(f"[done] {status}", flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
