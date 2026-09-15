"""Entry 15: quarterly futures expiry, index-arbitrage unwind at the
settlement open, on MES. Frozen at the entry 15 freeze commit.

    python backtests/run_entry15.py --reproduce          # pre-registered test 1, run first
    python backtests/run_entry15.py --run [--live]       # the two mechanism tests, the fade arm, the verdict

Mechanism tests: the absolute 09:30-to-10:00 move on quarterly expiry days
against every other eligible Friday (Welch t, one-sided for expiry larger),
and the reversal share (afternoon move opposite in sign to the opening
move) as a two-proportion z. The fade arm enters at the 10:00 open when the
opening move exceeds ``k`` control standard deviations, targets the day's
open, stops ``s`` standard deviations away, one contract, flat at 15:55.
``sd_open`` is derived from the control at run time and reported.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from opex import CONTROL_FRIDAY, SKIP_INSIDE_THRESHOLD  # noqa: E402
from quarterly import (  # noqa: E402
    QUARTERLY, QuarterlyFade, QuarterlyParams, session_calendar, session_moves,
)
import run_entry11 as e11  # noqa: E402
from run_entry11 import (  # noqa: E402
    BASE_SLIPPAGE_TICKS, END, LINE, MAX_BLOWUPS, MIN_PASS_PROBABILITY,
    MIN_PROFITABLE_FOLDS, SENSITIVITY_TICKS, START, TOTAL_FOLDS, YEARS, Criterion,
    Loaded, round_turn_points, stream_stats, welch,
)
from run_entry14 import reproduction_differences  # noqa: E402
from run_generated import hold_regression_line  # noqa: E402
from run_london import fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS = PROJECT_ROOT / "backtests" / "results"

#: The operator's specification, as frozen.
CONTRACTS = 1
PARAMS_K = 1.0
PARAMS_S = 1.0

#: Entry 15's kill criteria, as frozen. Any one failure kills the entry.
MIN_T = 2.0                  # opening impact, expiry larger, one-sided
MIN_REVERSAL_POINTS = 15.0   # reversal share, expiry minus control, percentage points
MIN_Z = 1.64                 # two-proportion z on the reversal share


# ---------------------------------------------------------------------------
# Loading and populations
# ---------------------------------------------------------------------------


def load(parquet: Path = PARQUET) -> Loaded:
    bars = loader.load_bars(str(parquet))
    return Loaded(bars=bars, win_bars=slice_by_date(bars, START, END),
                  rolls=set(loader.detect_roll_dates(bars)),
                  early=set(loader.detect_early_close_dates(bars)))


def scored_population(loaded: Loaded) -> pd.DataFrame:
    pop = session_moves(loaded.bars, loaded.rolls, loaded.early)
    keep = (pop["date"] >= START) & (pop["date"] <= END)
    return pop[keep].reset_index(drop=True)


def scored_calendar(loaded: Loaded) -> pd.DataFrame:
    cal = session_calendar(loaded.bars, loaded.rolls, loaded.early)
    keep = (cal["date"] >= START) & (cal["date"] <= END)
    return cal[keep].reset_index(drop=True)


def sd_open_from(pop: pd.DataFrame) -> float:
    """Spread of the control's opening move - no expiry day is read."""
    fri = pop[pop["label"] == CONTROL_FRIDAY]["opening_move"].astype(float)
    return float(fri.std(ddof=1))


# ---------------------------------------------------------------------------
# The mechanism tests
# ---------------------------------------------------------------------------


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """z for share k1/n1 minus k2/n2 under the pooled proportion."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


def mechanism_test(pop: pd.DataFrame) -> dict:
    expiry = pop[pop["label"] == QUARTERLY]
    control = pop[pop["label"] == CONTROL_FRIDAY]

    if len(expiry) >= 2 and len(control) >= 2:
        impact = welch(expiry["abs_open"].to_numpy(float), control["abs_open"].to_numpy(float))
        if max(impact["sd_a"], impact["sd_b"]) < 1e-9:
            impact.update(t=0.0, df=float("nan"), p_one_sided=0.5)
        impact["ratio_pct"] = ((impact["mean_a"] / impact["mean_b"] - 1.0) * 100.0
                               if impact["mean_b"] > 0 else float("nan"))
    else:
        impact = {"n_a": int(len(expiry)), "n_b": int(len(control)), "mean_a": float("nan"),
                  "mean_b": float("nan"), "median_a": float("nan"), "median_b": float("nan"),
                  "sd_a": float("nan"), "sd_b": float("nan"), "diff": float("nan"),
                  "t": float("nan"), "df": float("nan"), "p_one_sided": float("nan"),
                  "ratio_pct": float("nan")}

    k1, n1 = int(expiry["reversal"].astype(bool).sum()), int(len(expiry))
    k2, n2 = int(control["reversal"].astype(bool).sum()), int(len(control))
    s1 = k1 / n1 if n1 else float("nan")
    s2 = k2 / n2 if n2 else float("nan")
    reversal = {"reversals_expiry": k1, "n_expiry": n1, "reversals_control": k2, "n_control": n2,
                "share_expiry": s1, "share_control": s2,
                "diff_points": (s1 - s2) * 100.0 if n1 and n2 else float("nan"),
                "z": two_proportion_z(k1, n1, k2, n2)}

    def conditional(frame: pd.DataFrame) -> dict:
        up, down = frame[frame["opening_move"] > 0], frame[frame["opening_move"] < 0]
        return {"n_up": int(len(up)),
                "mean_afternoon_after_up": float(up["afternoon_move"].mean()) if len(up) else float("nan"),
                "n_down": int(len(down)),
                "mean_afternoon_after_down": float(down["afternoon_move"].mean()) if len(down) else float("nan")}

    per_year = []
    for year in YEARS:
        e = expiry[expiry["year"] == year]["abs_open"]
        c = control[control["year"] == year]["abs_open"]
        em = float(e.mean()) if len(e) else float("nan")
        cm = float(c.mean()) if len(c) else float("nan")
        per_year.append({"year": year, "n_expiry": int(len(e)), "n_control": int(len(c)),
                         "expiry_mean_abs": em, "control_mean_abs": cm,
                         "above": bool(np.isfinite(em) and np.isfinite(cm) and em > cm)})
    return {"impact": impact, "reversal": reversal,
            "conditional": {"expiry": conditional(expiry), "control": conditional(control)},
            "per_year": per_year,
            "years_expiry_impact_above": int(sum(1 for r in per_year if r["above"]))}


# ---------------------------------------------------------------------------
# The fade arm
# ---------------------------------------------------------------------------


def generate(loaded: Loaded, params: QuarterlyParams):
    strat = QuarterlyFade(params, roll_dates=loaded.rolls, early_close_dates=loaded.early)
    signals = strat.generate_signals(loaded.bars)
    return slice_by_date(signals, START, END), strat.diagnostics


def guarded_streams(signals: pd.DataFrame, loaded: Loaded, costs: CostModel):
    raw = price_trades(build_trades(signals, loaded.win_bars), MES, costs, CONTRACTS)
    standard, loss_halts, dd_halts = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=True)
    comparable, _, _ = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=False)
    return standard, comparable, len(loss_halts), len(dd_halts)


def reproduce(live: NullLive | None = None) -> bool:
    live = NullLive() if live is None else live
    with live:
        say = live.progress(lambda m: print(m, flush=True))
        print(LINE)
        say("ENTRY 15 reproduction check (pre-registered test 1)")
        print(LINE, flush=True)
        say("[progress] loading bars and building the calendar")
        loaded = load()
        cal = scored_calendar(loaded)
        expiries = set(cal[(cal["label"] == QUARTERLY) & cal["eligible"]]["date"])
        say("[progress] deriving sd_open from the Friday control")
        pop = scored_population(loaded)
        params = QuarterlyParams(sd_open=sd_open_from(pop), k=PARAMS_K, s=PARAMS_S)
        say(f"[progress] sd_open {params.sd_open:.2f} points -> threshold "
            f"{params.threshold_points:.2f}, stop {params.stop_points:.2f}")
        say("[progress] generating the fade arm's signals")
        _, diag = generate(loaded, params)
        diag = diag[[d >= START for d in diag.index]]
        problems = reproduction_differences(expiries, diag, params.threshold_points)
        for p in problems:
            say(f"  DIVERGES - {p}")
        if not problems:
            say(f"  IDENTICAL: {len(expiries)} eligible quarterly expiry days reproduced day for "
                f"day; {int(diag['entered'].astype(bool).sum())} entries, all at the 10:00 bar "
                f"beyond the threshold")
        print(LINE)
        print("REPRODUCTION " + ("PASSED" if not problems else "FAILED"))
        print(LINE)
        live.finish("REPRODUCED" if not problems else "DIVERGED")
        return not problems


# ---------------------------------------------------------------------------
# Criteria and the verdict
# ---------------------------------------------------------------------------


def evaluate_criteria(mechanism: dict, std_1t: dict, cmp_1t: dict, std_2t: dict) -> list[Criterion]:
    t = float(mechanism["impact"]["t"])
    diff_pts, z = float(mechanism["reversal"]["diff_points"]), float(mechanism["reversal"]["z"])
    p = float(std_1t["pass_probability"])
    blown = int(cmp_1t["blowups"])

    def rule13(s: dict) -> bool:
        return int(s["profitable_years"]) >= MIN_PROFITABLE_FOLDS and float(s["net_pnl"]) > 0

    return [
        Criterion("impact", "Opening impact: expiry-day mean |09:30-to-10:00 move| against the Friday control, Welch t",
                  f"t = {t:+.2f}", f"t >= {MIN_T:.1f}", bool(np.isfinite(t) and t >= MIN_T)),
        Criterion("reversal", "Reversal share, expiry minus control, and its two-proportion z",
                  f"{diff_pts:+.1f} points, z = {z:+.2f}",
                  f">= +{MIN_REVERSAL_POINTS:.0f} points and z >= {MIN_Z:.2f}",
                  bool(np.isfinite(diff_pts) and diff_pts >= MIN_REVERSAL_POINTS and z >= MIN_Z)),
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
    imp, rev, cond = mech["impact"], mech["reversal"], mech["conditional"]
    rt1 = round_turn_points(CostModel(slippage_ticks=BASE_SLIPPAGE_TICKS))
    lines = [f"### Verdict: {status}", ""]
    if status == "ACCEPTED":
        lines.append(
            "**Measured on 26 quarterly expiry days, and the fade arm's cost basis rests on a "
            "slippage assumption no backtest can measure.** Both were pre-registered; an "
            "ACCEPTED here is conditional on both and says so on its face.")
    else:
        lines.append(
            "**Rejected on the pre-registered criteria.** The settlement-open unwind, read on "
            "MES's opening move and its reversal and a fixed fade rule over 2020–2026, did not "
            "clear the lines set before the run. With 26 days the entry could detect a large "
            "effect and nothing else; that was stated before the run.")
    lines += [
        "",
        f"**Date:** {date_text}",
        "**Code:** `strategies/quarterly.py`, `backtests/run_entry15.py`, `research/power_check_quarterly.py`",
        "**Reports:** `backtests/results/entry15_report.txt`; `entry15_moves.csv` (every "
        "eligible session with its label, opening move, afternoon move and reversal flag); "
        "`entry15_fade_slip<1|2>.csv` (standard) and `_nohalt.csv` (comparable); "
        "`entry15_folds.csv` (fade arm, 1 tick, standard)",
        "",
        f"Quarterly expiry days by the cash calendar, monthly expiry days excluded from both "
        f"populations, control every other eligible Friday. Fade arm: **k = {PARAMS_K:g}, "
        f"s = {PARAMS_S:g}** in units of `sd_open`, entry at the 10:00 open, target the day's "
        f"open, flat at 15:55, **{CONTRACTS} contract**, ${rules.COMMISSION_PER_SIDE:.2f} a "
        f"side, **{BASE_SLIPPAGE_TICKS:g} tick a side as the base case** ({rt1:.2f} points a "
        f"round turn), {START} .. {END}, nothing selected. Guards: "
        f"`engine.apply_internal_guards` at {CONTRACTS} contract; halt OFF alongside.",
        "",
        f"**sd_open = {results['sd_open']:.2f} points** (Friday control, 09:30-to-10:00 move, "
        f"derived at run time) -> threshold {results['threshold_points']:.2f} points, stop "
        f"{results['stop_points']:.2f} points. Worst case of one trade: "
        f"${CONTRACTS * results['stop_points'] * MES.point_value:,.0f} plus costs against the "
        f"${rules.DAILY_LOSS_LIMIT:,.0f} daily limit.",
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
        "#### The mechanism tests: opening impact and reversal, quarterly expiry days against the Friday control",
        "",
        f"**Opening impact, |09:30-to-10:00 move| in points:** expiry n = {imp['n_a']}, mean "
        f"{_fmt(imp['mean_a'], '.2f')}, median {_fmt(imp['median_a'], '.2f')}; control n = "
        f"{imp['n_b']}, mean {_fmt(imp['mean_b'], '.2f')}, median {_fmt(imp['median_b'], '.2f')}. "
        f"**Difference {_fmt(imp['diff'])} points ({_fmt(imp['ratio_pct'], '+.1f')}%), Welch "
        f"t = {_fmt(imp['t'])}**, one-sided p (expiry larger) = {_fmt(imp['p_one_sided'], '.3f')}.",
        "",
        f"**Reversal share:** expiry {rev['reversals_expiry']} of {rev['n_expiry']} "
        f"({_fmt(100 * rev['share_expiry'], '.1f')}%), control {rev['reversals_control']} of "
        f"{rev['n_control']} ({_fmt(100 * rev['share_control'], '.1f')}%). **Difference "
        f"{_fmt(rev['diff_points'], '+.1f')} points, z = {_fmt(rev['z'])}.**",
        "",
        f"**Conditional reversion, reported:** expiry days after an up open (n = "
        f"{cond['expiry']['n_up']}) the afternoon moved {_fmt(cond['expiry']['mean_afternoon_after_up'])} "
        f"points on average, after a down open (n = {cond['expiry']['n_down']}) "
        f"{_fmt(cond['expiry']['mean_afternoon_after_down'])}; control days "
        f"{_fmt(cond['control']['mean_afternoon_after_up'])} and "
        f"{_fmt(cond['control']['mean_afternoon_after_down'])}.",
        "",
        "| Year | Expiry n | Expiry mean |o| | Control n | Control mean |o| | Expiry above |",
        "|---|---|---|---|---|---|",
    ]
    for r in mech["per_year"]:
        lines.append(f"| {r['year']} | {r['n_expiry']} | {_fmt(r['expiry_mean_abs'], '.2f')} | "
                     f"{r['n_control']} | {_fmt(r['control_mean_abs'], '.2f')} | "
                     f"{'yes' if r['above'] else 'no'} |")
    lines += ["", f"Expiry-day mean |o| above the control's in **{mech['years_expiry_impact_above']} "
                  f"of {TOTAL_FOLDS}** years (reported; not a criterion)."]
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
    skipped = diag["skipped"]
    lines += ["", "#### Diagnostics", "",
              f"Quarterly expiry days skipped in the scored span: {len(skipped)}"
              + (" (" + ", ".join(f"{r['date']} {r['skipped_reason']}" for _, r in skipped.iterrows()) + ")"
                 if len(skipped) else "")
              + f". Expiry days with no trade (|o| inside the threshold): {diag['inside_threshold']}. "
              f"Entry-bar stop breaches: {diag['entry_bar_breaches']}. Thursday expiries: "
              + (", ".join(str(d) for d in diag["thursday_expiries"]) or "none") + ".",
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

    progress("[progress] building the opening-move population and the mechanism tests")
    pop = scored_population(loaded)
    pop.to_csv(RESULTS / "entry15_moves.csv", index=False)
    mechanism = mechanism_test(pop)

    sd = sd_open_from(pop)
    params = QuarterlyParams(sd_open=sd, k=PARAMS_K, s=PARAMS_S)
    progress(f"[progress] sd_open {sd:.2f} points -> threshold {params.threshold_points:.2f}, "
             f"stop {params.stop_points:.2f}")

    progress("[progress] fade arm: generating signals")
    signals, diag = generate(loaded, params)
    diag = diag[[d >= START for d in diag.index]]
    cal = scored_calendar(loaded)
    skipped = cal[(cal["label"] == QUARTERLY) & ~cal["eligible"]][["date", "skipped_reason"]]
    diagnostics = {
        "skipped": skipped.reset_index(drop=True),
        "inside_threshold": int((diag["skipped_reason"] == SKIP_INSIDE_THRESHOLD).sum()),
        "entry_bar_breaches": int(diag["entry_bar_breach"].fillna(False).astype(bool).sum()),
        "thursday_expiries": [d for d in cal[cal["label"] == QUARTERLY]["date"] if d.weekday() == 3],
    }

    arms: dict = {}
    metrics_standard_1t: dict = {}
    for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
        progress(f"[progress] fade arm: pricing and guarding at {ticks:g} tick(s)")
        costs = CostModel(commission_per_side=commission, slippage_ticks=ticks)
        standard, comparable, loss_halts, dd_halts = guarded_streams(signals, loaded, costs)
        standard.to_csv(RESULTS / f"entry15_fade_slip{ticks:g}.csv", index=False)
        comparable.to_csv(RESULTS / f"entry15_fade_slip{ticks:g}_nohalt.csv", index=False)
        arms[ticks] = {
            "standard": stream_stats(standard, paths, comparable, dd_halts, loss_halts),
            "comparable": stream_stats(comparable, paths, None, 0, loss_halts),
        }
        if ticks == BASE_SLIPPAGE_TICKS:
            live.trades(standard, "standard")
            live.trades(comparable, "comparable")
            folds = fold_frame(standard, paths)
            folds.to_csv(RESULTS / "entry15_folds.csv", index=False)
            live.folds(folds)
            metrics_standard_1t = compute_metrics(standard) if len(standard) else {}

    criteria = evaluate_criteria(mechanism, arms[BASE_SLIPPAGE_TICKS]["standard"],
                                 arms[BASE_SLIPPAGE_TICKS]["comparable"],
                                 arms[SENSITIVITY_TICKS[0]]["standard"])
    status = verdict_status(criteria)
    if metrics_standard_1t.get("hold_regression"):
        status = "REJECTED"
    results = {"status": status, "criteria": criteria, "mechanism": mechanism,
               "sd_open": sd, "threshold_points": params.threshold_points,
               "stop_points": params.stop_points, "arms": arms, "diagnostics": diagnostics,
               "metrics_standard_1t": metrics_standard_1t}
    block = verdict_block(results, str(pd.Timestamp.now(tz=rules.ET).date()))
    (RESULTS / "entry15_verdict.md").write_text(block, encoding="utf-8")
    (RESULTS / "entry15_report.txt").write_text(
        "\n".join([LINE, "ENTRY 15 - quarterly futures expiry, index-arbitrage unwind at the open",
                   f"  ${commission:.2f}/side; base {BASE_SLIPPAGE_TICKS:g} tick/side; "
                   f"sensitivity {', '.join(f'{t:g}' for t in SENSITIVITY_TICKS)}", LINE, block]),
        encoding="utf-8")
    progress("[progress] verdict written to backtests/results/entry15_verdict.md")
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
        live_run = LiveRun("entry15", runner="run_entry15 --reproduce") if args.live else NullLive()
        return 0 if reproduce(live=live_run) else 1
    if args.run:
        live_run = LiveRun("entry15", runner="run_entry15 --run") if args.live else NullLive()
        _, status, _ = run(args.paths, args.commission,
                           progress=lambda m: print(m, flush=True), live=live_run)
        print(f"[done] {status}", flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
