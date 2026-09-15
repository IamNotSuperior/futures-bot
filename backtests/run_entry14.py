"""Entry 14: monthly option expiry, dealer gamma hedging, intraday on MES.
Frozen at ``f9f129c``; calendar addendum and power check at ``f79cd49``.

    python backtests/run_entry14.py --reproduce          # pre-registered test 1, run first
    python backtests/run_entry14.py --run [--live]       # the range test, the fade arm, the verdict

The mechanism test is on the log session range, expiry days against every
other eligible Friday (primary control), Welch t with the entry's sign
convention (expiry minus control; damping reads negative) and a one-sided p
for expiry smaller. The fade arm enters at the 10:30 open when the morning
move exceeds ``k`` control standard deviations, targets the day's open, stops
``s`` standard deviations away, three contracts, flat at 15:55. ``sd_move``
is derived from the primary control at run time and reported; no expiry-day
outcome is read to set it.

Everything the verdict quotes is written to ``backtests/results/entry14_*``
by this runner before anyone reads a number. ``--commission`` exists so a
verdict can be reproduced at another rate; it defaults to
``rules.COMMISSION_PER_SIDE``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date as date_type
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
from opex import (  # noqa: E402
    CONTROL_FRIDAY, CONTROL_OTHER, EXPIRY, SKIP_INSIDE_THRESHOLD, OpexFade, OpexParams,
    session_calendar, session_ranges,
)
import run_entry11 as e11  # noqa: E402
from run_entry11 import (  # noqa: E402
    BASE_SLIPPAGE_TICKS, END, LINE, MAX_BLOWUPS, MIN_PASS_PROBABILITY,
    MIN_PROFITABLE_FOLDS, SENSITIVITY_TICKS, START, TOTAL_FOLDS, YEARS, Criterion,
    Loaded, round_turn_points, stream_stats, welch,
)
from run_generated import hold_regression_line  # noqa: E402
from run_london import fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS = PROJECT_ROOT / "backtests" / "results"

#: The operator's specification, as frozen.
CONTRACTS = 3
PARAMS_K = 0.5
PARAMS_S = 1.0

#: Entry 14's kill criteria, as frozen. Any one failure kills the entry.
MAX_T = -2.0                    # expiry minus control, one-sided: damping
MIN_MEDIAN_DAMPING_PCT = 5.0    # expiry median range at least this much smaller
MIN_YEARS_EXPIRY_BELOW = 4      # of TOTAL_FOLDS


# ---------------------------------------------------------------------------
# Loading and populations
# ---------------------------------------------------------------------------


def load(parquet: Path = PARQUET) -> Loaded:
    bars = loader.load_bars(str(parquet))
    return Loaded(bars=bars, win_bars=slice_by_date(bars, START, END),
                  rolls=set(loader.detect_roll_dates(bars)),
                  early=set(loader.detect_early_close_dates(bars)))


def scored_population(loaded: Loaded) -> pd.DataFrame:
    """The range population over the whole file, sliced to the scored span."""
    pop = session_ranges(loaded.bars, loaded.rolls, loaded.early)
    keep = (pop["date"] >= START) & (pop["date"] <= END)
    return pop[keep].reset_index(drop=True)


def scored_calendar(loaded: Loaded) -> pd.DataFrame:
    cal = session_calendar(loaded.bars, loaded.rolls, loaded.early)
    keep = (cal["date"] >= START) & (cal["date"] <= END)
    return cal[keep].reset_index(drop=True)


def sd_move_from(pop: pd.DataFrame) -> float:
    """Spread of the primary control's morning move - no expiry day is read."""
    fri = pop[pop["label"] == CONTROL_FRIDAY]["morning_move"].astype(float)
    return float(fri.std(ddof=1))


# ---------------------------------------------------------------------------
# The mechanism test
# ---------------------------------------------------------------------------


def _damping(expiry: pd.DataFrame, control: pd.DataFrame, column: str = "log_range",
             raw: str = "range_points") -> dict:
    """Welch on ``column``, expiry minus control, one-sided p for expiry
    smaller, plus the median ratio in percent on the raw column. A side with
    fewer than two rows has no variance and reads as no result."""
    if len(expiry) < 2 or len(control) < 2:
        return {"n_a": int(len(expiry)), "n_b": int(len(control)), "mean_a": float("nan"),
                "mean_b": float("nan"), "sd_a": float("nan"), "sd_b": float("nan"),
                "median_a": float("nan"), "median_b": float("nan"), "diff": float("nan"),
                "t": float("nan"), "df": float("nan"), "p_one_sided": float("nan"),
                "median_ratio_pct": float("nan"), "median_expiry": float("nan"),
                "median_control": float("nan")}
    w = welch(expiry[column].to_numpy(float), control[column].to_numpy(float))
    if max(w["sd_a"], w["sd_b"]) < 1e-9:
        # Both sides constant to rounding: a difference of 1e-16 over a
        # standard error of 1e-17 is not a t statistic.
        w.update(t=0.0, df=float("nan"), p_one_sided=0.5)
    w["p_one_sided"] = 1.0 - w["p_one_sided"] if np.isfinite(w["t"]) else 0.5
    med_e, med_c = float(expiry[raw].median()), float(control[raw].median())
    w["median_ratio_pct"] = (med_e / med_c - 1.0) * 100.0 if med_c > 0 else float("nan")
    w["median_expiry"], w["median_control"] = med_e, med_c
    return w


def mechanism_test(pop: pd.DataFrame) -> dict:
    expiry = pop[pop["label"] == EXPIRY]
    friday = pop[pop["label"] == CONTROL_FRIDAY]
    other = pop[pop["label"] == CONTROL_OTHER]

    per_year = []
    for year in YEARS:
        e = expiry[expiry["year"] == year]["range_points"]
        f = friday[friday["year"] == year]["range_points"]
        em, fm = float(e.median()) if len(e) else float("nan"), float(f.median()) if len(f) else float("nan")
        per_year.append({"year": year, "n_expiry": int(len(e)), "n_friday": int(len(f)),
                         "expiry_median": em, "friday_median": fm,
                         "below": bool(np.isfinite(em) and np.isfinite(fm) and em < fm)})

    q, m = expiry[expiry["quarterly"]], expiry[~expiry["quarterly"]]
    quarterly = {
        "quarterly": {"n": int(len(q)), "median": float(q["range_points"].median()) if len(q) else float("nan"),
                      "mean_log": float(q["log_range"].mean()) if len(q) else float("nan")},
        "monthly": {"n": int(len(m)), "median": float(m["range_points"].median()) if len(m) else float("nan"),
                    "mean_log": float(m["log_range"].mean()) if len(m) else float("nan")},
    }
    return {
        "primary": _damping(expiry, friday),
        "secondary": _damping(expiry, other),
        "abs_move": _damping(expiry, friday, column="abs_move", raw="abs_move"),
        "quarterly": quarterly,
        "per_year": per_year,
        "years_expiry_below_friday": int(sum(1 for r in per_year if r["below"])),
    }


# ---------------------------------------------------------------------------
# The fade arm
# ---------------------------------------------------------------------------


def generate(loaded: Loaded, params: OpexParams):
    strat = OpexFade(params, roll_dates=loaded.rolls, early_close_dates=loaded.early)
    signals = strat.generate_signals(loaded.bars)
    return slice_by_date(signals, START, END), strat.diagnostics


def guarded_streams(signals: pd.DataFrame, loaded: Loaded, costs: CostModel):
    raw = price_trades(build_trades(signals, loaded.win_bars), MES, costs, CONTRACTS)
    standard, loss_halts, dd_halts = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=True)
    comparable, _, _ = apply_internal_guards(
        raw, loaded.win_bars, MES, costs, CONTRACTS, trailing_halt=False)
    return standard, comparable, len(loss_halts), len(dd_halts)


# ---------------------------------------------------------------------------
# Reproduction (pre-registered test 1)
# ---------------------------------------------------------------------------


def reproduction_differences(expiry_dates: set, diagnostics: pd.DataFrame,
                             threshold: float) -> list[str]:
    """The strategy's diagnostics against the calendar's eligible expiry days."""
    problems: list[str] = []
    diag_dates = set(diagnostics.index)
    missing = sorted(expiry_dates - diag_dates)
    if missing:
        problems.append(f"{len(missing)} expiry day(s) absent from the strategy's diagnostics: "
                        f"{[str(d) for d in missing[:5]]}")
    entered = diagnostics[diagnostics["entered"].astype(bool)]
    extra = sorted(set(entered.index) - expiry_dates)
    if extra:
        problems.append(f"entries on {len(extra)} non-expiry day(s): {[str(d) for d in extra[:5]]}")
    inside = entered[entered["morning_move"].abs() < threshold]
    if len(inside):
        problems.append(f"{len(inside)} entry(ies) with |morning move| inside the threshold "
                        f"{threshold:.2f}: {[str(d) for d in inside.index[:5]]}")
    skipped_inside = diagnostics[(~diagnostics["entered"].astype(bool))
                                 & (diagnostics["skipped_reason"] == SKIP_INSIDE_THRESHOLD)]
    beyond = skipped_inside[skipped_inside["morning_move"].abs() >= threshold]
    if len(beyond):
        problems.append(f"{len(beyond)} day(s) skipped as inside the threshold with |m| beyond it")
    return problems


def reproduce(live: NullLive | None = None) -> bool:
    live = NullLive() if live is None else live
    with live:
        say = live.progress(lambda m: print(m, flush=True))
        print(LINE)
        say("ENTRY 14 reproduction check (pre-registered test 1)")
        print(LINE, flush=True)
        say("[progress] loading bars and building the calendar")
        loaded = load()
        cal = scored_calendar(loaded)
        expiries = set(cal[(cal["label"] == EXPIRY) & cal["eligible"]]["date"])
        say("[progress] deriving sd_move from the Friday control")
        pop = scored_population(loaded)
        params = OpexParams(sd_move=sd_move_from(pop), k=PARAMS_K, s=PARAMS_S)
        say(f"[progress] sd_move {params.sd_move:.2f} points -> threshold "
            f"{params.threshold_points:.2f}, stop {params.stop_points:.2f}")
        say("[progress] generating the fade arm's signals")
        _, diag = generate(loaded, params)
        diag = diag[[d >= START for d in diag.index]]
        problems = reproduction_differences(expiries, diag, params.threshold_points)
        for p in problems:
            say(f"  DIVERGES - {p}")
        if not problems:
            n_entered = int(diag["entered"].astype(bool).sum())
            say(f"  IDENTICAL: {len(expiries)} eligible expiry days reproduced day for day; "
                f"{n_entered} entries, all at the 10:30 bar beyond the threshold")
        print(LINE)
        print("REPRODUCTION " + ("PASSED" if not problems else "FAILED"))
        print(LINE)
        live.finish("REPRODUCED" if not problems else "DIVERGED")
        return not problems


# ---------------------------------------------------------------------------
# Criteria and the verdict
# ---------------------------------------------------------------------------


def evaluate_criteria(mechanism: dict, std_1t: dict, cmp_1t: dict, std_2t: dict) -> list[Criterion]:
    prim = mechanism["primary"]
    t, ratio = float(prim["t"]), float(prim["median_ratio_pct"])
    years = int(mechanism["years_expiry_below_friday"])
    p = float(std_1t["pass_probability"])
    blown = int(cmp_1t["blowups"])

    def rule13(s: dict) -> bool:
        return int(s["profitable_years"]) >= MIN_PROFITABLE_FOLDS and float(s["net_pnl"]) > 0

    return [
        Criterion("damping", "Expiry-day log range against the Friday control: Welch t and median ratio",
                  f"t = {t:+.2f}, median {ratio:+.1f}%",
                  f"t <= {MAX_T:.1f} and median <= -{MIN_MEDIAN_DAMPING_PCT:.0f}%",
                  bool(np.isfinite(t) and t <= MAX_T and np.isfinite(ratio)
                       and ratio <= -MIN_MEDIAN_DAMPING_PCT)),
        Criterion("years", "Years in which the expiry-day median range is below the Friday control's",
                  f"{years} of {TOTAL_FOLDS}", f">= {MIN_YEARS_EXPIRY_BELOW} of {TOTAL_FOLDS}",
                  years >= MIN_YEARS_EXPIRY_BELOW),
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
    prim, sec, absm = mech["primary"], mech["secondary"], mech["abs_move"]
    rt1 = round_turn_points(CostModel(slippage_ticks=BASE_SLIPPAGE_TICKS))
    lines = [f"### Verdict: {status}", ""]
    if status == "ACCEPTED":
        lines.append(
            "**The damping was measured on the intraday 09:30-to-15:54 range only, on 80 "
            "expiry days, and the fade arm's cost basis rests on a slippage assumption no "
            "backtest can measure.** Both were pre-registered; an ACCEPTED here is "
            "conditional on both and says so on its face.")
    else:
        lines.append(
            "**Rejected on the pre-registered criteria.** Dealer gamma hedging on monthly "
            "expiry days, read on MES's intraday range and a fixed fade rule over 2020–2026, "
            "did not clear the lines set before the run.")
    lines += [
        "",
        f"**Date:** {date_text}",
        "**Code:** `strategies/opex.py`, `backtests/run_entry14.py`, `research/power_check_opex.py`",
        "**Reports:** `backtests/results/entry14_report.txt`; `entry14_ranges.csv` (every "
        "eligible session with its label, range, log range, |move| and morning move); "
        "`entry14_fade_slip<1|2>.csv` (standard) and `_nohalt.csv` (comparable); "
        "`entry14_folds.csv` (fade arm, 1 tick, standard)",
        "",
        f"Expiry days by the cash calendar (third Friday, Thursday when the cash market is "
        f"shut), primary control every other eligible Friday. Fade arm: **k = {PARAMS_K:g}, "
        f"s = {PARAMS_S:g}** in units of `sd_move`, entry at the 10:30 open, target the day's "
        f"open, flat at 15:55, **{CONTRACTS} contracts**, ${rules.COMMISSION_PER_SIDE:.2f} a "
        f"side, **{BASE_SLIPPAGE_TICKS:g} tick a side as the base case** ({rt1:.2f} points a "
        f"round turn), {START} .. {END}, nothing selected. Guards: "
        f"`engine.apply_internal_guards` at {CONTRACTS} contracts; halt OFF alongside.",
        "",
        f"**sd_move = {results['sd_move']:.2f} points** (Friday control, 09:30-to-10:30 move, "
        f"derived at run time) -> threshold {results['threshold_points']:.2f} points, stop "
        f"{results['stop_points']:.2f} points. Worst case of one trade at {CONTRACTS} contracts: "
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
        "#### The mechanism test: session range, expiry days against controls",
        "",
        f"**Primary (other Fridays):** expiry n = {prim['n_a']}, mean log range "
        f"{_fmt(prim['mean_a'], '.3f')}, median range {_fmt(prim.get('median_expiry'), '.2f')} points; "
        f"control n = {prim['n_b']}, mean log range {_fmt(prim['mean_b'], '.3f')}, median range "
        f"{_fmt(prim.get('median_control'), '.2f')}. **Difference {_fmt(prim['diff'], '+.3f')} in "
        f"log, Welch t = {_fmt(prim['t'])}**, one-sided p (expiry smaller) = "
        f"{_fmt(prim['p_one_sided'], '.3f')}; **median ratio {_fmt(prim['median_ratio_pct'], '+.1f')}%.**",
        "",
        f"**Secondary (all other sessions), reported:** control n = {sec['n_b']}, difference "
        f"{_fmt(sec['diff'], '+.3f')}, t = {_fmt(sec['t'])}, p = {_fmt(sec['p_one_sided'], '.3f')}, "
        f"median ratio {_fmt(sec['median_ratio_pct'], '+.1f')}%.",
        "",
        f"**|open-to-15:55 move|, expiry against Fridays, reported:** difference "
        f"{_fmt(absm['diff'], '+.2f')} points, t = {_fmt(absm['t'])}, p = "
        f"{_fmt(absm['p_one_sided'], '.3f')}, median ratio {_fmt(absm['median_ratio_pct'], '+.1f')}%.",
        "",
        "| Year | Expiry n | Expiry median range | Friday n | Friday median range | Expiry below |",
        "|---|---|---|---|---|---|",
    ]
    for r in mech["per_year"]:
        lines.append(f"| {r['year']} | {r.get('n_expiry', '')} | {_fmt(r['expiry_median'], '.2f')} | "
                     f"{r.get('n_friday', '')} | {_fmt(r['friday_median'], '.2f')} | "
                     f"{'yes' if r['below'] else 'no'} |")
    q = mech["quarterly"]
    lines += [
        "",
        f"Expiry-day median range below the Friday control's in **{mech['years_expiry_below_friday']} "
        f"of {TOTAL_FOLDS}** years.",
        "",
        f"**Quarterly against monthly expiry days, reported and not selected on:** quarterly n = "
        f"{q['quarterly']['n']}, median range {_fmt(q['quarterly']['median'], '.2f')}; monthly n = "
        f"{q['monthly']['n']}, median range {_fmt(q['monthly']['median'], '.2f')}.",
    ]
    for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
        r = results["arms"][ticks]
        tag = "base case" if ticks == BASE_SLIPPAGE_TICKS else "sensitivity"
        lines += ["", f"#### Fade arm — {ticks:g} tick{'s' if ticks != 1 else ''} ({tag})", ""]
        lines += e11._basis_table(f"2020–2026, {CONTRACTS} contracts, {ticks:g} "
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
              f"Expiry days skipped in the scored span: {len(skipped)}"
              + (" (" + ", ".join(f"{r['date']} {'quarterly' if r['quarterly'] else 'monthly'} "
                                  f"{r['skipped_reason']}" for _, r in skipped.iterrows()) + ")"
                 if len(skipped) else "")
              + f". Expiry days with no trade (|m| inside the threshold): {diag['inside_threshold']}. "
              f"Entry-bar stop breaches: {diag['entry_bar_breaches']}. Thursday expiries: "
              + ", ".join(str(d) for d in diag["thursday_expiries"]) + ".",
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
    """The range test, the fade arm at 1 and 2 ticks, the verdict.

    ``live`` is the optional event stream for ``bots/liveview.py``; the
    default writes nothing and changes nothing.
    """
    live = NullLive() if live is None else live
    with live:
        return _run(paths, commission, live.progress(progress), live)


def _run(paths: int, commission: float | None, progress, live: NullLive):
    RESULTS.mkdir(parents=True, exist_ok=True)
    commission = rules.COMMISSION_PER_SIDE if commission is None else commission
    progress("[progress] loading bars")
    loaded = load()

    progress("[progress] building the range population and the mechanism test")
    pop = scored_population(loaded)
    pop.to_csv(RESULTS / "entry14_ranges.csv", index=False)
    mechanism = mechanism_test(pop)

    sd = sd_move_from(pop)
    params = OpexParams(sd_move=sd, k=PARAMS_K, s=PARAMS_S)
    progress(f"[progress] sd_move {sd:.2f} points -> threshold {params.threshold_points:.2f}, "
             f"stop {params.stop_points:.2f}")

    progress("[progress] fade arm: generating signals")
    signals, diag = generate(loaded, params)
    diag = diag[[d >= START for d in diag.index]]
    cal = scored_calendar(loaded)
    skipped = cal[(cal["label"] == EXPIRY) & ~cal["eligible"]][["date", "quarterly", "skipped_reason"]]
    diagnostics = {
        "skipped": skipped.reset_index(drop=True),
        "inside_threshold": int((diag["skipped_reason"] == SKIP_INSIDE_THRESHOLD).sum()),
        "entry_bar_breaches": int(diag["entry_bar_breach"].fillna(False).astype(bool).sum()),
        "thursday_expiries": [d for d in cal[cal["label"] == EXPIRY]["date"] if d.weekday() == 3],
    }

    arms: dict = {}
    metrics_standard_1t: dict = {}
    for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
        progress(f"[progress] fade arm: pricing and guarding at {ticks:g} tick(s)")
        costs = CostModel(commission_per_side=commission, slippage_ticks=ticks)
        standard, comparable, loss_halts, dd_halts = guarded_streams(signals, loaded, costs)
        standard.to_csv(RESULTS / f"entry14_fade_slip{ticks:g}.csv", index=False)
        comparable.to_csv(RESULTS / f"entry14_fade_slip{ticks:g}_nohalt.csv", index=False)
        arms[ticks] = {
            "standard": stream_stats(standard, paths, comparable, dd_halts, loss_halts),
            "comparable": stream_stats(comparable, paths, None, 0, loss_halts),
        }
        if ticks == BASE_SLIPPAGE_TICKS:
            live.trades(standard, "standard")
            live.trades(comparable, "comparable")
            folds = fold_frame(standard, paths)
            folds.to_csv(RESULTS / "entry14_folds.csv", index=False)
            live.folds(folds)
            metrics_standard_1t = compute_metrics(standard) if len(standard) else {}

    criteria = evaluate_criteria(mechanism, arms[BASE_SLIPPAGE_TICKS]["standard"],
                                 arms[BASE_SLIPPAGE_TICKS]["comparable"],
                                 arms[SENSITIVITY_TICKS[0]]["standard"])
    status = verdict_status(criteria)
    if metrics_standard_1t.get("hold_regression"):
        status = "REJECTED"  # a stream not produced under rule 6 decides nothing
    results = {"status": status, "criteria": criteria, "mechanism": mechanism,
               "sd_move": sd, "threshold_points": params.threshold_points,
               "stop_points": params.stop_points, "arms": arms, "diagnostics": diagnostics,
               "metrics_standard_1t": metrics_standard_1t}
    block = verdict_block(results, str(pd.Timestamp.now(tz=rules.ET).date()))
    (RESULTS / "entry14_verdict.md").write_text(block, encoding="utf-8")
    (RESULTS / "entry14_report.txt").write_text(
        "\n".join([LINE, "ENTRY 14 - monthly option expiry, dealer gamma hedging, intraday",
                   f"  ${commission:.2f}/side; base {BASE_SLIPPAGE_TICKS:g} tick/side; "
                   f"sensitivity {', '.join(f'{t:g}' for t in SENSITIVITY_TICKS)}", LINE, block]),
        encoding="utf-8")
    progress("[progress] verdict written to backtests/results/entry14_verdict.md")
    live.finish(status)
    return results, status, block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", action="store_true", help="pre-registered test 1, run first")
    ap.add_argument("--run", action="store_true", help="the range test, the fade arm, the verdict")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--commission", type=float, default=rules.COMMISSION_PER_SIDE)
    ap.add_argument("--live", action="store_true",
                    help="write a live event stream for bots/liveview.py")
    args = ap.parse_args(argv)
    if args.reproduce:
        live_run = LiveRun("entry14", runner="run_entry14 --reproduce") if args.live else NullLive()
        return 0 if reproduce(live=live_run) else 1
    if args.run:
        live_run = LiveRun("entry14", runner="run_entry14 --run") if args.live else NullLive()
        _, status, _ = run(args.paths, args.commission,
                           progress=lambda m: print(m, flush=True), live=live_run)
        print(f"[done] {status}", flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
