"""Entry 12: turn-of-month intraday, replicated on MNQ (Part A) and on forward
MES data (Part B). Frozen at ``b632028``.

Entry 11's test, unchanged, on data it never touched: the cash trading
calendar, the four window days, long 09:30 to 15:55, the control of every
eligible non-window session, Welch's t at 2.0. What changes is what the
contract and the trail dictate: one contract, and a stop derived from the
instrument's own control standard deviation at the fraction entry 11's
15-point stop was of MES's.

    python backtests/run_entry12.py --part A --reproduce
    python backtests/run_entry12.py --part A --run
    python backtests/run_entry12.py --part B --parquet <forward MES file> --run

Part B refuses to run until the forward file holds at least
``MIN_FORWARD_WINDOW_SESSIONS`` eligible window sessions, as pre-registered.
Everything decided is imported from ``run_entry11`` or computed from the cost
model; nothing is restated.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from engine import (  # noqa: E402
    MES, MNQ, ContractSpec, CostModel, apply_internal_guards, build_trades,
    price_trades,
)
from live import LiveRun, NullLive  # noqa: E402
import run_entry11 as e11  # noqa: E402
from run_entry11 import (  # noqa: E402
    BASE_SLIPPAGE_TICKS, LINE, SENSITIVITY_TICKS, STOP_POINTS, TOTAL_FOLDS, YEARS,
    Criterion, evaluate_criteria, mechanism_test, round_turn_points, stream_stats,
    verdict_status,
)
from run_london import fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402
from tom import CONTROL_LABEL, TOMParams, TurnOfMonth, session_returns  # noqa: E402
from walkforward import DATA_END, build_folds  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"

#: The operator's specification, as frozen.
CONTRACTS = 1
#: Part B runs only once the forward MES file holds this many eligible window
#: sessions - about two calendar years.
MIN_FORWARD_WINDOW_SESSIONS = 96

#: MES's control-population standard deviation from entry 11's verdict
#: (commit 2bf4fc4: "control n = 1,332, mean +0.26, sd 40.99"). The stop
#: fraction is entry 11's 15 points over this figure. ``mes_control_sd_check``
#: recomputes it from the bars so the constant cannot drift from its source.
ENTRY11_MES_CONTROL_SD = 40.99


@dataclass(frozen=True)
class Part:
    key: str
    label: str
    spec: ContractSpec
    parquet: Path | None
    start: date
    end: date | None


PARTS: dict[str, Part] = {
    "A": Part("A", "MNQ", MNQ,
              PROJECT_ROOT / "data" / "mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet",
              build_folds()[0].test_start, DATA_END),
    # Forward MES: the day after the cached file ends, to whatever the forward
    # file holds. The parquet is supplied on the command line when it exists.
    "B": Part("B", "MES forward", MES, None, DATA_END + pd.Timedelta(days=1).to_pytimedelta(), None),
}


# ---------------------------------------------------------------------------
# The stop, instrument-aware
# ---------------------------------------------------------------------------


def stop_fraction() -> float:
    """Entry 11's stop as a fraction of MES's control standard deviation."""
    return STOP_POINTS / ENTRY11_MES_CONTROL_SD


def round_to_tick(points: float, spec: ContractSpec) -> float:
    return float(round(points / spec.tick_size) * spec.tick_size)


def derive_stop_points(control_sd: float, spec: ContractSpec) -> float:
    return round_to_tick(stop_fraction() * control_sd, spec)


def control_sd(returns: pd.DataFrame) -> float:
    """Spread of the control population only - no window outcome is read."""
    ctrl = returns[returns["label"] == CONTROL_LABEL]["points"]
    return float(ctrl.std(ddof=1))


# ---------------------------------------------------------------------------
# Loading, populations, streams
# ---------------------------------------------------------------------------


Loaded = e11.Loaded


def load(part: Part, parquet: Path | None = None) -> Loaded:
    path = parquet or part.parquet
    if path is None:
        raise ValueError(f"part {part.key} needs --parquet: no cached file is its source")
    bars = loader.load_bars(str(path))
    end = part.end or bars.index[-1].date()
    return Loaded(bars=bars, win_bars=slice_by_date(bars, part.start, end),
                  rolls=set(loader.detect_roll_dates(bars)),
                  early=set(loader.detect_early_close_dates(bars)))


def scored_returns(loaded: Loaded, part: Part) -> pd.DataFrame:
    """The population over the whole file, sliced to the part's span - the
    same order as the signals, for the reason entry 11 recorded."""
    returns = session_returns(loaded.bars, loaded.rolls, loaded.early)
    end = part.end or loaded.bars.index[-1].date()
    keep = (returns["date"] >= part.start) & (returns["date"] <= end)
    return returns[keep].reset_index(drop=True)


def forward_ready(loaded: Loaded) -> tuple[bool, int]:
    """Part B's trigger: enough eligible forward window sessions."""
    n = int(scored_returns(loaded, PARTS["B"])["window"].sum())
    return n >= MIN_FORWARD_WINDOW_SESSIONS, n


def generate(loaded: Loaded, part: Part, params: TOMParams):
    strat = TurnOfMonth(params, roll_dates=loaded.rolls, early_close_dates=loaded.early)
    signals = strat.generate_signals(loaded.bars)
    end = part.end or loaded.bars.index[-1].date()
    return slice_by_date(signals, part.start, end), strat.diagnostics


def guarded_streams(win_signals: pd.DataFrame, loaded: Loaded, spec: ContractSpec,
                    costs: CostModel):
    raw = price_trades(build_trades(win_signals, loaded.win_bars), spec, costs, CONTRACTS)
    standard, loss_halts, dd_halts = apply_internal_guards(
        raw, loaded.win_bars, spec, costs, CONTRACTS, trailing_halt=True)
    comparable, _, _ = apply_internal_guards(
        raw, loaded.win_bars, spec, costs, CONTRACTS, trailing_halt=False)
    return standard, comparable, len(loss_halts), len(dd_halts)


def mes_control_sd_check() -> float:
    """Recompute entry 11's MES control sd from the bars. Raises if it has
    drifted from the recorded constant by more than rounding."""
    loaded = e11.load()
    sd = control_sd(e11.scored_returns(loaded))
    if abs(sd - ENTRY11_MES_CONTROL_SD) > 0.01:
        raise RuntimeError(f"MES control sd recomputed as {sd:.4f}, recorded "
                           f"{ENTRY11_MES_CONTROL_SD}; the stop fraction's source has moved")
    return sd


# ---------------------------------------------------------------------------
# Reproduction (pre-registered test 1)
# ---------------------------------------------------------------------------


def reproduction_differences(loaded: Loaded, part: Part) -> list[str]:
    problems: list[str] = []
    window = scored_returns(loaded, part)
    window = window[window["window"]].sort_values("date")
    sig_a, _ = generate(loaded, part, TOMParams())
    one = price_trades(build_trades(sig_a, loaded.win_bars), part.spec, e11.FREE, 1)
    ours, theirs = list(one["session_date"]), list(window["date"])
    if ours != theirs:
        problems.append(f"trade dates differ: {len(ours)} trades against {len(theirs)} "
                        f"window sessions; missing {sorted(set(theirs) - set(ours))[:5]}, "
                        f"extra {sorted(set(ours) - set(theirs))[:5]}")
    else:
        gap = np.abs(one["gross_points"].to_numpy() - window["points"].to_numpy())
        if gap.max() > 1e-9:
            problems.append(f"gross points differ on {ours[int(np.argmax(gap))]}")
    stop = derive_stop_points(control_sd(scored_returns(loaded, part)), part.spec)
    sig_b, _ = generate(loaded, part, TOMParams(stop_points=stop))
    ea, eb = sig_a[sig_a["entry_long"]], sig_b[sig_b["entry_long"]]
    if not ea.index.equals(eb.index):
        problems.append(f"stop arm entries differ: {len(ea)} against {len(eb)}")
    elif not np.allclose(ea["entry_price"].to_numpy(float), eb["entry_price"].to_numpy(float)):
        problems.append("stop arm entry prices differ from the signal arm's")
    return problems


def reproduce(part: Part, parquet: Path | None = None,
              live: NullLive | None = None) -> bool:
    live = NullLive() if live is None else live
    with live:
        say = live.progress(lambda m: print(m, flush=True))
        print(LINE)
        say(f"ENTRY 12 - part {part.key} ({part.label}) reproduction check (pre-registered test 1)")
        print(LINE, flush=True)
        say("[progress] loading bars and rebuilding the population")
        loaded = load(part, parquet)
        problems = reproduction_differences(loaded, part)
        for p in problems:
            say(f"  DIVERGES - {p}")
        if not problems:
            n = int(scored_returns(loaded, part)["window"].sum())
            say(f"  IDENTICAL: {n} window sessions reproduced session for session; "
                f"stop-arm entries identical")
        print(LINE)
        print("REPRODUCTION " + ("PASSED" if not problems else "FAILED"))
        print(LINE)
        live.finish("REPRODUCED" if not problems else "DIVERGED")
        return not problems


# ---------------------------------------------------------------------------
# Criteria and the block
# ---------------------------------------------------------------------------


def criteria_for(mechanism: dict, stop_std_1t: dict, stop_cmp_1t: dict,
                 stop_std_2t: dict, spec: ContractSpec, base_costs: CostModel) -> list[Criterion]:
    """Entry 11's five criteria with the instrument's own round turn."""
    return evaluate_criteria(mechanism, stop_std_1t, stop_cmp_1t, stop_std_2t,
                             round_turn_points(base_costs, spec))


def _money(x) -> str:
    return f"${x:,.2f}"


def verdict_block(results: dict, date_text: str) -> str:
    part, status, mech = results["part"], results["status"], results["mechanism"]
    pooled = mech["pooled"]
    lines = [f"### Part {part} result: {status}", "",
             f"**Date:** {date_text}",
             "**Code:** `strategies/tom.py` (unchanged), `backtests/run_entry12.py`",
             f"**Reports:** `backtests/results/entry12_{part.lower()}_report.txt`, "
             f"`entry12_{part.lower()}_returns.csv`, per arm "
             f"`entry12_{part.lower()}_<signal|stop>_slip<1|2>.csv` and `_nohalt.csv`, "
             f"`entry12_{part.lower()}_folds.csv`",
             "",
             f"{results['label']}, one contract, ${rules.COMMISSION_PER_SIDE:.2f} a side, "
             f"{BASE_SLIPPAGE_TICKS:g} tick a side base case ({results['round_turn']:.2f} "
             f"points a round turn). **Stop derived as {stop_fraction():.3f} × the control "
             f"standard deviation of {results['control_sd']:.2f} points = "
             f"{results['stop_points']:.2f} points.** Guards: `engine.apply_internal_guards`, "
             f"halt OFF alongside.",
             "", "#### Kill criteria — any one failure kills the entry", "",
             "| # | Criterion | Result | Line | |", "|---|---|---|---|---|"]
    for i, c in enumerate(results["criteria"], start=1):
        lines.append(f"| {i} | {c.label} | {c.shown} | {c.threshold} | "
                     f"**{'PASS' if c.passed else 'FAIL'}** |")
    failed = [c.label for c in results["criteria"] if not c.passed]
    if status == "REJECTED":
        tail = (f"Failed on: {'; '.join(failed)}. " if failed else "") + (
            "Part B is not run: the entry is REJECTED at Part A." if part == "A"
            else "The entry is REJECTED at Part B.")
    else:
        tail = ("Every Part A criterion passed. The entry stays PROPOSED until Part B "
                "has run; Part B is not run before its trigger." if part == "A"
                else "Every Part B criterion passed.")
    lines += ["", f"**{status}.** {tail}", "",
              "#### The mechanism test, pre-cost, in points", "",
              f"**Pooled:** window n = {pooled['n_a']:,}, mean {pooled['mean_a']:+.2f}, sd "
              f"{pooled['sd_a']:.2f}, median {pooled['median_a']:+.2f}; control n = "
              f"{pooled['n_b']:,}, mean {pooled['mean_b']:+.2f}, sd {pooled['sd_b']:.2f}, "
              f"median {pooled['median_b']:+.2f}. **Difference {pooled['diff']:+.2f} points "
              f"({pooled['diff'] / pooled['sd_b']:+.3f} control sd), Welch t = "
              f"{pooled['t']:+.2f}** (df {pooled['df']:.0f}), one-sided p = "
              f"{pooled['p_one_sided']:.3f}.", "",
              "| Year | Window n | Window mean | Control n | Control mean | Difference | t | Window beats control |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in mech["by_year"].iterrows():
        lines.append(f"| {int(r['year'])} | {int(r['n_window'])} | {r['mean_window']:+.2f} | "
                     f"{int(r['n_control'])} | {r['mean_control']:+.2f} | {r['diff']:+.2f} | "
                     f"{r['t']:+.2f} | {'yes' if r['window_beats_control'] else 'no'} |")
    lines += ["", f"Window beats control in **{mech['years_window_beats_control']} of "
              f"{len(mech['by_year'])}** years.", "",
              "**Per window day, reported and not selected on:**", "",
              "| Day | n | Mean | sd | Difference vs control | t |", "|---|---|---|---|---|---|"]
    for _, r in mech["by_label"].iterrows():
        lines.append(f"| {r['label']} | {int(r['n'])} | {r['mean']:+.2f} | {r['sd']:.2f} | "
                     f"{r['diff']:+.2f} | {r['t']:+.2f} |")
    for arm, title in (("stop", f"Stop arm ({results['stop_points']:g}-point stop)"),
                       ("signal", "Signal arm (no stop)")):
        for ticks in (BASE_SLIPPAGE_TICKS, *SENSITIVITY_TICKS):
            r = results["arms"][arm][ticks]
            lines += ["", f"#### {title} — {ticks:g} tick{'s' if ticks != 1 else ''}", ""]
            lines += e11._basis_table(
                f"one contract, {ticks:g} tick{'s' if ticks != 1 else ''}, "
                f"${rules.COMMISSION_PER_SIDE:.2f}", r["standard"], r["comparable"])
            std = r["standard"]
            reasons = "; ".join(f"{k}: {n} trades, mean {_money(mu)}, total {_money(tot)}"
                                for k, (n, mu, tot) in sorted(std["by_reason"].items()))
            lines += ["", f"Exits, standard stream: {reasons or 'none'}. Daily-loss halts: "
                          f"{std['loss_halts']}.",
                      "Per fold, standard: " + "; ".join(
                          f"{y} {_money(v)}" for y, v in sorted(std["year_pnl"].items())) + "."]
    diag = results["diagnostics"]
    sk = diag["skipped"]
    lines += ["", "#### Diagnostics", "",
              f"Window sessions skipped: {len(sk)}"
              + (" (" + ", ".join(f"{r['date']} {r['label']} {r['skipped_reason']}"
                                  for _, r in sk.iterrows()) + ")" if len(sk) else "")
              + f". Entry-bar stop breaches on the stop arm: {diag['entry_bar_breaches']}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(part: Part, parquet: Path | None = None, paths: int = 20_000,
        commission: float | None = None, progress=print,
        live: NullLive | None = None):
    """One part's mechanism test, both arms, the verdict.

    ``live`` is the optional event stream for ``bots/liveview.py``; the
    default writes nothing and changes nothing.
    """
    live = NullLive() if live is None else live
    with live:
        return _run(part, parquet, paths, commission, live.progress(progress), live)


def _run(part: Part, parquet: Path | None, paths: int, commission: float | None,
         progress, live: NullLive):
    RESULTS.mkdir(parents=True, exist_ok=True)
    commission = rules.COMMISSION_PER_SIDE if commission is None else commission
    tag = f"entry12_{part.key.lower()}"

    progress(f"[progress] part {part.key}: loading {part.label} bars")
    loaded = load(part, parquet)
    if part.key == "B":
        ready, n = forward_ready(loaded)
        if not ready:
            raise RuntimeError(f"Part B needs {MIN_FORWARD_WINDOW_SESSIONS} eligible forward "
                               f"window sessions and has {n}; not run, as pre-registered")
    progress("[progress] checking entry 11's MES control sd against its source")
    mes_control_sd_check()

    returns = scored_returns(loaded, part)
    returns.to_csv(RESULTS / f"{tag}_returns.csv", index=False)
    sd = control_sd(returns)
    stop = derive_stop_points(sd, part.spec)
    progress(f"[progress] control sd {sd:.2f} points -> stop {stop:.2f} points")
    mechanism = mechanism_test(returns)

    arms_params = {"signal": TOMParams(), "stop": TOMParams(stop_points=stop)}
    arms: dict = {}
    diagnostics: dict = {}
    for arm, params in arms_params.items():
        progress(f"[progress] {arm} arm: generating signals")
        win_signals, diag = generate(loaded, part, params)
        if arm == "stop":
            oos = diag[[d >= part.start for d in diag.index]]
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
                win_signals, loaded, part.spec, costs)
            standard.to_csv(RESULTS / f"{tag}_{arm}_slip{ticks:g}.csv", index=False)
            comparable.to_csv(RESULTS / f"{tag}_{arm}_slip{ticks:g}_nohalt.csv", index=False)
            arms[arm][ticks] = {
                "standard": stream_stats(standard, paths, comparable, dd_halts, loss_halts),
                "comparable": stream_stats(comparable, paths, None, 0, loss_halts),
            }
            if arm == "stop" and ticks == BASE_SLIPPAGE_TICKS:
                live.trades(standard, "standard")
                live.trades(comparable, "comparable")
                folds = fold_frame(standard, paths)
                folds.to_csv(RESULTS / f"{tag}_folds.csv", index=False)
                live.folds(folds)

    base_costs = CostModel(commission_per_side=commission, slippage_ticks=BASE_SLIPPAGE_TICKS)
    criteria = criteria_for(mechanism, arms["stop"][BASE_SLIPPAGE_TICKS]["standard"],
                            arms["stop"][BASE_SLIPPAGE_TICKS]["comparable"],
                            arms["stop"][SENSITIVITY_TICKS[0]]["standard"],
                            part.spec, base_costs)
    results = {"part": part.key, "label": part.label, "criteria": criteria,
               "status": verdict_status(criteria), "mechanism": mechanism, "arms": arms,
               "diagnostics": diagnostics, "stop_points": stop, "control_sd": sd,
               "round_turn": round_turn_points(base_costs, part.spec)}
    block = verdict_block(results, str(pd.Timestamp.now(tz=rules.ET).date()))
    (RESULTS / f"{tag}_verdict.md").write_text(block, encoding="utf-8")
    (RESULTS / f"{tag}_report.txt").write_text("\n".join([LINE, f"ENTRY 12 - part {part.key}",
                                                          LINE, block]), encoding="utf-8")
    progress(f"[progress] result written to backtests/results/{tag}_verdict.md")
    live.finish(results["status"])
    return results, results["status"], block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=sorted(PARTS), required=True)
    ap.add_argument("--parquet", type=Path, default=None,
                    help="bar file; required for part B, defaults to the cached MNQ file for A")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--commission", type=float, default=rules.COMMISSION_PER_SIDE)
    ap.add_argument("--live", action="store_true",
                    help="write a live event stream for bots/liveview.py")
    args = ap.parse_args(argv)
    part = PARTS[args.part]
    name = f"entry12_{part.key.lower()}"
    if args.reproduce:
        live_run = (LiveRun(name, runner=f"run_entry12 --part {part.key} --reproduce")
                    if args.live else NullLive())
        return 0 if reproduce(part, args.parquet, live=live_run) else 1
    if args.run:
        live_run = (LiveRun(name, runner=f"run_entry12 --part {part.key} --run")
                    if args.live else NullLive())
        _, status, _ = run(part, args.parquet, args.paths, args.commission,
                           progress=lambda m: print(m, flush=True), live=live_run)
        print(f"[done] part {part.key} {status}", flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
