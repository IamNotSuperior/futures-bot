"""Walk-forward for a generated strategy. This one actually runs.

Existing entries read saved output - the ORB walk-forward takes about 55
minutes and a chat command that silently starts one is a worse answer than a
cached table. A generated strategy has no saved output by definition, so this
computes it: yearly folds at the base-case cost model, the evaluation
simulator, and the blow-up count, then writes the CSVs so later reads are
cheap.

Costs are **not** optional and **not** configurable from chat. CLAUDE.md rule
10: no backtest, parameter scan or performance report may run on a frictionless
fill model. The base case is 2 ticks of slippage per side, which is rule 13's
survival bar - a strategy is only accepted if it survives there - at the
commission in ``rules.COMMISSION_PER_SIDE``. The CLI's ``--commission`` exists
so a verdict scored before 2026-09-11 can be reproduced at the $1.25 it
carried (``rules.ASSUMED_COMMISSION_PER_SIDE``); the bot never passes it.

    python backtests/run_generated.py <name> --class-path <module:Class>
    python backtests/run_generated.py <name> --class-path <module:Class> --commission 1.25
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("data", "strategies", "strategies/generated", "backtests"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import eval_sim  # noqa: E402
import loader  # noqa: E402
import rules  # noqa: E402
import walkforward  # noqa: E402
from live import LiveRun, NullLive  # noqa: E402
from engine import (  # noqa: E402
    MES, MNQ, CostModel, apply_internal_guards, build_trades,
    count_evaluation_blowups, price_trades,
)
from metrics import compute_metrics  # noqa: E402
from run_london import fold_frame  # noqa: E402
from scan import slice_by_date  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"

#: The first scored day. Imported from walkforward.py's own fold construction
#: rather than written as a date, so a generated verdict is scored on exactly
#: the span every other entry in the log was scored on. The first run of this
#: module scored the whole parquet from 2019-05-05 and reported 588 trades
#: where entry 5 had 478 - a gap that was entirely this span, and is recorded
#: as a diagnostic on entry 8. Bars before this date are still handed to the
#: strategy: a 50-day EMA needs 2019 to be seeded by January 2020.
SCORE_START = walkforward.build_folds()[0].test_start


def score_slice(frame, data_end):
    """``frame`` restricted to the scored span. Pure; used on signals and bars."""
    return slice_by_date(frame, SCORE_START, data_end)

#: Entry 6's base case and rule 13's survival bar. Not a parameter.
BASE_SLIPPAGE_TICKS = 2.0
#: Read from rules, never restated: Lucid's confirmed $0.50 a side.
COMMISSION_PER_SIDE = rules.COMMISSION_PER_SIDE

#: Rule 13: profitable in a majority of yearly folds, positive total after
#: costs, surviving at 2 ticks. Restated nowhere else - imported from here.
MIN_PROFITABLE_FOLDS = 4
TOTAL_FOLDS = 7


@dataclass(frozen=True)
class BasisSummary:
    """The headline figures of one trade stream, for the comparable basis.

    The verdict is decided on the guarded stream; this is what the same
    signals did without the trailing halt, reported alongside it because a
    halt that fires in year one leaves the fold test with nothing to count.
    """

    trades: int
    net_pnl: float
    profitable_folds: int
    sharpe: float
    profit_factor: float
    max_drawdown: float
    max_daily_loss: float
    pass_probability: float
    payout_probability: float
    blowups: int


@dataclass
class WalkforwardResult:
    name: str
    folds: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    pass_probability: float
    blowups: int
    profitable_folds: int
    accepted: bool
    reasons: list
    contracts: int = 1
    size_note: str = ""
    span: tuple = ()
    #: Probability of touching the payout line ($52,100) before the trail.
    #: NaN rather than zero when unset, so a missing figure cannot read as a
    #: plausible one.
    payout_probability: float = float("nan")
    #: Sessions the trailing halt blocked on the standard stream. None when
    #: unset, for the same reason the payout figure is NaN.
    dd_halts: int | None = None
    #: The same signals without the trailing halt. None when not computed.
    comparable: BasisSummary | None = None
    #: The commission this run was priced at. NaN when unset, so a result
    #: built without one cannot print a plausible rate it did not use.
    commission_per_side: float = float("nan")

    @property
    def net_pnl(self) -> float:
        return float(self.trades["net_pnl"].sum()) if not self.trades.empty else 0.0


def parquet_for(symbol: str) -> Path:
    return (PROJECT_ROOT / "data" /
            f"{symbol.lower()}_v_0_ohlcv_1m_2019-05_2026-08.parquet")


def build_strategy(class_path: str, bars, rolls, early):
    """Instantiate from ``module:Class``, passing whatever it accepts.

    Generated strategies are written against a narrow interface, but the house
    style lets a strategy take roll and early-close dates. Rather than demand
    one signature, try the richer call and fall back - a generated strategy
    that takes no arguments is still valid.
    """
    module_name, _, class_name = class_path.partition(":")
    import importlib  # noqa: PLC0415

    cls = getattr(importlib.import_module(module_name), class_name)
    for kwargs in ({"roll_dates": rolls, "early_close_dates": early}, {}):
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    return cls()


def resolve_contracts(strategy, override: int | None = None) -> tuple[int, str]:
    """The size to trade, and a one-line note about where it came from.

    A generated strategy transcribes the size the submission stated into a
    ``contracts`` class attribute. It is the only quantity about position size
    a strategy carries, and it is still not the last word: rule 4 caps the
    internal position at 5 whatever a description says, so a larger number is
    **clamped here and reported** rather than honoured or silently ignored.

    Reporting the clamp matters as much as applying it. A backtest that quietly
    sized down would publish a P&L for a position the operator did not ask for,
    against a description that says otherwise.
    """
    if override is not None:
        return int(override), f"{int(override)} (given on the command line)"

    declared = getattr(strategy, "contracts", 1)
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        return 1, f"1 (the strategy's contracts attribute was {declared!r})"
    if declared < 1:
        return 1, f"1 (the strategy declared {declared}, which is not a size)"
    if declared > rules.POSITION_CAP:
        return rules.POSITION_CAP, (
            f"{rules.POSITION_CAP} - **CLAMPED** from the {declared} the "
            f"submission stated, by rule 4's internal cap "
            f"(the firm allows {rules.FIRM.max_contracts})"
        )
    return declared, f"{declared} (declared by the strategy)"


def guarded_streams(signals, bars, spec, costs: CostModel, contracts: int):
    """Price the signals, then apply the internal guards - both ways.

    Returns ``(standard, loss_halts, dd_halts, comparable)``. ``standard`` is
    the stream under both guards and is what the verdict is decided on;
    ``comparable`` is the same priced trades with the daily loss limit but no
    trailing halt, the basis entries 1 and 4 were measured on. Both come from
    :func:`engine.apply_internal_guards`, the same call entry 5's runner makes.
    """
    priced = price_trades(build_trades(signals, bars), spec, costs, contracts)
    standard, loss_halts, dd_halts = apply_internal_guards(
        priced, bars, spec, costs, contracts)
    comparable, _, _ = apply_internal_guards(
        priced, bars, spec, costs, contracts, trailing_halt=False)
    return standard, loss_halts, dd_halts, comparable


def summarise_basis(trades: pd.DataFrame, paths: int) -> BasisSummary:
    """Fold count, simulator and metrics for one stream, as a summary."""
    folds = fold_frame(trades, paths)
    sim = eval_sim.simulate(eval_sim.daily_pnl_from_trades(trades), paths=paths)
    blow = count_evaluation_blowups(trades)
    m = compute_metrics(trades)
    return BasisSummary(
        trades=len(trades), net_pnl=float(trades["net_pnl"].sum()),
        profitable_folds=int((folds["test_net_pnl"] > 0).sum()),
        sharpe=float(m["sharpe"]), profit_factor=float(m["profit_factor"]),
        max_drawdown=float(m["max_drawdown"]),
        max_daily_loss=float(m["max_daily_loss"]),
        pass_probability=sim.pass_probability,
        payout_probability=sim.payout_probability,
        blowups=int(blow["blowups"]),
    )


def run(name: str, class_path: str, symbol: str = "MES",
        contracts: int | None = None, paths: int = 20_000,
        progress=None, commission: float | None = None,
        live: NullLive | None = None) -> WalkforwardResult:
    """Generate signals over the full history, then score by year.

    ``commission`` defaults to :data:`COMMISSION_PER_SIDE`. It is a parameter
    only so a pre-2026-09-11 verdict can be reproduced at the $1.25 it was
    scored at; the bot's call site never passes it.

    ``live`` is the optional event stream for ``bots/liveview.py``. The
    default writes nothing; with a :class:`LiveRun` the same progress
    messages, the two priced streams and the fold table are also written to
    the live directory. Nothing about the computation, the ordering or the
    files under ``RESULTS`` depends on it.
    """
    live = NullLive() if live is None else live
    with live:
        return _run(name, class_path, symbol, contracts, paths,
                    live.progress(progress), commission, live)


def _run(name: str, class_path: str, symbol: str, contracts: int | None,
         paths: int, say, commission: float | None, live: NullLive) -> WalkforwardResult:
    if commission is None:
        commission = COMMISSION_PER_SIDE

    parquet = parquet_for(symbol)
    if not parquet.exists():
        raise FileNotFoundError(f"no bar data at {parquet.name}")

    say(f"loading {symbol} bars ...")
    bars = loader.load_bars(parquet)
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)

    say("building the strategy ...")
    strategy = build_strategy(class_path, bars, rolls, early)
    contracts, size_note = resolve_contracts(strategy, contracts)

    # Signals are generated over the FULL history so indicators are seeded,
    # then the scored span starts at SCORE_START. 2019 is history, not sample.
    say(f"generating signals over {len(bars):,} bars ...")
    signals = strategy.generate_signals(bars)
    data_end = rules.session_date(bars.index[-1])
    signal_bars = bars.loc[signals.index] if len(signals) != len(bars) else bars
    signals = score_slice(signals, data_end)
    signal_bars = score_slice(signal_bars, data_end)

    say(f"pricing {contracts} contract(s) at {BASE_SLIPPAGE_TICKS:g} ticks/side, "
        f"${commission:.2f}/side commission, {SCORE_START} .. {data_end} ...")
    spec = MES if symbol.upper() == "MES" else MNQ
    costs = CostModel(commission_per_side=commission,
                      slippage_ticks=BASE_SLIPPAGE_TICKS)
    trades, halts, dd_halts, comparable_trades = guarded_streams(
        signals, signal_bars, spec, costs, contracts)
    if comparable_trades.empty:
        raise ValueError(
            f"`{name}` took no trades over the whole history. There is nothing "
            f"to walk forward."
        )

    live.trades(trades, "standard")
    live.trades(comparable_trades, "comparable")

    say(f"{len(trades):,} trades, {len(halts)} daily-loss halts, "
        f"{len(dd_halts)} sessions blocked by the trailing halt; scoring folds ...")
    folds = fold_frame(trades, paths)
    live.folds(folds)
    profitable = int((folds["test_net_pnl"] > 0).sum())

    say("running the evaluation simulator ...")
    daily = eval_sim.daily_pnl_from_trades(trades)
    sim = eval_sim.simulate(daily, paths=paths)
    blow = count_evaluation_blowups(trades)
    metrics = compute_metrics(trades)

    # Rule 13 and the regression check, in `decide` so the acceptance
    # decision has exactly one implementation. Decided on the guarded stream.
    accepted, reasons = decide(profitable, float(trades["net_pnl"].sum()), metrics)
    live.finish("ACCEPTED" if accepted else "REJECTED")

    say(f"scoring the comparable basis, halt OFF: "
        f"{len(comparable_trades):,} trades ...")
    comparable_folds = fold_frame(comparable_trades, paths)
    comparable = summarise_basis(comparable_trades, paths)

    RESULTS.mkdir(parents=True, exist_ok=True)
    folds.to_csv(RESULTS / f"{name}_folds.csv", index=False)
    trades.to_csv(RESULTS / f"{name}_trades.csv", index=False)
    comparable_folds.to_csv(RESULTS / f"{name}_folds_nohalt.csv", index=False)
    comparable_trades.to_csv(RESULTS / f"{name}_trades_nohalt.csv", index=False)

    return WalkforwardResult(
        name=name, folds=folds, trades=trades, metrics=metrics,
        pass_probability=sim.pass_probability, blowups=int(blow["blowups"]),
        profitable_folds=profitable, accepted=accepted, reasons=reasons,
        contracts=contracts, size_note=size_note,
        span=(SCORE_START, data_end),
        payout_probability=sim.payout_probability,
        dd_halts=len(dd_halts), comparable=comparable,
        commission_per_side=commission,
    )


def hold_regression_line(metrics: dict) -> str:
    """One line every surface prints: the rule 6 / rule 7 regression check.

    Reads the counts ``compute_metrics`` produces and never recomputes a
    threshold. Missing keys read as clean so an older result renders.
    """
    under = int(metrics.get("min_hold_violation_count", 0) or 0)
    scalps = int(metrics.get("microscalp_trade_count", 0) or 0)
    if not under and not scalps:
        return (f"Rule 6/7 regression check: intact - no trade under the "
                f"{rules.MIN_HOLD_SECONDS}s floor, {scalps} held <= "
                f"{rules.MICROSCALP_SECONDS}s.")
    pct = float(metrics.get("microscalp_profit_pct") or 0.0)
    return (f"**REGRESSION - rule 6 enforcement is not working:** {under} trade(s) "
            f"held under {rules.MIN_HOLD_SECONDS}s, {scalps} held <= "
            f"{rules.MICROSCALP_SECONDS}s carrying {pct:.2f}% of profit. "
            f"Investigate the exit guard before reading any other figure.")


def decide(profitable_folds: int, net_pnl: float, metrics: dict) -> tuple[bool, list[str]]:
    """Rule 13 on the guarded stream, plus the regression check.

    One implementation of the acceptance decision. A stream with a hold
    under the floor is rejected whatever its P&L: it was produced by an exit
    path the rules did not govern, so it is evidence of nothing.
    """
    reasons: list[str] = []
    if profitable_folds < MIN_PROFITABLE_FOLDS:
        reasons.append(f"profitable in {profitable_folds} of {TOTAL_FOLDS} folds, "
                       f"needs {MIN_PROFITABLE_FOLDS}")
    if net_pnl <= 0:
        reasons.append(f"total walk-forward P&L ${net_pnl:,.2f} is not positive")
    under = int(metrics.get("min_hold_violation_count", 0) or 0)
    scalps = int(metrics.get("microscalp_trade_count", 0) or 0)
    if under or scalps:
        reasons.append(f"hold regression: {under} trade(s) under the "
                       f"{rules.MIN_HOLD_SECONDS}s floor, {scalps} held <= "
                       f"{rules.MICROSCALP_SECONDS}s - the stream was not produced "
                       f"under rule 6 and cannot be evidence")
    return not reasons, reasons


def verdict_block(result: WalkforwardResult, entry_number: int) -> str:
    """The markdown written into hypotheses.md. Frozen before anything posts."""
    status = "ACCEPTED" if result.accepted else "REJECTED"
    halted = "n/a" if result.dd_halts is None else str(result.dd_halts)
    lines = [
        f"### Verdict: {status}",
        "",
        f"**Date:** {pd.Timestamp.now(tz=rules.ET).date()}",
        f"**Code:** `strategies/generated/{result.name}.py`",
        f"**Reports:** `backtests/results/{result.name}_folds.csv`, "
        f"`{result.name}_trades.csv` (standard, trailing halt ON); "
        f"`{result.name}_folds_nohalt.csv`, `{result.name}_trades_nohalt.csv` "
        f"(comparable, halt OFF)",
        "",
        f"Walk-forward over {TOTAL_FOLDS} yearly folds at "
        f"{BASE_SLIPPAGE_TICKS:g} ticks of slippage per side and "
        f"${result.commission_per_side:.2f} commission per side, "
        f"**{result.contracts} contract(s)**, under both internal guards: the "
        f"${rules.DAILY_LOSS_LIMIT:,.0f} daily loss limit and the "
        f"${rules.TRAILING_DD_STOP:,.0f} end-of-day trailing halt.",
        "",
        "| | |",
        "|---|---|",
        f"| Scored span | {result.span[0]} .. {result.span[1]} "
        f"(earlier bars used as indicator history only) |"
        if result.span else "| Scored span | full data |",
        f"| Contracts | {result.size_note or result.contracts} |",
        f"| Trades | {len(result.trades):,} |",
        f"| Net P&L | ${result.net_pnl:,.2f} |",
        f"| Folds profitable | {result.profitable_folds} of {TOTAL_FOLDS} |",
        f"| Sharpe | {result.metrics['sharpe']:.2f} |",
        f"| Profit factor | {result.metrics['profit_factor']:.3f} |",
        f"| Max drawdown | ${result.metrics['max_drawdown']:,.2f} |",
        f"| Worst day | ${result.metrics['max_daily_loss']:,.2f} |",
        f"| Avg duration | {result.metrics['avg_duration_seconds'] / 60:.1f} min |",
        f"| Profit from <=5s holds | "
        f"{result.metrics.get('microscalp_profit_pct', 0.0):.2f}% |",
        f"| Pass probability | {result.pass_probability:.2%} |",
        f"| Payout probability | {result.payout_probability:.2%} |",
        f"| Evaluations blown | {result.blowups} |",
        f"| Sessions blocked by the trailing halt | {halted} |",
        "",
        hold_regression_line(result.metrics),
        "",
    ]
    if result.accepted:
        lines += ["Rule 13 is satisfied: profitable in a majority of yearly "
                  "folds, positive total after commission and slippage, at 2 "
                  "ticks per side.", ""]
    else:
        lines += ["**Rule 13 is not satisfied:**", ""]
        lines += [f"- {r}" for r in result.reasons]
        lines += ["", "In-sample results are never evidence, and this is the "
                      "out-of-sample answer.", ""]
    if result.comparable is not None:
        c = result.comparable
        lines += [
            "#### Comparable basis, trailing halt OFF",
            "",
            "The same signals with the daily loss limit but no trailing halt - "
            "the basis entries 1 and 4 were measured on. Reported for "
            "comparison only; rule 13 is decided on the guarded stream above. "
            "A blow-up count is only meaningful here, where trading continues "
            "past the internal line.",
            "",
            "| | |",
            "|---|---|",
            f"| Trades | {c.trades:,} |",
            f"| Net P&L | ${c.net_pnl:,.2f} |",
            f"| Folds profitable | {c.profitable_folds} of {TOTAL_FOLDS} |",
            f"| Sharpe | {c.sharpe:.2f} |",
            f"| Profit factor | {c.profit_factor:.3f} |",
            f"| Max drawdown | ${c.max_drawdown:,.2f} |",
            f"| Worst day | ${c.max_daily_loss:,.2f} |",
            f"| Pass probability | {c.pass_probability:.2%} |",
            f"| Payout probability | {c.payout_probability:.2%} |",
            f"| Evaluations blown | {c.blowups} |",
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_generated.py",
        description="Walk-forward a generated strategy at base-case costs.")
    ap.add_argument("name")
    ap.add_argument("--class-path", required=True, help="module:Class")
    ap.add_argument("--symbol", default="MES",
                    choices=sorted(rules.ALLOWED_INSTRUMENTS))
    ap.add_argument("--contracts", type=int, default=None,
                    help="override the size the strategy declares")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--commission", type=float, default=None,
                    help="per side; defaults to rules.COMMISSION_PER_SIDE. "
                         "Pass 1.25 to reproduce a verdict scored before "
                         "2026-09-11.")
    ap.add_argument("--live", action="store_true",
                    help="write a live event stream for bots/liveview.py")
    args = ap.parse_args(argv)

    live_run = (LiveRun(args.name, runner=f"run_generated {args.name}")
                if args.live else NullLive())
    result = run(args.name, args.class_path, args.symbol, args.contracts,
                 args.paths, progress=lambda m: print(f"  {m}", flush=True),
                 commission=args.commission, live=live_run)
    print()
    print(verdict_block(result, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
