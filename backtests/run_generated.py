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
survival bar - a strategy is only accepted if it survives there.
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
from engine import (  # noqa: E402
    MES, MNQ, CostModel, build_trades, count_evaluation_blowups,
    enforce_daily_loss_limit, price_trades,
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
COMMISSION_PER_SIDE = 1.25

#: Rule 13: profitable in a majority of yearly folds, positive total after
#: costs, surviving at 2 ticks. Restated nowhere else - imported from here.
MIN_PROFITABLE_FOLDS = 4
TOTAL_FOLDS = 7


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


def run(name: str, class_path: str, symbol: str = "MES",
        contracts: int | None = None, paths: int = 20_000,
        progress=None) -> WalkforwardResult:
    """Generate signals over the full history, then score by year."""
    def say(message: str) -> None:
        if progress:
            progress(message)

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

    say(f"pricing {contracts} contract(s) at 2 ticks/side, "
        f"{SCORE_START} .. {data_end} ...")
    spec = MES if symbol.upper() == "MES" else MNQ
    costs = CostModel(commission_per_side=COMMISSION_PER_SIDE,
                      slippage_ticks=BASE_SLIPPAGE_TICKS)
    trades = price_trades(build_trades(signals, signal_bars), spec, costs, contracts)
    if trades.empty:
        raise ValueError(
            f"`{name}` took no trades over the whole history. There is nothing "
            f"to walk forward."
        )
    trades, halts = enforce_daily_loss_limit(
        trades, signal_bars, spec, costs, contracts, rules.DAILY_LOSS_LIMIT)

    say(f"{len(trades):,} trades, {len(halts)} daily-loss halts; scoring folds ...")
    folds = fold_frame(trades, paths)
    profitable = int((folds["test_net_pnl"] > 0).sum())

    say("running the evaluation simulator ...")
    daily = eval_sim.daily_pnl_from_trades(trades)
    sim = eval_sim.simulate(daily, paths=paths)
    blow = count_evaluation_blowups(trades)
    metrics = compute_metrics(trades)

    # Rule 13, evaluated here rather than by the caller so the acceptance
    # decision has exactly one implementation.
    reasons = []
    if profitable < MIN_PROFITABLE_FOLDS:
        reasons.append(f"profitable in {profitable} of {TOTAL_FOLDS} folds, "
                       f"needs {MIN_PROFITABLE_FOLDS}")
    if float(trades["net_pnl"].sum()) <= 0:
        reasons.append(f"total walk-forward P&L "
                       f"${float(trades['net_pnl'].sum()):,.2f} is not positive")

    RESULTS.mkdir(parents=True, exist_ok=True)
    folds.to_csv(RESULTS / f"{name}_folds.csv", index=False)
    trades.to_csv(RESULTS / f"{name}_trades.csv", index=False)

    return WalkforwardResult(
        name=name, folds=folds, trades=trades, metrics=metrics,
        pass_probability=sim.pass_probability, blowups=int(blow["blowups"]),
        profitable_folds=profitable, accepted=not reasons, reasons=reasons,
        contracts=contracts, size_note=size_note,
        span=(SCORE_START, data_end),
        payout_probability=sim.payout_probability,
    )


def verdict_block(result: WalkforwardResult, entry_number: int) -> str:
    """The markdown written into hypotheses.md. Frozen before anything posts."""
    status = "ACCEPTED" if result.accepted else "REJECTED"
    lines = [
        f"### Verdict: {status}",
        "",
        f"**Date:** {pd.Timestamp.now(tz=rules.ET).date()}",
        f"**Code:** `strategies/generated/{result.name}.py`",
        f"**Reports:** `backtests/results/{result.name}_folds.csv`, "
        f"`{result.name}_trades.csv`",
        "",
        f"Walk-forward over {TOTAL_FOLDS} yearly folds at "
        f"{BASE_SLIPPAGE_TICKS:g} ticks of slippage per side and "
        f"${COMMISSION_PER_SIDE:.2f} commission per side, "
        f"**{result.contracts} contract(s)**.",
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
    args = ap.parse_args(argv)

    result = run(args.name, args.class_path, args.symbol, args.contracts,
                 args.paths, progress=lambda m: print(f"  {m}", flush=True))
    print()
    print(verdict_block(result, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
