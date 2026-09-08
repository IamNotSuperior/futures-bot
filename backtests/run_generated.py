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
from engine import (  # noqa: E402
    MES, MNQ, CostModel, build_trades, count_evaluation_blowups,
    enforce_daily_loss_limit, price_trades,
)
from metrics import compute_metrics  # noqa: E402
from run_london import fold_frame  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"

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


def run(name: str, class_path: str, symbol: str = "MES",
        contracts: int = 1, paths: int = 20_000,
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

    say(f"generating signals over {len(bars):,} bars ...")
    signals = strategy.generate_signals(bars)

    say("pricing trades at 2 ticks/side ...")
    spec = MES if symbol.upper() == "MES" else MNQ
    costs = CostModel(commission_per_side=COMMISSION_PER_SIDE,
                      slippage_ticks=BASE_SLIPPAGE_TICKS)
    signal_bars = bars.loc[signals.index] if len(signals) != len(bars) else bars
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
        f"${COMMISSION_PER_SIDE:.2f} commission per side.",
        "",
        "| | |",
        "|---|---|",
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
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--paths", type=int, default=20_000)
    args = ap.parse_args(argv)

    result = run(args.name, args.class_path, args.symbol, args.contracts,
                 args.paths, progress=lambda m: print(f"  {m}", flush=True))
    print()
    print(verdict_block(result, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
