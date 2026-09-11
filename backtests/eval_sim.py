"""Monte Carlo simulator for a prop-firm evaluation.

Given a trade-level P&L distribution, resample trade sequences and estimate:

  * the probability of reaching the profit target before the end-of-day
    trailing drawdown ends the account,
  * the probability of touching the payout line first - the balance at which
    a withdrawal becomes possible, reported alongside the pass and never
    below it, since the line sits under the target - and
  * how many paid attempts it takes, in expectation, to pass once.

Why resample rather than read the backtest's single path
-------------------------------------------------------
The historical sequence is one draw. It either passed or it did not, and that
single outcome says almost nothing about the odds - a strategy that passes on
the actual ordering can fail on 70% of equally plausible reorderings. Bootstrapping
the trade distribution asks the question the account actually faces: given trades
like these, arriving in some order, how often does the account survive to target?

What it assumes, and what that costs
------------------------------------
Trades are drawn i.i.d. from the empirical distribution. That throws away serial
structure - losing streaks that cluster in real markets are only reproduced at
their independent rate, so the estimate is **optimistic** for a strategy whose
losses bunch. Block resampling would preserve some of that; it is not implemented
here, and the daily-block mode (``block="day"``) is the partial mitigation:
resampling whole days keeps within-day clustering intact, which is where the
daily loss limit does its work.

    python backtests/eval_sim.py backtests/results/orb_oos_stream_slip1.csv
    python backtests/eval_sim.py journal/trades.jsonl --block day
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import rules  # noqa: E402

DEFAULT_ATTEMPT_COST = 115.0
DEFAULT_PATHS = 20_000
DEFAULT_MAX_DAYS = 250


@dataclass(frozen=True)
class EvalConfig:
    """One evaluation's terms. Defaults are the Lucid 50K Pro eval."""

    starting_balance: float = rules.ACCOUNT_SIZE
    profit_target: float = rules.PROFIT_TARGET
    trailing_drawdown: float = rules.FIRM.max_trailing_drawdown
    daily_loss_limit: float = rules.DAILY_LOSS_LIMIT
    #: The payout milestone: a balance this far above the start is the first
    #: point a withdrawal is possible. Reported, never enforced.
    payout_buffer: float = rules.FIRM.payout_buffer
    attempt_cost: float = DEFAULT_ATTEMPT_COST
    max_days: int = DEFAULT_MAX_DAYS


@dataclass(frozen=True)
class SimResult:
    paths: int
    pass_probability: float
    blowup_probability: float
    timeout_probability: float
    expected_attempts: float
    median_days_to_pass: float
    mean_final_balance: float
    ci_low: float
    ci_high: float
    #: Share of paths whose end-of-day balance touched the payout line before
    #: the trailing drawdown ended the account. A touch, not a survival: a path
    #: that reaches it and is later ended still counts. Never below
    #: ``pass_probability``, because the line sits below the target.
    payout_probability: float = float("nan")
    median_days_to_payout: float = float("nan")

    @property
    def expected_cost_to_pass(self) -> float:
        return self.expected_attempts * DEFAULT_ATTEMPT_COST


def daily_pnl_from_trades(trades: pd.DataFrame) -> np.ndarray:
    """Net P&L per session date."""
    if "session_date" not in trades.columns:
        trades = trades.assign(
            session_date=pd.to_datetime(trades["entry_time"]).dt.date
        )
    return (
        trades.groupby("session_date")["net_pnl"].sum().sort_index().to_numpy(float)
    )


def simulate(
    daily_pnl: np.ndarray,
    config: EvalConfig = EvalConfig(),
    paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
) -> SimResult:
    """Resample whole days and run each path to pass, blow-up, or timeout.

    Days are the unit because the trailing drawdown is scored end-of-day: a
    day's internal ordering cannot change whether it breaches the trail, only
    its net. Resampling trades individually and then re-bucketing them into
    days would invent day compositions that never occurred.
    """
    daily_pnl = np.asarray(daily_pnl, dtype=float)
    if daily_pnl.size == 0:
        raise ValueError("No daily P&L to resample from")

    rng = np.random.default_rng(seed)
    draws = rng.choice(daily_pnl, size=(paths, config.max_days), replace=True)

    balance = np.full(paths, config.starting_balance)
    peak = np.full(paths, config.starting_balance)
    status = np.zeros(paths, dtype=np.int8)  # 0 running, 1 passed, -1 blown
    days_taken = np.full(paths, config.max_days, dtype=int)
    paid = np.zeros(paths, dtype=bool)
    days_to_payout = np.full(paths, config.max_days, dtype=int)

    floor_gap = config.trailing_drawdown
    target_balance = config.starting_balance + config.profit_target
    payout_balance = config.starting_balance + config.payout_buffer

    for day in range(config.max_days):
        live = status == 0
        if not live.any():
            break
        balance[live] += draws[live, day]

        # Death first: an account already through the trailing line cannot be
        # rescued by the same day's close reaching target.
        blown = live & ((peak - balance) >= floor_gap)
        passed = live & ~blown & (balance >= target_balance)

        # The payout milestone is a touch on a live account. It does not end
        # the path - the evaluation carries on to pass, blow up or time out -
        # so both figures are measured on the same resampled days.
        reached = live & ~blown & ~paid & (balance >= payout_balance)
        paid[reached] = True
        days_to_payout[reached] = day + 1

        status[blown] = -1
        days_taken[blown] = day + 1
        status[passed] = 1
        days_taken[passed] = day + 1

        still = status == 0
        peak[still] = np.maximum(peak[still], balance[still])

    passes = int((status == 1).sum())
    blown_n = int((status == -1).sum())
    timeouts = paths - passes - blown_n
    paid_n = int(paid.sum())

    p = passes / paths
    # Wilson-free normal approximation is fine at these path counts.
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / paths))

    return SimResult(
        paths=paths,
        pass_probability=p,
        blowup_probability=blown_n / paths,
        timeout_probability=timeouts / paths,
        expected_attempts=(1.0 / p) if p > 0 else float("inf"),
        median_days_to_pass=(
            float(np.median(days_taken[status == 1])) if passes else float("nan")
        ),
        mean_final_balance=float(balance.mean()),
        ci_low=max(0.0, p - 1.96 * se),
        ci_high=min(1.0, p + 1.96 * se),
        payout_probability=paid_n / paths,
        median_days_to_payout=(
            float(np.median(days_to_payout[paid])) if paid_n else float("nan")
        ),
    )


def load_pnl(path: Path) -> tuple[np.ndarray, str]:
    """Daily P&L from a trade CSV or a journal .jsonl."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Point this at a backtest trade CSV in "
            f"backtests/results/, or a journal trades.jsonl."
        )
    if path.suffix == ".jsonl":
        trades = pd.read_json(path, lines=True)
    elif path.suffix == ".csv":
        trades = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type {path.suffix!r}")
    if "net_pnl" not in trades.columns:
        raise ValueError(f"{path} has no 'net_pnl' column")
    return daily_pnl_from_trades(trades), f"{len(trades):,} trades"


def format_result(result: SimResult, daily_pnl: np.ndarray,
                  config: EvalConfig, source: str) -> str:
    line = "=" * 88
    out = [line, "EVALUATION MONTE CARLO", line]
    out += [
        f"  Source                     {source}",
        f"  Distinct trading days      {len(daily_pnl):,}",
        f"  Mean day                   ${daily_pnl.mean():>10,.2f}",
        f"  Day standard deviation     ${daily_pnl.std(ddof=1):>10,.2f}",
        f"  Best / worst day           ${daily_pnl.max():>10,.2f} / "
        f"${daily_pnl.min():,.2f}",
        "",
        f"  Target                     +${config.profit_target:,.0f}",
        f"  Trailing drawdown          -${config.trailing_drawdown:,.0f} "
        f"(end-of-day, from peak)",
        f"  Payout line                +${config.payout_buffer:,.0f} "
        f"(balance ${config.starting_balance + config.payout_buffer:,.0f}; "
        f"${rules.FIRM.min_payout:,.0f} minimum withdrawal)",
        f"  Horizon                    {config.max_days} trading days",
        f"  Paths                      {result.paths:,}",
        "",
        f"  PASS probability           {result.pass_probability:>10.2%}   "
        f"95% CI [{result.ci_low:.2%}, {result.ci_high:.2%}]",
        f"  PAYOUT probability         {result.payout_probability:>10.2%}   "
        f"touched the payout line before the trail; never below PASS",
        f"  Blow-up probability        {result.blowup_probability:>10.2%}",
        f"  Ran out of time            {result.timeout_probability:>10.2%}",
        "",
        f"  Expected attempts to pass  {result.expected_attempts:>10.2f}",
        f"  Expected cost at ${config.attempt_cost:,.0f}/attempt   "
        f"${result.expected_attempts * config.attempt_cost:,.2f}"
        if np.isfinite(result.expected_attempts) else
        "  Expected cost              never passes in this model",
        f"  Median days to pass        {result.median_days_to_pass:>10.0f}",
        f"  Median days to payout      {result.median_days_to_payout:>10.0f}",
    ]
    out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trades", help="trade CSV or journal .jsonl")
    ap.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    ap.add_argument("--max-days", type=int, default=DEFAULT_MAX_DAYS)
    ap.add_argument("--attempt-cost", type=float, default=DEFAULT_ATTEMPT_COST)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    daily, source = load_pnl(Path(args.trades))
    config = EvalConfig(attempt_cost=args.attempt_cost, max_days=args.max_days)
    result = simulate(daily, config, paths=args.paths, seed=args.seed)
    print(format_result(result, daily, config, f"{args.trades} ({source})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
