"""Trade-level backtest engine.

The strategy already decides exactly where each trade leaves - a stop or target
touched intrabar, or the session-end close - so this engine does not re-derive
fills from the price series. It pairs entry signals with their exits, applies
commission and slippage, and produces a trade list. Portfolio analytics are
computed from that trade list in :mod:`metrics`.

Sizing is fixed at one contract. Position sizing is an execution-layer concern
(CLAUDE.md rule 4); here we are measuring the signal, not a money-management
scheme layered on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "direction",
    "entry_price",
    "exit_price",
    "exit_reason",
]


@dataclass(frozen=True)
class ContractSpec:
    """Contract specification. Defaults are MES (Micro E-mini S&P 500)."""

    symbol: str = "MES"
    tick_size: float = 0.25
    tick_value: float = 1.25
    point_value: float = 5.0

    def __post_init__(self) -> None:
        expected = self.tick_value / self.tick_size
        if abs(expected - self.point_value) > 1e-9:
            raise ValueError(
                f"Inconsistent spec: tick_value/tick_size = {expected} but "
                f"point_value = {self.point_value}"
            )


MES = ContractSpec()
MNQ = ContractSpec(symbol="MNQ", tick_size=0.25, tick_value=0.50, point_value=2.0)


@dataclass(frozen=True)
class CostModel:
    """Trading costs.

    ``commission_per_side`` is all-in per contract per side, roughly Tradovate's
    micro rate. ``slippage_ticks`` is applied adversely on both entry and exit,
    so the default of one tick costs two ticks over a round turn.
    """

    commission_per_side: float = 1.25
    slippage_ticks: float = 1.0

    def commission_round_turn(self, contracts: int = 1) -> float:
        return 2 * self.commission_per_side * contracts


def build_trades(signals: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Pair entry signals with their exits into a raw trade list.

    Follows the interface convention in :mod:`strategies.base`: an entry marked
    at ``T`` is filled at ``bars.loc[T, "open"]``. An exit uses the strategy's
    ``exit_price`` when it publishes one (the stop or target level actually
    touched), falling back to the bar open.

    Only one position is tracked at a time; a second entry while a trade is open
    is ignored, which also guards against a strategy that fails to suppress it.
    """
    rows: list[dict] = []
    open_trade: dict | None = None

    has_exit_price = "exit_price" in signals.columns
    has_reason = "exit_reason" in signals.columns

    for ts in signals.index:
        row = signals.loc[ts]

        # Exits are processed before entries so a same-bar flip is well defined.
        if open_trade is not None:
            closing = (
                row["exit_long"] if open_trade["direction"] == "long" else row["exit_short"]
            )
            if bool(closing):
                price = None
                if has_exit_price and pd.notna(row.get("exit_price")):
                    price = float(row["exit_price"])
                if price is None:
                    price = float(bars.loc[ts, "open"])
                open_trade["exit_time"] = ts
                open_trade["exit_price"] = price
                open_trade["exit_reason"] = (
                    str(row["exit_reason"])
                    if has_reason and pd.notna(row.get("exit_reason"))
                    else "signal"
                )
                rows.append(open_trade)
                open_trade = None

        if open_trade is None:
            if bool(row["entry_long"]):
                direction = "long"
            elif bool(row["entry_short"]):
                direction = "short"
            else:
                continue
            open_trade = {
                "entry_time": ts,
                "direction": direction,
                "entry_price": float(bars.loc[ts, "open"]),
                "exit_time": pd.NaT,
                "exit_price": float("nan"),
                "exit_reason": "",
            }

    # An entry with no exit is dropped rather than marked to the last close:
    # the strategy is responsible for closing every position, and inventing an
    # exit here would quietly manufacture P&L.
    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    if not trades.empty:
        trades = trades.sort_values("entry_time").reset_index(drop=True)
    return trades


def price_trades(
    trades: pd.DataFrame,
    spec: ContractSpec = MES,
    costs: CostModel = CostModel(),
    contracts: int = 1,
) -> pd.DataFrame:
    """Apply slippage and commission, and compute per-trade P&L.

    Slippage moves both fills against the trade. The returned frame keeps the
    signal prices alongside the filled prices so the cost drag is visible rather
    than baked invisibly into the result.
    """
    out = trades.copy()
    if out.empty:
        for col in (
            "entry_fill", "exit_fill", "gross_points", "net_points",
            "gross_pnl", "commission", "slippage_cost", "net_pnl",
            "duration_seconds", "session_date",
        ):
            out[col] = pd.Series(dtype="float64")
        return out

    slip = costs.slippage_ticks * spec.tick_size
    is_long = out["direction"].eq("long")
    sign = is_long.map({True: 1.0, False: -1.0})

    out["entry_fill"] = out["entry_price"] + sign * slip
    out["exit_fill"] = out["exit_price"] - sign * slip

    out["gross_points"] = (out["exit_price"] - out["entry_price"]) * sign
    out["net_points"] = (out["exit_fill"] - out["entry_fill"]) * sign

    out["gross_pnl"] = out["gross_points"] * spec.point_value * contracts
    out["commission"] = costs.commission_round_turn(contracts)
    out["slippage_cost"] = 2 * slip * spec.point_value * contracts
    out["net_pnl"] = out["net_points"] * spec.point_value * contracts - out["commission"]

    out["duration_seconds"] = (
        pd.to_datetime(out["exit_time"]) - pd.to_datetime(out["entry_time"])
    ).dt.total_seconds()
    out["session_date"] = pd.to_datetime(out["entry_time"]).dt.date

    return out


HALT_COLUMNS = [
    "session_date",
    "halt_time",
    "equity_at_halt",
    "forced_flatten",
    "trades_taken",
    "trades_blocked",
]

LOSS_LIMIT_EXIT = "loss_limit_flatten"


def enforce_daily_loss_limit(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    spec: ContractSpec = MES,
    costs: CostModel = CostModel(),
    contracts: int = 1,
    limit: float = 300.0,
    mark: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Halt a session once realised **plus open** P&L reaches ``limit``.

    CLAUDE.md rule 5 is an equity rule, not a realised-P&L rule: an open losing
    position counts against the limit while it is still open. So every bar an
    position is held is marked to market, and if realised P&L plus that mark
    reaches the limit the position is flattened at the **next bar's open** and
    the session halts. Anything the strategy wanted to trade later that day is
    dropped.

    The mark is liquidation value - what closing right now would realise -
    so it carries the full round-turn commission. That is deliberately the
    pessimistic reading: it halts marginally earlier than a mark that ignores
    the exit cost, which is the safer direction for a risk control.

    Args:
        trades: priced trades, as returned by :func:`price_trades`.
        bars: the bars the signals were generated on, used for the marks.
        mark: ``"close"`` marks each bar at its close, per the stated rule.
            ``"adverse"`` marks at the bar's low (long) or high (short), which
            catches an intrabar equity dip that a close-only mark misses. Real
            equity moves intrabar, so ``"adverse"`` is the stricter model.

    Returns:
        ``(kept_trades, halt_log)``. Kept trades are re-priced, so a forced
        flatten carries its actual exit rather than the strategy's intended one.
    """
    empty_log = pd.DataFrame(columns=HALT_COLUMNS)
    if trades.empty:
        return trades, empty_log
    if mark not in ("close", "adverse"):
        raise ValueError(f"mark must be 'close' or 'adverse', got {mark!r}")

    commission = costs.commission_round_turn(contracts)
    bar_dates = pd.Series(bars.index.date, index=bars.index)

    kept_raw: list[dict] = []
    halts: list[dict] = []

    for day, group in trades.groupby("session_date"):
        session_bars = bars[bar_dates == day]
        ordered = group.sort_values("entry_time")
        realised = 0.0
        taken = 0
        halted = False
        halt_row: dict | None = None

        for _, trade in ordered.iterrows():
            # A previous trade already closed the day out below the limit.
            if realised <= -limit:
                halted = True
                halt_row = {
                    "session_date": day,
                    "halt_time": pd.NaT,
                    "equity_at_halt": realised,
                    "forced_flatten": False,
                }
                break

            sign = 1.0 if trade["direction"] == "long" else -1.0
            entry_fill = float(trade["entry_fill"])

            held = session_bars[
                (session_bars.index >= trade["entry_time"])
                & (session_bars.index < trade["exit_time"])
            ]

            breach_pos = None
            if len(held):
                if mark == "adverse":
                    marks = (held["low"] if sign > 0 else held["high"]).to_numpy(float)
                else:
                    marks = held["close"].to_numpy(float)
                open_pnl = (marks - entry_fill) * sign * spec.point_value * contracts
                equity = realised + open_pnl - commission
                hits = np.flatnonzero(equity <= -limit)
                if hits.size:
                    breach_pos = int(hits[0])

            row = {c: trade[c] for c in TRADE_COLUMNS}

            if breach_pos is None:
                kept_raw.append(row)
                realised += float(trade["net_pnl"])
                taken += 1
                continue

            breach_ts = held.index[breach_pos]
            where = int(session_bars.index.get_indexer([breach_ts])[0])
            if where + 1 < len(session_bars):
                row["exit_time"] = session_bars.index[where + 1]
                row["exit_price"] = float(session_bars.iloc[where + 1]["open"])
            else:
                # Breach on the session's final bar: nothing left to open into.
                row["exit_time"] = breach_ts
                row["exit_price"] = float(session_bars.iloc[where]["close"])
            row["exit_reason"] = LOSS_LIMIT_EXIT

            kept_raw.append(row)
            taken += 1
            halted = True
            halt_row = {
                "session_date": day,
                "halt_time": breach_ts,
                "equity_at_halt": float(realised + open_pnl[breach_pos] - commission),
                "forced_flatten": True,
            }
            break

        if halted and halt_row is not None:
            halt_row["trades_taken"] = taken
            halt_row["trades_blocked"] = len(ordered) - taken
            halts.append(halt_row)

    kept = price_trades(
        pd.DataFrame(kept_raw, columns=TRADE_COLUMNS), spec, costs, contracts
    )
    if not kept.empty:
        kept = kept.sort_values("entry_time").reset_index(drop=True)
    return kept, pd.DataFrame(halts, columns=HALT_COLUMNS)


def run_backtest(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    spec: ContractSpec = MES,
    costs: CostModel = CostModel(),
    contracts: int = 1,
    enforce_loss_limit: bool = True,
    daily_loss_limit: float = 300.0,
    mark: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signals plus bars in, priced trade list out.

    Returns ``(trades, halt_log)``. With ``enforce_loss_limit`` the trade list
    is what the rules would actually have permitted; without it, it is the raw
    signal.
    """
    trades = price_trades(build_trades(signals, bars), spec, costs, contracts)
    if not enforce_loss_limit:
        return trades, pd.DataFrame(columns=HALT_COLUMNS)
    return enforce_daily_loss_limit(
        trades, bars, spec, costs, contracts, daily_loss_limit, mark
    )
