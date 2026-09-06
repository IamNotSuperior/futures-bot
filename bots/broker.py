"""Broker adapters. One interface, one implementation, and a refusal.

The adapter boundary exists so the paper and live paths cannot diverge: every
order goes through :meth:`BrokerAdapter.submit`, so a strategy cannot reach a
broker by a route that skipped a guard.

**Only :class:`PaperAdapter` exists.** There is no ``TradersPostAdapter``, not
even a stub raising ``NotImplementedError``. An empty class is an invitation,
and HANDOFF.md §6 records that the tempting failure here is wiring a broker
because the scaffolding looks ready. It is ready in the sense that the guards
work; it is not ready in the sense that nothing has demonstrated an edge.

The live check
--------------
:func:`require_live_eligible` is two independent conditions, evaluated in a
fixed order:

1. the strategy's registry status is ``live``; and
2. the account's ``live_enabled`` flag is on.

The status check runs **first and unconditionally**, so a flag turned on by
accident - or by someone who believed the flag was the switch - still refuses.
The flag can only ever subtract permission, never add it. This mirrors
``Registry.promote``: there is no override argument here either.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import rules  # noqa: E402
import store  # noqa: E402

#: Modes an adapter can run in. ``shadow`` and ``paper`` both simulate; they
#: differ only in what the ticket is allowed to claim about itself.
MODE_SHADOW = "shadow"
MODE_PAPER = "paper"
MODE_LIVE = "live"

SHADOW_LABEL = "SHADOW - strategy rejected, no orders"


class BrokerRefusal(RuntimeError):
    """An order was refused. The message says which condition refused it."""


@dataclass(frozen=True)
class AccountConfig:
    """One trading account the desk can route to.

    ``live_enabled`` defaults to False and is the *only* place the flag exists.
    It is necessary but never sufficient: see :func:`require_live_eligible`.
    """

    name: str = "paper-1"
    live_enabled: bool = False
    journal_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "journal" / "shadow_trades.jsonl"
    )

    def __post_init__(self) -> None:
        if self.live_enabled and self.journal_path.name == "shadow_trades.jsonl":
            # Not a rule so much as a contradiction: the shadow journal is the
            # one entry 3's count deliberately ignores.
            raise ValueError(
                f"account {self.name!r} has live_enabled=True but writes to the "
                f"shadow journal; a live account needs its own journal"
            )


@dataclass(frozen=True)
class Order:
    """A request to open a position. Prices are levels, not fills."""

    strategy: str
    instrument: str
    direction: str
    contracts: int
    entry_price: float
    stop_price: float
    target_price: float | None
    signal_time: pd.Timestamp
    mode: str


@dataclass(frozen=True)
class Fill:
    """What an adapter reports back. Simulated unless ``mode`` is live."""

    order: Order
    fill_price: float
    fill_time: pd.Timestamp
    simulated: bool
    adapter: str
    note: str = ""


def require_live_eligible(strategy_status: str, account: AccountConfig,
                          strategy_name: str = "?") -> None:
    """Raise unless this strategy may be routed to a real broker.

    Both conditions must hold, and the status is checked first so that the
    account flag is never the thing standing between a rejected strategy and a
    broker. Order matters more than it looks: it decides which message the
    operator sees, and "orb2 is rejected" is the useful one.
    """
    if strategy_status != "live":
        raise BrokerRefusal(
            f"{strategy_name} is at {strategy_status!r} status, not 'live'. "
            f"Only a strategy promoted to 'live' in the registry may be routed "
            f"to a broker, and Registry.promote will not put one there without "
            f"an ACCEPTED walk-forward in research/hypotheses.md. "
            f"account.live_enabled={account.live_enabled} does not change this."
        )
    if not account.live_enabled:
        raise BrokerRefusal(
            f"account {account.name!r} has live_enabled=False; refusing to route "
            f"{strategy_name} even though it is at 'live' status"
        )


class BrokerAdapter(ABC):
    """Where an order goes. Subclasses simulate or route; nothing else."""

    #: Reported on every fill, so a journal row says which adapter produced it.
    name: str = "unnamed"
    #: False for anything that can move money. Read by the desk before it
    #: writes a ticket, so a real fill can never land in the shadow journal.
    simulated: bool = True

    @abstractmethod
    def submit(self, order: Order, account: AccountConfig,
               strategy_status: str) -> Fill:
        """Accept ``order`` or raise :class:`BrokerRefusal`."""

    @abstractmethod
    def flatten(self, order: Order, account: AccountConfig, price: float,
                now: pd.Timestamp, reason: str) -> Fill:
        """Close an open position. Never refused - rules 2 and 5 outrank."""


class PaperAdapter(BrokerAdapter):
    """Fills simulated locally, reported at the **signal level**.

    Slippage and commission are deliberately **not** applied here. They belong
    to ``store.price_ticket`` -> ``engine.price_trades``, which is the cost
    model the backtests use, and routing every closed ticket through it is what
    makes a shadow result comparable to a backtest rather than approximately
    so. This adapter's job is to say *where* the fill happened, not what it
    cost.

    An earlier version slipped the entry by one tick here as well. Because
    ``price_trades`` computes ``entry_fill = entry_price + sign * slip``
    itself, that charged entry slippage twice - $1.25 per contract per trade,
    silently, in the pessimistic direction. The observed replay trade came out
    at -$56.25 where the correct figure is -$55.00. It was found by hand-
    checking one ticket against the engine, which is the only reason a
    one-tick discrepancy was visible at all.

    A future live adapter reports whatever price the broker actually filled at.
    That number is a *real* fill, so the cost model must not slip it again -
    ``price_trades`` would need a zero-slippage cost model for live tickets.
    That is a decision for whoever builds the live adapter; it is recorded here
    because the same double-count is waiting on that path.
    """

    name = "paper"
    simulated = True

    def submit(self, order: Order, account: AccountConfig,
               strategy_status: str) -> Fill:
        rules.require_allowed_instrument(order.instrument)
        rules.require_position_cap(abs(int(order.contracts)))
        if order.mode == MODE_LIVE:
            # A live order reaching the paper adapter is a routing bug. Check
            # eligibility anyway so the refusal is the same one either way.
            require_live_eligible(strategy_status, account, order.strategy)
            raise BrokerRefusal(
                f"PaperAdapter cannot route a live order for {order.strategy}; "
                f"no live adapter is implemented"
            )
        return Fill(
            order=order,
            fill_price=float(order.entry_price),
            fill_time=rules.to_et(order.signal_time),
            simulated=True,
            adapter=self.name,
            note=SHADOW_LABEL if order.mode == MODE_SHADOW else "",
        )

    def flatten(self, order: Order, account: AccountConfig, price: float,
                now: pd.Timestamp, reason: str) -> Fill:
        return Fill(
            order=order,
            fill_price=float(price),
            fill_time=rules.to_et(now),
            simulated=True,
            adapter=self.name,
            note=reason,
        )


def adapter_for(mode: str, **kwargs) -> BrokerAdapter:
    """The adapter for ``mode``. Live has none, and that is the point."""
    if mode in (MODE_SHADOW, MODE_PAPER):
        return PaperAdapter(**kwargs)
    if mode == MODE_LIVE:
        raise BrokerRefusal(
            "no live adapter is implemented. HANDOFF.md §6 gates live routing "
            "on a strategy reaching 'paper' status in the registry; none has."
        )
    raise ValueError(f"unknown mode {mode!r}")
