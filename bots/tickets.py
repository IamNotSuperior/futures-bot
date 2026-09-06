"""Signal -> guards -> ticket. The desk's decision path.

Every signal the desk produces is routed through
:func:`pretrade.evaluate` - **the same function** ``journal/pretrade.py`` calls
for a manual trade, imported rather than reimplemented.

That choice is the point of this module. CLAUDE.md rule 9 wants the risk limits
enforced by runtime logic with no bypass path, and HANDOFF.md records that
restating a threshold has already caused a real bug once. A second copy of the
checks here would satisfy neither: it would pass its own tests, drift on the
first change to ``rules.py``, and drift silently. ``tests/test_desk_rules.py``
asserts the equivalence by calling both paths and comparing the ``Decision``.

Journal separation
------------------
Shadow tickets are written to ``journal/shadow_trades.jsonl``, never to
``journal/trades.jsonl``. Entry 3's 60-trade gate counts the manual journal, so
a shadow ticket must not be able to reach it. The separation is a different
*file*, not a filtered column: ``store`` already takes a path everywhere, and a
column the reader has to remember to filter is one bug away from inflating the
count that licenses buying an evaluation.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Collection

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import pretrade  # noqa: E402
import rules  # noqa: E402
import store  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "bots"))
import broker  # noqa: E402

log = logging.getLogger("desk.tickets")

SHADOW_JOURNAL = PROJECT_ROOT / "journal" / "shadow_trades.jsonl"

#: Prefixed to every shadow ticket, in the embed and in the journal row.
SHADOW_LABEL = broker.SHADOW_LABEL


@dataclass(frozen=True)
class Signal:
    """What a strategy asked for, before any guard has seen it."""

    strategy: str
    instrument: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float | None
    contracts: int
    bar_time: pd.Timestamp
    mode: str
    thesis: str = ""

    def as_request(self) -> pretrade.TicketRequest:
        """The shape ``pretrade.evaluate`` takes.

        The thesis is mandatory upstream - ``pretrade`` blocks an empty one -
        so a strategy signal supplies a generated one naming the strategy and
        the bar. That is not a human's reasoning and does not pretend to be;
        it exists so the row is reviewable later.
        """
        thesis = self.thesis.strip() or (
            f"{self.strategy} signal on the {self.bar_time:%Y-%m-%d %H:%M} bar "
            f"({self.mode} mode, no discretion applied)"
        )
        return pretrade.TicketRequest(
            instrument=self.instrument,
            direction=self.direction,
            entry_price=float(self.entry_price),
            stop_price=float(self.stop_price),
            contracts=int(self.contracts),
            thesis=thesis,
        )


@dataclass
class Outcome:
    """The result of putting one signal through the guards."""

    signal: Signal
    decision: pretrade.Decision
    ticket: store.Ticket | None = None
    fill: broker.Fill | None = None
    refusal: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision.allowed and self.refusal is None

    @property
    def label(self) -> str:
        return SHADOW_LABEL if self.signal.mode == broker.MODE_SHADOW else ""


def account_state(day: date_type, journal_path: Path,
                  open_position: int | None = None) -> pretrade.AccountState:
    """Account state read from ``journal_path``.

    ``open_position`` overrides the journal's view when the desk's own state
    file disagrees - which it can, briefly, between a fill and the journal
    append. The desk passes its own number so the position cap is checked
    against what the desk actually holds.
    """
    state = pretrade.AccountState.from_journal(day, journal_path)
    if open_position is not None:
        state = pretrade.AccountState(
            open_position=int(open_position),
            today_realised_pnl=state.today_realised_pnl,
            balance=state.balance,
            peak_eod_balance=state.peak_eod_balance,
        )
    return state


def evaluate_signal(
    signal: Signal,
    state: pretrade.AccountState,
    now: pd.Timestamp,
    early_close_dates: Collection[date_type] = (),
    roll_dates: Collection[date_type] = (),
    closed_dates: Collection[date_type] = (),
) -> pretrade.Decision:
    """Run the guards. A thin, deliberate delegation - see the module docstring.

    Nothing is added here and nothing is skipped. If this function ever grows a
    check of its own, the manual and automated paths have diverged and the
    equivalence test will fail, which is the intended outcome.
    """
    return pretrade.evaluate(
        signal.as_request(), state, now,
        early_close_dates, roll_dates, closed_dates,
    )


def submit(
    signal: Signal,
    decision: pretrade.Decision,
    adapter: broker.BrokerAdapter,
    account: broker.AccountConfig,
    strategy_status: str,
    now: pd.Timestamp,
    journal_path: Path = SHADOW_JOURNAL,
) -> Outcome:
    """Place an allowed signal, or record why it was not placed.

    A blocked decision writes nothing. ``pretrade`` takes the same position -
    "the journal records trades you were permitted to take, so a blocked idea
    leaves no ticket to close" - and a desk that logged its blocks as tickets
    would corrupt every count that reads the file.
    """
    outcome = Outcome(signal=signal, decision=decision)
    if not decision.allowed:
        return outcome

    order = broker.Order(
        strategy=signal.strategy,
        instrument=signal.instrument,
        direction=signal.direction,
        contracts=int(signal.contracts),
        entry_price=float(signal.entry_price),
        stop_price=float(signal.stop_price),
        target_price=signal.target_price,
        signal_time=rules.to_et(signal.bar_time),
        mode=signal.mode,
    )
    try:
        fill = adapter.submit(order, account, strategy_status)
    except broker.BrokerRefusal as exc:
        outcome.refusal = str(exc)
        decision.block(f"broker refused: {exc}")
        return outcome

    if not fill.simulated and journal_path == SHADOW_JOURNAL:
        # Unreachable today - PaperAdapter is the only adapter and it always
        # simulates. Kept because the day that stops being true, a real fill
        # landing in the shadow journal would silently contaminate the file
        # entry 3's gate is defined against.
        raise RuntimeError(
            f"refusing to write a non-simulated fill from {fill.adapter!r} to "
            f"the shadow journal"
        )

    ticket = pretrade.build_ticket(signal.as_request(), decision, rules.to_et(now))
    # Re-stamp the fields the desk owns. `build_ticket` is reused for its
    # id/field construction; entry price and time come from the fill, and the
    # provenance fields below are what keep a shadow row identifiable forever.
    ticket.entry_price = float(fill.fill_price)
    ticket.entry_time = rules.to_et(fill.fill_time).isoformat()
    ticket.session_date = str(rules.session_date(fill.fill_time))
    ticket.checks = dict(decision.checks) | {
        "mode": signal.mode,
        "strategy": signal.strategy,
        "adapter": fill.adapter,
        "simulated": str(fill.simulated),
        "label": SHADOW_LABEL if signal.mode == broker.MODE_SHADOW else "",
        "target_price": ("" if signal.target_price is None
                         else f"{float(signal.target_price):.2f}"),
    }
    store.append(ticket, Path(journal_path))
    outcome.ticket = ticket
    outcome.fill = fill
    return outcome


def close_position(
    ticket_id: str,
    exit_price: float,
    exit_time,
    reason: str,
    journal_path: Path = SHADOW_JOURNAL,
) -> store.Ticket:
    """Append the closing record for a shadow ticket.

    Delegates to ``close.find_open_ticket`` and ``close.close_ticket`` - the
    same two functions ``journal/close.py`` uses for a manual trade. P&L
    therefore comes from ``store.price_ticket`` -> ``engine.price_trades``,
    which is what makes a shadow result and a backtest result comparable rather
    than approximately so, commission and slippage included.

    ``min_hold_ok`` is set by that shared path too. Rule 7 says a non-zero
    microscalp share is a regression signal rather than merely a risk warning,
    so the daily summary counts this field to detect rule 6's enforcement
    failing.
    """
    import close as journal_close  # noqa: PLC0415 - journal/ is on sys.path

    path = Path(journal_path)
    ticket = journal_close.find_open_ticket(ticket_id, path)
    closed = journal_close.close_ticket(
        ticket, float(exit_price), rules.to_et(exit_time), reason
    )
    store.append(closed, path)
    return closed


def block_reasons(decision: pretrade.Decision) -> list[str]:
    return list(decision.blocks)
