"""Close a logged ticket and record the outcome.

    python journal/close.py --ticket a1b2c3d4e5f6 --exit 6812.50 --reason target

P&L comes from :func:`engine.price_trades` - the same function, cost model and
slippage the backtests use - so a paper result sits on the same scale as a
backtest result rather than merely near it.

Closing appends a new record; the original open ticket stays on disk.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("strategies", "backtests", "journal"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import rules  # noqa: E402
import store  # noqa: E402

REASONS = ("target", "stop", "manual", "session_end", "loss_limit_flatten", "other")


def find_open_ticket(ticket_id: str, path: Path) -> store.Ticket:
    tickets = store.load_tickets(path)
    if tickets.empty:
        raise LookupError("The journal is empty - nothing to close.")

    match = tickets[tickets["ticket_id"] == ticket_id]
    if match.empty:
        open_ids = list(tickets[tickets["status"] == store.STATUS_OPEN]["ticket_id"])
        raise LookupError(
            f"No ticket {ticket_id!r}. Open tickets: {open_ids or 'none'}"
        )
    row = match.iloc[-1]
    if row["status"] == store.STATUS_CLOSED:
        raise LookupError(
            f"Ticket {ticket_id} is already closed "
            f"(exit {row['exit_price']} at {row['exit_time']})."
        )

    return store.Ticket(
        ticket_id=row["ticket_id"],
        status=store.STATUS_OPEN,
        instrument=row["instrument"],
        direction=row["direction"],
        contracts=int(row["contracts"]),
        entry_price=float(row["entry_price"]),
        stop_price=float(row["stop_price"]),
        thesis=row["thesis"],
        risk_dollars=float(row["risk_dollars"]),
        entry_time=pd.Timestamp(row["entry_time"]).isoformat(),
        session_date=str(row["session_date"]),
        checks=row.get("checks") or {},
    )


def close_ticket(
    ticket: store.Ticket,
    exit_price: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
    entry_price: float | None = None,
    entry_time: pd.Timestamp | None = None,
) -> store.Ticket:
    """Return the closed record. ``entry_*`` override the planned fill."""
    if entry_price is not None:
        ticket = replace(ticket, entry_price=float(entry_price))
    if entry_time is not None:
        ticket = replace(ticket, entry_time=rules.to_et(entry_time).isoformat())

    exit_time = rules.to_et(exit_time)
    ticket = replace(ticket, exit_reason=exit_reason)
    priced = store.price_ticket(ticket, exit_price, exit_time)

    held = priced["duration_seconds"]
    return replace(
        ticket,
        status=store.STATUS_CLOSED,
        exit_price=float(exit_price),
        exit_time=exit_time.isoformat(),
        exit_reason=exit_reason,
        min_hold_ok=bool(held >= rules.MIN_HOLD_SECONDS),
        **priced,
    )


def format_close(closed: store.Ticket) -> str:
    line = "=" * 74
    held = closed.duration_seconds or 0.0
    minutes, seconds = divmod(int(held), 60)
    out = [
        line,
        f"  CLOSED {closed.ticket_id}   {closed.direction.upper()} "
        f"{closed.contracts} {closed.instrument}",
        line,
        f"  Entry           {closed.entry_price}  at {closed.entry_time}",
        f"  Exit            {closed.exit_price}  at {closed.exit_time}  "
        f"({closed.exit_reason})",
        f"  Held            {minutes}m {seconds}s",
        "",
        f"  Gross P&L       ${closed.gross_pnl:>10,.2f}",
        f"  Commission      ${-closed.commission:>10,.2f}",
        f"  Slippage        ${-closed.slippage_cost:>10,.2f}",
        f"  NET P&L         ${closed.net_pnl:>10,.2f}",
        "",
        f"  Planned risk    ${closed.risk_dollars:,.2f}",
        f"  Thesis          {closed.thesis}",
    ]
    if closed.min_hold_ok is False:
        out += [
            "",
            f"  *** RULE 6 VIOLATION: held {int(held)}s, floor is "
            f"{rules.MIN_HOLD_SECONDS}s ***",
            "  This resets the clean-trade count in research/hypotheses.md.",
        ]
    out.append(line)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticket", required=True)
    ap.add_argument("--exit", dest="exit_price", type=float, required=True)
    ap.add_argument("--reason", default="manual", choices=REASONS)
    ap.add_argument("--exit-time", default=None,
                    help="defaults to now; accepts any timestamp")
    ap.add_argument("--entry-price", type=float, default=None,
                    help="actual fill, if it differed from the planned entry")
    ap.add_argument("--entry-time", default=None)
    ap.add_argument("--journal", default=str(store.TRADES_PATH))
    args = ap.parse_args(argv)

    journal_path = Path(args.journal)
    try:
        ticket = find_open_ticket(args.ticket, journal_path)
    except LookupError as exc:
        print(f"  {exc}")
        return 1

    exit_time = (pd.Timestamp(args.exit_time) if args.exit_time
                 else pd.Timestamp.now(tz=rules.ET))
    closed = close_ticket(
        ticket,
        exit_price=args.exit_price,
        exit_time=exit_time,
        exit_reason=args.reason,
        entry_price=args.entry_price,
        entry_time=pd.Timestamp(args.entry_time) if args.entry_time else None,
    )
    store.append(closed, journal_path)
    print(format_close(closed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
