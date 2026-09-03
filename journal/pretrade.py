"""Pre-trade check. Nothing gets placed without a logged ticket.

    python journal/pretrade.py --instrument MES --direction long \
        --entry 6800.25 --stop 6795.00 --contracts 2 \
        --thesis "failed breakdown, reclaimed VWAP"

Prints ALLOW or BLOCK with every reason, and on ALLOW appends the ticket to
``journal/trades.jsonl``. A BLOCK writes nothing: the journal records trades
you were permitted to take, so a blocked idea leaves no ticket to close.

Every threshold comes from :mod:`rules`. There are no numbers in this file.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import Collection

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("strategies", "backtests", "journal"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import rules  # noqa: E402
import store  # noqa: E402

DIRECTIONS = ("long", "short")


@dataclass(frozen=True)
class TicketRequest:
    instrument: str
    direction: str
    entry_price: float
    stop_price: float
    contracts: int
    thesis: str


@dataclass(frozen=True)
class AccountState:
    """Everything the decision needs about where the account already stands."""

    open_position: int = 0
    today_realised_pnl: float = 0.0
    balance: float = rules.ACCOUNT_SIZE
    peak_eod_balance: float = rules.ACCOUNT_SIZE

    @classmethod
    def from_journal(cls, day: date_type, path: Path = store.TRADES_PATH) -> "AccountState":
        balance, peak = store.account_balance(path=path)
        return cls(
            open_position=store.net_open_position(path),
            today_realised_pnl=store.realised_pnl_on(day, path),
            balance=balance,
            peak_eod_balance=peak,
        )


@dataclass
class Decision:
    allowed: bool = True
    blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    risk_dollars: float = 0.0
    checks: dict = field(default_factory=dict)

    def block(self, reason: str) -> None:
        self.allowed = False
        self.blocks.append(reason)

    def warn(self, reason: str) -> None:
        self.warnings.append(reason)


def evaluate(
    request: TicketRequest,
    state: AccountState,
    now: pd.Timestamp,
    early_close_dates: Collection[date_type] = (),
    roll_dates: Collection[date_type] = (),
    closed_dates: Collection[date_type] = (),
) -> Decision:
    """Run every rule against a proposed trade. Pure: no I/O, no clock.

    ``now`` is injected rather than read from the system clock so the boundary
    cases can be tested at the same instants as the rules themselves.
    """
    decision = Decision()
    now = rules.to_et(now)
    day = now.date()

    if day in set(closed_dates):
        decision.block(f"{day} is an exchange holiday - there is no session to trade")
        decision.checks["exchange_open"] = "BLOCK"
        return decision
    decision.checks["exchange_open"] = "ok"

    # --- shape of the request itself ---------------------------------------
    if not rules.is_allowed_instrument(request.instrument):
        decision.block(
            f"Instrument {request.instrument!r} is not permitted "
            f"(rule 1 allows {sorted(rules.ALLOWED_INSTRUMENTS)})"
        )
        # Everything below needs a contract spec, so stop here.
        decision.checks["instrument"] = "BLOCK"
        return decision
    decision.checks["instrument"] = "ok"

    if request.direction not in DIRECTIONS:
        decision.block(f"Direction must be one of {DIRECTIONS}, got {request.direction!r}")
        return decision

    if int(request.contracts) < 1:
        decision.block(f"Contracts must be at least 1, got {request.contracts}")
        return decision

    if not str(request.thesis).strip():
        decision.block(
            "No thesis given. A trade you cannot describe in one line is a trade "
            "you cannot review later."
        )

    # A stop on the wrong side is a typo, and it would silently invert the risk.
    wrong_side = (
        request.direction == "long" and request.stop_price >= request.entry_price
    ) or (
        request.direction == "short" and request.stop_price <= request.entry_price
    )
    if wrong_side:
        decision.block(
            f"Stop {request.stop_price} is on the wrong side of entry "
            f"{request.entry_price} for a {request.direction}"
        )
        decision.checks["stop_side"] = "BLOCK"
        return decision
    decision.checks["stop_side"] = "ok"

    risk = store.risk_dollars(
        request.instrument, request.entry_price, request.stop_price, request.contracts
    )
    decision.risk_dollars = risk

    # --- rules 2 and 3: the clock ------------------------------------------
    if rules.must_flatten(now, early_close_dates):
        decision.block(
            f"Past the {rules.flatten_deadline(now, early_close_dates)} forced "
            f"flatten - positions should be closed, not opened"
        )
        decision.checks["flatten"] = "BLOCK"
    else:
        decision.checks["flatten"] = "ok"

    if not rules.is_entry_allowed(now, early_close_dates, roll_dates):
        if rules.is_roll_day(day, roll_dates):
            decision.block(f"{day} is a contract roll date - no new entries")
            decision.checks["entry_window"] = "BLOCK (roll day)"
        else:
            cutoff = rules.entry_deadline(now, early_close_dates)
            decision.block(
                f"Entry cutoff {cutoff} has passed (now {now.time().strftime('%H:%M:%S')})"
                + (" - early-close session" if day in set(early_close_dates) else "")
            )
            decision.checks["entry_window"] = "BLOCK (after cutoff)"
    else:
        decision.checks["entry_window"] = "ok"

    # --- rule 4: position cap ----------------------------------------------
    signed = int(request.contracts) * (1 if request.direction == "long" else -1)
    allowed_size = rules.clamp_order_size(state.open_position, signed)
    if allowed_size != signed:
        decision.block(
            f"Position cap: holding {state.open_position}, requested {signed:+d}, "
            f"only {allowed_size:+d} fits under the {rules.POSITION_CAP}-contract cap"
        )
        decision.checks["position_cap"] = "BLOCK"
    else:
        decision.checks["position_cap"] = "ok"

    # --- rule 5: daily loss budget -----------------------------------------
    budget = rules.DAILY_LOSS_LIMIT + state.today_realised_pnl
    if rules.is_daily_loss_breached(state.today_realised_pnl):
        decision.block(
            f"Daily loss limit already reached: today is "
            f"${state.today_realised_pnl:,.2f} against a "
            f"${rules.DAILY_LOSS_LIMIT:,.0f} limit. Done for the day."
        )
        decision.checks["daily_loss"] = "BLOCK"
    elif risk > budget:
        decision.block(
            f"Risk ${risk:,.2f} exceeds the ${budget:,.2f} left in today's loss "
            f"budget (limit ${rules.DAILY_LOSS_LIMIT:,.0f}, "
            f"today ${state.today_realised_pnl:,.2f})"
        )
        decision.checks["daily_loss"] = "BLOCK"
    else:
        decision.checks["daily_loss"] = f"ok (${budget - risk:,.2f} left after this)"

    # --- rule 5b: trailing drawdown ----------------------------------------
    dd = rules.drawdown_from_peak(state.balance, state.peak_eod_balance)
    dd_state = rules.trailing_drawdown_state(state.balance, state.peak_eod_balance)
    room = rules.INTERNAL.trailing_drawdown_stop - dd
    if dd_state == "stop":
        decision.block(
            f"Trailing drawdown ${dd:,.2f} is at or past the internal stop of "
            f"${rules.INTERNAL.trailing_drawdown_stop:,.0f}. Stop trading. "
            f"(firm terminates at ${rules.FIRM.max_trailing_drawdown:,.0f})"
        )
        decision.checks["trailing_drawdown"] = "BLOCK"
    elif risk > room:
        decision.block(
            f"Risk ${risk:,.2f} exceeds the ${room:,.2f} left before the internal "
            f"trailing-drawdown stop"
        )
        decision.checks["trailing_drawdown"] = "BLOCK"
    else:
        decision.checks["trailing_drawdown"] = f"ok (${room - risk:,.2f} left after this)"
        if dd_state == "warn":
            decision.warn(
                f"Trailing drawdown ${dd:,.2f} is past the "
                f"${rules.INTERNAL.trailing_drawdown_warn:,.0f} warning line"
            )

    # --- rule 6, as a reminder rather than a gate --------------------------
    decision.warn(
        f"Minimum hold {rules.MIN_HOLD_SECONDS}s - do not close before "
        f"{(now + pd.Timedelta(seconds=rules.MIN_HOLD_SECONDS)).time().strftime('%H:%M:%S')}"
    )
    return decision


def format_decision(request: TicketRequest, state: AccountState,
                    decision: Decision, now: pd.Timestamp) -> str:
    verdict = "ALLOW" if decision.allowed else "BLOCK"
    line = "=" * 74
    out = [line, f"  {verdict}   {request.direction.upper()} {request.contracts} "
                 f"{request.instrument} @ {request.entry_price} stop {request.stop_price}",
           line,
           f"  Time            {rules.to_et(now).strftime('%Y-%m-%d %H:%M:%S %Z')}",
           f"  Risk if stopped ${decision.risk_dollars:,.2f}  "
           f"(stop distance + 2 ticks slippage + commission)",
           f"  Open position   {state.open_position:+d} contracts",
           f"  Today realised  ${state.today_realised_pnl:,.2f}",
           f"  Balance         ${state.balance:,.2f}   peak EOD "
           f"${state.peak_eod_balance:,.2f}",
           "",
           "  Checks:"]
    for name, status in decision.checks.items():
        out.append(f"    {name:<20} {status}")
    if decision.blocks:
        out.append("")
        out.append("  BLOCKED because:")
        out += [f"    - {r}" for r in decision.blocks]
    if decision.warnings:
        out.append("")
        out.append("  Notes:")
        out += [f"    - {w}" for w in decision.warnings]
    out.append(line)
    return "\n".join(out)


def build_ticket(request: TicketRequest, decision: Decision,
                 now: pd.Timestamp) -> store.Ticket:
    now = rules.to_et(now)
    return store.Ticket(
        ticket_id=store.new_ticket_id(),
        status=store.STATUS_OPEN,
        instrument=request.instrument.upper(),
        direction=request.direction,
        contracts=int(request.contracts),
        entry_price=float(request.entry_price),
        stop_price=float(request.stop_price),
        thesis=request.thesis.strip(),
        risk_dollars=round(decision.risk_dollars, 2),
        entry_time=now.isoformat(),
        session_date=str(now.date()),
        checks=dict(decision.checks),
    )


def calendar_dates(early_close_flag: bool, day: date_type):
    """Early-close, roll and closed dates from every source available.

    Three sources, unioned:

    * ``data/cme_calendar.py`` for future dates - the published holiday rules;
    * the cached bars, for historical half-days actually observed in the data;
    * ``--early-close``, the operator's manual override, which still works when
      the calendar is stale or the date is past its coverage.

    Returns ``(early_closes, roll_dates, closed_dates, warnings)``.
    """
    early: set = {day} if early_close_flag else set()
    rolls: set = set()
    closed: set = set()
    warnings: list[str] = []

    sys.path.insert(0, str(PROJECT_ROOT / "data"))
    try:
        import cme_calendar  # noqa: PLC0415
        early |= cme_calendar.early_close_dates()
        closed |= cme_calendar.closed_dates()
        note = cme_calendar.coverage_warning(day)
        if note:
            warnings.append(note)
        label = cme_calendar.describe(day)
        if label:
            warnings.append(f"Calendar says today is an {label}")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"CME calendar unavailable ({exc}); using --early-close only")

    parquet = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
    if parquet.exists():
        try:
            import loader  # noqa: PLC0415
            bars = loader.load_bars(parquet)
            early |= loader.detect_early_close_dates(bars)
            rolls |= loader.detect_roll_dates(bars)
        except Exception:  # noqa: BLE001 - the calendar and flag still work
            warnings.append("Cached bars unreadable; roll dates not checked")

    return early, rolls, closed, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--direction", required=True, choices=DIRECTIONS)
    ap.add_argument("--entry", type=float, required=True)
    ap.add_argument("--stop", type=float, required=True)
    ap.add_argument("--contracts", type=int, required=True)
    ap.add_argument("--thesis", required=True)
    ap.add_argument("--early-close", action="store_true",
                    help="declare today an exchange half-day (13:00 ET close)")
    ap.add_argument("--now", default=None, help="override the clock, for testing")
    ap.add_argument("--journal", default=str(store.TRADES_PATH))
    args = ap.parse_args(argv)

    now = rules.to_et(pd.Timestamp(args.now) if args.now
                      else pd.Timestamp.now(tz=rules.ET))
    journal_path = Path(args.journal)

    request = TicketRequest(
        instrument=args.instrument.upper(),
        direction=args.direction,
        entry_price=args.entry,
        stop_price=args.stop,
        contracts=args.contracts,
        thesis=args.thesis,
    )
    state = AccountState.from_journal(now.date(), journal_path)
    early, rolls, closed, cal_warnings = calendar_dates(args.early_close, now.date())
    decision = evaluate(request, state, now, early, rolls, closed)
    for note in cal_warnings:
        decision.warn(note)

    print(format_decision(request, state, decision, now))

    if decision.allowed:
        ticket = build_ticket(request, decision, now)
        store.append(ticket, journal_path)
        try:
            shown = journal_path.relative_to(PROJECT_ROOT)
        except ValueError:
            # A journal outside the project is legitimate (a temp file in a
            # test, a path on another drive); it just cannot be relativised.
            shown = journal_path
        print(f"\n  Ticket {ticket.ticket_id} logged to {shown}")
        print(f"  Close it with:  python journal/close.py --ticket {ticket.ticket_id} "
              f"--exit <price> --reason <reason>")
        return 0

    print("\n  No ticket written. Do not place this trade.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
