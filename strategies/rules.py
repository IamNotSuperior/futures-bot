"""Hard trading rules from CLAUDE.md, implemented as enforceable functions.

This module is the single source of truth for the project's risk rules. Both
the backtest and the execution layer import it, so a rule cannot be satisfied
in one place and quietly skipped in the other (CLAUDE.md rule 9).

Timestamp convention
--------------------
Every function takes exchange-local wall-clock time. A tz-aware timestamp is
converted to America/New_York; a naive timestamp is *interpreted as* ET, which
matches the bar index produced by ``data.loader.load_bars``. Passing a naive
UTC timestamp will therefore be read as ET and give wrong answers - localise
before calling.

Working in ET wall-clock terms is what makes the rules DST-correct: 16:20 ET
is 20:20 UTC in winter and 21:20 UTC in summer, and the tz database resolves
that without any offset arithmetic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, time, timedelta
from typing import Collection

import pandas as pd

ET = "America/New_York"


# ===========================================================================
# Limits: the firm's ceilings, and the tighter lines this code enforces
# ===========================================================================
#
# Two separate sets of numbers, deliberately never merged:
#
#   FIRM     - Lucid's published limits for the 50K Pro evaluation. Breaching
#              any of these ends the account. Recorded here as documentation
#              and as the denominator for the buffer report. **Nothing in this
#              codebase enforces against FIRM values** - if code ever stops at
#              a firm number it means the internal guard above it failed.
#
#   INTERNAL - what the guards actually enforce. Every value sits strictly
#              inside its firm counterpart, so an internal stop fires with
#              room left. That gap is the whole point: a limit you hit exactly
#              is a limit you eventually cross.
#
# Keeping them in separate objects means a future edit cannot loosen an
# internal limit to the firm's number by accident - it would have to be
# retyped, in the internal block, on purpose.


@dataclass(frozen=True)
class FirmLimits:
    """Lucid 50K Pro evaluation - the firm's hard ceilings. Never enforced."""

    name: str = "Lucid 50K Pro eval"
    account_size: float = 50_000.0
    #: Account ends if a single day loses this much.
    daily_loss: float = 1_200.0
    #: Trailing drawdown measured from the peak *end-of-day* balance.
    max_trailing_drawdown: float = 2_000.0
    #: Micro contracts, aggregate.
    max_contracts: int = 40
    #: On funded accounts, no single day may exceed this share of total profit.
    consistency_pct: float = 40.0
    #: Profit needed to pass the evaluation.
    profit_target: float = 3_000.0
    #: Account terms, not ceilings. A withdrawal needs the balance this far
    #: above the start (the drawdown plus $100), and takes at least
    #: ``min_payout``. Nothing enforces against either; the simulator reports
    #: the line as a milestone alongside the pass.
    payout_buffer: float = 2_100.0
    min_payout: float = 500.0


@dataclass(frozen=True)
class InternalLimits:
    """What the code enforces. Every value is tighter than the firm's."""

    #: Stop trading for the day here (firm: 1,200).
    daily_loss: float = 400.0
    #: Warn when trailing drawdown reaches this (firm's line: 2,000).
    trailing_drawdown_warn: float = 1_000.0
    #: Stop trading entirely here, well short of the firm's 2,000.
    trailing_drawdown_stop: float = 1_500.0
    #: Micro contracts, aggregate (firm: 40).
    max_contracts: int = 5
    #: Flag a day above this share of total profit (firm's line: 40%).
    consistency_warn_pct: float = 30.0


FIRM = FirmLimits()
INTERNAL = InternalLimits()


def buffer_report() -> list[tuple[str, float, float, str]]:
    """(limit, internal, firm, headroom) rows, for logging and for the report.

    Makes the gap between what we stop at and what ends the account visible in
    one place rather than implied across two modules.
    """
    return [
        ("Daily loss", INTERNAL.daily_loss, FIRM.daily_loss,
         f"{FIRM.daily_loss - INTERNAL.daily_loss:,.0f} of room"),
        ("Trailing drawdown (stop)", INTERNAL.trailing_drawdown_stop,
         FIRM.max_trailing_drawdown,
         f"{FIRM.max_trailing_drawdown - INTERNAL.trailing_drawdown_stop:,.0f} of room"),
        ("Max contracts", INTERNAL.max_contracts, FIRM.max_contracts,
         f"{FIRM.max_contracts - INTERNAL.max_contracts} contracts of room"),
        ("Consistency", INTERNAL.consistency_warn_pct, FIRM.consistency_pct,
         f"{FIRM.consistency_pct - INTERNAL.consistency_warn_pct:.0f} points of room"),
    ]


# --- CLAUDE.md rule 1: instruments -----------------------------------------
ALLOWED_INSTRUMENTS = frozenset({"MES", "MNQ"})

# --- CLAUDE.md rules 2 and 3: intraday time limits -------------------------
FLATTEN_TIME = time(16, 30)  # rule 2: everything closed by 16:30 ET
ENTRY_CUTOFF = time(16, 20)  # rule 3: no new entries after 16:20 ET

# Gap between the entry cutoff and the forced flatten. Rule 3 exists so that
# every position has at least this much runway, which is what keeps the
# minimum-hold rule from ever colliding with the flatten.
ENTRY_RUNWAY = timedelta(minutes=10)

# CME equity-index futures close 17:00 ET on a regular day; on an early-close
# holiday session they close 13:00 ET.
REGULAR_SESSION_CLOSE = time(17, 0)
EARLY_SESSION_CLOSE = time(13, 0)

# --- Enforced values -------------------------------------------------------
# These aliases are what the guards and the backtest read. They point at
# INTERNAL, never at FIRM. Kept as module-level names because every call site
# already reads them, and because a single indirection here is easier to audit
# than the same constant repeated across modules.

# CLAUDE.md rule 4: position cap
POSITION_CAP = INTERNAL.max_contracts

# CLAUDE.md rule 5: daily loss limit
DAILY_LOSS_LIMIT = INTERNAL.daily_loss

# CLAUDE.md rule 5b: end-of-day trailing drawdown
TRAILING_DD_WARN = INTERNAL.trailing_drawdown_warn
TRAILING_DD_STOP = INTERNAL.trailing_drawdown_stop

# CLAUDE.md rule 6: minimum hold
MIN_HOLD_SECONDS = 30

# CLAUDE.md rule 7: microscalping measure
MICROSCALP_SECONDS = 5
MICROSCALP_PROFIT_FLAG_PCT = 30.0
MICROSCALP_FIRM_LIMIT_PCT = 50.0

# CLAUDE.md rule 8: consistency
WORST_DAY_FLAG_PCT = INTERNAL.consistency_warn_pct

# The evaluation target, for the simulator and the journal.
PROFIT_TARGET = FIRM.profit_target
ACCOUNT_SIZE = FIRM.account_size

# The balance at which a payout first becomes possible. Below the target, so
# every passing path reaches it first; the simulator's payout probability is
# therefore never lower than its pass probability.
PAYOUT_BALANCE = ACCOUNT_SIZE + FIRM.payout_buffer

# ===========================================================================
# Costs: the commission Lucid confirmed, and the figure it replaces
# ===========================================================================
#
# MES commission per contract per side on a 50K LucidPro evaluation is $0.50 -
# $1.00 a round turn. Confirmed by Lucid support on 2026-09-11; the source is
# support.lucidtrading.com article 11508978. Every cost model in the project
# reads this value and nothing restates it (CLAUDE.md rule 9 applied to a cost
# rather than a limit). Slippage is unaffected and remains an assumption: at
# one tick a side it is now the larger part of a round turn.
COMMISSION_PER_SIDE = 0.50

# What every verdict before 2026-09-11 was scored at - the assumption, not a
# rate anyone quoted. Kept by name so those runs can be reproduced exactly
# (`--commission 1.25` on any runner). It is never a default.
ASSUMED_COMMISSION_PER_SIDE = 1.25


class RuleViolation(Exception):
    """Raised when an action would breach a hard rule."""


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def to_et(ts: pd.Timestamp | datetime | str) -> pd.Timestamp:
    """Normalise a timestamp to exchange local time (America/New_York).

    Naive input is interpreted as ET. See the module docstring.
    """
    stamp = pd.Timestamp(ts)
    if stamp.tz is None:
        return stamp.tz_localize(ET)
    return stamp.tz_convert(ET)


def session_date(ts: pd.Timestamp | datetime | str) -> date_type:
    """Calendar date of a timestamp in exchange local time."""
    return to_et(ts).date()


def session_close(
    ts: pd.Timestamp | datetime | str | date_type,
    early_close_dates: Collection[date_type] = (),
) -> time:
    """Exchange close time for the session containing ``ts``."""
    day = ts if isinstance(ts, date_type) and not isinstance(ts, datetime) else session_date(ts)
    return EARLY_SESSION_CLOSE if day in set(early_close_dates) else REGULAR_SESSION_CLOSE


def _minus(t: time, delta: timedelta) -> time:
    """Subtract a timedelta from a wall-clock time (same-day arithmetic)."""
    anchor = datetime(2000, 1, 1, t.hour, t.minute, t.second)
    return (anchor - delta).time()


def flatten_deadline(
    ts: pd.Timestamp | datetime | str | date_type,
    early_close_dates: Collection[date_type] = (),
) -> time:
    """Wall-clock time by which all positions must be flat for this session.

    Normally 16:30 ET. On an early-close session the exchange closes at 13:00,
    so the deadline moves earlier - you cannot flatten into a closed market.
    """
    return min(FLATTEN_TIME, session_close(ts, early_close_dates))


def entry_deadline(
    ts: pd.Timestamp | datetime | str | date_type,
    early_close_dates: Collection[date_type] = (),
) -> time:
    """Wall-clock time after which no new position may be opened.

    Normally 16:20 ET. On an early-close session it is the flatten deadline
    less the standard runway, preserving the same guarantee that every entry
    has room to satisfy the minimum-hold rule before the flatten.
    """
    return min(ENTRY_CUTOFF, _minus(flatten_deadline(ts, early_close_dates), ENTRY_RUNWAY))


# ---------------------------------------------------------------------------
# Rule 1: instruments
# ---------------------------------------------------------------------------


def is_allowed_instrument(symbol: str) -> bool:
    """True if ``symbol`` refers to MES or MNQ.

    Accepts continuous/parent forms such as ``MES.v.0`` and ``MNQ.FUT`` as well
    as raw contract codes such as ``MESZ4``, by taking the leading root.
    """
    if not symbol:
        return False
    root = str(symbol).split(".")[0][:3].upper()
    return root in ALLOWED_INSTRUMENTS


def require_allowed_instrument(symbol: str) -> str:
    if not is_allowed_instrument(symbol):
        raise RuleViolation(
            f"Instrument {symbol!r} is not permitted; "
            f"CLAUDE.md rule 1 allows only {sorted(ALLOWED_INSTRUMENTS)}."
        )
    return symbol


# ---------------------------------------------------------------------------
# Rules 2 and 3: intraday time limits
# ---------------------------------------------------------------------------


def is_entry_allowed(
    ts: pd.Timestamp | datetime | str,
    early_close_dates: Collection[date_type] = (),
    roll_dates: Collection[date_type] = (),
) -> bool:
    """True if a new position may be opened at ``ts``.

    Blocked at or after the entry deadline - the cutoff time itself is already
    too late - and blocked outright on a contract roll date.
    """
    et = to_et(ts)
    if is_roll_day(et, roll_dates):
        return False
    return et.time() < entry_deadline(et, early_close_dates)


def must_flatten(
    ts: pd.Timestamp | datetime | str,
    early_close_dates: Collection[date_type] = (),
) -> bool:
    """True if any open position must be force-closed at or after ``ts``."""
    et = to_et(ts)
    return et.time() >= flatten_deadline(et, early_close_dates)


# ---------------------------------------------------------------------------
# Roll days
# ---------------------------------------------------------------------------


def is_roll_day(
    ts: pd.Timestamp | datetime | str | date_type,
    roll_dates: Collection[date_type] = (),
) -> bool:
    """True if ``ts`` falls on a session where the underlying contract changed.

    A continuous series is not back-adjusted across a roll, so a position held
    over the boundary sees a price jump that is an artefact, not a move.
    """
    if not roll_dates:
        return False
    day = ts if isinstance(ts, date_type) and not isinstance(ts, datetime) else session_date(ts)
    return day in set(roll_dates)


# ---------------------------------------------------------------------------
# Rule 4: position cap
# ---------------------------------------------------------------------------


def within_position_cap(size: int) -> bool:
    return abs(int(size)) <= POSITION_CAP


def clamp_order_size(current_position: int, requested_delta: int) -> int:
    """Largest part of ``requested_delta`` that keeps the net position capped.

    Guarantees, in order of importance:

    * the result never has the opposite sign to the request - clamping must
      never turn a buy into a sell;
    * the result is never larger in magnitude than the request - a risk control
      may shrink an order, never enlarge one;
    * the resulting net position satisfies the cap, or, if the current position
      already breaches it, at least moves toward flat without crossing to an
      over-cap position on the other side.

    The third clause only matters if a position somehow got past the cap - a
    state that should be impossible, but which the clamp must still be able to
    unwind rather than refusing every order.
    """
    current = int(current_position)
    delta = int(requested_delta)
    if delta == 0:
        return 0

    step = 1 if delta > 0 else -1
    best = 0
    for size in range(1, abs(delta) + 1):
        candidate = step * size
        net = current + candidate
        acceptable = abs(net) <= POSITION_CAP or (
            abs(net) < abs(current) and (net >= 0) == (current >= 0)
        )
        if not acceptable:
            break
        best = candidate
    return best


def require_position_cap(size: int) -> int:
    if not within_position_cap(size):
        raise RuleViolation(
            f"Position size {size} exceeds the hard cap of {POSITION_CAP} contracts "
            "(CLAUDE.md rule 4)."
        )
    return int(size)


# ---------------------------------------------------------------------------
# Rule 5: daily loss limit
# ---------------------------------------------------------------------------


def is_daily_loss_breached(day_pnl: float) -> bool:
    """True once the day's P&L has hit a loss of DAILY_LOSS_LIMIT or worse."""
    return float(day_pnl) <= -DAILY_LOSS_LIMIT


def can_open_new_position(
    ts: pd.Timestamp | datetime | str,
    day_pnl: float,
    early_close_dates: Collection[date_type] = (),
    roll_dates: Collection[date_type] = (),
) -> bool:
    """Combined gate: time cutoff, roll day, and daily loss limit."""
    if is_daily_loss_breached(day_pnl):
        return False
    return is_entry_allowed(ts, early_close_dates, roll_dates)


# ---------------------------------------------------------------------------
# Rule 6: minimum hold
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rule 5b: end-of-day trailing drawdown
# ---------------------------------------------------------------------------


def trailing_floor(peak_eod_balance: float, limit: float | None = None) -> float:
    """The balance at which the account is dead, given the peak EOD balance.

    The drawdown trails the highest *end-of-day* balance ever reached, so
    intraday spikes do not raise it - only a close does. Modelled as a pure
    trail with no lock-in: some firms freeze the floor once it reaches the
    starting balance, and if Lucid does, this is the conservative reading.
    """
    if limit is None:
        limit = INTERNAL.trailing_drawdown_stop
    return float(peak_eod_balance) - float(limit)


def drawdown_from_peak(balance: float, peak_eod_balance: float) -> float:
    """How far below the peak EOD balance we are, as a positive number."""
    return max(0.0, float(peak_eod_balance) - float(balance))


def trailing_drawdown_state(balance: float, peak_eod_balance: float) -> str:
    """``"ok"``, ``"warn"`` or ``"stop"`` for the current balance.

    ``"stop"`` is the internal line, not the firm's - by the time this fires
    there is still headroom before the account would actually be terminated.
    """
    dd = drawdown_from_peak(balance, peak_eod_balance)
    if dd >= INTERNAL.trailing_drawdown_stop:
        return "stop"
    if dd >= INTERNAL.trailing_drawdown_warn:
        return "warn"
    return "ok"


def is_account_terminated(balance: float, peak_eod_balance: float) -> bool:
    """True if the FIRM would have ended the account at this balance.

    Only for measuring what would have happened - never a trading gate. The
    guards stop at :data:`TRAILING_DD_STOP` long before this is reachable.
    """
    return drawdown_from_peak(balance, peak_eod_balance) >= FIRM.max_trailing_drawdown


def headroom_to_firm_termination(balance: float, peak_eod_balance: float) -> float:
    """Dollars of loss still available before the firm's line is crossed."""
    return FIRM.max_trailing_drawdown - drawdown_from_peak(balance, peak_eod_balance)


def hold_seconds(entry_ts, exit_ts) -> float:
    return (to_et(exit_ts) - to_et(entry_ts)).total_seconds()


def can_close(entry_ts, exit_ts, *, override: bool = False) -> bool:
    """True if a position opened at ``entry_ts`` may be closed at ``exit_ts``.

    ``override`` is for the hard risk controls only - the forced flatten and a
    daily-loss breach. If an override is ever needed in practice it means the
    entry guard let through a position it should not have, so callers should
    log it rather than treat it as routine (CLAUDE.md rule 6).
    """
    if override:
        return True
    return hold_seconds(entry_ts, exit_ts) >= MIN_HOLD_SECONDS


def seconds_until_closable(entry_ts, now_ts) -> float:
    """Seconds remaining before the minimum hold is satisfied (0 if already)."""
    return max(0.0, MIN_HOLD_SECONDS - hold_seconds(entry_ts, now_ts))
