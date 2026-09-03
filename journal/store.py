"""Append-only ticket store for the manual trading journal.

Format
------
``journal/trades.jsonl``, one JSON object per line. Records are **appended,
never rewritten**: opening a ticket appends a ``status="open"`` record, and
closing it appends a second, complete ``status="closed"`` record carrying the
same ``ticket_id``. Readers fold by ticket id, keeping the last record.

The append-only shape is the point. A journal whose past can be edited is a
journal you can quietly correct after a bad day, and the whole purpose here is
to make that awkward. The full history stays on disk even though the reader
only surfaces the folded view.

Reuse
-----
P&L is computed with :func:`engine.price_trades`, the same function and the
same cost model the backtests use, so a paper result and a backtest result are
directly comparable rather than approximately so.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("strategies", "backtests"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import rules  # noqa: E402
from engine import MES, MNQ, ContractSpec, CostModel, price_trades  # noqa: E402

JOURNAL_DIR = PROJECT_ROOT / "journal"
TRADES_PATH = JOURNAL_DIR / "trades.jsonl"

#: Contract specs by instrument root. Only rule 1's allowlist appears here.
SPECS: dict[str, ContractSpec] = {"MES": MES, "MNQ": MNQ}

#: The journal uses the engine's default cost model so paper and backtest
#: figures are produced by identical arithmetic.
COSTS = CostModel()

STATUS_OPEN = "open"
STATUS_CLOSED = "closed"


def spec_for(instrument: str) -> ContractSpec:
    rules.require_allowed_instrument(instrument)
    return SPECS[instrument.split(".")[0][:3].upper()]


def new_ticket_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Ticket:
    """One planned trade. Written on ALLOW, amended on close."""

    ticket_id: str
    status: str
    instrument: str
    direction: str
    contracts: int
    entry_price: float
    stop_price: float
    thesis: str
    risk_dollars: float
    entry_time: str
    session_date: str
    checks: dict = field(default_factory=dict)

    # Filled in at close.
    exit_price: float | None = None
    exit_time: str | None = None
    exit_reason: str | None = None
    net_pnl: float | None = None
    gross_pnl: float | None = None
    commission: float | None = None
    slippage_cost: float | None = None
    duration_seconds: float | None = None
    min_hold_ok: bool | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def append(record: Ticket, path: Path = TRADES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json() + "\n")


def read_raw(path: Path = TRADES_PATH) -> pd.DataFrame:
    """Every record ever written, in order. Nothing folded, nothing dropped."""
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def fold(raw: pd.DataFrame) -> pd.DataFrame:
    """Latest record per ticket."""
    if raw.empty or "ticket_id" not in raw.columns:
        return raw
    return raw.drop_duplicates("ticket_id", keep="last").reset_index(drop=True)


def _localise(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
            frame[col] = frame[col].dt.tz_convert(rules.ET)
    return frame


def load_tickets(path: Path = TRADES_PATH) -> pd.DataFrame:
    """All tickets, folded, with timestamps localised to ET."""
    folded = fold(read_raw(path))
    if folded.empty:
        return folded
    folded = _localise(folded, ("entry_time", "exit_time"))
    if "session_date" in folded.columns:
        folded["session_date"] = pd.to_datetime(folded["session_date"]).dt.date
    return folded.sort_values("entry_time").reset_index(drop=True)


def load_closed(path: Path = TRADES_PATH) -> pd.DataFrame:
    """Closed tickets only - the frame the metrics and review code expects."""
    tickets = load_tickets(path)
    if tickets.empty:
        return tickets
    closed = tickets[tickets["status"] == STATUS_CLOSED].copy()
    return closed.reset_index(drop=True)


def load_open(path: Path = TRADES_PATH) -> pd.DataFrame:
    tickets = load_tickets(path)
    if tickets.empty:
        return tickets
    return tickets[tickets["status"] == STATUS_OPEN].reset_index(drop=True)


def net_open_position(path: Path = TRADES_PATH) -> int:
    """Signed contracts currently open, long positive."""
    opened = load_open(path)
    if opened.empty:
        return 0
    signed = opened.apply(
        lambda r: int(r["contracts"]) * (1 if r["direction"] == "long" else -1),
        axis=1,
    )
    return int(signed.sum())


def realised_pnl_on(day: date_type, path: Path = TRADES_PATH) -> float:
    """Realised P&L for one session date."""
    closed = load_closed(path)
    if closed.empty:
        return 0.0
    today = closed[closed["session_date"] == day]
    if today.empty:
        return 0.0
    return float(today["net_pnl"].sum())


def account_balance(
    starting_balance: float | None = None, path: Path = TRADES_PATH
) -> tuple[float, float]:
    """``(balance, peak_end_of_day_balance)`` from all closed tickets.

    The peak trails end-of-day closes only, matching
    :func:`engine.equity_curve_by_day`. Today's running P&L moves the balance
    but cannot raise the peak until the day is closed out - which is what makes
    a mid-day surge unable to buy extra drawdown room.
    """
    if starting_balance is None:
        starting_balance = rules.ACCOUNT_SIZE
    closed = load_closed(path)
    if closed.empty:
        return float(starting_balance), float(starting_balance)

    daily = closed.groupby("session_date")["net_pnl"].sum().sort_index()
    balance = float(starting_balance)
    peak = float(starting_balance)
    days = list(daily.items())
    for i, (_day, pnl) in enumerate(days):
        balance += float(pnl)
        if i < len(days) - 1:
            # Every day but the most recent is a completed session.
            peak = max(peak, balance)
    return balance, peak


def price_ticket(
    ticket: Ticket, exit_price: float, exit_time: pd.Timestamp
) -> dict:
    """P&L for a closed ticket, via the engine's pricing function.

    Deliberately routed through :func:`engine.price_trades` rather than
    reimplemented: the moment the journal computes its own P&L, paper results
    stop being comparable to backtests and nobody notices until the numbers
    disagree.
    """
    spec = spec_for(ticket.instrument)
    frame = pd.DataFrame([
        {
            "entry_time": pd.Timestamp(ticket.entry_time),
            "exit_time": exit_time,
            "direction": ticket.direction,
            "entry_price": float(ticket.entry_price),
            "exit_price": float(exit_price),
            "exit_reason": ticket.exit_reason or "manual",
        }
    ])
    priced = price_trades(frame, spec, COSTS, int(ticket.contracts))
    row = priced.iloc[0]
    return {
        "net_pnl": float(row["net_pnl"]),
        "gross_pnl": float(row["gross_pnl"]),
        "commission": float(row["commission"]),
        "slippage_cost": float(row["slippage_cost"]),
        "duration_seconds": float(row["duration_seconds"]),
    }


def risk_dollars(
    instrument: str, entry_price: float, stop_price: float, contracts: int
) -> float:
    """What this trade loses if the stop fills, including costs.

    Slippage moves both fills against the trade, so the loss is the stop
    distance *plus* two ticks, and commission is on top. Quoting the bare stop
    distance would understate the real risk on every single trade.
    """
    spec = spec_for(instrument)
    slip = COSTS.slippage_ticks * spec.tick_size
    points = abs(float(entry_price) - float(stop_price)) + 2 * slip
    return points * spec.point_value * int(contracts) + COSTS.commission_round_turn(
        int(contracts)
    )


# ---------------------------------------------------------------------------
# Git: the journal is tracked so its history is tamper-evident
# ---------------------------------------------------------------------------


def git_commit_journal(
    path: Path, message: str, repo_root: Path | None = None
) -> tuple[bool, str]:
    """Stage and commit just the journal file.

    Committing after every close puts each outcome into git history at the time
    it happened. Editing a past result later then shows up as a rewrite of a
    committed file rather than a silent change to an untracked one - which is
    the entire reason the journal is tracked.

    Failure here must never lose a trade: the record is already on disk before
    this runs, so any git problem is reported and swallowed.
    """
    import subprocess  # noqa: PLC0415 - only needed on this path

    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    try:
        rel = path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False, f"{path} is outside the repository; not committed"

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        )

    try:
        inside = run("rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0:
            return False, "not a git repository; journal not committed"

        staged = run("add", "--", str(rel))
        if staged.returncode != 0:
            return False, f"git add failed: {staged.stderr.strip()}"

        # Nothing staged means the file was already committed unchanged.
        if run("diff", "--cached", "--quiet", "--", str(rel)).returncode == 0:
            return False, "no journal change to commit"

        committed = run("commit", "-m", message, "--only", "--", str(rel))
        if committed.returncode != 0:
            return False, f"git commit failed: {committed.stderr.strip()}"

        sha = run("rev-parse", "--short", "HEAD").stdout.strip()
        return True, sha
    except FileNotFoundError:
        return False, "git not found on PATH; journal not committed"
    except Exception as exc:  # noqa: BLE001 - never lose the trade over this
        return False, f"git error: {exc}"
