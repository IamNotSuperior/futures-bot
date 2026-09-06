"""Crash-safe desk state.

The desk holds things a restart must not forget: which shadow positions are
open, when the last heartbeat went out, whether today's pre-market check and
daily summary have already run. Losing that on a crash produces a bot that
double-posts a summary, or worse, forgets a position it opened.

Durability
----------
Writes are atomic: the payload goes to a sibling ``.tmp`` file, is flushed and
``fsync``-ed, then :func:`os.replace` moves it over the real path. ``os.replace``
is atomic on Windows and POSIX alike, so a crash mid-write leaves either the
old file or the new one, never a half-written one. A plain ``write_text`` would
truncate first and leave nothing recoverable if the process died in between.

``encoding="utf-8"`` is passed explicitly on every read and write. On Windows
``Path.write_text`` defaults to cp1252 and raises on the first non-ASCII
character *after* truncating the file - which is the same failure this module
exists to prevent.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import rules  # noqa: E402

log = logging.getLogger("desk.state")

STATE_PATH = PROJECT_ROOT / "journal" / "desk_state.json"

#: Bumped when the on-disk shape changes incompatibly. A state file from an
#: older version is discarded rather than guessed at.
SCHEMA_VERSION = 1


@dataclass
class OpenPosition:
    """A shadow position the desk believes is open."""

    ticket_id: str
    strategy: str
    instrument: str
    direction: str
    contracts: int
    entry_price: float
    stop_price: float
    target_price: float | None
    entry_time: str
    mode: str
    session_date: str

    @property
    def entry_ts(self) -> pd.Timestamp:
        return rules.to_et(self.entry_time)

    def held_seconds(self, now) -> float:
        return rules.hold_seconds(self.entry_ts, rules.to_et(now))


@dataclass
class DeskState:
    """Everything a restart needs to pick up where it left off."""

    schema_version: int = SCHEMA_VERSION
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    last_heartbeat: str | None = None
    last_bar_at: str | None = None
    premarket_done_on: str | None = None
    summary_done_on: str | None = None
    bars_seen: int = 0
    tickets_allowed: int = 0
    tickets_blocked: int = 0
    started_at: str | None = None

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "DeskState":
        """Read state, or return a fresh one.

        A missing, unreadable, or version-mismatched file yields a fresh state
        rather than an exception: the desk must be able to start. What it must
        *not* do is start believing it has no open positions when it does, and
        that is what the startup reconciliation in ``desk.py`` is for - it
        checks the journal, not this file.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("desk state unreadable (%s); starting fresh", exc)
            return cls()
        if raw.get("schema_version") != SCHEMA_VERSION:
            log.warning("desk state schema %r != %d; starting fresh",
                        raw.get("schema_version"), SCHEMA_VERSION)
            return cls()
        positions = {
            tid: OpenPosition(**row)
            for tid, row in (raw.get("open_positions") or {}).items()
        }
        raw["open_positions"] = positions
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path = STATE_PATH) -> None:
        """Atomically persist. See the module docstring for why."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["open_positions"] = {
            tid: asdict(p) for tid, p in self.open_positions.items()
        }
        # ensure_ascii=False keeps the file readable rather than \u-escaped.
        # It also means the explicit utf-8 below is load-bearing rather than
        # decorative: with escaping on, the payload is ASCII and any encoding
        # would happen to work, hiding the cp1252 trap until a real non-ASCII
        # thesis arrived.
        text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)

        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # -- mutation ----------------------------------------------------------

    def add_position(self, position: OpenPosition) -> None:
        self.open_positions[position.ticket_id] = position

    def drop_position(self, ticket_id: str) -> OpenPosition | None:
        return self.open_positions.pop(ticket_id, None)

    def net_position(self, instrument: str | None = None) -> int:
        """Signed contracts held, optionally for one instrument.

        Rule 4 caps net position per instrument *and* in aggregate, so both
        readings are needed and the caller says which one it wants.
        """
        total = 0
        for p in self.open_positions.values():
            if instrument and p.instrument != instrument:
                continue
            total += p.contracts * (1 if p.direction == "long" else -1)
        return total

    def positions_for(self, strategy: str) -> list[OpenPosition]:
        return [p for p in self.open_positions.values() if p.strategy == strategy]

    # -- once-a-day guards -------------------------------------------------

    def needs_premarket(self, day: date_type) -> bool:
        return self.premarket_done_on != str(day)

    def needs_summary(self, day: date_type) -> bool:
        return self.summary_done_on != str(day)

    def mark_premarket(self, day: date_type) -> None:
        self.premarket_done_on = str(day)

    def mark_summary(self, day: date_type) -> None:
        self.summary_done_on = str(day)

    def due_for_heartbeat(self, now, interval_minutes: int) -> bool:
        if self.last_heartbeat is None:
            return True
        elapsed = rules.to_et(now) - rules.to_et(self.last_heartbeat)
        return elapsed >= pd.Timedelta(minutes=interval_minutes)
