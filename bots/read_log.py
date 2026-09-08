"""Logging a `/read` bracket as a manual trade. "Log long" / "Log short".

Pressing a button opens a modal asking for a one-line thesis, then runs the
bracket through ``pretrade.evaluate`` and appends the ticket to
``journal/trades.jsonl`` - the manual journal.

**These count toward entry 3, and the desk's ticket buttons do not.**
That is not an inconsistency, and the distinction is worth stating because a
future session will otherwise read it as one:

* A desk ticket is an ``orb2`` signal. The mechanism is REJECTED, the thesis
  is the bot's, and pressing Execute is agreeing with a strategy the log has
  already killed. Counting those would let a rejected idea pass the gate.
* A ``/read`` is explicitly a description, not a signal - no bias label, no
  score, nothing to agree with. The operator picks the direction, writes their
  own thesis, and the ticket goes through ``pretrade.evaluate`` into
  ``trades.jsonl``. That is *literally* the mechanism entry 3 pre-registered:
  "a human reading order flow, levels and context in real time".

The guards are re-run at submit time, not reused from the read. Minutes can
pass between the embed being posted and the modal being submitted, and the
16:20 cutoff can fall inside that gap - so the decision that matters is the
one taken when the ticket is actually written.

Only ``DESK_OWNER_ID`` may press, same lock as the desk buttons.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

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
import approvals  # noqa: E402
import chart_read  # noqa: E402

log = logging.getLogger("desk.read_log")

THESIS_PROMPT = "Why this trade? One line."
NOT_OWNER = approvals.NOT_OWNER_MESSAGE


@dataclass
class LogResult:
    logged: bool
    message: str
    ticket: object | None = None


def log_trade(bracket: chart_read.Bracket, symbol: str, thesis: str,
              now: pd.Timestamp, journal_path: Path | None = None,
              early_close_dates=(), roll_dates=(), closed_dates=()) -> LogResult:
    """Evaluate and, on ALLOW, append the ticket. The whole logging path.

    Separated from every Discord type so it can be tested directly; the modal
    is a shell over this.
    """
    journal_path = Path(journal_path) if journal_path else store.TRADES_PATH
    now = rules.to_et(now)
    thesis = (thesis or "").strip()
    if not thesis:
        return LogResult(False, "A thesis is required. Nothing was logged.")
    if bracket.stop is None or bracket.entry is None:
        return LogResult(False, "This read produced no bracket to log.")
    if bracket.contracts < 1:
        return LogResult(False, "This read sized the trade at 0 contracts.")

    request = pretrade.TicketRequest(
        instrument=symbol, direction=bracket.direction,
        entry_price=float(bracket.entry), stop_price=float(bracket.stop),
        contracts=int(bracket.contracts), thesis=thesis,
    )
    state = pretrade.AccountState.from_journal(rules.session_date(now),
                                               journal_path)
    decision = pretrade.evaluate(request, state, now, early_close_dates,
                                 roll_dates, closed_dates)
    if not decision.allowed:
        # Same position pretrade.py takes: a block writes nothing, so there is
        # no ticket to close and no row to explain later.
        return LogResult(False, "**BLOCKED - nothing logged**\n"
                                + "\n".join(f"- {r}" for r in decision.blocks))

    ticket = pretrade.build_ticket(request, decision, now)
    ticket.checks = dict(decision.checks) | {
        "source": "/read",
        "target_price": ("" if bracket.target is None
                         else f"{float(bracket.target):.2f}"),
        "stop_level": bracket.stop_level,
        "target_level": bracket.target_level,
    }
    store.append(ticket, journal_path)
    target = "-" if bracket.target is None else f"{bracket.target:,.2f}"
    return LogResult(
        True,
        f"**Logged `{ticket.ticket_id}`** - counts toward entry 3.\n"
        f"{bracket.direction.upper()} {ticket.contracts} {symbol} @ "
        f"{ticket.entry_price:,.2f}  stop {ticket.stop_price:,.2f}  "
        f"target {target}\n"
        f"risk ${ticket.risk_dollars:,.2f}\n"
        f"Close it with: `python journal/close.py --ticket "
        f"{ticket.ticket_id} --exit <price> --reason <reason>`",
        ticket,
    )


class ReadLogView:
    """Core state for the two buttons. Discord-free so it can be tested."""

    def __init__(self, read: chart_read.ChartRead, owner_id: int | None,
                 journal_path: Path | None = None,
                 early_close_dates=(), roll_dates=(), closed_dates=()) -> None:
        self.read = read
        self.owner_id = owner_id
        self.journal_path = Path(journal_path) if journal_path else store.TRADES_PATH
        self.calendar = dict(early_close_dates=early_close_dates,
                             roll_dates=roll_dates, closed_dates=closed_dates)
        self.logged: dict[str, str] = {}

    def bracket(self, direction: str) -> chart_read.Bracket:
        return self.read.long if direction == "long" else self.read.short

    def authorised(self, user_id: int | None) -> bool:
        return self.owner_id is not None and user_id == self.owner_id

    def button_specs(self) -> list[dict]:
        """A direction the rules already block gets a disabled button.

        The bracket is recomputed at submit anyway, so this is presentation:
        offering a button that can only refuse wastes a click and reads as a
        malfunction.
        """
        out = []
        for direction, style in (("long", "success"), ("short", "danger")):
            bracket = self.bracket(direction)
            blocked = bool(bracket.blocks) or bracket.contracts < 1
            out.append({
                "label": f"Log {direction}",
                "style": style,
                "disabled": blocked or direction in self.logged,
                "custom_id": f"read:{self.read.symbol}:{direction}",
            })
        return out

    def submit(self, user_id: int | None, direction: str, thesis: str,
               now: pd.Timestamp | None = None) -> LogResult:
        if not self.authorised(user_id):
            return LogResult(False, NOT_OWNER)
        if direction in self.logged:
            return LogResult(False, f"Already logged `{self.logged[direction]}`.")
        result = log_trade(
            self.bracket(direction), self.read.symbol, thesis,
            now or pd.Timestamp.now(tz=rules.ET), self.journal_path,
            **self.calendar,
        )
        if result.logged:
            self.logged[direction] = result.ticket.ticket_id
        return result

    # -- the discord.py objects -------------------------------------------

    def as_discord_view(self, timeout: float = 900.0):
        """Buttons that open a thesis modal. Needs a running event loop."""
        import discord  # noqa: PLC0415

        core = self

        class _Modal(discord.ui.Modal):
            def __init__(self, direction: str) -> None:
                super().__init__(title=f"Log {direction} - "
                                       f"{core.read.symbol}"[:45])
                self.direction = direction
                self.thesis = discord.ui.TextInput(
                    label=THESIS_PROMPT[:45], required=True, max_length=200,
                    placeholder="failed breakdown, reclaimed VWAP",
                )
                self.add_item(self.thesis)

            async def on_submit(self, interaction) -> None:
                result = core.submit(getattr(interaction.user, "id", None),
                                     self.direction, str(self.thesis))
                await interaction.response.send_message(result.message,
                                                        ephemeral=True)

        class _View(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=timeout)
                styles = {"success": discord.ButtonStyle.success,
                          "danger": discord.ButtonStyle.danger}
                for spec in core.button_specs():
                    self.add_item(discord.ui.Button(
                        label=spec["label"], style=styles[spec["style"]],
                        disabled=spec["disabled"], custom_id=spec["custom_id"]))
                self.children[0].callback = self._long
                self.children[1].callback = self._short

            async def _open(self, interaction, direction: str) -> None:
                if not core.authorised(getattr(interaction.user, "id", None)):
                    await interaction.response.send_message(NOT_OWNER,
                                                            ephemeral=True)
                    return
                await interaction.response.send_modal(_Modal(direction))

            async def _long(self, interaction) -> None:
                await self._open(interaction, "long")

            async def _short(self, interaction) -> None:
                await self._open(interaction, "short")

        return _View()
