"""Ticket approval buttons: Execute / Don't trade.

Every ticket embed the desk posts carries a :class:`TicketView` with two
buttons. What each does depends on the strategy's registry status, and the
rule that matters most is the one about what *cannot* happen:

**There is no path from a rejected strategy to an order.** For a shadow ticket
the Execute button is rendered disabled with the label
:data:`EXECUTE_UNAVAILABLE_LABEL`, and - because a disabled button is a UI
promise rather than a guard - :meth:`TicketView.decide` refuses an approve on
any status other than ``paper`` or ``live`` even if the callback is reached by
some other route. ``tests/test_desk_discord.py`` asserts both layers.

Who may press
-------------
Only the Discord user whose id is ``DESK_OWNER_ID`` in ``.env``. Anyone else
receives an ephemeral refusal and nothing is recorded. The check is by id,
not name: names are changeable and ids are not.

Expiry
------
The view times out when the strategy would cancel its unfilled entry order -
``ORB2Params.entry_cancel_time``, 10:30 ET, for orb2 - at which point both
buttons are disabled and the embed is retitled EXPIRED. A ticket that nobody
decided on before then is recorded as ``expired``.

Views with a timeout are not persistent across process restarts: if the desk
is restarted, buttons on already-posted tickets stop responding. With expiry
inside the same session that is an acceptable loss, and it is stated here so
it is not mistaken for a bug.

Testability
-----------
The discord.py callbacks are thin wrappers around :meth:`TicketView.decide`,
which takes plain values and returns a plain result. Tests exercise ``decide``
directly and inspect the buttons offline; nothing here needs a gateway.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import time as time_type
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import decisions as decisions_mod  # noqa: E402
import rules  # noqa: E402

log = logging.getLogger("desk.approvals")

EXECUTE_LABEL = "Execute"
EXECUTE_UNAVAILABLE_LABEL = "Execute — unavailable (strategy rejected)"
DECLINE_LABEL = "Don't trade"

#: Statuses for which Execute is live. Everything else is disabled *and*
#: refused - see the module docstring.
EXECUTABLE_STATUSES = frozenset({"paper", "live"})

NOT_OWNER_MESSAGE = ("These buttons belong to the desk operator. Your press was "
                     "not recorded.")


def owner_id_from_env() -> int | None:
    raw = (os.environ.get("DESK_OWNER_ID") or "").strip()
    return int(raw) if raw.isdigit() else None


def seconds_until(now: pd.Timestamp, cancel_time: time_type) -> float:
    """Seconds from ``now`` to today's ``cancel_time`` in ET, floored at 1.

    A ticket posted after the cancel time (which the entry guards should make
    impossible, but see rules.py's lack of a session-open guard) gets one
    second rather than a negative timeout discord.py would reject.
    """
    now = rules.to_et(now)
    deadline = pd.Timestamp.combine(now.date(), cancel_time).tz_localize(rules.ET)
    return max(1.0, (deadline - now).total_seconds())


@dataclass
class DecideResult:
    recorded: bool
    decision: str | None
    message: str
    embed_title: str | None = None


class TicketView:
    """The two-button view for one ticket.

    Constructed as a plain object so tests can build it without discord.py's
    event loop; :meth:`as_discord_view` produces the real ``discord.ui.View``
    when there is a gateway to attach it to.
    """

    def __init__(
        self,
        ticket_id: str,
        strategy: str,
        strategy_status: str,
        mode: str,
        instrument: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        owner_id: int | None,
        expires_at: pd.Timestamp,
        on_execute: Callable[[], Awaitable[str]] | None = None,
        journal_path: Path = decisions_mod.DECISIONS_PATH,
    ) -> None:
        self.ticket_id = ticket_id
        self.strategy = strategy
        self.strategy_status = strategy_status
        self.mode = mode
        self.instrument = instrument
        self.direction = direction
        self.entry_price = float(entry_price)
        self.stop_price = float(stop_price)
        self.owner_id = owner_id
        self.expires_at = rules.to_et(expires_at)
        self.on_execute = on_execute
        self.journal_path = Path(journal_path)
        self.decided: str | None = None
        self.message = None  # set by the poster once the embed is sent

    # -- what the buttons look like ---------------------------------------

    @property
    def executable(self) -> bool:
        return self.strategy_status in EXECUTABLE_STATUSES

    @property
    def execute_label(self) -> str:
        return EXECUTE_LABEL if self.executable else EXECUTE_UNAVAILABLE_LABEL

    def button_specs(self) -> list[dict]:
        """Label / style / disabled for each button, in order."""
        return [
            {"label": self.execute_label, "style": "success",
             "disabled": (not self.executable) or self.decided is not None,
             "custom_id": f"desk:{self.ticket_id}:execute"},
            {"label": DECLINE_LABEL, "style": "danger",
             "disabled": self.decided is not None,
             "custom_id": f"desk:{self.ticket_id}:decline"},
        ]

    # -- the decision itself ----------------------------------------------

    def authorised(self, user_id: int | None) -> bool:
        return self.owner_id is not None and user_id == self.owner_id

    async def decide(self, user_id: int | None, user_name: str, choice: str,
                     now: pd.Timestamp | None = None) -> DecideResult:
        """Record a press. Pure apart from the journal append and on_execute.

        Order of checks: owner, already-decided, expiry, then the status guard
        on approve. The status guard is last so that a rejected strategy's
        decline still works - declining is always allowed.
        """
        now = rules.to_et(now or pd.Timestamp.now(tz=rules.ET))
        if not self.authorised(user_id):
            return DecideResult(False, None, NOT_OWNER_MESSAGE)
        if self.decided is not None:
            return DecideResult(False, None,
                                f"Already {self.decided} - nothing changed.")
        if now >= self.expires_at:
            return DecideResult(False, None,
                                f"This ticket expired at {self.expires_at:%H:%M} ET.")

        if choice == decisions_mod.APPROVE and not self.executable:
            # Defence in depth behind the disabled button. This is the line
            # the "no path to an order for a rejected strategy" test aims at.
            return DecideResult(
                False, None,
                f"{self.strategy} is at '{self.strategy_status}' status; only a "
                f"strategy at paper or live status can be executed. No order "
                f"path exists for it.",
            )
        if choice not in (decisions_mod.APPROVE, decisions_mod.DECLINE):
            return DecideResult(False, None, f"unknown choice {choice!r}")

        self._record(choice, user_id, user_name, now)
        if choice == decisions_mod.APPROVE:
            note = ""
            if self.on_execute is not None:
                try:
                    note = await self.on_execute()
                except Exception as exc:  # noqa: BLE001 - surface, never crash
                    log.error("execute callback failed for %s: %s", self.ticket_id, exc)
                    note = f"routing failed: {exc}"
            return DecideResult(True, choice, f"Approved. {note}".strip(),
                                embed_title="APPROVED")
        return DecideResult(True, choice, "Declined and recorded.",
                            embed_title="DECLINED")

    def expire(self, now: pd.Timestamp | None = None) -> DecideResult:
        """Called on timeout. Records ``expired`` only if nobody pressed."""
        if self.decided is not None:
            return DecideResult(False, self.decided, "already decided")
        now = rules.to_et(now or pd.Timestamp.now(tz=rules.ET))
        self._record(decisions_mod.EXPIRED, None, "(timeout)", now)
        return DecideResult(True, decisions_mod.EXPIRED, "expired",
                            embed_title="EXPIRED")

    def _record(self, choice: str, user_id: int | None, user_name: str,
                now: pd.Timestamp) -> None:
        self.decided = choice
        decisions_mod.append(decisions_mod.Decision(
            ticket_id=self.ticket_id,
            strategy=self.strategy,
            strategy_status=self.strategy_status,
            mode=self.mode,
            decision=choice,
            user_id=user_id,
            user_name=user_name,
            decided_at=now.isoformat(),
            instrument=self.instrument,
            direction=self.direction,
            entry_price=self.entry_price,
            stop_price=self.stop_price,
        ), self.journal_path)

    # -- the discord.py object ---------------------------------------------

    def as_discord_view(self, now: pd.Timestamp | None = None):
        """Build the real ``discord.ui.View``. Needs a running event loop."""
        import discord  # noqa: PLC0415

        core = self
        timeout = max(1.0, (self.expires_at - rules.to_et(
            now or pd.Timestamp.now(tz=rules.ET))).total_seconds())

        class _View(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=timeout)
                styles = {"success": discord.ButtonStyle.success,
                          "danger": discord.ButtonStyle.danger}
                for spec in core.button_specs():
                    self.add_item(discord.ui.Button(
                        label=spec["label"], style=styles[spec["style"]],
                        disabled=spec["disabled"], custom_id=spec["custom_id"]))
                self.children[0].callback = self._execute
                self.children[1].callback = self._decline

            async def _handle(self, interaction, choice: str) -> None:
                user = interaction.user
                result = await core.decide(getattr(user, "id", None),
                                           str(user), choice)
                if not result.recorded:
                    await interaction.response.send_message(result.message,
                                                            ephemeral=True)
                    return
                await _retitle(interaction.message, result.embed_title, core, self)
                await interaction.response.send_message(result.message,
                                                        ephemeral=True)

            async def _execute(self, interaction) -> None:
                await self._handle(interaction, decisions_mod.APPROVE)

            async def _decline(self, interaction) -> None:
                await self._handle(interaction, decisions_mod.DECLINE)

            async def on_timeout(self) -> None:
                result = core.expire()
                if result.recorded and core.message is not None:
                    await _retitle(core.message, result.embed_title, core, self)

        return _View()


async def _retitle(message, title: str | None, core: TicketView, view) -> None:
    """Prefix the embed title and disable both buttons."""
    if message is None or not title:
        return
    try:
        embeds = list(message.embeds)
        if embeds:
            embed = embeds[0]
            base = (embed.title or "").split(" — ", 1)[-1]
            embed.title = f"{title} — {base}"[:256]
        for child in view.children:
            child.disabled = True
        await message.edit(embeds=embeds, view=view)
    except Exception as exc:  # noqa: BLE001 - an edit failing must not lose the record
        log.error("could not retitle ticket %s: %s", core.ticket_id, exc)
