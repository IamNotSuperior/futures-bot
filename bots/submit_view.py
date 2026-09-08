"""The Discord surface for /submit: the modal, and the approval buttons.

Kept apart from ``bots/submissions.py`` for the same reason ``runners.py`` is
kept apart from ``research.py`` - the pipeline is testable without a gateway,
and this file is a shell over it.

The approval gate
-----------------
:class:`ApprovalView` is the only thing that calls ``submissions.approve``,
and it refuses anyone but ``DESK_OWNER_ID``. That check is duplicated in
:meth:`ApprovalView.decide` rather than left to the button being disabled,
because a disabled button is a rendering decision and approval is a gate: it
moves a strategy to ``testing`` and makes it eligible to be walked forward.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "bots"))

import approvals  # noqa: E402
import generate as generate_mod  # noqa: E402
import submissions  # noqa: E402

log = logging.getLogger("bot.submit_view")

NOT_OWNER = approvals.NOT_OWNER_MESSAGE


@dataclass
class Decision:
    acted: bool
    message: str
    status: str = ""


class ApprovalView:
    """Approve / Reject for one submission. Discord-free core."""

    def __init__(self, run: submissions.SubmissionRun,
                 owner_id: int | None) -> None:
        self.run = run
        self.owner_id = owner_id

    def authorised(self, user_id: int | None) -> bool:
        return self.owner_id is not None and user_id == self.owner_id

    def button_specs(self) -> list[dict]:
        done = self.run.status in (submissions.STATUS_APPROVED,
                                   submissions.STATUS_REJECTED)
        return [
            {"label": "Approve", "style": "success", "disabled": done,
             "custom_id": f"submit:{self.run.registry_name}:approve"},
            {"label": "Reject", "style": "danger", "disabled": done,
             "custom_id": f"submit:{self.run.registry_name}:reject"},
        ]

    def decide(self, user_id: int | None, choice: str) -> Decision:
        """The gate. Owner-only, once, and only from awaiting_approval."""
        if not self.authorised(user_id):
            return Decision(False, NOT_OWNER)
        if self.run.status in (submissions.STATUS_APPROVED,
                               submissions.STATUS_REJECTED):
            return Decision(False, f"Already {self.run.status}.")
        if self.run.status != submissions.STATUS_AWAITING:
            return Decision(
                False,
                f"`{self.run.registry_name}` is {self.run.status}; only a "
                f"submission whose tests passed can be approved."
            )
        try:
            if choice == "approve":
                commit = submissions.approve(self.run)
                return Decision(
                    True,
                    f"**Approved.** `{self.run.registry_name}` is now at "
                    f"`testing`, frozen at commit `{commit}`.\n"
                    f"Run `/walkforward {self.run.registry_name}` when ready - "
                    f"it computes rather than reading a cached table.",
                    submissions.STATUS_APPROVED,
                )
            submissions.reject(self.run)
            return Decision(
                True,
                f"**Rejected.** Entry {self.run.entry_number} stays in the log "
                f"with its verdict, and rejection is terminal.",
                submissions.STATUS_REJECTED,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            log.exception("approval failed")
            return Decision(False, f"Could not {choice}: {exc}")

    def as_discord_view(self, timeout: float = 86_400.0):
        import discord  # noqa: PLC0415

        core = self

        class _View(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=timeout)
                styles = {"success": discord.ButtonStyle.success,
                          "danger": discord.ButtonStyle.danger}
                for spec in core.button_specs():
                    self.add_item(discord.ui.Button(
                        label=spec["label"], style=styles[spec["style"]],
                        disabled=spec["disabled"], custom_id=spec["custom_id"]))
                self.children[0].callback = self._approve
                self.children[1].callback = self._reject

            async def _handle(self, interaction, choice: str) -> None:
                result = core.decide(getattr(interaction.user, "id", None),
                                     choice)
                if result.acted:
                    for child in self.children:
                        child.disabled = True
                    try:
                        await interaction.message.edit(view=self)
                    except Exception:  # noqa: BLE001
                        pass
                await interaction.response.send_message(result.message,
                                                        ephemeral=not result.acted)

            async def _approve(self, interaction) -> None:
                await self._handle(interaction, "approve")

            async def _reject(self, interaction) -> None:
                await self._handle(interaction, "reject")

        return _View()


def build_modal(name: str, description: str, pine_source: str, on_submit):
    """The three required fields. Returns a ``discord.ui.Modal`` subclass."""
    import discord  # noqa: PLC0415

    class _Modal(discord.ui.Modal):
        def __init__(self) -> None:
            super().__init__(title=f"Submit: {name}"[:45])
            self.mechanism = discord.ui.TextInput(
                label="Mechanism - why should this edge exist?",
                style=discord.TextStyle.paragraph, required=True,
                max_length=1200,
            )
            self.counterparty = discord.ui.TextInput(
                label="Who is on the other side of the trade?",
                style=discord.TextStyle.paragraph, required=True,
                max_length=1000,
                placeholder="An idea that cannot name its counterparty is a "
                            "pattern, not a hypothesis.",
            )
            self.kill = discord.ui.TextInput(
                label="Kill criteria - what would falsify it?",
                style=discord.TextStyle.paragraph, required=True,
                max_length=1000,
            )
            for item in (self.mechanism, self.counterparty, self.kill):
                self.add_item(item)

        async def on_submit(self, interaction) -> None:
            submission = generate_mod.Submission(
                name=name,
                mechanism=str(self.mechanism),
                counterparty=str(self.counterparty),
                kill_criteria=str(self.kill),
                description=description,
                pine_source=pine_source,
            )
            await on_submit(interaction, submission)

    return _Modal()
