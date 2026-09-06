"""Research bot: read-only access to the backtests, the log and the registry.

    venv\\Scripts\\python.exe bots\\research.py

Reads ``DISCORD_TOKEN`` from ``.env``. Slash commands are synced to every guild
the bot is in on startup, which is instant; a global sync can take up to an
hour to appear and is only used as a fallback when the bot is in no guild yet.

Three properties this shell is built for:

**It never blocks the gateway.** Every command answers immediately with a
"running..." message, does its work on a worker thread via ``asyncio.to_thread``
so the event loop keeps beating, then edits that message with the result. A
backtest takes tens of seconds; done inline, discord.py would miss heartbeats
and the connection would drop mid-answer.

**It never dies.** Every command body is wrapped, and anything unexpected is
posted as a traceback rather than raised into the event loop.

**It changes nothing.** This bot reads. There is no command here that promotes a
strategy, writes to the journal, or places an order - promotion goes through
``Registry.promote``, which is deliberately not exposed over chat.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import traceback
from pathlib import Path

import discord
from discord import app_commands
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runners  # noqa: E402
from runners import WorkError  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

TOKEN = os.getenv("DISCORD_TOKEN")
COLOUR_OK = discord.Colour.blurple()
COLOUR_BAD = discord.Colour.red()

log = logging.getLogger("research-bot")


def _embed(title: str, fields: dict, colour=COLOUR_OK) -> discord.Embed:
    embed = discord.Embed(title=title, colour=colour)
    for key, value in fields.items():
        embed.add_field(name=key, value=str(value)[:1024], inline=True)
    return embed


def _traceback_block(exc: BaseException) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Discord caps a message at 2,000 characters; keep the tail, which is where
    # the actual error is.
    if len(text) > 1850:
        text = "...\n" + text[-1840:]
    return f"```py\n{text}\n```"


class ResearchBot(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        self.tree.on_error = self.on_tree_error

    async def on_ready(self) -> None:
        guilds = list(self.guilds)
        if guilds:
            for guild in guilds:
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                log.info("synced %d commands to %s (%s)",
                         len(synced), guild.name, guild.id)
                print(f"  synced {len(synced)} commands to "
                      f"{guild.name} ({guild.id})", flush=True)
        else:
            synced = await self.tree.sync()
            print(f"  no guilds yet - synced {len(synced)} commands globally; "
                  f"global sync can take up to an hour to appear", flush=True)
        print(f"Logged in as {self.user} (id {self.user.id})", flush=True)
        print("Ready.", flush=True)

    async def on_tree_error(self, interaction: discord.Interaction,
                            error: app_commands.AppCommandError) -> None:
        """Last line of defence: report, never propagate."""
        await report_error(interaction, getattr(error, "original", error))


bot = ResearchBot()


async def report_error(interaction: discord.Interaction, exc: BaseException) -> None:
    """Post the failure to the channel and keep the bot alive."""
    friendly = isinstance(exc, WorkError)
    body = str(exc) if friendly else _traceback_block(exc)
    title = "Cannot do that" if friendly else "Command failed"
    embed = discord.Embed(title=title, description=body[:4000], colour=COLOUR_BAD)
    try:
        if interaction.response.is_done():
            await interaction.edit_original_response(embed=embed, attachments=[])
        else:
            await interaction.response.send_message(embed=embed)
    except Exception:  # the channel itself is unreachable; log and move on
        log.exception("could not report error to the channel")
    if not friendly:
        log.error("command failed", exc_info=exc)


async def ack(interaction: discord.Interaction, what: str) -> None:
    """Answer inside Discord's 3-second window, before any work starts."""
    await interaction.response.send_message(f"running {what} ...")


async def strategy_autocomplete(interaction: discord.Interaction, current: str):
    try:
        names = runners.known_strategies()
    except Exception:
        names = []
    return [
        app_commands.Choice(name=n, value=n)
        for n in names if current.lower() in n.lower()
    ][:25]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@bot.tree.command(name="backtest",
                  description="Backtest a strategy over a date range")
@app_commands.describe(strategy="registry name",
                       start="YYYY-MM-DD", end="YYYY-MM-DD")
@app_commands.autocomplete(strategy=strategy_autocomplete)
async def backtest(interaction: discord.Interaction, strategy: str,
                   start: str, end: str) -> None:
    await ack(interaction, f"backtest `{strategy}` {start} to {end}")
    try:
        fields, png = await asyncio.to_thread(
            runners.run_backtest, strategy, start, end
        )
        embed = _embed(f"{strategy}  backtest", fields)
        embed.set_image(url="attachment://equity.png")
        await interaction.edit_original_response(
            content=None, embed=embed,
            attachments=[discord.File(io.BytesIO(png), filename="equity.png")],
        )
    except Exception as exc:
        await report_error(interaction, exc)


@bot.tree.command(name="walkforward",
                  description="Per-fold walk-forward table for a strategy")
@app_commands.describe(strategy="registry name")
@app_commands.autocomplete(strategy=strategy_autocomplete)
async def walkforward(interaction: discord.Interaction, strategy: str) -> None:
    await ack(interaction, f"walk-forward `{strategy}`")
    try:
        text = await asyncio.to_thread(runners.run_walkforward, strategy)
        await interaction.edit_original_response(content=text[:1990])
    except Exception as exc:
        await report_error(interaction, exc)


@bot.tree.command(name="evalsim",
                  description="Evaluation pass probability and expected attempts")
@app_commands.describe(strategy="registry name")
@app_commands.autocomplete(strategy=strategy_autocomplete)
async def evalsim(interaction: discord.Interaction, strategy: str) -> None:
    await ack(interaction, f"eval_sim `{strategy}`")
    try:
        fields = await asyncio.to_thread(runners.run_evalsim, strategy)
        await interaction.edit_original_response(
            content=None, embed=_embed(f"{strategy}  evaluation simulator", fields)
        )
    except Exception as exc:
        await report_error(interaction, exc)


@bot.tree.command(name="hypotheses",
                  description="Hypothesis log: all entries, or one in detail")
@app_commands.describe(n="entry number; omit for the list")
async def hypotheses(interaction: discord.Interaction, n: int | None = None) -> None:
    await ack(interaction, "hypotheses")
    try:
        text = await asyncio.to_thread(runners.hypothesis_text, n)
        await interaction.edit_original_response(content=text[:1990])
    except Exception as exc:
        await report_error(interaction, exc)


@bot.tree.command(name="status", description="The strategy registry")
async def status(interaction: discord.Interaction) -> None:
    await ack(interaction, "status")
    try:
        text = await asyncio.to_thread(runners.status_text)
        await interaction.edit_original_response(content=text[:1990])
    except Exception as exc:
        await report_error(interaction, exc)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
    if not TOKEN:
        print("DISCORD_TOKEN is not set. Put it in .env next to "
              "DATABENTO_API_KEY.", file=sys.stderr)
        return 1
    print("Starting research bot ...", flush=True)
    try:
        bot.run(TOKEN, log_handler=None)
    except discord.LoginFailure:
        print("Discord rejected the token. Regenerate it in the developer "
              "portal and update .env.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
