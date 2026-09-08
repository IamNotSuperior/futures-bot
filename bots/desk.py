"""The desk bot. Shadow mode: it decides, it posts, it places nothing.

    venv\\Scripts\\python.exe bots\\desk.py                        # live webhook
    venv\\Scripts\\python.exe bots\\desk.py --replay --start 2026-08-24 \\
                                            --end 2026-08-28 --speed 60

What it is
----------
A long-running process that consumes 1-minute MES bars, runs every strategy the
registry says it should, puts each resulting signal through the same guards a
manual trade goes through, and posts the outcome to Discord. Allowed signals
are simulated by :class:`broker.PaperAdapter` and written to
``journal/shadow_trades.jsonl``. Blocked ones are posted with their reasons and
written nowhere.

**Nothing here can place an order.** ``PaperAdapter`` is the only adapter, and
``broker.require_live_eligible`` refuses any strategy not at ``live`` status
regardless of the per-account flag.

Why it exists before the gate
-----------------------------
HANDOFF.md §6 gates the desk bot on a strategy reaching ``paper`` status, and
none has. This is a deliberate exception, taken to test the plumbing rather
than to trade: the strategy it runs (``orb2``) is *rejected*, every ticket says
so, and the shadow journal is a separate file from the one entry 3's 60-trade
gate counts.

**The existence of this file is not evidence that the gate was met.** It was
not. See §6 and the note this build added to HANDOFF.md §2.

Clock discipline
----------------
Every guard takes an injected ``now``. In live mode that is the wall clock; in
replay it is the **bar's** timestamp. A replay judged against the wall clock
would block every bar of an August session as after-hours, and the fact that
replay and live share one code path is the only reason a replay tests anything.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date as date_type, time as time_type
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("data", "journal", "backtests", "strategies", "bots"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import rules  # noqa: E402
import store  # noqa: E402
from registry import Registry  # noqa: E402

import broker  # noqa: E402
import desk_state  # noqa: E402
import feed as feed_mod  # noqa: E402
import tickets as tickets_mod  # noqa: E402
import approvals  # noqa: E402

log = logging.getLogger("desk")

MES_PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"

#: Ops cadence. All of these are session-relative, not wall-clock-relative, so
#: a replay produces the same ops posts a live day would.
HEARTBEAT_MINUTES = 15
PREMARKET_TIME = time_type(9, 25)
SUMMARY_TIME = time_type(16, 35)

#: The heartbeat runs **during the session only**, from the pre-market check to
#: the daily summary. The desk consumes overnight bars too - the feed is 23
#: hours a day - so an unbounded heartbeat posts around the clock: a five-day
#: replay produced 436 of them, roughly 87 a day, which is noise rather than
#: monitoring. Outside this window silence is the correct signal.
SESSION_START = PREMARKET_TIME
SESSION_END = SUMMARY_TIME


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

DEFAULT_CHANNEL_NAME = "general"


def channel_name_from_env() -> str:
    return (os.environ.get("DESK_CHANNEL_NAME") or "").strip() or DEFAULT_CHANNEL_NAME


class Poster:
    """Every desk post, to one named channel, buffered while the gateway is down.

    Three behaviours, each deliberate:

    * **One channel, resolved by name.** ``DESK_CHANNEL_NAME`` is looked up in
      the bot's guild on first ``on_ready``. If it cannot be found, or the bot
      cannot send to it, :meth:`resolve_channel` returns the reason and the
      client treats that as fatal. There is no silent fallback to stdout: a
      desk that is quietly printing to a console nobody is watching is worse
      than one that refuses to start.
    * **Buffering.** A post made while the gateway is disconnected - or before
      the channel is resolved - is queued and flushed, in order, on the next
      ``on_ready``/``on_resumed``. The feed does not stop for a Discord outage
      and neither does the record of what it did.
    * **Console always.** Every post is also printed, so the local window is a
      complete transcript regardless of Discord's state. In replay and with
      ``--no-discord`` the console is the *only* destination, and that is
      explicit rather than a fallback.
    """

    def __init__(self, client=None, channel_name: str = DEFAULT_CHANNEL_NAME,
                 echo: bool = True) -> None:
        self.client = client
        self.channel_name = channel_name
        self.echo = echo
        self.channel = None
        self.ready = False
        self.posted: list[tuple[str, str]] = []
        self.buffer: list[tuple] = []
        self.dropped = 0

    async def send(self, kind: str, title: str, body: str,
                   colour: int = 0x5A6672, view=None):
        """Post, or queue. Returns the Discord message when one was sent."""
        self.posted.append((kind, f"{title}\n{body}"))
        if self.echo:
            print(f"\n[{kind}] {title}\n{body}", flush=True)
        if self.client is None:
            return None
        if not self.ready or self.channel is None:
            self.buffer.append((kind, title, body, colour, view))
            return None
        return await self._post(kind, title, body, colour, view)

    async def _post(self, kind: str, title: str, body: str, colour: int, view):
        try:
            import discord  # noqa: PLC0415

            embed = discord.Embed(title=title[:256], description=body[:4000],
                                  colour=colour)
            dview = view.as_discord_view() if view is not None else None
            message = await self.channel.send(embed=embed, view=dview)
            if view is not None:
                view.message = message
            return message
        except Exception as exc:  # noqa: BLE001 - a post must never kill the desk
            log.error("discord post failed (%s: %s): %s", kind, title, exc)
            self.dropped += 1
            return None

    async def flush(self) -> int:
        """Send everything queued, in order. Returns how many went out."""
        sent = 0
        while self.buffer and self.ready and self.channel is not None:
            kind, title, body, colour, view = self.buffer.pop(0)
            if await self._post(kind, title, body, colour, view) is not None:
                sent += 1
        return sent

    async def resolve_channel(self) -> str | None:
        """Find ``channel_name`` in the bot's guilds. Returns an error, or None."""
        try:
            import discord  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            return f"discord.py unavailable: {exc}"
        guilds = list(getattr(self.client, "guilds", []) or [])
        if not guilds:
            return ("the bot is not in any guild. Invite it to your server "
                    "first (OAuth2 URL with the bot scope).")
        matches = []
        for guild in guilds:
            channel = discord.utils.get(guild.text_channels, name=self.channel_name)
            if channel is not None:
                matches.append((guild, channel))
        if not matches:
            names = sorted({c.name for g in guilds for c in g.text_channels})
            return (f"no text channel named {self.channel_name!r} in "
                    f"{', '.join(g.name for g in guilds)}. Set DESK_CHANNEL_NAME "
                    f"in .env to one of: {', '.join(names) or '(none visible)'}")
        guild, channel = matches[0]
        if len(matches) > 1:
            log.warning("channel %r exists in %d guilds; using %s",
                        self.channel_name, len(matches), guild.name)
        perms = channel.permissions_for(guild.me)
        if not (perms.send_messages and perms.embed_links):
            return (f"the bot cannot post to #{channel.name} in {guild.name}: "
                    f"it needs Send Messages and Embed Links there.")
        self.channel = channel
        log.info("posting to #%s in %s", channel.name, guild.name)
        return None


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

@dataclass
class StrategyRunner:
    """One registry strategy, driven bar by bar.

    ``generate_signals`` takes a frame, not a bar, so the runner accumulates
    the session and regenerates on each new bar, reading only the newest row.
    That is exactly equivalent to per-window generation - ``tests/test_scan.py``
    and ``tests/test_orb2.py`` already assert the session-independence this
    relies on - and at ~390 bars a session the cost is irrelevant.
    """

    name: str
    strategy: object
    status: str
    mode: str
    instrument: str = "MES"
    #: When the strategy would cancel an unfilled entry. Ticket buttons expire
    #: here: a decision after the order would have been pulled is not a
    #: decision the strategy could have acted on.
    entry_cancel_time: time_type = rules.ENTRY_CUTOFF
    resample_minutes: int = 1
    aggregator: feed_mod.BarAggregator | None = None
    session: list[feed_mod.Bar] = field(default_factory=list)
    session_day: date_type | None = None
    fired_on: set = field(default_factory=set)

    def reset_session(self, day: date_type) -> None:
        self.session = []
        self.session_day = day
        if self.resample_minutes > 1:
            self.aggregator = feed_mod.BarAggregator(self.resample_minutes)

    def frame(self) -> pd.DataFrame:
        rows = [
            {"open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume}
            for b in self.session
        ]
        index = pd.DatetimeIndex([b.timestamp for b in self.session])
        return pd.DataFrame(rows, index=index)

    def push(self, bar: feed_mod.Bar) -> pd.DataFrame | None:
        """Accumulate ``bar``; return the session frame when it is worth rerunning."""
        day = rules.session_date(bar.timestamp)
        if self.session_day != day:
            self.reset_session(day)
        if self.aggregator is not None:
            completed = self.aggregator.push(bar)
            if completed is None:
                return None
            self.session.append(completed)
        else:
            self.session.append(bar)
        return self.frame()


def build_runners(registry: Registry, trend_ema: pd.Series | None,
                  roll_dates, early_close_dates) -> list[StrategyRunner]:
    """Instantiate everything :meth:`Registry.desk_strategies` names.

    A strategy that cannot be constructed is skipped with a loud log line
    rather than taking the desk down. A desk that refuses to start because one
    of several strategies has a bad config is worse than one that runs the rest
    and says so.
    """
    runners: list[StrategyRunner] = []
    for record, mode in registry.desk_strategies():
        if not record.class_path:
            log.warning("%s has no class_path; skipping", record.name)
            continue
        try:
            cls = record.load_class()
        except Exception as exc:  # noqa: BLE001
            log.error("cannot load %s (%s); skipping", record.name, exc)
            continue
        try:
            if record.name in ("orb2", "orb_flat_1030"):
                import orb2  # noqa: PLC0415

                params = orb2.ORB2Params()
                strategy = cls(params=params, trend_ema=trend_ema,
                               roll_dates=roll_dates,
                               early_close_dates=early_close_dates)
            else:
                strategy = cls()
        except Exception as exc:  # noqa: BLE001
            log.error("cannot construct %s (%s); skipping", record.name, exc)
            continue
        params = getattr(strategy, "params", None)
        cancel_time = getattr(params, "entry_cancel_time", None) or rules.ENTRY_CUTOFF
        runners.append(StrategyRunner(
            name=record.name, strategy=strategy, status=record.status,
            mode=broker.MODE_SHADOW if mode == "shadow" else broker.MODE_PAPER,
            entry_cancel_time=cancel_time,
        ))
        log.info("running %s (status=%s, mode=%s)", record.name, record.status, mode)
    return runners


def signal_from_row(runner: StrategyRunner, frame: pd.DataFrame,
                    signals: pd.DataFrame) -> tickets_mod.Signal | None:
    """Read the newest signal row, if it fires an entry.

    Only the last row is consulted. Earlier rows were already offered on the
    bars that produced them, and reacting to them again would fire a signal
    twice from one session.
    """
    if signals is None or signals.empty:
        return None
    ts = signals.index[-1]
    row = signals.iloc[-1]
    if not (bool(row.get("entry_long", False)) or bool(row.get("entry_short", False))):
        return None
    direction = "long" if bool(row.get("entry_long", False)) else "short"

    def _num(key):
        value = row.get(key, None)
        return None if value is None or pd.isna(value) else float(value)

    entry = _num("entry_price")
    if entry is None:
        # Base-interface strategies publish no levels; the convention is to act
        # at the open of the signalled bar.
        entry = float(frame.loc[ts, "open"])
    stop = _num("stop_price")
    if stop is None:
        return None  # a signal with no stop cannot be sized or risk-checked
    return tickets_mod.Signal(
        strategy=runner.name,
        instrument=runner.instrument,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=_num("target_price"),
        contracts=1,
        bar_time=ts,
        mode=runner.mode,
    )


# ---------------------------------------------------------------------------
# The desk
# ---------------------------------------------------------------------------

class Desk:
    """Consumes bars, produces tickets and ops posts. Places nothing."""

    def __init__(self, registry: Registry, runners: list[StrategyRunner],
                 poster: Poster, state: desk_state.DeskState,
                 adapter: broker.BrokerAdapter, account: broker.AccountConfig,
                 journal_path: Path, state_path: Path,
                 early_close_dates=(), roll_dates=(), closed_dates=()) -> None:
        self.registry = registry
        self.runners = runners
        self.poster = poster
        self.state = state
        self.adapter = adapter
        self.account = account
        self.journal_path = Path(journal_path)
        self.state_path = Path(state_path)
        self.early_close_dates = set(early_close_dates)
        self.roll_dates = set(roll_dates)
        self.closed_dates = set(closed_dates)
        self.day_bars: dict[date_type, list[feed_mod.Bar]] = {}
        self.current_day: date_type | None = None
        #: Set when the webhook server failed to start. Surfaced in every
        #: heartbeat and posted once Discord is up - never silent.
        self.feed_error: str | None = None
        self.owner_id: int | None = approvals.owner_id_from_env()
        #: Human-gated tickets awaiting Execute, by ticket id.
        self.pending: dict[str, tuple[tickets_mod.Signal, object]] = {}
        self.decisions_path: Path = approvals.decisions_mod.DECISIONS_PATH

    # -- ops ---------------------------------------------------------------

    def in_session(self, now: pd.Timestamp) -> bool:
        """Is ``now`` inside the monitored window? See :data:`SESSION_START`.

        A holiday is out of session by definition - there is nothing to
        monitor - and an early close pulls the end forward with the flatten
        deadline rather than staying at 16:35.
        """
        now = rules.to_et(now)
        day = rules.session_date(now)
        if day in self.closed_dates:
            return False
        end = SESSION_END
        if day in self.early_close_dates:
            close = rules.flatten_deadline(now, self.early_close_dates)
            end = min(end, (pd.Timestamp.combine(day, close)
                            + pd.Timedelta(minutes=5)).time())
        return SESSION_START <= now.time() <= end

    async def premarket_check(self, now: pd.Timestamp, bars_seen: int) -> None:
        day = rules.session_date(now)
        lines = [f"**Session** {day}"]
        lines.append(f"**Bars received** {bars_seen} since start"
                     + ("" if bars_seen else "  **- NO DATA**"))
        if day in self.closed_dates:
            lines.append("**Calendar** exchange holiday - no session")
        elif day in self.early_close_dates:
            lines.append(f"**Calendar** early close, flatten "
                         f"{rules.flatten_deadline(now, self.early_close_dates)}")
        else:
            lines.append(f"**Calendar** normal session, flatten "
                         f"{rules.flatten_deadline(now, self.early_close_dates)}, "
                         f"entry cutoff {rules.entry_deadline(now, self.early_close_dates)}")
        if rules.is_roll_day(day, self.roll_dates):
            lines.append("**Roll day** - no new entries")
        try:
            import cme_calendar  # noqa: PLC0415

            note = cme_calendar.coverage_warning(day)
            if note:
                lines.append(f"**Calendar coverage** {note}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"**Calendar coverage** unavailable ({exc})")
        lines.append(f"**Strategies** " + (", ".join(
            f"{r.name} [{r.mode}]" for r in self.runners) or "none"))
        lines.append(f"**Open shadow positions** {len(self.state.open_positions)}")
        await self.poster.send("ops", f"09:25 pre-market check - {day}",
                               "\n".join(lines))
        self.state.mark_premarket(day)
        self.save()

    async def heartbeat(self, now: pd.Timestamp) -> None:
        net = self.state.net_position()
        body = (
            f"**Now** {now:%Y-%m-%d %H:%M:%S %Z}\n"
            f"**Bars** {self.state.bars_seen}  "
            f"**last** {self.state.last_bar_at or 'none'}\n"
            f"**Tickets** {self.state.tickets_allowed} allowed, "
            f"{self.state.tickets_blocked} blocked\n"
            f"**Open** {len(self.state.open_positions)} position(s), net {net:+d}\n"
            f"**Mode** shadow - no orders are placed"
        )
        if self.feed_error:
            # Repeated on every beat on purpose. A feed that is down is the
            # single most important fact about the desk, and one notice at
            # startup scrolls away.
            body += f"\n\n**FEED DOWN** {self.feed_error}"
        await self.poster.send("ops", "Heartbeat", body)
        self.state.last_heartbeat = rules.to_et(now).isoformat()
        self.save()

    async def daily_summary(self, now: pd.Timestamp) -> None:
        day = rules.session_date(now)
        body = self.summary_text(day)
        await self.poster.send("summary", f"Daily summary - {day}", body)
        self.state.mark_summary(day)
        self.save()

    def summary_text(self, day: date_type) -> str:
        """The 16:35 report. Reads the shadow journal, not in-memory counters."""
        try:
            closed = store.load_closed(self.journal_path)
        except FileNotFoundError:
            closed = pd.DataFrame()
        if not closed.empty and "session_date" in closed:
            today = closed[closed["session_date"].astype(str) == str(day)]
        else:
            today = pd.DataFrame()

        lines = [tickets_mod.SHADOW_LABEL, ""]
        if today.empty:
            lines.append(f"**Trades** none closed on {day}")
        else:
            net = float(today["net_pnl"].sum())
            wins = int((today["net_pnl"] > 0).sum())
            durations = today["duration_seconds"].astype(float)
            short = today[durations <= rules.MICROSCALP_SECONDS]
            lines += [
                f"**Trades** {len(today)} closed, {wins} winners",
                f"**Net P&L** ${net:,.2f} (commission and slippage applied)",
                f"**Avg duration** {durations.mean() / 60:,.1f} min "
                f"(min {durations.min():,.0f}s)",
            ]
            # Rule 6/7: a sub-30s close means the minimum-hold enforcement
            # failed. Rule 7 says treat that as a bug, not a risk warning.
            violations = today[durations < rules.MIN_HOLD_SECONDS]
            if len(violations):
                lines.append(
                    f"**RULE 6 VIOLATION** {len(violations)} close(s) under "
                    f"{rules.MIN_HOLD_SECONDS}s - this is a bug in the exit "
                    f"guard and must be investigated, not accepted"
                )
            if len(short):
                profit = float(today[today["net_pnl"] > 0]["net_pnl"].sum())
                share = (float(short[short["net_pnl"] > 0]["net_pnl"].sum())
                         / profit * 100.0) if profit > 0 else 0.0
                lines.append(
                    f"**Microscalp (rule 7)** {share:.1f}% of profit from trades "
                    f"<= {rules.MICROSCALP_SECONDS}s"
                    + ("  **- OVER THE 30% LINE**"
                       if share > rules.MICROSCALP_PROFIT_FLAG_PCT else "")
                )
        lines += [
            "",
            f"**Signals** {self.state.tickets_allowed} allowed, "
            f"{self.state.tickets_blocked} blocked",
            f"**Bars** {self.state.bars_seen}",
            f"**Still open** {len(self.state.open_positions)}",
        ]
        return "\n".join(lines)

    async def reconcile(self, now: pd.Timestamp) -> None:
        """Post anything open at startup that the state file does not explain.

        The journal is the authority, not the state file. A crash between the
        journal append and the state save leaves a position on disk that the
        desk has no memory of, and that is precisely the case worth shouting
        about.
        """
        try:
            open_rows = store.load_open(self.journal_path)
        except FileNotFoundError:
            open_rows = pd.DataFrame()
        known = set(self.state.open_positions)
        found = set(open_rows["ticket_id"].astype(str)) if not open_rows.empty else set()

        unknown = found - known
        missing = known - found
        if not unknown and not missing:
            return
        lines = []
        if unknown:
            for tid in sorted(unknown):
                row = open_rows[open_rows["ticket_id"].astype(str) == tid].iloc[-1]
                lines.append(
                    f"- `{tid}` {row['direction']} {int(row['contracts'])} "
                    f"{row['instrument']} @ {float(row['entry_price']):.2f} "
                    f"opened {row['entry_time']} - **not in desk state**"
                )
        if missing:
            lines += [f"- `{tid}` in desk state but not open in the journal"
                      for tid in sorted(missing)]
        await self.poster.send(
            "ops", "Startup reconciliation - unrecognised positions",
            "\n".join([
                "The journal and the desk state file disagree. Nothing has been "
                "closed automatically; resolve before trusting the position cap.",
                "",
                *lines,
            ]),
            colour=0xC1662F,
        )

    def save(self) -> None:
        try:
            self.state.save(self.state_path)
        except OSError as exc:
            log.error("could not save desk state: %s", exc)

    # -- the decision path -------------------------------------------------

    async def handle_signal(self, signal: tickets_mod.Signal,
                            now: pd.Timestamp) -> tickets_mod.Outcome:
        state = tickets_mod.account_state(
            rules.session_date(now), self.journal_path,
            open_position=self.state.net_position(signal.instrument),
        )
        decision = tickets_mod.evaluate_signal(
            signal, state, now,
            self.early_close_dates, self.roll_dates, self.closed_dates,
        )
        if not decision.allowed:
            outcome = tickets_mod.Outcome(signal=signal, decision=decision)
            self.state.tickets_blocked += 1
            await self.post_block(outcome)
            self.save()
            return outcome

        status = self._status_for(signal.strategy)
        if signal.mode == broker.MODE_SHADOW:
            # Shadow is the plumbing test: simulate immediately, exactly as
            # before. The buttons on its embed record the operator's
            # judgement; Execute is disabled because there is no order path
            # for a rejected strategy, and Don't trade records a decline
            # without touching the bot's own shadow record.
            outcome = tickets_mod.submit(
                signal, decision, self.adapter, self.account, status, now,
                self.journal_path,
            )
            if outcome.allowed and outcome.ticket is not None:
                self._open_from(outcome, signal)
                self.state.tickets_allowed += 1
                await self.post_ticket(outcome, now, status, ticket_id=None)
            else:
                self.state.tickets_blocked += 1
                await self.post_block(outcome)
            self.save()
            return outcome

        # paper / live: human-gated. Nothing is submitted until Execute is
        # pressed; the ticket is posted under an id minted now so the journal
        # row written later carries the id the operator saw.
        outcome = tickets_mod.Outcome(signal=signal, decision=decision)
        ticket_id = store.new_ticket_id()
        self.pending[ticket_id] = (signal, decision)
        self.state.tickets_allowed += 1
        await self.post_ticket(outcome, now, status, ticket_id=ticket_id)
        self.save()
        return outcome

    def _open_from(self, outcome: tickets_mod.Outcome,
                   signal: tickets_mod.Signal) -> None:
        self.state.add_position(desk_state.OpenPosition(
            ticket_id=outcome.ticket.ticket_id,
            strategy=signal.strategy,
            instrument=signal.instrument,
            direction=signal.direction,
            contracts=int(signal.contracts),
            entry_price=float(outcome.ticket.entry_price),
            stop_price=float(signal.stop_price),
            target_price=signal.target_price,
            entry_time=outcome.ticket.entry_time,
            mode=signal.mode,
            session_date=outcome.ticket.session_date,
        ))

    async def execute_pending(self, ticket_id: str) -> str:
        """Route a human-approved ticket. Called from the Execute button.

        Returns a one-line note for the operator. A ``live`` approval with no
        live adapter is **refused**, not quietly paper-filled: an approval the
        operator believes went to a broker must never have gone somewhere
        else instead.
        """
        item = self.pending.pop(ticket_id, None)
        if item is None:
            return f"ticket `{ticket_id}` is not pending"
        signal, decision = item
        status = self._status_for(signal.strategy)
        now = pd.Timestamp.now(tz=rules.ET)
        if status == "live":
            try:
                broker.adapter_for(broker.MODE_LIVE)
            except broker.BrokerRefusal as exc:
                return f"NOT routed - {exc}"
        outcome = tickets_mod.submit(
            signal, decision, self.adapter, self.account, status, now,
            self.journal_path, ticket_id=ticket_id,
        )
        if outcome.allowed and outcome.ticket is not None:
            self._open_from(outcome, signal)
            self.save()
            return (f"routed to {outcome.fill.adapter} (simulated="
                    f"{outcome.fill.simulated}), ticket `{ticket_id}` open")
        return f"NOT routed - {outcome.refusal or '; '.join(outcome.decision.blocks)}"

    def _status_for(self, name: str) -> str:
        try:
            return self.registry.get(name).status
        except KeyError:
            return "unknown"

    def _cancel_time_for(self, strategy: str) -> time_type:
        for runner in self.runners:
            if runner.name == strategy:
                return runner.entry_cancel_time
        return rules.ENTRY_CUTOFF

    def ticket_view(self, signal: tickets_mod.Signal, status: str,
                    ticket_id: str, now: pd.Timestamp) -> approvals.TicketView:
        cancel = self._cancel_time_for(signal.strategy)
        expires = pd.Timestamp.combine(rules.session_date(now), cancel).tz_localize(rules.ET)
        on_execute = None
        if status in approvals.EXECUTABLE_STATUSES:
            async def on_execute(_tid=ticket_id) -> str:
                return await self.execute_pending(_tid)
        return approvals.TicketView(
            ticket_id=ticket_id, strategy=signal.strategy, strategy_status=status,
            mode=signal.mode, instrument=signal.instrument,
            direction=signal.direction, entry_price=signal.entry_price,
            stop_price=signal.stop_price, owner_id=self.owner_id,
            expires_at=expires, on_execute=on_execute,
            journal_path=self.decisions_path,
        )

    async def post_ticket(self, outcome: tickets_mod.Outcome, now: pd.Timestamp,
                          status: str, ticket_id: str | None) -> None:
        s = outcome.signal
        t = outcome.ticket
        tid = ticket_id or (t.ticket_id if t is not None else "?")
        entry = t.entry_price if t is not None else s.entry_price
        risk = t.risk_dollars if t is not None else outcome.decision.risk_dollars
        target = ("-" if s.target_price is None else f"{s.target_price:,.2f}")
        header = outcome.label or "PAPER"
        if t is not None and outcome.fill is not None:
            mode_line = (f"**Mode** {s.mode} - simulated by {outcome.fill.adapter}, "
                         f"no order was placed")
        else:
            mode_line = (f"**Mode** {s.mode} - awaiting your decision; nothing "
                         f"is routed until Execute is pressed")
        cancel = self._cancel_time_for(s.strategy)
        body = (
            f"**{header}**\n\n"
            f"**Strategy** {s.strategy} ({status})\n"
            f"**Direction** {s.direction.upper()}\n"
            f"**Entry** {entry:,.2f}   **Stop** {s.stop_price:,.2f}   "
            f"**Target** {target}\n"
            f"**Size** {s.contracts} {s.instrument}   "
            f"**Risk** ${risk:,.2f}\n"
            f"{mode_line}\n"
            f"**Ticket** `{tid}`\n"
            f"**Buttons expire** {cancel:%H:%M} ET"
        )
        if outcome.decision.warnings:
            body += "\n\n**Notes**\n" + "\n".join(
                f"- {w}" for w in outcome.decision.warnings)
        view = self.ticket_view(s, status, tid, now)
        await self.poster.send("tickets", f"{s.strategy} {s.direction} "
                                          f"{s.contracts} {s.instrument}",
                               body, colour=0x3A7D44, view=view)

    async def post_block(self, outcome: tickets_mod.Outcome) -> None:
        s = outcome.signal
        body = (
            f"**{outcome.label or 'PAPER'}**\n\n"
            f"**Strategy** {s.strategy}\n"
            f"**Would have been** {s.direction.upper()} {s.contracts} "
            f"{s.instrument} @ {s.entry_price:,.2f} stop {s.stop_price:,.2f}\n"
            f"**Bar** {s.bar_time:%Y-%m-%d %H:%M}\n\n"
            f"**Blocked because**\n"
            + "\n".join(f"- {r}" for r in outcome.decision.blocks)
        )
        await self.poster.send("tickets", f"BLOCKED - {s.strategy} {s.direction}",
                               body, colour=0x9E2B25)

    # -- exits -------------------------------------------------------------

    async def manage_positions(self, bar: feed_mod.Bar, now: pd.Timestamp) -> None:
        """Stop, target, and the rule 2 flatten - checked on every bar.

        Rule 6's 30-second floor blocks a close that is too young, with two
        exceptions CLAUDE.md names explicitly: the rule 2 forced flatten and a
        rule 5 daily-loss breach outrank it. Rule 3's entry cutoff is supposed
        to make that collision impossible, so if the override ever fires it is
        logged as the bug indicator it is.
        """
        for position in list(self.state.open_positions.values()):
            if position.instrument != bar.symbol:
                continue
            exit_price, reason = self._exit_for(position, bar, now)
            if exit_price is None:
                continue

            forced = reason in ("flatten", "daily_loss")
            held = position.held_seconds(now)
            if held < rules.MIN_HOLD_SECONDS and not forced:
                log.info("holding %s: %0.0fs of %ds minimum",
                         position.ticket_id, held, rules.MIN_HOLD_SECONDS)
                continue
            if held < rules.MIN_HOLD_SECONDS and forced:
                log.error(
                    "RULE 6 OVERRIDE: %s forced out at %0.0fs (< %ds) by %r. "
                    "Rule 3's entry cutoff should make this impossible - "
                    "investigate the entry guard.",
                    position.ticket_id, held, rules.MIN_HOLD_SECONDS, reason,
                )
                await self.poster.send(
                    "ops", "RULE 6 OVERRIDE - investigate",
                    f"`{position.ticket_id}` was forced out after {held:.0f}s by "
                    f"`{reason}`, under the {rules.MIN_HOLD_SECONDS}s floor.\n"
                    f"CLAUDE.md rule 6: this indicates a bug in the entry guard "
                    f"and must be investigated, not accepted as normal.",
                    colour=0xC1662F,
                )

            closed = tickets_mod.close_position(
                position.ticket_id, exit_price, now, reason, self.journal_path)
            self.state.drop_position(position.ticket_id)
            self.save()
            await self.poster.send(
                "tickets", f"CLOSED {position.strategy} {position.direction}",
                f"**{tickets_mod.SHADOW_LABEL if position.mode == broker.MODE_SHADOW else ''}**\n\n"
                f"**Ticket** `{position.ticket_id}`\n"
                f"**Exit** {closed.exit_price:,.2f} ({reason})\n"
                f"**Held** {(closed.duration_seconds or 0) / 60:,.1f} min\n"
                f"**Net P&L** ${closed.net_pnl:,.2f}",
                colour=0x3A7D44 if (closed.net_pnl or 0) >= 0 else 0x9E2B25,
            )

    def _exit_for(self, position: desk_state.OpenPosition, bar: feed_mod.Bar,
                  now: pd.Timestamp) -> tuple[float | None, str]:
        """Which level this bar hit, if any.

        When a bar spans both the stop and the target, the **stop** is taken.
        One-minute bars carry no intrabar path, so the choice is an assumption
        rather than a measurement, and assuming the worse fill keeps a shadow
        result from flattering itself.
        """
        if rules.must_flatten(now, self.early_close_dates):
            return bar.close, "flatten"
        long = position.direction == "long"
        stop, target = position.stop_price, position.target_price
        hit_stop = bar.low <= stop if long else bar.high >= stop
        hit_target = (target is not None
                      and (bar.high >= target if long else bar.low <= target))
        if hit_stop:
            return stop, "stop"
        if hit_target:
            return target, "target"
        return None, ""

    # -- the loop ----------------------------------------------------------

    async def on_bar(self, bar: feed_mod.Bar) -> None:
        now = bar.timestamp
        day = rules.session_date(now)
        self.state.bars_seen += 1
        self.state.last_bar_at = now.isoformat()

        if self.current_day != day:
            self.current_day = day
        if self.state.needs_premarket(day) and now.time() >= PREMARKET_TIME:
            await self.premarket_check(now, self.state.bars_seen)
        if (self.in_session(now)
                and self.state.due_for_heartbeat(now, HEARTBEAT_MINUTES)):
            await self.heartbeat(now)

        await self.manage_positions(bar, now)

        for runner in self.runners:
            if runner.instrument != bar.symbol:
                continue
            frame = runner.push(bar)
            if frame is None or len(frame) < 2:
                continue
            try:
                signals = await asyncio.to_thread(
                    runner.strategy.generate_signals, frame)
            except Exception as exc:  # noqa: BLE001 - one bad strategy, not the desk
                log.error("%s.generate_signals failed at %s: %s",
                          runner.name, now, exc)
                continue
            signal = signal_from_row(runner, frame, signals)
            if signal is None:
                continue
            key = (runner.name, day)
            if key in runner.fired_on:
                continue
            runner.fired_on.add(key)
            await self.handle_signal(signal, now)

        if self.state.needs_summary(day) and now.time() >= SUMMARY_TIME:
            await self.daily_summary(now)
        self.save()

    async def end_of_replay(self, now: pd.Timestamp) -> None:
        """Flush the last session's summary even if the feed stopped early."""
        if now is None:
            return
        day = rules.session_date(now)
        if self.state.needs_summary(day):
            await self.daily_summary(now)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def calendar_sets(day: date_type):
    """Early-close, roll and closed dates. Same three sources as pretrade.py."""
    return __import__("pretrade").calendar_dates(False, day)


def build_desk(journal_path: Path, state_path: Path, echo: bool = True,
               client=None, registry: Registry | None = None,
               day: date_type | None = None) -> Desk:
    registry = registry or Registry.load()
    problems = registry.verify()
    if problems:
        log.warning("registry disagrees with the log: %s", "; ".join(problems))

    day = day or pd.Timestamp.now(tz=rules.ET).date()
    early, rolls, closed, warnings = calendar_sets(day)
    for note in warnings:
        log.info("calendar: %s", note)

    trend_ema = None
    if MES_PARQUET.exists():
        try:
            import loader  # noqa: PLC0415
            import trend  # noqa: PLC0415

            trend_ema = trend.trend_filter(loader.load_bars(MES_PARQUET))
        except Exception as exc:  # noqa: BLE001
            log.error("could not build the trend EMA (%s); "
                      "trend-filtered strategies will be skipped", exc)

    runners = build_runners(registry, trend_ema, rolls, early)
    state = desk_state.DeskState.load(state_path)
    state.started_at = pd.Timestamp.now(tz=rules.ET).isoformat()
    account = broker.AccountConfig(journal_path=Path(journal_path))
    return Desk(
        registry=registry,
        runners=runners,
        poster=Poster(client=client, channel_name=channel_name_from_env(), echo=echo),
        state=state,
        adapter=broker.PaperAdapter(),
        account=account,
        journal_path=Path(journal_path),
        state_path=Path(state_path),
        early_close_dates=early, roll_dates=rolls, closed_dates=closed,
    )


async def run_replay(desk: Desk, replay: feed_mod.ReplayFeed) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    producer = asyncio.create_task(replay.run(queue))
    last: pd.Timestamp | None = None
    try:
        async for bar in feed_mod.drain(queue):
            last = bar.timestamp
            await desk.on_bar(bar)
    finally:
        producer.cancel()
    await desk.end_of_replay(last)


async def run_live(desk: Desk, host: str, port: int, token: str | None) -> None:
    """The feed: webhook server plus the bar loop. Started exactly once.

    A bind failure does not kill the desk and is never silent. uvicorn
    responds to a port already in use by logging and calling ``sys.exit`` from
    inside the serving task - which is why the reconnect bug presented as the
    whole process exiting. Both ``OSError`` and that ``SystemExit`` are caught
    here, recorded on ``desk.feed_error``, shouted to the log, and posted to
    Discord (buffered until the gateway is up). The bar loop then waits on a
    queue nothing will ever fill, which is the correct behaviour: the desk
    stays up, keeps heartbeating with FEED DOWN in every beat, and the
    operator can see exactly what is wrong.
    """
    queue: asyncio.Queue = asyncio.Queue()
    webhook = feed_mod.WebhookFeed(queue, host=host, port=port, token=token)

    async def serve() -> None:
        try:
            await webhook.run()
        except (OSError, SystemExit) as exc:
            desk.feed_error = (f"webhook server failed to start on {host}:{port} "
                               f"({type(exc).__name__}: {exc}). No bars will "
                               f"arrive until the desk is restarted.")
            log.critical("FEED DOWN: %s", desk.feed_error)
            await desk.poster.send(
                "ops", "FEED DOWN - webhook failed to start",
                desk.feed_error + "\n\nIs another desk already running on this "
                "port? Check with:  netstat -ano | findstr :" + str(port),
                colour=0x9E2B25,
            )

    server = asyncio.create_task(serve())
    await desk.reconcile(pd.Timestamp.now(tz=rules.ET))
    try:
        async for bar in feed_mod.drain(queue):
            if bar.is_stale(pd.Timestamp.now(tz=rules.ET)):
                log.warning("stale bar %s ignored", bar.timestamp)
                continue
            await desk.on_bar(bar)
    finally:
        # Graceful first, cancel only if it will not go. See WebhookFeed.stop.
        webhook.stop()
        try:
            await asyncio.wait_for(asyncio.shield(server), timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
            server.cancel()


class DeskClient:
    """The Discord side of the desk, built so it cannot restart the feed.

    The feed is started from ``setup_hook``, which discord.py calls **exactly
    once**, after login and before the first gateway connection. ``on_ready``
    fires on the initial connection *and again on every reconnect*, and the
    previous version started the feed from there - so a reconnect tried to
    bind port 8787 a second time and uvicorn's ``sys.exit`` took the process
    down. ``start_feed_once`` is additionally idempotent, and
    ``tests/test_desk_discord.py`` calls ``on_ready`` twice and asserts the
    feed started once and nothing exited.

    Written as a mixin over ``discord.Client`` rather than a subclass at
    module level so the module imports without discord.py (replay does not
    need it); :func:`make_client` binds the two.
    """

    def _init_desk(self, desk: Desk, feed_factory) -> None:
        self.desk = desk
        self._feed_factory = feed_factory
        self.feed_task: asyncio.Task | None = None
        self.feed_starts = 0
        self.ready_count = 0
        self.fatal: str | None = None

    def start_feed_once(self) -> bool:
        if self.feed_task is not None:
            return False
        self.feed_starts += 1
        self.feed_task = asyncio.create_task(self._feed_factory())
        return True

    async def setup_hook(self) -> None:
        self.start_feed_once()

    async def stop_feed(self) -> None:
        """Wind the feed down before a fatal close, so the exit is quiet."""
        task = self.feed_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutting down
            pass

    async def on_ready(self) -> None:
        self.ready_count += 1
        poster = self.desk.poster
        if self.ready_count == 1:
            error = await poster.resolve_channel()
            if error:
                self.fatal = error
                log.critical("cannot post to Discord: %s", error)
                await self.stop_feed()
                await self.close()
                return
            poster.ready = True
            await poster.send(
                "ops", "Desk online",
                f"Connected as {self.user}. Posting to #{poster.channel.name}.\n"
                f"Strategies: {', '.join(f'{r.name} [{r.mode}]' for r in self.desk.runners) or 'none'}\n"
                f"Owner id: {self.desk.owner_id or 'NOT SET - no one can press ticket buttons'}",
            )
        else:
            poster.ready = True
            await poster.send(
                "ops", f"Discord reconnected (#{self.ready_count - 1})",
                f"Gateway session re-established as {self.user}. The bar feed "
                f"was not interrupted; {len(poster.buffer)} buffered post(s) "
                f"follow.",
            )
        await poster.flush()
        # Never starts a second feed. Kept explicit so the invariant is a
        # line of code rather than an absence of one.
        assert not self.start_feed_once(), "feed must already be running"

    async def on_resumed(self) -> None:
        self.desk.poster.ready = True
        await self.desk.poster.flush()

    async def on_disconnect(self) -> None:
        # Posts made from here until the next on_ready/on_resumed are buffered.
        self.desk.poster.ready = False


def make_client(desk: Desk, feed_factory):
    """Bind :class:`DeskClient` to a real ``discord.Client``."""
    import discord  # noqa: PLC0415

    class _Client(DeskClient, discord.Client):
        def __init__(self) -> None:
            discord.Client.__init__(self, intents=discord.Intents.default())
            self._init_desk(desk, feed_factory)

    return _Client()


def main(argv=None) -> int:
    # A short ASCII description rather than __doc__: the module docstring is
    # long, and its section marks render as mojibake in a cp1252 console.
    ap = argparse.ArgumentParser(
        prog="desk.py",
        description="Desk bot, SHADOW mode - decides and posts, places nothing. "
                    "The only strategy it runs (orb2) is REJECTED; see "
                    "docs/HANDOFF.md section 6.",
        epilog="example: desk.py --replay --start 2026-08-24 --end 2026-08-28 "
               "--speed 60",
    )
    ap.add_argument("--replay", action="store_true",
                    help="feed cached parquet bars instead of listening")
    ap.add_argument("--start", default=None, help="replay start date")
    ap.add_argument("--end", default=None, help="replay end date")
    ap.add_argument("--speed", type=float, default=60.0,
                    help="replay speed-up; 0 runs as fast as possible")
    ap.add_argument("--symbol", default="MES", choices=sorted(rules.ALLOWED_INSTRUMENTS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--journal", default=str(tickets_mod.SHADOW_JOURNAL))
    ap.add_argument("--state", default=str(desk_state.STATE_PATH))
    ap.add_argument("--no-discord", action="store_true",
                    help="print posts to stdout only")
    ap.add_argument("--quiet", action="store_true", help="suppress stdout echo")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass

    if args.replay:
        if not (args.start and args.end):
            ap.error("--replay needs --start and --end")
        parquet = (PROJECT_ROOT / "data" /
                   f"{args.symbol.lower()}_v_0_ohlcv_1m_2019-05_2026-08.parquet")
        replay = feed_mod.ReplayFeed.from_parquet(
            parquet, args.start, args.end, symbol=args.symbol, speed=args.speed)
        desk = build_desk(Path(args.journal), Path(args.state),
                          echo=not args.quiet,
                          day=pd.Timestamp(args.start).date())
        print(f"Replaying {len(replay):,} bars, {args.start} to {args.end}, "
              f"speed {args.speed}x\n")
        asyncio.run(run_replay(desk, replay))
        return 0

    webhook_token = os.environ.get("DESK_WEBHOOK_TOKEN")
    token = (os.environ.get("DISCORD_TOKEN") or "").strip()
    if not args.no_discord and not token:
        # Checked before build_desk loads 40 MB of bars: a configuration
        # error should fail in a second, not a minute. And it is not a
        # fallback to stdout - a desk you believe is posting to Discord while
        # it prints to a window is the failure mode being refused.
        print("\nERROR: DISCORD_TOKEN is not set in .env.\n"
              "  Either add it, or pass --no-discord to run console-only on "
              "purpose.\n", file=sys.stderr)
        return 2

    desk = build_desk(Path(args.journal), Path(args.state), echo=not args.quiet)

    if args.no_discord:
        # Explicitly console-only. This is the one legitimate stdout mode.
        asyncio.run(run_live(desk, args.host, args.port, webhook_token))
        return 0

    if desk.owner_id is None:
        log.warning("DESK_OWNER_ID is not set: ticket buttons will refuse "
                    "everyone. Set it in .env (Discord > Developer Mode > "
                    "Copy User ID).")

    client = make_client(desk, lambda: run_live(desk, args.host, args.port,
                                                webhook_token))
    desk.poster.client = client
    client.run(token, log_handler=None)

    if client.fatal:
        print(f"\nERROR: {client.fatal}\n", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
