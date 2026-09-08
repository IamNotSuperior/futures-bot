"""The Discord side of the desk: reconnects, the one channel, and the buttons.

Three things this file exists to pin, each of which was a real defect or a
real rule:

* **A reconnect must not restart the feed.** discord.py fires ``on_ready`` on
  every reconnect. The feed used to start from there, so a reconnect bound
  the port twice and uvicorn's ``sys.exit`` took the process down.
* **No silent fallback.** If the named channel cannot be found or posted to,
  the desk exits with a reason. Posts made while the gateway is down are
  buffered and flushed, in order, when it returns.
* **No order path for a rejected strategy.** The Execute button is disabled
  for a shadow ticket *and* the decision core refuses an approve on any
  status but paper/live, so a disabled button is not the only guard.

Nothing here opens a gateway. discord.py's ``utils.get`` and ``Embed`` work
on plain objects, and the decision core is deliberately free of discord.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

import approvals
import broker
import decisions as decisions_mod
import desk as desk_mod
import desk_state
import feed as feed_mod
import rules
import store
import tickets as tickets_mod
from registry import Registry

OWNER = 123456789012345678
STRANGER = 999999999999999999
TEN_AM = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
CANCEL_1030 = pd.Timestamp("2026-08-24 10:30", tz=rules.ET)


# -- fakes ------------------------------------------------------------------

class FakeMessage:
    def __init__(self, embed=None, view=None) -> None:
        self.embeds = [embed] if embed is not None else []
        self.view = view
        self.edits: list[dict] = []

    async def edit(self, **kwargs) -> None:
        self.edits.append(kwargs)


class FakePerms:
    send_messages = True
    embed_links = True


class FakeChannel:
    def __init__(self, name: str, can_post: bool = True) -> None:
        self.name = name
        self.sent: list[tuple] = []
        self._can_post = can_post

    def permissions_for(self, _member):
        perms = FakePerms()
        perms.send_messages = self._can_post
        return perms

    async def send(self, embed=None, view=None):
        self.sent.append((embed, view))
        return FakeMessage(embed, view)


class FakeGuild:
    def __init__(self, name: str, channels: list[FakeChannel]) -> None:
        self.name = name
        self.text_channels = channels
        self.me = object()


class FakeUser:
    def __init__(self, user_id: int, name: str = "farhan") -> None:
        self.id = user_id
        self.name = name

    def __str__(self) -> str:
        return self.name


class StubClient(desk_mod.DeskClient):
    """DeskClient's logic over a stub gateway: no discord.Client involved."""

    def __init__(self, desk, feed_factory, guilds) -> None:
        self._init_desk(desk, feed_factory)
        self.guilds = guilds
        self.user = "desk-bot#0001"
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def build(tmp_path: Path, runners=(), client=None, channel="general") -> desk_mod.Desk:
    poster = desk_mod.Poster(client=client, channel_name=channel, echo=False)
    desk = desk_mod.Desk(
        registry=Registry.load(),
        runners=list(runners),
        poster=poster,
        state=desk_state.DeskState(),
        adapter=broker.PaperAdapter(),
        account=broker.AccountConfig(journal_path=tmp_path / "shadow_trades.jsonl"),
        journal_path=tmp_path / "shadow_trades.jsonl",
        state_path=tmp_path / "desk_state.json",
    )
    desk.owner_id = OWNER
    desk.decisions_path = tmp_path / "decisions.jsonl"
    return desk


def make_signal(mode=broker.MODE_SHADOW, strategy="orb2") -> tickets_mod.Signal:
    return tickets_mod.Signal(
        strategy=strategy, instrument="MES", direction="long",
        entry_price=6800.0, stop_price=6790.0, target_price=6818.0,
        contracts=1, bar_time=TEN_AM, mode=mode,
    )


async def _never_ends() -> None:
    await asyncio.Event().wait()


# -- 1. reconnects ----------------------------------------------------------

def test_second_on_ready_does_not_restart_the_feed_or_exit(tmp_path):
    """The bug: on_ready on reconnect bound the port twice and the process died."""
    starts = []

    async def feed_factory() -> None:
        starts.append(1)
        await _never_ends()

    channel = FakeChannel("general")
    desk = build(tmp_path)
    client = StubClient(desk, feed_factory, [FakeGuild("g", [channel])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()           # initial connect
        await client.on_disconnect()      # gateway drops
        await client.on_ready()           # reconnect
        await client.on_ready()           # and again
        await asyncio.sleep(0)
        client.feed_task.cancel()

    asyncio.run(go())                     # no SystemExit, no exception
    assert client.feed_starts == 1
    assert len(starts) == 1
    assert client.ready_count == 3
    assert client.fatal is None
    assert not client.closed


def test_reconnect_is_announced_in_the_channel(tmp_path):
    channel = FakeChannel("general")
    desk = build(tmp_path)
    client = StubClient(desk, _never_ends, [FakeGuild("g", [channel])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        await client.on_disconnect()
        await client.on_ready()
        client.feed_task.cancel()

    asyncio.run(go())
    titles = [e.title for e, _ in channel.sent]
    assert any("Desk online" in t for t in titles)
    assert any("reconnected" in t.lower() for t in titles)


def test_feed_bind_failure_is_recorded_posted_and_not_fatal(tmp_path, monkeypatch):
    """uvicorn calls sys.exit on a bind failure. That must not be the desk's exit."""

    class ExplodingFeed:
        """Models WebhookFeed's interface: run() and stop()."""

        def __init__(self, *a, **k) -> None:
            pass

        async def run(self) -> None:
            raise SystemExit(1)

        def stop(self) -> None:
            pass

    monkeypatch.setattr(feed_mod, "WebhookFeed", ExplodingFeed)
    desk = build(tmp_path)

    async def go():
        task = asyncio.create_task(desk_mod.run_live(desk, "127.0.0.1", 1, None))
        await asyncio.sleep(0.2)
        assert not task.done(), "the desk loop exited - the bind failure was fatal"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert desk.feed_error and "failed to start" in desk.feed_error
    assert any("FEED DOWN" in body for _, body in desk.poster.posted)


def test_heartbeat_repeats_the_feed_error(tmp_path):
    desk = build(tmp_path)
    desk.feed_error = "webhook server failed to start on 127.0.0.1:8787"
    asyncio.run(desk.heartbeat(TEN_AM))
    beat = [b for _, b in desk.poster.posted if b.startswith("Heartbeat")][0]
    assert "FEED DOWN" in beat


# -- 2. the one channel -----------------------------------------------------

def test_missing_channel_is_fatal_not_a_stdout_fallback(tmp_path):
    desk = build(tmp_path, channel="desk")
    client = StubClient(desk, _never_ends, [FakeGuild("g", [FakeChannel("general")])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        client.feed_task.cancel()

    asyncio.run(go())
    assert client.fatal and "'desk'" in client.fatal
    assert "general" in client.fatal            # tells you what exists
    assert client.closed
    assert desk.poster.ready is False


def test_channel_without_send_permission_is_fatal(tmp_path):
    desk = build(tmp_path)
    client = StubClient(desk, _never_ends,
                        [FakeGuild("g", [FakeChannel("general", can_post=False)])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        client.feed_task.cancel()

    asyncio.run(go())
    assert client.fatal and "cannot post" in client.fatal
    assert client.closed


def test_no_guild_is_fatal(tmp_path):
    desk = build(tmp_path)
    client = StubClient(desk, _never_ends, [])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        client.feed_task.cancel()

    asyncio.run(go())
    assert client.fatal and "not in any guild" in client.fatal


def test_every_kind_of_post_goes_to_the_one_channel(tmp_path):
    channel = FakeChannel("general")
    desk = build(tmp_path)
    client = StubClient(desk, _never_ends, [FakeGuild("g", [channel])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        await desk.premarket_check(TEN_AM, 5)
        await desk.heartbeat(TEN_AM)
        await desk.daily_summary(TEN_AM)
        client.feed_task.cancel()

    asyncio.run(go())
    titles = [e.title for e, _ in channel.sent]
    assert any("pre-market" in t for t in titles)
    assert any(t == "Heartbeat" for t in titles)
    assert any("Daily summary" in t for t in titles)


def test_posts_are_buffered_while_disconnected_and_flushed_in_order(tmp_path):
    channel = FakeChannel("general")
    desk = build(tmp_path)
    client = StubClient(desk, _never_ends, [FakeGuild("g", [channel])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        before = len(channel.sent)
        await client.on_disconnect()
        await desk.poster.send("ops", "first while down", "a")
        await desk.poster.send("ops", "second while down", "b")
        assert len(channel.sent) == before, "posted while disconnected"
        assert len(desk.poster.buffer) == 2
        await client.on_resumed()
        client.feed_task.cancel()
        return before

    before = asyncio.run(go())
    assert desk.poster.buffer == []
    titles = [e.title for e, _ in channel.sent[before:]]
    assert titles == ["first while down", "second while down"]


def test_console_echo_is_kept_alongside_discord(tmp_path, capsys):
    channel = FakeChannel("general")
    desk = build(tmp_path)
    desk.poster.echo = True
    client = StubClient(desk, _never_ends, [FakeGuild("g", [channel])])
    desk.poster.client = client

    async def go():
        await client.setup_hook()
        await client.on_ready()
        await desk.poster.send("ops", "both places", "body")
        client.feed_task.cancel()

    asyncio.run(go())
    assert "both places" in capsys.readouterr().out
    assert any(e.title == "both places" for e, _ in channel.sent)


# -- 3. buttons: the decision core -----------------------------------------

def view(tmp_path, status="rejected", mode=broker.MODE_SHADOW, on_execute=None,
         owner=OWNER, expires=CANCEL_1030) -> approvals.TicketView:
    return approvals.TicketView(
        ticket_id="t1", strategy="orb2", strategy_status=status, mode=mode,
        instrument="MES", direction="long", entry_price=6800.0, stop_price=6790.0,
        owner_id=owner, expires_at=expires, on_execute=on_execute,
        journal_path=tmp_path / "decisions.jsonl",
    )


def test_shadow_ticket_execute_is_disabled_with_the_unavailable_label(tmp_path):
    execute, decline = view(tmp_path).button_specs()
    assert execute["disabled"] is True
    assert execute["label"] == "Execute — unavailable (strategy rejected)"
    assert decline["disabled"] is False
    assert decline["label"] == "Don't trade"


@pytest.mark.parametrize("status", ["rejected", "proposed", "testing", "unknown"])
def test_no_order_path_exists_for_a_non_executable_strategy(tmp_path, status):
    """Defence in depth: even reaching the callback directly must refuse."""
    routed = []

    async def on_execute() -> str:
        routed.append(1)
        return "routed"

    v = view(tmp_path, status=status, on_execute=on_execute)
    result = asyncio.run(v.decide(OWNER, "farhan", decisions_mod.APPROVE, TEN_AM))
    assert result.recorded is False
    assert "No order path" in result.message
    assert routed == []
    assert v.decided is None
    assert not (tmp_path / "decisions.jsonl").exists()


def test_paper_ticket_execute_is_enabled_and_routes(tmp_path):
    routed = []

    async def on_execute() -> str:
        routed.append(1)
        return "routed to paper"

    v = view(tmp_path, status="paper", mode=broker.MODE_PAPER, on_execute=on_execute)
    assert v.button_specs()[0] == {
        "label": "Execute", "style": "success", "disabled": False,
        "custom_id": "desk:t1:execute",
    }
    result = asyncio.run(v.decide(OWNER, "farhan", decisions_mod.APPROVE, TEN_AM))
    assert result.recorded and result.embed_title == "APPROVED"
    assert routed == [1]
    row = decisions_mod.load(tmp_path / "decisions.jsonl").iloc[0]
    assert row["decision"] == "approve" and row["user_id"] == OWNER
    assert row["strategy_status"] == "paper"


def test_decline_is_journaled_as_the_operators_decision(tmp_path):
    v = view(tmp_path)
    result = asyncio.run(v.decide(OWNER, "farhan", decisions_mod.DECLINE, TEN_AM))
    assert result.recorded and result.embed_title == "DECLINED"
    frame = decisions_mod.load(tmp_path / "decisions.jsonl")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert (row["ticket_id"], row["decision"], row["user_id"], row["user_name"]) \
        == ("t1", "decline", OWNER, "farhan")
    # Both buttons disabled after a decision.
    assert all(spec["disabled"] for spec in v.button_specs())


def test_only_the_owner_may_press(tmp_path):
    v = view(tmp_path)
    for uid in (STRANGER, None):
        result = asyncio.run(v.decide(uid, "someone", decisions_mod.DECLINE, TEN_AM))
        assert result.recorded is False
        assert result.message == approvals.NOT_OWNER_MESSAGE
    assert v.decided is None
    assert not (tmp_path / "decisions.jsonl").exists()


def test_unset_owner_id_means_nobody_can_press(tmp_path):
    v = view(tmp_path, owner=None)
    result = asyncio.run(v.decide(OWNER, "farhan", decisions_mod.DECLINE, TEN_AM))
    assert result.recorded is False


def test_second_press_is_ignored(tmp_path):
    v = view(tmp_path)
    asyncio.run(v.decide(OWNER, "farhan", decisions_mod.DECLINE, TEN_AM))
    again = asyncio.run(v.decide(OWNER, "farhan", decisions_mod.DECLINE, TEN_AM))
    assert again.recorded is False and "Already" in again.message
    assert len(decisions_mod.load(tmp_path / "decisions.jsonl")) == 1


def test_press_after_expiry_is_refused(tmp_path):
    v = view(tmp_path)
    late = CANCEL_1030 + pd.Timedelta(minutes=1)
    result = asyncio.run(v.decide(OWNER, "farhan", decisions_mod.DECLINE, late))
    assert result.recorded is False and "expired" in result.message.lower()


def test_expiry_is_recorded_only_when_nobody_pressed(tmp_path):
    undecided = view(tmp_path)
    result = undecided.expire(CANCEL_1030)
    assert result.recorded and result.embed_title == "EXPIRED"

    decided = view(tmp_path)
    decided.ticket_id = "t2"
    asyncio.run(decided.decide(OWNER, "farhan", decisions_mod.DECLINE, TEN_AM))
    assert decided.expire(CANCEL_1030).recorded is False

    frame = decisions_mod.load(tmp_path / "decisions.jsonl")
    assert sorted(frame["decision"]) == ["decline", "expired"]


def test_seconds_until_cancel_time():
    from datetime import time

    assert approvals.seconds_until(TEN_AM, time(10, 30)) == 30 * 60
    assert approvals.seconds_until(CANCEL_1030 + pd.Timedelta(hours=1), time(10, 30)) == 1.0


def test_discord_view_reflects_the_specs(tmp_path):
    """The real discord.ui.View carries the same labels and disabled flags."""
    async def go():
        dv = view(tmp_path).as_discord_view(TEN_AM)
        return [(c.label, c.disabled) for c in dv.children], dv.timeout

    children, timeout = asyncio.run(go())
    assert children == [("Execute — unavailable (strategy rejected)", True),
                        ("Don't trade", False)]
    assert timeout == pytest.approx(30 * 60)


# -- 3. buttons: through the desk -------------------------------------------

def runner(status, mode, name="orb2"):
    from datetime import time

    class Silent:
        def generate_signals(self, bars):  # pragma: no cover - not driven here
            raise AssertionError

    return desk_mod.StrategyRunner(name=name, strategy=Silent(), status=status,
                                   mode=mode, entry_cancel_time=time(10, 30))


def test_shadow_signal_is_simulated_immediately_with_disabled_execute(tmp_path):
    desk = build(tmp_path, [runner("rejected", broker.MODE_SHADOW)])
    asyncio.run(desk.handle_signal(make_signal(), TEN_AM))
    assert len(desk.state.open_positions) == 1          # simulated, as before
    assert desk.pending == {}
    v = desk.ticket_view(make_signal(), "rejected", "x", TEN_AM)
    assert v.executable is False
    assert v.expires_at == CANCEL_1030


def test_paper_signal_waits_for_execute(tmp_path, monkeypatch):
    desk = build(tmp_path, [runner("paper", broker.MODE_PAPER)])
    monkeypatch.setattr(desk, "_status_for", lambda name: "paper")
    asyncio.run(desk.handle_signal(make_signal(mode=broker.MODE_PAPER), TEN_AM))

    assert desk.state.open_positions == {}               # nothing routed yet
    assert len(desk.pending) == 1
    assert not desk.journal_path.exists()                # no journal row yet
    (ticket_id,) = desk.pending
    assert f"`{ticket_id}`" in desk.poster.posted[-1][1]
    assert "awaiting your decision" in desk.poster.posted[-1][1]


def test_execute_pending_routes_under_the_posted_id(tmp_path, monkeypatch):
    desk = build(tmp_path, [runner("paper", broker.MODE_PAPER)])
    monkeypatch.setattr(desk, "_status_for", lambda name: "paper")
    asyncio.run(desk.handle_signal(make_signal(mode=broker.MODE_PAPER), TEN_AM))
    (ticket_id,) = desk.pending

    note = asyncio.run(desk.execute_pending(ticket_id))
    assert "routed to paper" in note
    assert ticket_id in desk.state.open_positions
    rows = store.load_open(desk.journal_path)
    assert list(rows["ticket_id"]) == [ticket_id]
    assert desk.pending == {}


def test_live_approval_without_a_live_adapter_is_refused_not_paper_filled(tmp_path, monkeypatch):
    desk = build(tmp_path, [runner("live", broker.MODE_PAPER)])
    monkeypatch.setattr(desk, "_status_for", lambda name: "live")
    asyncio.run(desk.handle_signal(make_signal(mode=broker.MODE_PAPER), TEN_AM))
    (ticket_id,) = desk.pending
    note = asyncio.run(desk.execute_pending(ticket_id))
    assert note.startswith("NOT routed")
    assert "no live adapter" in note
    assert desk.state.open_positions == {}


def test_execute_on_shadow_desk_ticket_has_no_callback(tmp_path):
    desk = build(tmp_path, [runner("rejected", broker.MODE_SHADOW)])
    v = desk.ticket_view(make_signal(), "rejected", "t9", TEN_AM)
    assert v.on_execute is None
    assert v.button_specs()[0]["disabled"] is True


# -- 3. decisions are reported, not counted -------------------------------

def test_decisions_do_not_count_toward_the_entry_3_gate(tmp_path):
    """Pre-registered gate: trades through pretrade.py. A click is not one."""
    # Two modules are called `review`; a bare import gets backtests' one.
    # Load journal's by path, the way strategies/registry.py does.
    from registry import _journal_review

    journal_review = _journal_review()

    path = tmp_path / "decisions.jsonl"
    for i in range(70):
        decisions_mod.append(decisions_mod.Decision(
            ticket_id=f"t{i}", strategy="orb2", strategy_status="rejected",
            mode="shadow", decision="decline", user_id=OWNER, user_name="farhan",
            decided_at=TEN_AM.isoformat(), instrument="MES", direction="long",
            entry_price=6800.0, stop_price=6790.0,
        ), path)
    assert len(decisions_mod.load(path)) == 70

    empty_manual = store.load_closed(tmp_path / "trades.jsonl") \
        if (tmp_path / "trades.jsonl").exists() else pd.DataFrame()
    r = journal_review.readiness(empty_manual)
    assert r["clean_trades"] == 0
    assert r["ready"] is False


def test_review_reports_decisions_in_their_own_section(tmp_path):
    path = tmp_path / "decisions.jsonl"
    decisions_mod.append(decisions_mod.Decision(
        ticket_id="t1", strategy="orb2", strategy_status="rejected", mode="shadow",
        decision="decline", user_id=OWNER, user_name="farhan",
        decided_at=TEN_AM.isoformat(), instrument="MES", direction="long",
        entry_price=6800.0, stop_price=6790.0,
    ), path)
    text = decisions_mod.format_decisions(path)
    assert "OPERATOR DECISIONS" in text
    assert "declined          1" in text
    assert "farhan" in text
    assert "do NOT count toward" in text


def test_latest_per_ticket_keeps_the_human_decision_over_a_later_expiry(tmp_path):
    path = tmp_path / "decisions.jsonl"
    common = dict(strategy="orb2", strategy_status="rejected", mode="shadow",
                  instrument="MES", direction="long", entry_price=1.0, stop_price=0.5)
    decisions_mod.append(decisions_mod.Decision(
        ticket_id="t1", decision="decline", user_id=OWNER, user_name="farhan",
        decided_at=TEN_AM.isoformat(), **common), path)
    decisions_mod.append(decisions_mod.Decision(
        ticket_id="t1", decision="expired", user_id=None, user_name="(timeout)",
        decided_at=CANCEL_1030.isoformat(), **common), path)
    latest = decisions_mod.latest_per_ticket(decisions_mod.load(path))
    assert list(latest["decision"]) == ["decline"]
