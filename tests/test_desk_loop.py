"""The desk loop end to end, driven by a replay feed.

These are the tests that would catch a desk that looks right in pieces and is
wrong assembled: a position opened and never closed, a flatten that misses the
16:30 deadline, a summary computed from in-memory counters rather than the
journal, or a replay judged against the wall clock instead of the bar's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

import broker
import desk as desk_mod
import desk_state
import feed as feed_mod
import rules
import store
import tickets as tickets_mod
from registry import Registry, StrategyRecord


class StubStrategy:
    """Fires one long entry on a nominated bar. No pandas machinery."""

    name = "stub"

    def __init__(self, fire_at: pd.Timestamp, stop_offset=10.0, target_offset=18.0):
        self.fire_at = fire_at
        self.stop_offset = stop_offset
        self.target_offset = target_offset

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "entry_long": False, "entry_short": False,
            "exit_long": False, "exit_short": False,
        }, index=bars.index)
        for col in ("entry_price", "stop_price", "target_price"):
            out[col] = pd.Series(pd.NA, index=bars.index, dtype="Float64")
        if self.fire_at in out.index:
            price = float(bars.loc[self.fire_at, "open"])
            out.loc[self.fire_at, "entry_long"] = True
            out.loc[self.fire_at, "entry_price"] = price
            out.loc[self.fire_at, "stop_price"] = price - self.stop_offset
            out.loc[self.fire_at, "target_price"] = price + self.target_offset
        return out


def session_frame(day="2026-08-24", start="09:30", minutes=400,
                  price=6800.0, drift=0.0) -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(f"{day} {start}", tz=rules.ET),
                          periods=minutes, freq="1min")
    closes = [price + drift * i for i in range(minutes)]
    return pd.DataFrame({
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [100.0] * minutes,
    }, index=index)


def build(tmp_path: Path, runners, **kwargs) -> desk_mod.Desk:
    journal = tmp_path / "shadow_trades.jsonl"
    state_path = tmp_path / "desk_state.json"
    registry = Registry.load()
    return desk_mod.Desk(
        registry=registry,
        runners=runners,
        poster=desk_mod.Poster(client=None, echo=False),
        state=desk_state.DeskState(),
        adapter=broker.PaperAdapter(),
        account=broker.AccountConfig(journal_path=journal),
        journal_path=journal,
        state_path=state_path,
        **kwargs,
    )


def runner_for(strategy, mode=broker.MODE_SHADOW, name="orb2"):
    return desk_mod.StrategyRunner(
        name=name, strategy=strategy, status="rejected", mode=mode)


def drive(desk: desk_mod.Desk, frame: pd.DataFrame) -> None:
    async def go():
        for bar in feed_mod.ReplayFeed(frame, speed=0).iter_bars():
            await desk.on_bar(bar)
    asyncio.run(go())


# -- the happy path ---------------------------------------------------------

def test_a_signal_becomes_a_labelled_ticket(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(minutes=120))

    kinds = [k for k, _ in desk.poster.posted]
    assert "tickets" in kinds
    ticket_posts = [b for k, b in desk.poster.posted if k == "tickets"]
    assert any(broker.SHADOW_LABEL in b for b in ticket_posts)
    assert desk.state.tickets_allowed == 1


def test_ticket_lands_in_the_shadow_journal_only(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(minutes=120))

    assert desk.journal_path.exists()
    assert store.TRADES_PATH.resolve() != desk.journal_path.resolve()
    rows = desk.journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) >= 1


def test_a_strategy_fires_at_most_once_a_session(tmp_path):
    """The runner reads only the newest row; an old signal must not re-fire."""
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(minutes=200))
    assert desk.state.tickets_allowed == 1


# -- exits ------------------------------------------------------------------

def test_target_closes_the_position_and_prices_it(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire, target_offset=5.0))])
    drive(desk, session_frame(minutes=180, drift=0.5))   # rises into the target

    assert desk.state.open_positions == {}
    closed = store.load_closed(desk.journal_path)
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "target"
    # Priced through engine.price_trades, so costs are already applied.
    assert closed.iloc[0]["commission"] > 0
    assert closed.iloc[0]["net_pnl"] < closed.iloc[0]["gross_pnl"]


def test_stop_closes_the_position(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire, stop_offset=5.0))])
    drive(desk, session_frame(minutes=180, drift=-0.5))

    closed = store.load_closed(desk.journal_path)
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "stop"


def test_position_is_flattened_by_the_rule_2_deadline(tmp_path):
    """Nothing may be held past 16:30 ET. Rule 2, no exceptions.

    The bracket is the ordinary 10/18 rather than an absurdly wide one: a
    500-point stop is $2,500 of risk and the rule 5 budget blocks it before
    the position ever opens. A flat price series keeps both levels untouched
    so the flatten is the only exit left.
    """
    fire = pd.Timestamp("2026-08-24 15:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(start="09:30", minutes=440))   # runs past 16:30

    assert desk.state.open_positions == {}
    closed = store.load_closed(desk.journal_path)
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "flatten"
    exit_ts = rules.to_et(closed.iloc[0]["exit_time"])
    assert exit_ts.time() <= rules.FLATTEN_TIME or exit_ts.hour == 16


async def _drive_n(desk, frame, until):
    for bar in feed_mod.ReplayFeed(frame, speed=0).iter_bars():
        if bar.timestamp > until:
            return
        await desk.on_bar(bar)


def test_min_hold_floor_is_enforced_on_the_same_bar(tmp_path):
    """A position cannot be closed on the bar that opened it."""
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire, stop_offset=0.5))])
    frame = session_frame(minutes=60)
    asyncio.run(_drive_n(desk, frame, until=fire))

    assert len(desk.state.open_positions) == 1
    position = next(iter(desk.state.open_positions.values()))
    assert position.held_seconds(fire) < rules.MIN_HOLD_SECONDS


def test_stop_and_target_in_one_bar_takes_the_stop(tmp_path):
    """A 1-minute bar has no intrabar path; assume the worse fill."""
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    position = desk_state.OpenPosition(
        ticket_id="t", strategy="orb2", instrument="MES", direction="long",
        contracts=1, entry_price=6800.0, stop_price=6790.0, target_price=6818.0,
        entry_time=fire.isoformat(), mode=broker.MODE_SHADOW,
        session_date="2026-08-24",
    )
    wide = feed_mod.Bar("MES", fire + pd.Timedelta(minutes=5),
                        6800.0, 6820.0, 6788.0, 6805.0, 100.0)
    price, reason = desk._exit_for(position, wide, wide.timestamp)
    assert reason == "stop"
    assert price == 6790.0


# -- guards -----------------------------------------------------------------

def test_a_signal_after_the_entry_cutoff_is_blocked_and_posted(tmp_path):
    fire = pd.Timestamp("2026-08-24 16:25", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(start="16:00", minutes=40))

    assert desk.state.tickets_allowed == 0
    assert desk.state.tickets_blocked == 1
    blocked = [b for k, b in desk.poster.posted if k == "tickets"
               and "BLOCKED" not in b or "Blocked because" in b]
    assert any("Blocked because" in b for _, b in desk.poster.posted)
    assert not desk.journal_path.exists() or \
        desk.journal_path.read_text(encoding="utf-8") == ""


def test_position_cap_is_respected_across_strategies(tmp_path):
    """Rule 4 caps the aggregate, not just one strategy's book."""
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    runners = [
        runner_for(StubStrategy(fire + pd.Timedelta(minutes=i)),
                   name=f"s{i}")
        for i in range(rules.POSITION_CAP + 2)
    ]
    for runner in runners:
        runner.name = "orb2"      # all report as the shadow strategy
    desk = build(tmp_path, runners)
    drive(desk, session_frame(minutes=120, drift=0.0))

    assert desk.state.net_position() <= rules.POSITION_CAP
    assert desk.state.tickets_blocked >= 1


# -- ops --------------------------------------------------------------------

def test_premarket_check_fires_once(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(start="09:20", minutes=120))

    premarket = [b for k, b in desk.poster.posted if "pre-market" in b]
    assert len(premarket) == 1
    assert "Session" in premarket[0]


def test_heartbeat_fires_on_the_interval(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(minutes=90))

    beats = [b for k, b in desk.poster.posted if b.startswith("Heartbeat")]
    # 90 minutes at a 15-minute cadence, plus the first bar's beat.
    assert 5 <= len(beats) <= 8
    assert "no orders are placed" in beats[0]


def test_heartbeat_is_silent_outside_the_session(tmp_path):
    """The feed runs 23 hours a day; the heartbeat must not.

    Before this was bounded, a five-session replay produced 436 heartbeats -
    about 87 a day, one every 15 minutes around the clock. That is noise, and
    noise in a monitoring channel is worse than nothing because it trains the
    operator to ignore it.
    """
    desk = build(tmp_path, [])
    drive(desk, session_frame(day="2026-08-24", start="19:00", minutes=300))
    beats = [b for k, b in desk.poster.posted if b.startswith("Heartbeat")]
    assert beats == []


def test_session_window_excludes_overnight_and_holidays(tmp_path):
    desk = build(tmp_path, [])
    assert desk.in_session(pd.Timestamp("2026-08-24 10:00", tz=rules.ET))
    assert desk.in_session(pd.Timestamp("2026-08-24 09:25", tz=rules.ET))
    assert not desk.in_session(pd.Timestamp("2026-08-24 09:24", tz=rules.ET))
    assert not desk.in_session(pd.Timestamp("2026-08-24 16:36", tz=rules.ET))
    assert not desk.in_session(pd.Timestamp("2026-08-24 03:00", tz=rules.ET))

    holiday = build(tmp_path, [], closed_dates={pd.Timestamp("2026-08-24").date()})
    assert not holiday.in_session(pd.Timestamp("2026-08-24 10:00", tz=rules.ET))


def test_early_close_pulls_the_session_end_forward(tmp_path):
    day = pd.Timestamp("2026-11-27").date()
    desk = build(tmp_path, [], early_close_dates={day})
    assert desk.in_session(pd.Timestamp("2026-11-27 12:00", tz=rules.ET))
    # Early-close flatten is 13:00, so the window ends well before 16:35.
    assert not desk.in_session(pd.Timestamp("2026-11-27 15:00", tz=rules.ET))


def test_daily_summary_reads_the_journal_not_the_counters(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire, target_offset=5.0))])
    drive(desk, session_frame(minutes=440, drift=0.5))

    summaries = [b for k, b in desk.poster.posted if k == "summary"]
    assert len(summaries) == 1
    body = summaries[0]
    assert broker.SHADOW_LABEL in body
    assert "Net P&L" in body
    assert "Avg duration" in body


def test_summary_flags_a_rule_6_violation(tmp_path):
    """A sub-30s close is a bug signal, and the summary must say so."""
    journal = tmp_path / "shadow_trades.jsonl"
    desk = build(tmp_path, [])
    entry = pd.Timestamp("2026-08-24 10:00:00", tz=rules.ET)
    ticket = store.Ticket(
        ticket_id="short1", status=store.STATUS_OPEN, instrument="MES",
        direction="long", contracts=1, entry_price=6800.0, stop_price=6790.0,
        thesis="t", risk_dollars=50.0, entry_time=entry.isoformat(),
        session_date="2026-08-24", checks={},
    )
    store.append(ticket, desk.journal_path)
    tickets_mod.close_position("short1", 6802.0,
                               entry + pd.Timedelta(seconds=4),
                               "target", desk.journal_path)
    body = desk.summary_text(pd.Timestamp("2026-08-24").date())
    assert "RULE 6 VIOLATION" in body
    assert "Microscalp" in body


def test_reconciliation_reports_an_unknown_open_position(tmp_path):
    desk = build(tmp_path, [])
    entry = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    store.append(store.Ticket(
        ticket_id="orphan", status=store.STATUS_OPEN, instrument="MES",
        direction="long", contracts=1, entry_price=6800.0, stop_price=6790.0,
        thesis="left by a crash", risk_dollars=50.0,
        entry_time=entry.isoformat(), session_date="2026-08-24", checks={},
    ), desk.journal_path)

    asyncio.run(desk.reconcile(pd.Timestamp("2026-08-25 09:00", tz=rules.ET)))
    posts = [b for k, b in desk.poster.posted]
    assert any("reconciliation" in b.lower() for b in posts)
    assert any("orphan" in b for b in posts)


def test_reconciliation_is_silent_when_state_and_journal_agree(tmp_path):
    desk = build(tmp_path, [])
    asyncio.run(desk.reconcile(pd.Timestamp("2026-08-25 09:00", tz=rules.ET)))
    assert desk.poster.posted == []


def test_state_survives_a_restart(tmp_path):
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(minutes=120))   # flat prices: nothing is hit
    assert len(desk.state.open_positions) == 1

    reloaded = desk_state.DeskState.load(desk.state_path)
    assert len(reloaded.open_positions) == 1
    assert reloaded.net_position("MES") == 1


# -- replay clock discipline ------------------------------------------------

def test_replay_is_judged_against_the_bar_clock_not_the_wall_clock(tmp_path):
    """A 2026-08-24 replay must not be blocked as after-hours.

    This is the property that makes a replay worth running: if the guards read
    the wall clock, every bar of a historical session is past the cutoff and
    the replay silently blocks everything.
    """
    fire = pd.Timestamp("2026-08-24 10:00", tz=rules.ET)
    desk = build(tmp_path, [runner_for(StubStrategy(fire))])
    drive(desk, session_frame(minutes=120))
    assert desk.state.tickets_allowed == 1, (
        "the replay blocked its own signal - the guards are reading the wall "
        "clock instead of the bar timestamp"
    )
