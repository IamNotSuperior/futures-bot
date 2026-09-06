"""Shadow labelling, journal separation, and the adapter's refusal.

Three properties, each of which is load-bearing for a different reason:

* **Labelling** - a shadow ticket must say it came from a rejected strategy,
  everywhere it appears. A ticket that does not is one screenshot away from
  being read as a result.
* **Journal separation** - shadow tickets must never reach
  ``journal/trades.jsonl``. Entry 3's gate counts that file, and a contaminated
  count licenses buying an evaluation on evidence that does not exist.
* **Adapter refusal** - no strategy below ``live`` status reaches a broker,
  whatever the account flag says.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import broker
import desk_state
import pretrade
import rules
import store
import tickets as tickets_mod
from registry import Registry


@pytest.fixture
def shadow_journal(tmp_path: Path) -> Path:
    return tmp_path / "shadow_trades.jsonl"


@pytest.fixture
def manual_journal(tmp_path: Path) -> Path:
    path = tmp_path / "trades.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def make_signal(mode=broker.MODE_SHADOW, **overrides) -> tickets_mod.Signal:
    base = dict(
        strategy="orb2", instrument="MES", direction="long",
        entry_price=6800.0, stop_price=6790.0, target_price=6818.0,
        contracts=1, bar_time=pd.Timestamp("2026-08-24 10:05", tz=rules.ET),
        mode=mode,
    )
    base.update(overrides)
    return tickets_mod.Signal(**base)


def place(signal, journal, account=None):
    now = signal.bar_time
    decision = tickets_mod.evaluate_signal(signal, pretrade.AccountState(), now)
    assert decision.allowed, decision.blocks
    return tickets_mod.submit(
        signal, decision, broker.PaperAdapter(),
        account or broker.AccountConfig(journal_path=journal),
        "rejected", now, journal,
    )


# -- labelling --------------------------------------------------------------

def test_shadow_ticket_carries_the_label(shadow_journal):
    outcome = place(make_signal(), shadow_journal)
    assert outcome.allowed
    assert outcome.label == broker.SHADOW_LABEL
    assert "rejected" in outcome.label.lower()
    assert "no orders" in outcome.label.lower()


def test_label_is_written_into_the_journal_row(shadow_journal):
    place(make_signal(), shadow_journal)
    rows = [json.loads(line) for line in
            shadow_journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    checks = rows[0]["checks"]
    assert checks["label"] == broker.SHADOW_LABEL
    assert checks["mode"] == broker.MODE_SHADOW
    assert checks["strategy"] == "orb2"
    assert checks["simulated"] == "True"


def test_paper_mode_ticket_is_not_labelled_shadow(shadow_journal):
    outcome = place(make_signal(mode=broker.MODE_PAPER), shadow_journal)
    assert outcome.label == ""
    assert outcome.fill.note == ""


def test_fill_is_always_simulated(shadow_journal):
    outcome = place(make_signal(), shadow_journal)
    assert outcome.fill.simulated is True
    assert outcome.fill.adapter == "paper"


def test_adapter_does_not_apply_slippage_itself(shadow_journal):
    """The fill is the signal level. ``price_trades`` owns the cost model.

    Regression test. The adapter used to slip the entry by a tick, and
    ``engine.price_trades`` slips it again - so every shadow ticket carried
    $1.25 per contract of phantom cost. Comparability with backtests is the
    whole reason the journal routes through the engine, and a second slippage
    application quietly breaks it.
    """
    for direction, stop in (("long", 6790.0), ("short", 6810.0)):
        fill = place(make_signal(direction=direction, stop_price=stop,
                                 target_price=None), shadow_journal).fill
        assert fill.fill_price == 6800.0, (
            f"{direction} fill was adjusted to {fill.fill_price}; the cost "
            f"model applies slippage, the adapter must not"
        )


def test_closed_shadow_pnl_matches_the_engine_exactly(shadow_journal):
    """One trade, hand-checked against engine.price_trades.

    A 10-point loss on 1 MES at 1 tick a side: entry fills at 6800.25, exit at
    6789.75, so 10.5 points against at $5 a point is -$52.50, less a $2.50
    round turn = **-$55.00**. Before the double-count was fixed this produced
    -$56.25, and the replay posted that figure to Discord.
    """
    signal = make_signal(direction="long", entry_price=6800.0,
                         stop_price=6790.0, target_price=None)
    outcome = place(signal, shadow_journal)
    assert outcome.ticket.entry_price == 6800.0

    closed = tickets_mod.close_position(
        outcome.ticket.ticket_id, 6790.0,
        signal.bar_time + pd.Timedelta(minutes=5), "stop", shadow_journal)

    assert closed.entry_price == 6800.0
    assert closed.net_pnl == pytest.approx(-55.00)
    assert closed.commission == pytest.approx(2.50)
    assert closed.min_hold_ok is True


# -- journal separation -----------------------------------------------------

def test_shadow_tickets_never_touch_the_manual_journal(shadow_journal, manual_journal):
    for _ in range(3):
        place(make_signal(), shadow_journal)
    assert manual_journal.read_text(encoding="utf-8") == ""
    assert store.net_open_position(manual_journal) == 0


def test_entry_3_count_is_untouched_by_shadow_activity(shadow_journal, manual_journal):
    """The 60-trade gate reads the manual journal and must stay at zero."""
    before = len(store.load_closed(manual_journal))
    for _ in range(5):
        place(make_signal(), shadow_journal)
    assert len(store.load_closed(manual_journal)) == before == 0


def test_shadow_journal_default_is_a_separate_file():
    assert tickets_mod.SHADOW_JOURNAL != store.TRADES_PATH
    assert tickets_mod.SHADOW_JOURNAL.name == "shadow_trades.jsonl"


def test_blocked_signal_writes_nothing(shadow_journal):
    """A block leaves no ticket - same position pretrade.py takes."""
    late = pd.Timestamp("2026-08-24 16:45", tz=rules.ET)
    signal = make_signal(bar_time=late)
    decision = tickets_mod.evaluate_signal(signal, pretrade.AccountState(), late)
    assert not decision.allowed
    outcome = tickets_mod.submit(
        signal, decision, broker.PaperAdapter(),
        broker.AccountConfig(journal_path=shadow_journal),
        "rejected", late, shadow_journal,
    )
    assert outcome.ticket is None
    assert not shadow_journal.exists() or shadow_journal.read_text(encoding="utf-8") == ""


def test_live_account_cannot_be_configured_onto_the_shadow_journal():
    with pytest.raises(ValueError, match="shadow journal"):
        broker.AccountConfig(name="acct", live_enabled=True,
                             journal_path=tickets_mod.SHADOW_JOURNAL)


# -- adapter refusal --------------------------------------------------------

@pytest.mark.parametrize("status", ["rejected", "proposed", "testing", "paper"])
def test_adapter_refuses_any_strategy_below_live(status):
    """Even with the account flag ON. The status check runs first."""
    account = broker.AccountConfig(name="acct", live_enabled=True,
                                   journal_path=Path("nonshadow.jsonl"))
    with pytest.raises(broker.BrokerRefusal, match="not 'live'"):
        broker.require_live_eligible(status, account, "orb2")


def test_adapter_refuses_live_status_when_the_flag_is_off():
    account = broker.AccountConfig(name="acct", live_enabled=False,
                                   journal_path=Path("nonshadow.jsonl"))
    with pytest.raises(broker.BrokerRefusal, match="live_enabled=False"):
        broker.require_live_eligible("live", account, "someday")


def test_live_flag_defaults_to_off():
    assert broker.AccountConfig().live_enabled is False


def test_status_is_checked_before_the_flag():
    """The refusal message names the status, not the flag.

    Order matters: the flag must never be the thing standing between a rejected
    strategy and a broker, and the message the operator sees should say so.
    """
    account = broker.AccountConfig(name="acct", live_enabled=True,
                                   journal_path=Path("nonshadow.jsonl"))
    with pytest.raises(broker.BrokerRefusal) as exc:
        broker.require_live_eligible("rejected", account, "orb2")
    assert "orb2 is at 'rejected' status" in str(exc.value)
    assert "does not change this" in str(exc.value)


def test_no_live_adapter_exists():
    """There is no TradersPost implementation, not even a stub."""
    with pytest.raises(broker.BrokerRefusal, match="no live adapter"):
        broker.adapter_for(broker.MODE_LIVE)
    adapters = {c.__name__ for c in broker.BrokerAdapter.__subclasses__()}
    assert adapters == {"PaperAdapter"}, (
        f"a second adapter appeared: {adapters}. HANDOFF.md §6 gates live "
        f"routing on a strategy reaching 'paper' status."
    )


def test_paper_adapter_refuses_a_live_order(shadow_journal):
    order = broker.Order(
        strategy="orb2", instrument="MES", direction="long", contracts=1,
        entry_price=6800.0, stop_price=6790.0, target_price=None,
        signal_time=pd.Timestamp("2026-08-24 10:05", tz=rules.ET),
        mode=broker.MODE_LIVE,
    )
    with pytest.raises(broker.BrokerRefusal):
        broker.PaperAdapter().submit(
            order, broker.AccountConfig(journal_path=shadow_journal), "rejected")


def test_adapter_enforces_the_instrument_allowlist(shadow_journal):
    order = broker.Order(
        strategy="orb2", instrument="ES", direction="long", contracts=1,
        entry_price=6800.0, stop_price=6790.0, target_price=None,
        signal_time=pd.Timestamp("2026-08-24 10:05", tz=rules.ET),
        mode=broker.MODE_SHADOW,
    )
    with pytest.raises(rules.RuleViolation):
        broker.PaperAdapter().submit(
            order, broker.AccountConfig(journal_path=shadow_journal), "rejected")


def test_adapter_enforces_the_position_cap(shadow_journal):
    order = broker.Order(
        strategy="orb2", instrument="MES", direction="long",
        contracts=rules.POSITION_CAP + 1,
        entry_price=6800.0, stop_price=6790.0, target_price=None,
        signal_time=pd.Timestamp("2026-08-24 10:05", tz=rules.ET),
        mode=broker.MODE_SHADOW,
    )
    with pytest.raises(rules.RuleViolation):
        broker.PaperAdapter().submit(
            order, broker.AccountConfig(journal_path=shadow_journal), "rejected")


# -- the registry's shadow list --------------------------------------------

def test_shadow_is_not_a_status():
    from registry import STATUSES

    assert "shadow" not in STATUSES


def test_orb2_is_listed_for_shadow_and_still_rejected():
    reg = Registry.load()
    assert "orb2" in reg.shadow_names()
    assert reg.get("orb2").status == "rejected"
    assert reg.verify() == []


def test_shadow_listing_does_not_permit_promotion():
    reg = Registry.load()
    with pytest.raises(Exception, match="terminal|rejected"):
        reg.promote("orb2", "paper")


def test_desk_runs_shadow_entries_in_shadow_mode():
    reg = Registry.load()
    modes = dict((r.name, m) for r, m in reg.desk_strategies())
    assert modes.get("orb2") == "shadow"
    for name, mode in modes.items():
        if mode == "live-eligible":
            assert reg.get(name).status in ("paper", "live")


def test_verify_flags_a_shadow_name_that_does_not_exist(tmp_path):
    from registry import RegistryError  # noqa: F401

    reg = Registry.load()
    reg._shadow = ("orb2", "does_not_exist")
    problems = reg.verify()
    assert any("does_not_exist" in p for p in problems)


def test_verify_flags_a_live_strategy_listed_for_shadow():
    reg = Registry.load()
    record = reg.get("orb2")
    reg._records["orb2"] = type(record)(**{**record.__dict__, "status": "live"})
    problems = reg.verify()
    assert any("live" in p and "shadow" in p for p in problems)


def test_shadow_key_rejects_a_status_name(tmp_path):
    from registry import RegistryError

    path = tmp_path / "registry.yaml"
    path.write_text(
        "strategies:\n"
        "  - name: x\n    class_path: null\n    hypothesis_entry: 1\n"
        "    status: rejected\n"
        "shadow:\n  - paper\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="not a status"):
        Registry.load(path)


def test_save_preserves_the_shadow_list(tmp_path):
    """A promote() elsewhere must not silently empty the desk's run set."""
    import yaml

    src = Registry.load()
    path = tmp_path / "registry.yaml"
    copy = Registry(src.all(), path, src.hypotheses_path, src.shadow_names())
    copy.save()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["shadow"] == ["orb2"]


# -- state file -------------------------------------------------------------

def test_desk_state_round_trips(tmp_path):
    path = tmp_path / "desk_state.json"
    state = desk_state.DeskState()
    state.add_position(desk_state.OpenPosition(
        ticket_id="t1", strategy="orb2", instrument="MES", direction="long",
        contracts=2, entry_price=6800.0, stop_price=6790.0, target_price=6818.0,
        entry_time=pd.Timestamp("2026-08-24 10:05", tz=rules.ET).isoformat(),
        mode=broker.MODE_SHADOW, session_date="2026-08-24",
    ))
    state.tickets_allowed = 3
    state.save(path)

    loaded = desk_state.DeskState.load(path)
    assert loaded.tickets_allowed == 3
    assert loaded.net_position() == 2
    assert loaded.net_position("MNQ") == 0
    assert loaded.open_positions["t1"].strategy == "orb2"


def test_corrupt_state_file_starts_fresh_rather_than_crashing(tmp_path):
    path = tmp_path / "desk_state.json"
    path.write_text("{not json", encoding="utf-8")
    assert desk_state.DeskState.load(path).open_positions == {}


def test_state_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "desk_state.json"
    for _ in range(3):
        desk_state.DeskState().save(path)
    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_state_file_is_utf8(tmp_path):
    """Windows write_text defaults to cp1252 and truncates before raising."""
    path = tmp_path / "desk_state.json"
    state = desk_state.DeskState()
    state.add_position(desk_state.OpenPosition(
        ticket_id="t1", strategy="orb2 – en-dash", instrument="MES",
        direction="long", contracts=1, entry_price=1.0, stop_price=0.5,
        target_price=None, entry_time="2026-08-24T10:05:00-04:00",
        mode=broker.MODE_SHADOW, session_date="2026-08-24",
    ))
    state.save(path)
    assert "–" in path.read_text(encoding="utf-8")
    # The property that actually matters: the value survives the round trip.
    assert desk_state.DeskState.load(path).open_positions["t1"].strategy \
        == "orb2 – en-dash"
