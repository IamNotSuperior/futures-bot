"""The desk's guard path is the manual one, not a copy of it.

The property under test is equivalence: for the same request, state and clock,
``tickets.evaluate_signal`` must produce exactly what ``pretrade.evaluate``
produces. If someone adds a check to one path and not the other - which is the
realistic way this breaks - these tests fail.

CLAUDE.md rule 9 wants the limits enforced by runtime logic with no bypass
path, and HANDOFF.md records that restating a threshold has already caused a
real bug once (``enforce_daily_loss_limit`` carried a hardcoded ``limit=300.0``).
A second copy of the checks in ``bots/`` would be that bug again.
"""

from __future__ import annotations

import pandas as pd
import pytest

import broker
import pretrade
import rules
import tickets as tickets_mod


def make_signal(**overrides) -> tickets_mod.Signal:
    base = dict(
        strategy="orb2",
        instrument="MES",
        direction="long",
        entry_price=6800.0,
        stop_price=6790.0,
        target_price=6818.0,
        contracts=1,
        bar_time=pd.Timestamp("2026-08-24 10:05", tz=rules.ET),
        mode=broker.MODE_SHADOW,
    )
    base.update(overrides)
    return tickets_mod.Signal(**base)


FLAT = pretrade.AccountState()


def test_evaluate_signal_matches_pretrade_exactly():
    """The two paths return the same decision, check for check."""
    signal = make_signal()
    now = signal.bar_time

    desk_decision = tickets_mod.evaluate_signal(signal, FLAT, now)
    manual_decision = pretrade.evaluate(signal.as_request(), FLAT, now)

    assert desk_decision.allowed == manual_decision.allowed
    assert desk_decision.blocks == manual_decision.blocks
    assert desk_decision.checks == manual_decision.checks
    assert desk_decision.risk_dollars == pytest.approx(manual_decision.risk_dollars)


@pytest.mark.parametrize("when,expect_allowed", [
    ("2026-08-24 10:05", True),    # mid-session
    ("2026-08-24 16:19", True),    # one minute inside the entry cutoff
    ("2026-08-24 16:21", False),   # rule 3: past 16:20
    ("2026-08-24 16:31", False),   # rule 2: past the flatten
])
def test_entry_cutoff_is_enforced_on_the_desk_path(when, expect_allowed):
    """Rule 3 rejects a late signal rather than deferring it."""
    now = pd.Timestamp(when, tz=rules.ET)
    signal = make_signal(bar_time=now)
    decision = tickets_mod.evaluate_signal(signal, FLAT, now)
    assert decision.allowed is expect_allowed
    if not expect_allowed:
        assert decision.blocks, "a blocked decision must say why"


def test_premarket_signal_is_not_blocked_by_rules_py():
    """**Known gap, pinned deliberately.** ``rules.py`` has no session-open guard.

    HANDOFF.md §5 records this: ``is_entry_allowed`` returns True at 03:00 ET
    because 03:00 is numerically before a cutoff designed for the afternoon,
    not because the module models a session open. There is one RTH session per
    calendar day and no lower bound on it.

    The desk does not currently expose the gap - ``orb2`` slices to
    ``between_time("09:30", "15:59")`` and cannot produce a pre-market signal -
    so nothing here is unsafe today. This test exists so that the day a
    strategy *can* signal at 08:00, the failure is a red test rather than a
    surprise fill. If ``rules.py`` gains a session-open guard, invert it.
    """
    now = pd.Timestamp("2026-08-24 08:00", tz=rules.ET)
    decision = tickets_mod.evaluate_signal(make_signal(bar_time=now), FLAT, now)
    assert decision.allowed is True, (
        "rules.py gained a session-open guard - good. Update this test and the "
        "note in HANDOFF.md §5."
    )
    assert rules.is_entry_allowed(now) is True


def test_late_signal_block_names_the_cutoff():
    now = pd.Timestamp("2026-08-24 16:25", tz=rules.ET)
    decision = tickets_mod.evaluate_signal(make_signal(bar_time=now), FLAT, now)
    joined = " ".join(decision.blocks)
    assert "cutoff" in joined.lower() or "flatten" in joined.lower()


def test_instrument_allowlist_rejects_anything_outside_rule_1():
    now = pd.Timestamp("2026-08-24 10:05", tz=rules.ET)
    for symbol in ("ES", "NQ", "CL", "SPY"):
        decision = tickets_mod.evaluate_signal(
            make_signal(instrument=symbol), FLAT, now)
        assert decision.allowed is False
        assert "not permitted" in " ".join(decision.blocks)


def test_position_cap_blocks_beyond_the_internal_limit():
    """Rule 4: the desk clamps at INTERNAL, never at the firm's 40."""
    now = pd.Timestamp("2026-08-24 10:05", tz=rules.ET)
    state = pretrade.AccountState(open_position=rules.POSITION_CAP)
    decision = tickets_mod.evaluate_signal(make_signal(), state, now)
    assert decision.allowed is False
    assert "cap" in " ".join(decision.blocks).lower()


def test_daily_loss_limit_blocks_at_internal_not_firm():
    now = pd.Timestamp("2026-08-24 10:05", tz=rules.ET)
    state = pretrade.AccountState(today_realised_pnl=-rules.DAILY_LOSS_LIMIT)
    decision = tickets_mod.evaluate_signal(make_signal(), state, now)
    assert decision.allowed is False
    assert decision.allowed is not True
    # The firm's line is $1,200; stopping there would mean a guard above it
    # failed. Confirm we stopped at the internal number.
    assert rules.DAILY_LOSS_LIMIT < rules.FIRM.daily_loss


def test_roll_day_blocks_entries():
    now = pd.Timestamp("2026-08-24 10:05", tz=rules.ET)
    decision = tickets_mod.evaluate_signal(
        make_signal(), FLAT, now, roll_dates={now.date()})
    assert decision.allowed is False
    assert "roll" in " ".join(decision.blocks).lower()


def test_exchange_holiday_blocks_entries():
    now = pd.Timestamp("2026-08-24 10:05", tz=rules.ET)
    decision = tickets_mod.evaluate_signal(
        make_signal(), FLAT, now, closed_dates={now.date()})
    assert decision.allowed is False
    assert "holiday" in " ".join(decision.blocks).lower()


def test_signal_generates_a_thesis_so_the_row_is_reviewable():
    """pretrade blocks an empty thesis; the desk must not trip its own guard."""
    request = make_signal().as_request()
    assert request.thesis.strip()
    assert "orb2" in request.thesis


def test_evaluate_signal_adds_no_checks_of_its_own():
    """A check here that pretrade does not have means the paths have diverged."""
    signal = make_signal()
    now = signal.bar_time
    desk_keys = set(tickets_mod.evaluate_signal(signal, FLAT, now).checks)
    manual_keys = set(pretrade.evaluate(signal.as_request(), FLAT, now).checks)
    assert desk_keys == manual_keys
