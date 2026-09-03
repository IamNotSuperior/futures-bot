"""Tests for the pre-trade ALLOW/BLOCK decision.

The boundaries mirror tests/test_rules.py exactly - 16:19 versus 16:21, the
early-close 12:49/12:50 pair, the position cap edge, the daily-loss edge - so
that a rule and its gate cannot drift apart. Every threshold is read from
`rules`; none is restated here.
"""

from datetime import date

import pandas as pd
import pytest

import rules
import store
from close import close_ticket
from pretrade import (
    AccountState, Decision, TicketRequest, build_ticket, evaluate,
)

ET = rules.ET
SUMMER_DAY = "2025-07-16"      # regular Wednesday
EARLY = date(2025, 7, 3)       # 13:00 ET close
ROLL = date(2025, 6, 18)


def et(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz=ET)


def req(**kwargs) -> TicketRequest:
    """A normally-sized, well-formed MES long unless overridden."""
    base = dict(
        instrument="MES",
        direction="long",
        entry_price=6800.00,
        stop_price=6795.00,   # 5 points = $25/contract + costs
        contracts=1,
        thesis="reclaimed the range low",
    )
    base.update(kwargs)
    return TicketRequest(**base)


def flat() -> AccountState:
    return AccountState()


class TestInstrument:
    @pytest.mark.parametrize("symbol", ["MES", "MNQ", "mes"])
    def test_allowed(self, symbol):
        d = evaluate(req(instrument=symbol), flat(), et(f"{SUMMER_DAY} 10:00"))
        assert d.allowed is True

    @pytest.mark.parametrize("symbol", ["ES", "NQ", "MYM", "CL"])
    def test_blocked(self, symbol):
        d = evaluate(req(instrument=symbol), flat(), et(f"{SUMMER_DAY} 10:00"))
        assert d.allowed is False
        assert "not permitted" in d.blocks[0]


class TestEntryCutoff:
    """Same instants as TestEntryCutoff in test_rules.py."""

    @pytest.mark.parametrize(
        "clock, allowed",
        [("09:30", True), ("16:00", True), ("16:19", True),
         ("16:20", False), ("16:21", False)],
    )
    def test_boundaries(self, clock, allowed):
        d = evaluate(req(), flat(), et(f"{SUMMER_DAY} {clock}"))
        assert d.allowed is allowed

    def test_4_19_allows_and_4_21_blocks(self):
        assert evaluate(req(), flat(), et(f"{SUMMER_DAY} 16:19")).allowed is True
        blocked = evaluate(req(), flat(), et(f"{SUMMER_DAY} 16:21"))
        assert blocked.allowed is False
        assert any("cutoff" in b for b in blocked.blocks)

    def test_after_the_flatten_deadline_also_blocks(self):
        d = evaluate(req(), flat(), et(f"{SUMMER_DAY} 16:31"))
        assert d.allowed is False
        assert any("flatten" in b for b in d.blocks)


class TestEarlyClose:
    """Same 12:49/12:50 boundary as the rules tests."""

    @pytest.mark.parametrize(
        "clock, allowed", [("12:49", True), ("12:50", False), ("12:51", False)]
    )
    def test_cutoff_moves(self, clock, allowed):
        d = evaluate(req(), flat(), et(f"{EARLY} {clock}"), early_close_dates={EARLY})
        assert d.allowed is allowed

    def test_without_the_flag_a_late_entry_looks_legal(self):
        """The failure mode the --early-close flag exists to prevent."""
        assert evaluate(req(), flat(), et(f"{EARLY} 15:00")).allowed is True
        assert evaluate(req(), flat(), et(f"{EARLY} 15:00"),
                        early_close_dates={EARLY}).allowed is False

    def test_block_names_the_early_close(self):
        d = evaluate(req(), flat(), et(f"{EARLY} 13:30"), early_close_dates={EARLY})
        assert any("early-close" in b for b in d.blocks)


class TestRollDay:
    def test_roll_day_blocks(self):
        d = evaluate(req(), flat(), et(f"{ROLL} 10:00"), roll_dates={ROLL})
        assert d.allowed is False
        assert any("roll" in b for b in d.blocks)

    def test_neighbouring_days_allowed(self):
        for day in ("2025-06-17", "2025-06-19"):
            assert evaluate(req(), flat(), et(f"{day} 10:00"),
                            roll_dates={ROLL}).allowed is True


class TestPositionCap:
    """Cap is 5; the edge cases match test_rules.py's clamp table."""

    @pytest.mark.parametrize(
        "open_position, contracts, allowed",
        [(0, 1, True), (0, 5, True), (0, 6, False),
         (3, 2, True), (4, 2, False), (5, 1, False)],
    )
    def test_long_side(self, open_position, contracts, allowed):
        d = evaluate(req(contracts=contracts),
                     AccountState(open_position=open_position),
                     et(f"{SUMMER_DAY} 10:00"))
        assert d.allowed is allowed

    def test_short_side(self):
        short = req(direction="short", entry_price=6800.0, stop_price=6805.0,
                    contracts=2)
        assert evaluate(short, AccountState(open_position=-3),
                        et(f"{SUMMER_DAY} 10:00")).allowed is True
        assert evaluate(short, AccountState(open_position=-4),
                        et(f"{SUMMER_DAY} 10:00")).allowed is False

    def test_block_message_reports_what_would_fit(self):
        d = evaluate(req(contracts=3), AccountState(open_position=4),
                     et(f"{SUMMER_DAY} 10:00"))
        assert any("only +1 fits" in b for b in d.blocks)

    def test_uses_the_rules_cap_not_a_literal(self):
        d = evaluate(req(contracts=rules.POSITION_CAP), flat(),
                     et(f"{SUMMER_DAY} 10:00"))
        assert d.allowed is True
        d2 = evaluate(req(contracts=rules.POSITION_CAP + 1), flat(),
                      et(f"{SUMMER_DAY} 10:00"))
        assert d2.allowed is False


class TestDailyLossBudget:
    NOW = f"{SUMMER_DAY} 10:00"

    def test_blocked_once_the_limit_is_reached(self):
        d = evaluate(req(), AccountState(today_realised_pnl=-rules.DAILY_LOSS_LIMIT),
                     et(self.NOW))
        assert d.allowed is False
        assert any("already reached" in b for b in d.blocks)

    def test_just_inside_the_limit_still_trades(self):
        d = evaluate(req(), AccountState(today_realised_pnl=-100.0), et(self.NOW))
        assert d.allowed is True

    def test_a_trade_larger_than_the_remaining_budget_is_blocked(self):
        """$350 already lost leaves $50; a ~$40 risk fits, a ~$140 one does not."""
        state = AccountState(today_realised_pnl=-350.0)
        small = evaluate(req(stop_price=6794.00, contracts=1), state, et(self.NOW))
        big = evaluate(req(stop_price=6773.00, contracts=1), state, et(self.NOW))
        assert small.allowed is True
        assert big.allowed is False
        assert any("loss budget" in b for b in big.blocks)

    def test_budget_is_measured_against_the_rules_limit(self):
        state = AccountState(today_realised_pnl=-(rules.DAILY_LOSS_LIMIT - 1))
        d = evaluate(req(), state, et(self.NOW))
        assert d.allowed is False  # $1 of budget cannot cover any real trade


class TestTrailingDrawdown:
    NOW = f"{SUMMER_DAY} 10:00"

    def test_ok_well_inside_the_line(self):
        state = AccountState(balance=49_800.0, peak_eod_balance=50_000.0)
        assert evaluate(req(), state, et(self.NOW)).allowed is True

    def test_warn_state_allows_but_notes_it(self):
        state = AccountState(balance=48_900.0, peak_eod_balance=50_000.0)
        d = evaluate(req(), state, et(self.NOW))
        assert d.allowed is True
        assert any("warning line" in w for w in d.warnings)

    def test_internal_stop_blocks(self):
        state = AccountState(
            balance=50_000.0 - rules.INTERNAL.trailing_drawdown_stop,
            peak_eod_balance=50_000.0,
        )
        d = evaluate(req(), state, et(self.NOW))
        assert d.allowed is False
        assert any("internal stop" in b for b in d.blocks)

    def test_blocks_before_the_firm_line_is_anywhere_near(self):
        """The buffer doing its job: stopped with $500 still to spare."""
        balance = 50_000.0 - rules.INTERNAL.trailing_drawdown_stop
        assert rules.is_account_terminated(balance, 50_000.0) is False
        assert evaluate(req(), AccountState(balance=balance,
                                            peak_eod_balance=50_000.0),
                        et(self.NOW)).allowed is False

    def test_risk_exceeding_remaining_room_is_blocked(self):
        """$1,450 down leaves $50 of room; a $140 risk does not fit."""
        state = AccountState(balance=48_550.0, peak_eod_balance=50_000.0)
        d = evaluate(req(stop_price=6773.00), state, et(self.NOW))
        assert d.allowed is False
        assert any("trailing-drawdown stop" in b for b in d.blocks)

    def test_drawdown_trails_the_peak_not_the_start(self):
        """Up then down: blocked while still above the starting balance."""
        state = AccountState(balance=50_400.0, peak_eod_balance=51_900.0)
        assert evaluate(req(), state, et(self.NOW)).allowed is False


class TestRequestSanity:
    NOW = f"{SUMMER_DAY} 10:00"

    def test_stop_on_the_wrong_side_of_a_long(self):
        d = evaluate(req(stop_price=6805.0), flat(), et(self.NOW))
        assert d.allowed is False
        assert any("wrong side" in b for b in d.blocks)

    def test_stop_on_the_wrong_side_of_a_short(self):
        d = evaluate(req(direction="short", stop_price=6795.0), flat(), et(self.NOW))
        assert d.allowed is False

    def test_stop_equal_to_entry(self):
        d = evaluate(req(stop_price=6800.0), flat(), et(self.NOW))
        assert d.allowed is False

    def test_empty_thesis_blocks(self):
        for thesis in ("", "   "):
            d = evaluate(req(thesis=thesis), flat(), et(self.NOW))
            assert d.allowed is False
            assert any("thesis" in b for b in d.blocks)

    def test_zero_contracts_blocks(self):
        assert evaluate(req(contracts=0), flat(), et(self.NOW)).allowed is False


class TestRiskCalculation:
    def test_includes_slippage_and_commission(self):
        """5 points on 1 MES is $25, plus 2 ticks ($2.50) and $2.50 commission."""
        d = evaluate(req(entry_price=6800.0, stop_price=6795.0, contracts=1),
                     flat(), et(f"{SUMMER_DAY} 10:00"))
        assert d.risk_dollars == pytest.approx(25.0 + 2.50 + 2.50)

    def test_scales_with_contracts(self):
        one = evaluate(req(contracts=1), flat(), et(f"{SUMMER_DAY} 10:00"))
        three = evaluate(req(contracts=3), flat(), et(f"{SUMMER_DAY} 10:00"))
        assert three.risk_dollars == pytest.approx(3 * one.risk_dollars)

    def test_bare_stop_distance_would_understate_it(self):
        d = evaluate(req(entry_price=6800.0, stop_price=6795.0), flat(),
                     et(f"{SUMMER_DAY} 10:00"))
        assert d.risk_dollars > 25.0


class TestBlockedTradesLeaveNoTicket:
    def test_multiple_reasons_all_reported(self):
        state = AccountState(open_position=5, today_realised_pnl=-500.0)
        d = evaluate(req(), state, et(f"{SUMMER_DAY} 16:25"), roll_dates={date(2025, 7, 16)})
        assert d.allowed is False
        assert len(d.blocks) >= 3

    def test_allow_produces_a_well_formed_ticket(self, tmp_path):
        now = et(f"{SUMMER_DAY} 10:00")
        d = evaluate(req(), flat(), now)
        ticket = build_ticket(req(), d, now)
        assert ticket.status == store.STATUS_OPEN
        assert ticket.instrument == "MES"
        assert ticket.contracts == 1
        assert ticket.thesis
        assert ticket.risk_dollars > 0
        assert len(ticket.ticket_id) == 12


class TestJournalRoundTrip:
    def test_open_then_close_prices_with_the_engine(self, tmp_path):
        path = tmp_path / "trades.jsonl"
        now = et(f"{SUMMER_DAY} 10:00")

        request = req(entry_price=6800.0, stop_price=6795.0, contracts=1)
        decision = evaluate(request, flat(), now)
        ticket = build_ticket(request, decision, now)
        store.append(ticket, path)

        assert store.net_open_position(path) == 1
        assert store.load_closed(path).empty

        closed = close_ticket(
            ticket, exit_price=6810.0,
            exit_time=now + pd.Timedelta(minutes=20), exit_reason="target",
        )
        store.append(closed, path)

        # +10 points less 2 ticks slippage = 9.50 pts = $47.50, less $2.50.
        assert closed.net_pnl == pytest.approx(45.00)
        assert closed.min_hold_ok is True

        loaded = store.load_closed(path)
        assert len(loaded) == 1
        assert store.net_open_position(path) == 0
        assert store.realised_pnl_on(date(2025, 7, 16), path) == pytest.approx(45.00)

    def test_append_only_history_is_preserved(self, tmp_path):
        path = tmp_path / "trades.jsonl"
        now = et(f"{SUMMER_DAY} 10:00")
        request = req()
        ticket = build_ticket(request, evaluate(request, flat(), now), now)
        store.append(ticket, path)
        store.append(
            close_ticket(ticket, 6810.0, now + pd.Timedelta(minutes=20), "target"),
            path,
        )
        raw = store.read_raw(path)
        assert len(raw) == 2                       # both records still on disk
        assert len(store.load_tickets(path)) == 1  # folded to one

    def test_short_hold_is_flagged_as_a_violation(self, tmp_path):
        now = et(f"{SUMMER_DAY} 10:00")
        request = req()
        ticket = build_ticket(request, evaluate(request, flat(), now), now)
        closed = close_ticket(ticket, 6801.0, now + pd.Timedelta(seconds=10), "manual")
        assert closed.min_hold_ok is False

    def test_realised_pnl_feeds_the_next_decision(self, tmp_path):
        """A losing day logged through the journal blocks the next trade."""
        path = tmp_path / "trades.jsonl"
        now = et(f"{SUMMER_DAY} 10:00")
        request = req(entry_price=6800.0, stop_price=6700.0, contracts=2)
        ticket = build_ticket(request, evaluate(request, flat(), now), now)
        store.append(ticket, path)
        store.append(
            close_ticket(ticket, 6760.0, now + pd.Timedelta(minutes=30), "stop"),
            path,
        )
        state = AccountState.from_journal(date(2025, 7, 16), path)
        assert state.today_realised_pnl < -rules.DAILY_LOSS_LIMIT
        assert evaluate(req(), state, et(f"{SUMMER_DAY} 14:00")).allowed is False
