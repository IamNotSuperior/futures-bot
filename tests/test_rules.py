"""Tests for the hard rules in strategies/rules.py.

The time-based rules get the most attention because they are the ones with
real edge cases: the cutoff boundaries themselves, daylight saving, and
early-close sessions where the deadlines move.
"""

from datetime import date, time, timedelta

import pandas as pd
import pytest

import rules


def et(stamp: str) -> pd.Timestamp:
    """An ET-localised timestamp."""
    return pd.Timestamp(stamp, tz=rules.ET)


# A regular Wednesday in summer (EDT) and in winter (EST).
SUMMER_DAY = "2025-07-16"
WINTER_DAY = "2025-01-15"


# ---------------------------------------------------------------------------
# Rule 3: entry cutoff at 16:20 ET
# ---------------------------------------------------------------------------


class TestEntryCutoff:
    @pytest.mark.parametrize(
        "clock, allowed",
        [
            ("09:30", True),
            ("16:00", True),
            ("16:19", True),
            ("16:19:59", True),
            ("16:20", False),  # the cutoff instant is already too late
            ("16:21", False),
            ("16:30", False),
        ],
    )
    def test_boundaries(self, clock, allowed):
        assert rules.is_entry_allowed(et(f"{SUMMER_DAY} {clock}")) is allowed

    def test_4_19_allowed_and_4_21_blocked(self):
        assert rules.is_entry_allowed(et(f"{SUMMER_DAY} 16:19")) is True
        assert rules.is_entry_allowed(et(f"{SUMMER_DAY} 16:21")) is False


# ---------------------------------------------------------------------------
# Rule 2: forced flatten at 16:30 ET
# ---------------------------------------------------------------------------


class TestFlattenDeadline:
    @pytest.mark.parametrize(
        "clock, flatten",
        [
            ("16:00", False),
            ("16:29", False),
            ("16:29:59", False),
            ("16:30", True),  # deadline reached, not merely passed
            ("16:31", True),
            ("17:00", True),
        ],
    )
    def test_boundaries(self, clock, flatten):
        assert rules.must_flatten(et(f"{SUMMER_DAY} {clock}")) is flatten

    def test_4_29_and_4_31(self):
        assert rules.must_flatten(et(f"{SUMMER_DAY} 16:29")) is False
        assert rules.must_flatten(et(f"{SUMMER_DAY} 16:31")) is True

    def test_entry_cutoff_precedes_flatten_by_the_runway(self):
        assert rules.entry_deadline(date(2025, 7, 16)) == time(16, 20)
        assert rules.flatten_deadline(date(2025, 7, 16)) == time(16, 30)

    def test_no_entry_survives_into_the_flatten_window(self):
        """The gap between cutoff and flatten must exceed the minimum hold.

        This is the property that makes rule 6 and rule 2 non-conflicting.
        """
        gap = rules.ENTRY_RUNWAY.total_seconds()
        assert gap > rules.MIN_HOLD_SECONDS


# ---------------------------------------------------------------------------
# Daylight saving
# ---------------------------------------------------------------------------


class TestDaylightSaving:
    def test_same_utc_instant_differs_by_season(self):
        """20:20 UTC is 15:20 EST in winter but 16:20 EDT in summer."""
        winter = pd.Timestamp(f"{WINTER_DAY} 20:20", tz="UTC")
        summer = pd.Timestamp(f"{SUMMER_DAY} 20:20", tz="UTC")

        assert rules.to_et(winter).time() == time(15, 20)
        assert rules.to_et(summer).time() == time(16, 20)

        assert rules.is_entry_allowed(winter) is True  # 15:20 EST, still open
        assert rules.is_entry_allowed(summer) is False  # 16:20 EDT, cutoff hit

    def test_winter_cutoff_is_2120_utc(self):
        assert rules.is_entry_allowed(pd.Timestamp(f"{WINTER_DAY} 21:19", tz="UTC")) is True
        assert rules.is_entry_allowed(pd.Timestamp(f"{WINTER_DAY} 21:20", tz="UTC")) is False

    def test_summer_cutoff_is_2020_utc(self):
        assert rules.is_entry_allowed(pd.Timestamp(f"{SUMMER_DAY} 20:19", tz="UTC")) is True
        assert rules.is_entry_allowed(pd.Timestamp(f"{SUMMER_DAY} 20:20", tz="UTC")) is False

    @pytest.mark.parametrize("day", ["2025-03-10", "2025-11-03"])
    def test_day_after_a_transition_still_uses_wall_clock(self, day):
        """The Mondays after spring-forward and fall-back behave normally."""
        assert rules.is_entry_allowed(et(f"{day} 16:19")) is True
        assert rules.is_entry_allowed(et(f"{day} 16:21")) is False
        assert rules.must_flatten(et(f"{day} 16:31")) is True

    def test_transition_days_themselves(self):
        """DST changes land on Sundays; the rules still evaluate cleanly."""
        for day in ("2025-03-09", "2025-11-02"):
            assert rules.is_entry_allowed(et(f"{day} 16:19")) is True
            assert rules.must_flatten(et(f"{day} 16:31")) is True

    def test_ambiguous_fall_back_hour_is_resolved(self):
        """01:30 occurs twice on 2025-11-02; both instants must localise."""
        early = pd.Timestamp("2025-11-02 05:30", tz="UTC").tz_convert(rules.ET)
        late = pd.Timestamp("2025-11-02 06:30", tz="UTC").tz_convert(rules.ET)
        assert early.time() == time(1, 30)
        assert late.time() == time(1, 30)
        assert early.utcoffset() != late.utcoffset()
        assert rules.is_entry_allowed(early) is True
        assert rules.is_entry_allowed(late) is True


# ---------------------------------------------------------------------------
# Early-close sessions
# ---------------------------------------------------------------------------


EARLY = date(2025, 7, 3)  # 13:00 ET close, the day before Independence Day


class TestEarlyClose:
    def test_deadlines_move_earlier(self):
        assert rules.flatten_deadline(EARLY, {EARLY}) == time(13, 0)
        assert rules.entry_deadline(EARLY, {EARLY}) == time(12, 50)

    def test_regular_day_is_unaffected(self):
        assert rules.flatten_deadline(EARLY, set()) == time(16, 30)
        assert rules.entry_deadline(EARLY, set()) == time(16, 20)

    @pytest.mark.parametrize(
        "clock, allowed",
        [("12:49", True), ("12:50", False), ("12:51", False), ("16:19", False)],
    )
    def test_entry_cutoff(self, clock, allowed):
        assert rules.is_entry_allowed(et(f"{EARLY} {clock}"), {EARLY}) is allowed

    @pytest.mark.parametrize(
        "clock, flatten",
        [("12:59", False), ("13:00", True), ("13:01", True)],
    )
    def test_flatten(self, clock, flatten):
        assert rules.must_flatten(et(f"{EARLY} {clock}"), {EARLY}) is flatten

    def test_an_entry_at_1619_would_be_after_the_close(self):
        """Without early-close awareness this entry looks legal but isn't."""
        assert rules.is_entry_allowed(et(f"{EARLY} 16:19"), set()) is True
        assert rules.is_entry_allowed(et(f"{EARLY} 16:19"), {EARLY}) is False

    def test_runway_preserved_on_early_close(self):
        gap = timedelta(
            hours=rules.flatten_deadline(EARLY, {EARLY}).hour,
            minutes=rules.flatten_deadline(EARLY, {EARLY}).minute,
        ) - timedelta(
            hours=rules.entry_deadline(EARLY, {EARLY}).hour,
            minutes=rules.entry_deadline(EARLY, {EARLY}).minute,
        )
        assert gap == rules.ENTRY_RUNWAY


# ---------------------------------------------------------------------------
# Roll days
# ---------------------------------------------------------------------------


ROLL = date(2025, 6, 18)


class TestRollDays:
    def test_entries_blocked_on_roll_day(self):
        assert rules.is_entry_allowed(et(f"{ROLL} 10:00"), (), {ROLL}) is False

    def test_entries_allowed_on_neighbouring_days(self):
        assert rules.is_entry_allowed(et("2025-06-17 10:00"), (), {ROLL}) is True
        assert rules.is_entry_allowed(et("2025-06-19 10:00"), (), {ROLL}) is True

    def test_flatten_is_unaffected_by_roll_day(self):
        """A roll day blocks new entries; it does not change the exit deadline."""
        assert rules.must_flatten(et(f"{ROLL} 16:00")) is False
        assert rules.must_flatten(et(f"{ROLL} 16:31")) is True

    def test_empty_roll_set(self):
        assert rules.is_roll_day(et(f"{ROLL} 10:00"), ()) is False


# ---------------------------------------------------------------------------
# Naive timestamps
# ---------------------------------------------------------------------------


def test_naive_timestamps_are_read_as_et():
    assert rules.to_et(pd.Timestamp(f"{SUMMER_DAY} 16:21")).time() == time(16, 21)
    assert rules.is_entry_allowed(f"{SUMMER_DAY} 16:19") is True
    assert rules.is_entry_allowed(f"{SUMMER_DAY} 16:21") is False


# ---------------------------------------------------------------------------
# Rule 1: instruments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol, ok",
    [
        ("MES", True),
        ("MNQ", True),
        ("MES.v.0", True),
        ("MNQ.FUT", True),
        ("MESZ4", True),
        ("mes.v.0", True),
        ("ES", False),
        ("NQ", False),
        ("ESZ4", False),
        ("MYM", False),
        ("", False),
    ],
)
def test_allowed_instruments(symbol, ok):
    assert rules.is_allowed_instrument(symbol) is ok


def test_require_allowed_instrument_raises():
    with pytest.raises(rules.RuleViolation, match="not permitted"):
        rules.require_allowed_instrument("ES")


# ---------------------------------------------------------------------------
# Rule 4: position cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current, requested, expected",
    [
        (0, 1, 1),
        (0, 5, 5),
        (0, 6, 5),  # clamped down to the cap
        (3, 2, 2),
        (4, 2, 1),  # only one more contract fits
        (5, 1, 0),  # already at the cap
        (0, -6, -5),
        (-5, -1, 0),
        (-4, -5, -1),
        (5, -1, -1),  # reducing is always fine
        (5, -10, -10),  # reversing to -5 stays inside the cap
        (5, -11, -10),  # reversal clamped at the far side
        (0, 0, 0),
    ],
)
def test_clamp_order_size(current, requested, expected):
    assert rules.clamp_order_size(current, requested) == expected


def test_clamp_never_flips_order_direction():
    for current in range(-4, 5):
        for requested in range(-5, 6):
            clamped = rules.clamp_order_size(current, requested)
            if clamped:
                assert (clamped > 0) == (requested > 0)
                assert abs(clamped) <= abs(requested)


def test_position_cap_boundaries():
    assert rules.within_position_cap(5) is True
    assert rules.within_position_cap(-5) is True
    assert rules.within_position_cap(6) is False
    with pytest.raises(rules.RuleViolation, match="exceeds the hard cap"):
        rules.require_position_cap(6)


# ---------------------------------------------------------------------------
# Rule 5: daily loss limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pnl, breached",
    [(0.0, False), (-399.99, False), (-400.0, True), (-400.01, True), (250.0, False)],
)
def test_daily_loss_limit(pnl, breached):
    assert rules.is_daily_loss_breached(pnl) is breached


def test_loss_limit_blocks_entry_even_inside_the_window():
    ts = et(f"{SUMMER_DAY} 10:00")
    assert rules.can_open_new_position(ts, day_pnl=-50.0) is True
    assert rules.can_open_new_position(ts, day_pnl=-400.0) is False


def test_time_cutoff_blocks_entry_even_when_profitable():
    assert rules.can_open_new_position(et(f"{SUMMER_DAY} 16:21"), day_pnl=500.0) is False


# ---------------------------------------------------------------------------
# Rule 6: minimum hold
# ---------------------------------------------------------------------------


class TestMinimumHold:
    entry = et(f"{SUMMER_DAY} 10:00:00")

    @pytest.mark.parametrize(
        "seconds, allowed",
        [(0, False), (29, False), (29.9, False), (30, True), (31, True), (600, True)],
    )
    def test_boundaries(self, seconds, allowed):
        exit_ts = self.entry + pd.Timedelta(seconds=seconds)
        assert rules.can_close(self.entry, exit_ts) is allowed

    def test_override_bypasses_the_floor(self):
        exit_ts = self.entry + pd.Timedelta(seconds=5)
        assert rules.can_close(self.entry, exit_ts) is False
        assert rules.can_close(self.entry, exit_ts, override=True) is True

    def test_seconds_until_closable(self):
        assert rules.seconds_until_closable(self.entry, self.entry) == 30
        assert rules.seconds_until_closable(
            self.entry, self.entry + pd.Timedelta(seconds=10)
        ) == 20
        assert rules.seconds_until_closable(
            self.entry, self.entry + pd.Timedelta(seconds=60)
        ) == 0

    def test_hold_floor_exceeds_the_microscalp_threshold(self):
        """Rule 6 must keep every trade clear of the rule 7 measure."""
        assert rules.MIN_HOLD_SECONDS > rules.MICROSCALP_SECONDS


# ---------------------------------------------------------------------------
# Firm ceilings vs internal limits
# ---------------------------------------------------------------------------


class TestLimitSeparation:
    """The buffer is the point: every internal limit must sit inside the firm's."""

    def test_firm_values_match_the_lucid_50k_pro_eval(self):
        assert rules.FIRM.account_size == 50_000
        assert rules.FIRM.daily_loss == 1_200
        assert rules.FIRM.max_trailing_drawdown == 2_000
        assert rules.FIRM.max_contracts == 40
        assert rules.FIRM.consistency_pct == 40.0
        assert rules.FIRM.profit_target == 3_000

    def test_internal_values(self):
        assert rules.INTERNAL.daily_loss == 400
        assert rules.INTERNAL.trailing_drawdown_warn == 1_000
        assert rules.INTERNAL.trailing_drawdown_stop == 1_500
        assert rules.INTERNAL.max_contracts == 5
        assert rules.INTERNAL.consistency_warn_pct == 30.0

    def test_every_internal_limit_is_strictly_tighter(self):
        assert rules.INTERNAL.daily_loss < rules.FIRM.daily_loss
        assert rules.INTERNAL.trailing_drawdown_stop < rules.FIRM.max_trailing_drawdown
        assert rules.INTERNAL.max_contracts < rules.FIRM.max_contracts
        assert rules.INTERNAL.consistency_warn_pct < rules.FIRM.consistency_pct

    def test_warn_precedes_stop(self):
        assert (
            rules.INTERNAL.trailing_drawdown_warn
            < rules.INTERNAL.trailing_drawdown_stop
        )

    def test_enforced_aliases_point_at_internal_never_firm(self):
        """The guards must read internal values. This is the regression that
        would matter most: an alias silently repointed at a firm number."""
        assert rules.DAILY_LOSS_LIMIT == rules.INTERNAL.daily_loss
        assert rules.POSITION_CAP == rules.INTERNAL.max_contracts
        assert rules.TRAILING_DD_STOP == rules.INTERNAL.trailing_drawdown_stop
        assert rules.TRAILING_DD_WARN == rules.INTERNAL.trailing_drawdown_warn
        assert rules.WORST_DAY_FLAG_PCT == rules.INTERNAL.consistency_warn_pct

        assert rules.DAILY_LOSS_LIMIT != rules.FIRM.daily_loss
        assert rules.POSITION_CAP != rules.FIRM.max_contracts
        assert rules.TRAILING_DD_STOP != rules.FIRM.max_trailing_drawdown

    def test_buffer_report_covers_each_limit(self):
        rows = rules.buffer_report()
        assert len(rows) == 4
        for _label, internal, firm, _note in rows:
            assert internal < firm

    def test_limits_are_frozen(self):
        with pytest.raises(Exception):
            rules.INTERNAL.daily_loss = 999

    def test_payout_terms_match_the_lucid_50k_pro_account(self):
        """A withdrawal needs the balance $2,100 over the start, $500 minimum.

        These are account terms, not ceilings: nothing enforces against them,
        and the simulator reports the line as a milestone alongside the pass.
        """
        assert rules.FIRM.payout_buffer == 2_100
        assert rules.FIRM.min_payout == 500
        assert rules.PAYOUT_BALANCE == rules.ACCOUNT_SIZE + rules.FIRM.payout_buffer
        assert rules.PAYOUT_BALANCE == 52_100

    def test_payout_line_sits_below_the_profit_target(self):
        """Every passing path reaches payout first; the ordering is load-bearing
        for the simulator's payout-never-below-pass property."""
        assert rules.FIRM.payout_buffer < rules.FIRM.profit_target


class TestCommission:
    """The MES commission Lucid support confirmed, and the figure it replaces."""

    def test_verified_rate_is_fifty_cents_a_side(self):
        assert rules.COMMISSION_PER_SIDE == 0.50

    def test_the_source_is_recorded_next_to_the_number(self):
        import inspect

        source = inspect.getsource(rules)
        line = next(l for l in source.splitlines()
                    if l.startswith("COMMISSION_PER_SIDE ="))
        block = source[max(0, source.index(line) - 800):source.index(line) + 200]
        assert "11508978" in block, "the support article is the number's provenance"

    def test_the_historical_assumption_stays_reachable(self):
        """Every verdict before 2026-09-11 was scored at $1.25 a side. It has
        to remain callable by name so those runs can be reproduced exactly."""
        assert rules.ASSUMED_COMMISSION_PER_SIDE == 1.25
        assert rules.COMMISSION_PER_SIDE < rules.ASSUMED_COMMISSION_PER_SIDE

    def test_no_runner_restates_a_commission_default(self):
        """The old rate was a literal default in eight runners. A restated
        default is a value that does not move when rules.py does."""
        from pathlib import Path

        root = Path(rules.__file__).resolve().parent.parent
        offenders = []
        for folder in ("backtests", "bots", "journal", "strategies"):
            for path in sorted((root / folder).glob("*.py")):
                text = path.read_text(encoding="utf-8")
                for needle in ("default=1.25", "default=0.5",
                               "commission_per_side=1.25",
                               "commission_per_side=0.5"):
                    if needle in text:
                        offenders.append(f"{path.name}: {needle}")
        assert offenders == []


# ---------------------------------------------------------------------------
# Rule 5b: end-of-day trailing drawdown
# ---------------------------------------------------------------------------


class TestTrailingDrawdown:
    START = 50_000.0

    def test_floor_starts_below_the_opening_balance(self):
        assert rules.trailing_floor(self.START) == 48_500.0  # internal stop
        assert rules.trailing_floor(self.START, rules.FIRM.max_trailing_drawdown) \
            == 48_000.0

    def test_floor_trails_a_new_peak(self):
        assert rules.trailing_floor(52_000.0) == 50_500.0

    def test_drawdown_is_zero_at_or_above_the_peak(self):
        assert rules.drawdown_from_peak(50_000.0, 50_000.0) == 0.0
        assert rules.drawdown_from_peak(51_000.0, 50_000.0) == 0.0

    def test_drawdown_measures_from_the_peak(self):
        assert rules.drawdown_from_peak(49_100.0, 50_000.0) == 900.0

    @pytest.mark.parametrize(
        "balance, state",
        [
            (50_000.0, "ok"),
            (49_100.0, "ok"),      # -900, inside the warn line
            (49_000.0, "warn"),    # -1,000 exactly
            (48_600.0, "warn"),    # -1,400
            (48_500.0, "stop"),    # -1,500 exactly, internal stop
            (48_400.0, "stop"),
        ],
    )
    def test_state_boundaries(self, balance, state):
        assert rules.trailing_drawdown_state(balance, 50_000.0) == state

    def test_internal_stop_fires_before_the_firm_would_terminate(self):
        """At the internal stop there is still $500 of the firm's line left."""
        at_stop = 50_000.0 - rules.INTERNAL.trailing_drawdown_stop
        assert rules.trailing_drawdown_state(at_stop, 50_000.0) == "stop"
        assert rules.is_account_terminated(at_stop, 50_000.0) is False
        assert rules.headroom_to_firm_termination(at_stop, 50_000.0) == 500.0

    @pytest.mark.parametrize(
        "balance, dead",
        [(48_100.0, False), (48_000.0, True), (47_900.0, True)],
    )
    def test_firm_termination_line(self, balance, dead):
        assert rules.is_account_terminated(balance, 50_000.0) is dead

    def test_termination_line_trails_the_peak(self):
        """After a good day the account dies at a balance that was once fine.

        Peak 52,000 puts the firm's floor at exactly 50,000 - the starting
        balance. Being flat on the account is a termination once you have been
        up $2,000 and given it all back.
        """
        assert rules.is_account_terminated(50_000.0, 52_000.0) is True
        assert rules.is_account_terminated(50_100.0, 52_000.0) is False  # -1,900
        assert rules.is_account_terminated(50_000.0, 50_000.0) is False

    def test_headroom_shrinks_as_the_drawdown_grows(self):
        assert rules.headroom_to_firm_termination(50_000.0, 50_000.0) == 2_000.0
        assert rules.headroom_to_firm_termination(49_000.0, 50_000.0) == 1_000.0
