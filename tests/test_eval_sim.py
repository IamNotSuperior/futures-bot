"""Tests for the evaluation Monte Carlo.

Every case uses a hand-built distribution whose answer is known by
construction, so a wrong number is a bug rather than a judgement call.
"""

import numpy as np
import pytest

import rules
from eval_sim import EvalConfig, daily_pnl_from_trades, simulate

import pandas as pd

ET = "America/New_York"


class TestDeterministicDistributions:
    """Single-valued distributions have exact, arithmetic answers."""

    def test_constant_winner_always_passes(self):
        """+$1,000 a day reaches +$3,000 on day 3, every path."""
        r = simulate(np.array([1_000.0]), EvalConfig(max_days=10), paths=500)
        assert r.pass_probability == 1.0
        assert r.blowup_probability == 0.0
        assert r.median_days_to_pass == 3
        assert r.expected_attempts == pytest.approx(1.0)

    def test_constant_loser_always_blows_up(self):
        """-$500 a day crosses -$2,000 on day 4, every path."""
        r = simulate(np.array([-500.0]), EvalConfig(max_days=10), paths=500)
        assert r.blowup_probability == 1.0
        assert r.pass_probability == 0.0
        assert np.isinf(r.expected_attempts)

    def test_flat_distribution_times_out(self):
        """Zero every day: never passes, never dies, always runs out of time."""
        r = simulate(np.array([0.0]), EvalConfig(max_days=20), paths=200)
        assert r.timeout_probability == 1.0
        assert r.pass_probability == 0.0
        assert r.blowup_probability == 0.0

    def test_horizon_can_prevent_a_pass(self):
        """+$1,000/day passes on day 3, so a 2-day horizon must time out."""
        assert simulate(np.array([1_000.0]), EvalConfig(max_days=2),
                        paths=200).pass_probability == 0.0
        assert simulate(np.array([1_000.0]), EvalConfig(max_days=3),
                        paths=200).pass_probability == 1.0


class TestTrailingDrawdownInTheSim:
    def test_drawdown_trails_the_peak_not_the_start(self):
        """+$2,500 then -$2,000 a day: dies despite never being down on the start.

        Day 1 closes at 52,500 and sets the peak. Day 2 closes at 50,500,
        exactly $2,000 below it, so the account is dead while still $500 up.
        """
        r = simulate(np.array([2_500.0, -2_000.0]), EvalConfig(max_days=2),
                     paths=4_000, seed=1)
        # The +,- ordering blows up; +,+ passes on day 2 (5,000 >= 3,000).
        assert r.blowup_probability > 0.2
        assert r.pass_probability > 0.2

    def test_exact_line_is_a_blowup(self):
        r = simulate(np.array([-2_000.0]), EvalConfig(max_days=1), paths=100)
        assert r.blowup_probability == 1.0

    def test_just_inside_the_line_survives(self):
        r = simulate(np.array([-1_999.0]), EvalConfig(max_days=1), paths=100)
        assert r.blowup_probability == 0.0
        assert r.timeout_probability == 1.0

    def test_target_checked_before_blowup_on_the_same_day(self):
        """A day that both hits target and would breach is scored as a pass.

        It cannot be both, and reaching +3,000 ends the evaluation.
        """
        r = simulate(np.array([3_000.0]), EvalConfig(max_days=1), paths=100)
        assert r.pass_probability == 1.0


class TestKnownCoinFlip:
    """A symmetric two-point distribution with a computable answer."""

    def test_two_thousand_up_or_down_is_a_fair_race(self):
        """+/-$2,000 with equal odds: day 1 either kills or leaves 52,000.

        Day 1 down is instant death (-2,000 from the 50,000 peak), so at least
        half of all paths die on the first day.
        """
        r = simulate(np.array([2_000.0, -2_000.0]), EvalConfig(max_days=1),
                     paths=40_000, seed=7)
        assert r.blowup_probability == pytest.approx(0.5, abs=0.02)

    def test_pass_probability_is_stable_across_seeds(self):
        cfg = EvalConfig(max_days=30)
        dist = np.array([400.0, -200.0, 150.0, -350.0])
        a = simulate(dist, cfg, paths=20_000, seed=1)
        b = simulate(dist, cfg, paths=20_000, seed=2)
        assert a.pass_probability == pytest.approx(b.pass_probability, abs=0.02)

    def test_expected_attempts_is_the_reciprocal(self):
        r = simulate(np.array([400.0, -200.0, 150.0, -350.0]),
                     EvalConfig(max_days=30), paths=20_000, seed=3)
        if r.pass_probability > 0:
            assert r.expected_attempts == pytest.approx(1 / r.pass_probability)

    def test_probabilities_sum_to_one(self):
        r = simulate(np.array([500.0, -400.0, 100.0]), EvalConfig(max_days=40),
                     paths=5_000, seed=5)
        total = (r.pass_probability + r.blowup_probability
                 + r.timeout_probability)
        assert total == pytest.approx(1.0)


class TestConfigDefaults:
    def test_defaults_come_from_the_rules_module(self):
        cfg = EvalConfig()
        assert cfg.starting_balance == rules.ACCOUNT_SIZE
        assert cfg.profit_target == rules.FIRM.profit_target
        assert cfg.trailing_drawdown == rules.FIRM.max_trailing_drawdown

    def test_simulator_uses_the_firm_line_not_the_internal_one(self):
        """The sim answers "would the account survive", so it uses the firm's
        line. The internal stop is a trading guard, not a termination."""
        assert EvalConfig().trailing_drawdown == 2_000
        assert EvalConfig().trailing_drawdown != rules.INTERNAL.trailing_drawdown_stop


class TestDailyAggregation:
    def test_trades_collapse_to_days(self):
        trades = pd.DataFrame([
            {"entry_time": pd.Timestamp("2025-01-02 10:00", tz=ET),
             "net_pnl": 100.0},
            {"entry_time": pd.Timestamp("2025-01-02 11:00", tz=ET),
             "net_pnl": -40.0},
            {"entry_time": pd.Timestamp("2025-01-03 10:00", tz=ET),
             "net_pnl": 25.0},
        ])
        got = daily_pnl_from_trades(trades)
        assert list(got) == [60.0, 25.0]

    def test_empty_distribution_is_rejected(self):
        with pytest.raises(ValueError, match="No daily P&L"):
            simulate(np.array([]))
