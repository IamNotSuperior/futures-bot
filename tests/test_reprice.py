"""Tests for re-pricing a saved trade stream at a different commission.

Every verdict before 2026-09-11 was scored at $1.25 a side; Lucid's verified
rate is $0.50. Commission does not touch fills, so a saved stream can be
re-priced exactly: each trade moves by twice the per-side difference times
its size, and the size is recovered from the commission the trade carried.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import rules
from engine import MES, CostModel, price_trades
from reprice import reprice_stream, score_stream

ET = "America/New_York"
OLD = rules.ASSUMED_COMMISSION_PER_SIDE
NEW = rules.COMMISSION_PER_SIDE


def stream(sizes, moves, start=date(2025, 7, 14)):
    """One trade a day at the given sizes and point moves, priced at ``OLD``.

    Mixed sizes are priced row by row, the way entry 6's per-session sizing
    produced its stream, so the ``commission`` column varies across rows.
    """
    frames = []
    for i, (size, move) in enumerate(zip(sizes, moves)):
        day = (pd.Timestamp(start) + pd.Timedelta(days=i)).date()
        raw = pd.DataFrame([{
            "entry_time": pd.Timestamp(f"{day} 10:00", tz=ET),
            "exit_time": pd.Timestamp(f"{day} 10:30", tz=ET),
            "direction": "long", "entry_price": 100.0,
            "exit_price": 100.0 + move, "exit_reason": "target",
        }])
        frames.append(price_trades(raw, MES, CostModel(commission_per_side=OLD), size))
    return pd.concat(frames, ignore_index=True)


class TestReprice:
    def test_each_trade_moves_by_the_round_turn_difference_times_its_size(self):
        before = stream([1, 1, 1], [2.0, -1.0, 3.0])
        after = reprice_stream(before, OLD, NEW)
        shift = 2 * (OLD - NEW)
        assert shift == pytest.approx(1.50)
        assert list((after["net_pnl"] - before["net_pnl"]).round(6)) == [shift] * 3
        assert list(after["commission"]) == pytest.approx([2 * NEW] * 3)

    def test_size_is_recovered_from_the_commission_column(self):
        before = stream([1, 5, 3, 2], [1.0, 1.0, 1.0, 1.0])
        after = reprice_stream(before, OLD, NEW)
        expected = [2 * (OLD - NEW) * n for n in (1, 5, 3, 2)]
        assert list((after["net_pnl"] - before["net_pnl"]).round(6)) == \
            pytest.approx(expected)
        assert list(after["commission"]) == pytest.approx([2 * NEW * n for n in (1, 5, 3, 2)])

    def test_fills_points_and_slippage_are_untouched(self):
        before = stream([4, 4], [2.0, -2.0])
        after = reprice_stream(before, OLD, NEW)
        for col in ("entry_fill", "exit_fill", "gross_points", "net_points",
                    "gross_pnl", "slippage_cost", "exit_reason", "duration_seconds"):
            pd.testing.assert_series_equal(after[col], before[col], check_names=False)

    def test_repricing_to_the_same_commission_is_the_identity(self):
        before = stream([1, 2], [1.0, -1.0])
        pd.testing.assert_frame_equal(reprice_stream(before, OLD, OLD), before)

    def test_agrees_with_pricing_from_scratch(self):
        """Re-pricing the saved stream must equal what the engine would have
        produced had it been run at the new commission in the first place."""
        raw = pd.DataFrame([{
            "entry_time": pd.Timestamp("2025-07-14 10:00", tz=ET),
            "exit_time": pd.Timestamp("2025-07-14 10:30", tz=ET),
            "direction": "short", "entry_price": 100.0,
            "exit_price": 97.0, "exit_reason": "target",
        }])
        at_old = price_trades(raw, MES, CostModel(commission_per_side=OLD), 3)
        at_new = price_trades(raw, MES, CostModel(commission_per_side=NEW), 3)
        pd.testing.assert_frame_equal(reprice_stream(at_old, OLD, NEW), at_new)

    def test_the_input_is_not_mutated(self):
        before = stream([1], [1.0])
        original = before.copy()
        reprice_stream(before, OLD, NEW)
        pd.testing.assert_frame_equal(before, original)

    def test_refuses_a_stream_without_a_commission_column(self):
        with pytest.raises(ValueError, match="commission"):
            reprice_stream(pd.DataFrame({"net_pnl": [1.0]}), OLD, NEW)

    def test_refuses_a_stream_whose_commission_does_not_match_the_old_rate(self):
        """A stream that was not priced at ``old`` cannot be re-priced from it;
        the recovered size would not be a whole number."""
        before = stream([1], [1.0])
        with pytest.raises(ValueError, match="not priced at"):
            reprice_stream(before, 0.7, NEW)


class TestScore:
    def test_headline_figures_and_fold_count(self):
        s = stream([1] * 4, [10.0, -2.0, 10.0, -2.0],
                   start=date(2024, 12, 30))  # two trades in 2024, two in 2025
        got = score_stream(s, paths=200)
        assert got["trades"] == 4
        assert got["net_pnl"] == pytest.approx(s["net_pnl"].sum())
        assert got["years"] == [2024, 2025]
        assert got["profitable_years"] == 2
        for key in ("sharpe", "profit_factor", "max_drawdown", "max_daily_loss",
                    "pass_probability", "payout_probability", "blowups",
                    "median_year_sharpe", "mean_per_trade"):
            assert key in got
        assert got["payout_probability"] >= got["pass_probability"]

    def test_trailing_halt_can_be_applied(self):
        """Re-scoring a halt-ON basis re-applies the engine's halt to the
        re-priced stream, so a halt that fired at $1.25 may fire later at $0.50."""
        s = stream([4] * 3, [-40.0, -40.0, 10.0])  # -820, -820, then blocked
        without = score_stream(s, paths=100, trailing_halt=False)
        with_halt = score_stream(s, paths=100, trailing_halt=True)
        assert without["trades"] == 3
        assert with_halt["trades"] == 2
        assert with_halt["dd_halts"] == 1
        assert without["dd_halts"] == 0
