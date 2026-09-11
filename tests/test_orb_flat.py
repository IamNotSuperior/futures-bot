"""Tests for entry 5: ORB flat by 10:30.

Two things are new here and both are tested: the forced flatten landing on the
*open of the flatten bar* rather than the close before it, and the $1,500
trailing drawdown halt that entry 5's specification requires.

Entry 4's behaviour must be untouched by either, since its results are frozen.
"""

from datetime import date, time, timedelta

import pandas as pd
import pytest

import rules
from engine import MES, CostModel, build_trades, price_trades
from orb2 import ORB2, ORB2Params
from run_orb_flat import (
    BREAK_EVEN, enforce_trailing_drawdown_halt, entry5_params,
)

ET = "America/New_York"
DAY = date(2025, 7, 16)

BASE = 101.0
OR_HIGH, OR_LOW = 102.0, 100.0
BUY_STOP, SELL_STOP = 103.0, 99.0


def session(day: date = DAY, base: float = BASE, minutes: int = 390) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=minutes, freq="1min", tz=ET)
    return pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base,
         "volume": 100, "instrument_id": 1},
        index=idx, dtype="float64",
    ).assign(volume=100, instrument_id=1)


def put(bars, hhmm, **values):
    ts = pd.Timestamp(f"{bars.index[0].date()} {hhmm}", tz=ET)
    for col, value in values.items():
        bars.loc[ts, col] = float(value)
    return bars


def with_range(bars, high=OR_HIGH, low=OR_LOW):
    return put(bars, "09:30", open=BASE, high=high, low=low, close=BASE)


class TestEntry5Params:
    """The frozen configuration, checked against the entry rather than assumed."""

    def test_matches_the_frozen_spec(self):
        p = entry5_params()
        assert p.opening_range_end == time(9, 45)
        assert p.entry_offset == 1.0
        assert p.max_range_points == 10.0
        assert p.stop_points == 10.0
        assert p.target_points == 18.0
        assert p.entry_cancel_time == time(10, 30)
        assert p.flatten_time == time(10, 30)
        assert p.use_trend_filter is False
        assert p.flatten_at_next_open is True

    def test_break_even_constant_is_the_bracket_arithmetic(self):
        assert BREAK_EVEN == pytest.approx(55.0 / 140.0)
        assert BREAK_EVEN == pytest.approx(0.392857, abs=1e-6)


class TestFlattenAtNextOpen:
    def _bars(self):
        bars = with_range(session())
        bars = put(bars, "10:00", high=BUY_STOP)      # fill, then nothing
        return put(bars, "10:30", open=95.0, high=95.0, low=95.0, close=95.0)

    def test_exit_lands_on_the_flatten_bar_open(self):
        strat = ORB2(entry5_params())
        signals = strat.generate_signals(self._bars())
        trades = build_trades(signals, self._bars())
        assert len(trades) == 1
        row = trades.iloc[0]
        assert row["exit_time"].time() == time(10, 30)
        assert row["exit_price"] == pytest.approx(95.0)
        assert row["exit_reason"] == "session_end"

    def test_entry_four_behaviour_is_unchanged(self):
        """Default False must still exit on the close of the prior bar."""
        params = ORB2Params(
            use_trend_filter=False, entry_cancel_time=time(10, 30),
            flatten_time=time(10, 30),
        )
        assert params.flatten_at_next_open is False
        strat = ORB2(params)
        bars = self._bars()
        trades = build_trades(strat.generate_signals(bars), bars)
        row = trades.iloc[0]
        assert row["exit_time"].time() == time(10, 29)
        assert row["exit_price"] == pytest.approx(BASE)

    def test_a_resolved_trade_is_unaffected(self):
        bars = with_range(session())
        bars = put(bars, "10:00", high=BUY_STOP)
        bars = put(bars, "10:05", high=BUY_STOP + 18.0)
        strat = ORB2(entry5_params())
        trades = build_trades(strat.generate_signals(bars), bars)
        row = trades.iloc[0]
        assert row["exit_reason"] == "target"
        assert row["exit_time"].time() == time(10, 5)

    def test_nothing_is_held_past_1030(self):
        strat = ORB2(entry5_params())
        signals = strat.generate_signals(self._bars())
        exits = signals.index[signals["exit_long"] | signals["exit_short"]]
        assert (exits.time <= time(10, 30)).all()

    def test_no_entry_at_or_after_1030(self):
        bars = with_range(session())
        bars = put(bars, "10:30", high=120.0)
        strat = ORB2(entry5_params())
        signals = strat.generate_signals(bars)
        assert not signals["entry_long"].any()

    def test_entry_just_before_1030_is_taken(self):
        bars = with_range(session())
        bars = put(bars, "10:29", high=BUY_STOP)
        strat = ORB2(entry5_params())
        signals = strat.generate_signals(bars)
        entries = signals.index[signals["entry_long"]]
        assert len(entries) == 1 and entries[0].time() == time(10, 29)


class TestPayoutInTheReport:
    """Entry 5's window stats carry the payout milestone next to the pass."""

    def _priced(self):
        raw = pd.DataFrame([
            {"entry_time": pd.Timestamp(f"2025-07-{d:02d} 09:46", tz=ET),
             "exit_time": pd.Timestamp(f"2025-07-{d:02d} 10:30", tz=ET),
             "direction": "long", "entry_price": 100.0,
             "exit_price": 100.0 + move, "exit_reason": "session_end"}
            for d, move in ((14, 2.0), (15, -1.0), (16, 3.0), (17, -2.0))
        ])
        return price_trades(raw, MES, CostModel(), 4)

    def test_window_stats_carry_the_payout_probability(self):
        from run_orb_flat import window_stats

        stats = window_stats(self._priced(), "fixture", paths=200)
        assert "payout_probability" in stats
        assert 0.0 <= stats["payout_probability"] <= 1.0
        assert stats["payout_probability"] >= stats["pass_probability"]

    def test_side_by_side_prints_payout_under_pass(self):
        from run_orb_flat import side_by_side, window_stats

        a = window_stats(self._priced(), "A", paths=200)
        b = window_stats(self._priced(), "B", paths=200)
        text = side_by_side(a, b)
        assert "Pass probability" in text
        assert "Payout probability" in text
        assert text.index("Payout probability") > text.index("Pass probability")


class TestTrailingDrawdownHalt:
    def _trades(self, pnls, start=date(2025, 7, 14)):
        return pd.DataFrame([
            {"session_date": start + timedelta(days=i), "net_pnl": float(pnl)}
            for i, pnl in enumerate(pnls)
        ])

    def test_no_halt_while_inside_the_limit(self):
        trades = self._trades([-200, -300, -400])
        kept, halts = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 3
        assert halts.empty

    def test_halts_once_the_trail_is_breached(self):
        # -1,600 by the end of day 2, so day 3 opens past the $1,500 line.
        trades = self._trades([-800, -800, +500, +500])
        kept, halts = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 2
        assert len(halts) == 2
        assert halts.iloc[0]["drawdown"] == pytest.approx(1600.0)

    def test_the_day_that_breaches_is_kept(self):
        """The guard blocks the next entry, it does not undo the day just had."""
        trades = self._trades([-800, -800])
        kept, _ = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 2
        assert kept["net_pnl"].sum() == pytest.approx(-1600.0)

    def test_the_trail_follows_the_peak_not_the_start(self):
        # +2,000 first, so the floor rises; -1,600 from there is not yet a halt
        # relative to the starting balance but is relative to the peak.
        trades = self._trades([+2000, -800, -800, +100])
        kept, halts = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 3
        assert len(halts) == 1
        assert halts.iloc[0]["peak"] == pytest.approx(rules.ACCOUNT_SIZE + 2000)

    def test_the_halt_is_not_permanent(self):
        """Matching the script: recovery inside the limit resumes trading."""
        trades = self._trades([-800, -800, +1, +1])
        kept, halts = enforce_trailing_drawdown_halt(trades, limit=1550.0)
        # -1,600 halts day 3; nothing recovers, so day 4 stays halted too.
        assert len(kept) == 2 and len(halts) == 2

        # Now give it a peak to recover toward: the trail is measured from the
        # running peak, so a smaller drawdown lets trading continue.
        trades = self._trades([+1000, -800, -400, +50])
        kept, halts = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 4 and halts.empty

    def test_defaults_come_from_the_rules_module(self):
        """Rule 9: no restated threshold, no bypass path."""
        trades = self._trades([-(rules.TRAILING_DD_STOP + 1), 100])
        kept, halts = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 1 and len(halts) == 1

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["session_date", "net_pnl"])
        kept, halts = enforce_trailing_drawdown_halt(empty)
        assert kept.empty and halts.empty

    def test_multiple_trades_on_a_halted_day_all_go(self):
        trades = pd.DataFrame([
            {"session_date": date(2025, 7, 14), "net_pnl": -1600.0},
            {"session_date": date(2025, 7, 15), "net_pnl": 100.0},
            {"session_date": date(2025, 7, 15), "net_pnl": 200.0},
        ])
        kept, halts = enforce_trailing_drawdown_halt(trades)
        assert len(kept) == 1
        assert len(halts) == 1
