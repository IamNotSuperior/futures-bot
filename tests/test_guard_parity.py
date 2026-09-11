"""Both runners apply the same internal guards, from the same engine code.

Entry 5's runner carried its own trailing-drawdown halt for a while and the
generated-strategy runner had none, so a generated verdict was scored on a
more permissive basis than a hand-built one. The guards now live in
``engine.apply_internal_guards`` and both runners call it. This file pins
that: a synthetic stream that trips the halt must come out identical through
either runner, and neither runner may carry a halt of its own.
"""

import inspect
from datetime import date, timedelta

import pandas as pd
import pytest

import rules
from engine import MES, CostModel

ET = "America/New_York"
START = date(2025, 7, 14)  # a Monday
CONTRACTS = 4
#: $1.25 a side, 1 tick - the arithmetic below was worked out at this model.
COSTS = CostModel(commission_per_side=rules.ASSUMED_COMMISSION_PER_SIDE)


def synthetic_days(moves: list[float]):
    """One trade per session, entered 09:46 at 100 and exited 10:30 at 100+move.

    Bars are flat at 100 so the daily loss limit never marks a breach intrabar;
    the only guard that can fire is the trailing halt. At 4 contracts and
    ``COSTS`` a move of ``m`` points nets ``20 * m - 20`` dollars.
    """
    bar_frames, signal_frames = [], []
    for i, move in enumerate(moves):
        day = START + timedelta(days=i)
        idx = pd.date_range(f"{day} 09:30", f"{day} 10:35", freq="1min", tz=ET)
        bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                             "close": 100.0, "volume": 100, "instrument_id": 1},
                            index=idx)
        signals = pd.DataFrame({"entry_long": False, "entry_short": False,
                                "exit_long": False, "exit_short": False,
                                "entry_price": float("nan"),
                                "exit_price": float("nan"),
                                "exit_reason": None}, index=idx)
        entry = pd.Timestamp(f"{day} 09:46", tz=ET)
        exit_ = pd.Timestamp(f"{day} 10:30", tz=ET)
        signals.loc[entry, ["entry_long", "entry_price"]] = [True, 100.0]
        signals.loc[exit_, ["exit_long", "exit_price", "exit_reason"]] = \
            [True, 100.0 + move, "session_end"]
        bar_frames.append(bars)
        signal_frames.append(signals)
    return pd.concat(bar_frames), pd.concat(signal_frames)


#: -800, -800, +500, +500 at 4 contracts: day 2 closes $1,600 under the start,
#: past the $1,500 trail, so days 3 and 4 are blocked.
MOVES = [-39.0, -39.0, 26.0, 26.0]


class TestBothRunnersApplyTheSameHalt:
    def test_the_fixture_nets_what_it_claims(self):
        from engine import build_trades, price_trades

        bars, signals = synthetic_days(MOVES)
        priced = price_trades(build_trades(signals, bars), MES, COSTS, CONTRACTS)
        assert list(priced["net_pnl"]) == [-800.0, -800.0, 500.0, 500.0]

    def test_entry_5_runner_and_generated_runner_agree_trade_for_trade(self):
        import run_generated
        import run_orb_flat

        bars, signals = synthetic_days(MOVES)
        costs = COSTS
        end = START + timedelta(days=len(MOVES))

        by_hand, hand_loss, hand_dd = run_orb_flat.build_stream(
            bars, signals, START, end, costs)
        generated, gen_loss, gen_dd, _ = run_generated.guarded_streams(
            signals, bars, MES, costs, CONTRACTS)

        pd.testing.assert_frame_equal(by_hand, generated)
        assert len(by_hand) == 2, "days 3 and 4 open past the trail"
        assert hand_loss.empty and gen_loss.empty
        assert len(hand_dd) == len(gen_dd) == 2
        assert hand_dd.iloc[0]["drawdown"] == pytest.approx(1600.0)

    def test_the_comparable_stream_is_the_same_signals_without_the_halt(self):
        import run_generated
        import run_orb_flat

        bars, signals = synthetic_days(MOVES)
        costs = COSTS
        end = START + timedelta(days=len(MOVES))

        by_hand, _, _ = run_orb_flat.build_stream(
            bars, signals, START, end, costs, dd_halt=False)
        _, _, _, comparable = run_generated.guarded_streams(
            signals, bars, MES, costs, CONTRACTS)

        pd.testing.assert_frame_equal(by_hand, comparable)
        assert len(comparable) == 4

    def test_neither_runner_carries_a_halt_of_its_own(self):
        """The halt is engine code. A runner defining one is the drift that
        produced two scoring bases in the first place."""
        import run_generated
        import run_orb_flat

        for module in (run_orb_flat, run_generated):
            source = inspect.getsource(module)
            assert "def enforce_trailing_drawdown_halt" not in source
            assert "def enforce_daily_loss_limit" not in source
            assert "apply_internal_guards(" in source

    def test_the_halt_threshold_is_not_restated_by_either_runner(self):
        import run_generated
        import run_orb_flat

        for module in (run_orb_flat, run_generated):
            source = inspect.getsource(module)
            assert str(int(rules.TRAILING_DD_STOP)) not in source.replace(
                f"{rules.TRAILING_DD_STOP:,.0f}", "")
