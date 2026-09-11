"""Tests for the backtest engine and metrics.

The core fixture is five trades whose P&L can be checked with a calculator:

    MES: $5.00/point, tick 0.25
    commission $1.25/side  -> $2.50 round turn
    slippage 1 tick/side   -> 0.25 points against each fill

    #  dir    signal in -> out    fills            points    P&L
    1  long   5000.00 -> 5010.00  5000.25/5009.75   +9.50   +45.00
    2  long   5000.00 -> 4995.00  5000.25/4994.75   -5.50   -30.00
    3  short  5000.00 -> 4990.00  4999.75/4990.25   +9.50   +45.00
    4  short  5000.00 -> 5004.00  4999.75/5004.25   -4.50   -25.00
    5  long   5000.00 -> 5000.50  5000.25/5000.25    0.00    -2.50

    net = 45 - 30 + 45 - 25 - 2.50 = +32.50
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine import (
    MES, MNQ, ContractSpec, CostModel, apply_internal_guards, build_trades,
    enforce_daily_loss_limit, enforce_trailing_drawdown_halt, price_trades,
)
import rules
from metrics import compute_metrics, max_drawdown, sharpe_ratio

ET = "America/New_York"

RAW_TRADES = [
    # (day, entry_clock, exit_clock, direction, entry, exit, reason)
    ("2025-07-14", "10:00", "10:30", "long", 5000.00, 5010.00, "target"),
    ("2025-07-14", "11:00", "11:20", "long", 5000.00, 4995.00, "stop"),
    ("2025-07-15", "10:00", "10:45", "short", 5000.00, 4990.00, "target"),
    ("2025-07-16", "10:00", "10:15", "short", 5000.00, 5004.00, "stop"),
    ("2025-07-16", "11:00", "15:55", "long", 5000.00, 5000.50, "session_end"),
]

EXPECTED_NET = [45.00, -30.00, 45.00, -25.00, -2.50]
EXPECTED_TOTAL = 32.50


@pytest.fixture
def raw_trades() -> pd.DataFrame:
    rows = []
    for day, t_in, t_out, direction, entry, exit_, reason in RAW_TRADES:
        rows.append(
            {
                "entry_time": pd.Timestamp(f"{day} {t_in}", tz=ET),
                "exit_time": pd.Timestamp(f"{day} {t_out}", tz=ET),
                "direction": direction,
                "entry_price": entry,
                "exit_price": exit_,
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(rows)


#: The cost model every figure in this file was worked out at: $1.25 a side,
#: 1 tick. The default commission is now Lucid's verified $0.50, so tests that
#: pin arithmetic name the historical figure rather than lean on the default.
HISTORICAL = CostModel(commission_per_side=rules.ASSUMED_COMMISSION_PER_SIDE)


@pytest.fixture
def priced(raw_trades) -> pd.DataFrame:
    return price_trades(raw_trades, spec=MES, costs=HISTORICAL, contracts=1)


class TestContractSpec:
    def test_mes_defaults(self):
        assert MES.tick_size == 0.25
        assert MES.tick_value == 1.25
        assert MES.point_value == 5.0

    def test_mnq_defaults(self):
        assert MNQ.point_value == 2.0

    def test_inconsistent_spec_is_rejected(self):
        with pytest.raises(ValueError, match="Inconsistent spec"):
            ContractSpec(tick_size=0.25, tick_value=1.25, point_value=4.0)


class TestCostModel:
    def test_round_turn_commission(self):
        assert HISTORICAL.commission_round_turn() == 2.50
        assert HISTORICAL.commission_round_turn(2) == 5.00

    def test_default_commission_is_the_verified_lucid_rate(self):
        """Rule 9: the default is read from rules, not restated. $0.50 a side,
        $1.00 a round turn, per Lucid support (article 11508978)."""
        assert CostModel().commission_per_side == rules.COMMISSION_PER_SIDE
        assert CostModel().commission_round_turn() == pytest.approx(1.00)

    def test_default_slippage_is_unchanged_at_one_tick(self):
        assert CostModel().slippage_ticks == 1.0


class TestBracketBreakEven:
    """Hit rate at which a fixed stop/target bracket nets zero after costs.

    Entry 4's 10/18 bracket at $1.25 and 1 tick: a winner nets $85, a loser
    $55, so the break-even is 55/140 = 39.29%. At the verified $0.50 the
    winner nets $86.50 and the loser $53.50: 53.5/140 = 38.21%.
    """

    def test_entry_four_bracket_at_the_historical_cost(self):
        from engine import bracket_break_even

        assert bracket_break_even(10.0, 18.0, MES, HISTORICAL) == pytest.approx(55.0 / 140.0)

    def test_entry_four_bracket_at_the_verified_cost(self):
        from engine import bracket_break_even

        assert bracket_break_even(10.0, 18.0, MES, CostModel()) == pytest.approx(53.5 / 140.0)
        assert bracket_break_even(10.0, 18.0) == pytest.approx(53.5 / 140.0)

    def test_free_trading_is_the_bare_ratio(self):
        from engine import bracket_break_even

        free = CostModel(commission_per_side=0.0, slippage_ticks=0.0)
        assert bracket_break_even(10.0, 18.0, MES, free) == pytest.approx(10.0 / 28.0)

    def test_size_cancels(self):
        from engine import bracket_break_even

        assert bracket_break_even(10.0, 18.0, MES, HISTORICAL, contracts=4) == \
            pytest.approx(bracket_break_even(10.0, 18.0, MES, HISTORICAL))


class TestTradePricing:
    def test_fills_include_slippage(self, priced):
        # Long pays up on entry, sells down on exit.
        assert priced.loc[0, "entry_fill"] == pytest.approx(5000.25)
        assert priced.loc[0, "exit_fill"] == pytest.approx(5009.75)
        # Short sells down on entry, buys up on exit.
        assert priced.loc[2, "entry_fill"] == pytest.approx(4999.75)
        assert priced.loc[2, "exit_fill"] == pytest.approx(4990.25)

    def test_net_points(self, priced):
        assert list(priced["net_points"].round(4)) == [9.50, -5.50, 9.50, -4.50, 0.00]

    def test_gross_points_exclude_slippage(self, priced):
        assert list(priced["gross_points"].round(4)) == [10.0, -5.0, 10.0, -4.0, 0.5]

    def test_net_pnl_per_trade(self, priced):
        assert list(priced["net_pnl"].round(2)) == EXPECTED_NET

    def test_total_net_pnl(self, priced):
        assert priced["net_pnl"].sum() == pytest.approx(EXPECTED_TOTAL)

    def test_cost_components(self, priced):
        assert (priced["commission"] == 2.50).all()
        assert priced["slippage_cost"].round(6).eq(2.50).all()

    def test_costs_account_for_the_gap_to_gross(self, priced):
        gross = priced["gross_pnl"].sum()
        costs = priced["commission"].sum() + priced["slippage_cost"].sum()
        assert priced["net_pnl"].sum() == pytest.approx(gross - costs)

    def test_durations(self, priced):
        assert priced.loc[0, "duration_seconds"] == 30 * 60
        assert priced.loc[3, "duration_seconds"] == 15 * 60

    def test_two_contracts_double_pnl_and_commission(self, raw_trades):
        one = price_trades(raw_trades, MES, CostModel(), contracts=1)
        two = price_trades(raw_trades, MES, CostModel(), contracts=2)
        assert two["commission"].sum() == pytest.approx(2 * one["commission"].sum())
        assert two["net_pnl"].sum() == pytest.approx(2 * one["net_pnl"].sum())

    def test_zero_costs_recovers_gross(self, raw_trades):
        free = price_trades(
            raw_trades, MES, CostModel(commission_per_side=0.0, slippage_ticks=0.0)
        )
        assert free["net_pnl"].sum() == pytest.approx(free["gross_pnl"].sum())
        assert free["net_pnl"].sum() == pytest.approx(57.50)  # 32.50 + 5 x 5.00 costs


class TestMetrics:
    def test_headline_numbers(self, priced):
        m = compute_metrics(priced)
        assert m["trade_count"] == 5
        assert m["net_pnl"] == pytest.approx(EXPECTED_TOTAL)
        assert m["win_count"] == 2
        assert m["loss_count"] == 3
        assert m["win_rate_pct"] == pytest.approx(40.0)
        assert m["avg_win"] == pytest.approx(45.00)
        assert m["avg_loss"] == pytest.approx(-19.1666667)
        # 90.00 of wins against 57.50 of losses
        assert m["profit_factor"] == pytest.approx(90.0 / 57.5)

    def test_daily_aggregation(self, priced):
        m = compute_metrics(priced)
        assert m["trading_days"] == 3
        # day1 = 45 - 30 = 15, day2 = 45, day3 = -25 - 2.50 = -27.50
        assert m["max_daily_loss"] == pytest.approx(-27.50)
        assert m["max_daily_loss_date"] == date(2025, 7, 16)
        assert m["best_day"] == pytest.approx(45.00)

    def test_percentages_of_total_profit(self, priced):
        m = compute_metrics(priced)
        assert m["worst_day_pct_of_total_profit"] == pytest.approx(-27.5 / 32.5 * 100)
        assert m["best_day_pct_of_total_profit"] == pytest.approx(45.0 / 32.5 * 100)
        assert m["consistency_breach"] is True  # 138% of total profit in one day

    def test_duration_metrics(self, priced):
        m = compute_metrics(priced)
        expected = np.mean([1800, 1200, 2700, 900, 17700])
        assert m["avg_duration_seconds"] == pytest.approx(expected)
        assert m["shortest_hold_violates_rule"] is False

    def test_microscalp_is_zero_when_no_short_trades(self, priced):
        m = compute_metrics(priced)
        assert m["microscalp_trade_count"] == 0
        assert m["microscalp_profit_pct"] == pytest.approx(0.0)
        assert m["microscalp_breach"] is False

    def test_by_exit_reason(self, priced):
        m = compute_metrics(priced)
        by = m["by_exit_reason"]
        assert by.loc["target", "net_pnl"] == pytest.approx(90.0)
        assert by.loc["stop", "net_pnl"] == pytest.approx(-55.0)
        assert by.loc["session_end", "net_pnl"] == pytest.approx(-2.5)
        assert by["trades"].sum() == 5

    def test_no_daily_loss_breach_in_fixture(self, priced):
        m = compute_metrics(priced)
        assert len(m["daily_loss_breaches"]) == 0

    def test_percentages_are_na_when_unprofitable(self, raw_trades):
        expensive = price_trades(raw_trades, MES, CostModel(commission_per_side=50.0))
        m = compute_metrics(expensive)
        assert m["net_pnl"] < 0
        assert m["worst_day_pct_of_total_profit"] is None
        assert m["consistency_breach"] is None

    def test_empty_trades(self):
        m = compute_metrics(pd.DataFrame())
        assert m["trade_count"] == 0


class TestDailyLossBreach:
    def test_breach_detected_on_running_intraday_total(self):
        """A day that dips past the limit counts even if it recovers by the close."""
        rows = [
            {
                "entry_time": pd.Timestamp("2025-07-14 10:00", tz=ET),
                "exit_time": pd.Timestamp("2025-07-14 10:30", tz=ET),
                "direction": "long",
                "entry_price": 5000.0,
                "exit_price": 4915.0,  # -85 pts = -$425 gross, past the $400 limit
                "exit_reason": "stop",
            },
            {
                "entry_time": pd.Timestamp("2025-07-14 11:00", tz=ET),
                "exit_time": pd.Timestamp("2025-07-14 11:30", tz=ET),
                "direction": "long",
                "entry_price": 5000.0,
                "exit_price": 5100.0,  # big recovery
                "exit_reason": "target",
            },
        ]
        priced = price_trades(pd.DataFrame(rows))
        m = compute_metrics(priced)
        breaches = m["daily_loss_breaches"]
        assert len(breaches) == 1
        assert breaches.iloc[0]["worst_running_pnl"] <= -rules.DAILY_LOSS_LIMIT
        assert breaches.iloc[0]["closing_pnl"] > 0


class TestDailyLossEnforcement:
    """Trading halts once realised + open P&L reaches the limit.

    Bars are flat at the entry price except where a test moves them, so the
    mark-to-market value of an open position is easy to reason about.
    """

    DAY = "2025-07-14"

    def bars(self, closes, default=5000.0):
        """5-minute session bars, flat at `default` except where overridden."""
        idx = pd.date_range(f"{self.DAY} 09:30", f"{self.DAY} 15:55",
                            freq="5min", tz=ET)
        px = pd.Series(default, index=idx, dtype=float)
        for clock, value in closes.items():
            px.loc[f"{self.DAY} {clock}"] = value
        return pd.DataFrame(
            {"open": px, "high": px, "low": px, "close": px, "volume": 1}, index=idx
        )

    def trade(self, t_in, t_out, direction="long", entry=5000.0, exit_=5000.0,
              reason="target"):
        return {
            "entry_time": pd.Timestamp(f"{self.DAY} {t_in}", tz=ET),
            "exit_time": pd.Timestamp(f"{self.DAY} {t_out}", tz=ET),
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_,
            "exit_reason": reason,
        }

    def test_open_loss_alone_triggers_the_halt(self):
        """A single position never closed by the strategy still halts the day.

        Long filled at 5000.25 after slippage. At 4935 the open loss is
        (4935 - 5000.25) x $5 = -$326.25, less the round-turn commission: past
        -$300 at either the $1.25 or the confirmed $0.50 rate.
        """
        trades = price_trades(pd.DataFrame([
            self.trade("10:00", "15:55", exit_=5000.0, reason="session_end")
        ]))
        bars = self.bars({"10:30": 4935.0})
        kept, halts = enforce_daily_loss_limit(trades, bars, limit=300.0)

        assert len(halts) == 1
        assert bool(halts.iloc[0]["forced_flatten"]) is True
        assert halts.iloc[0]["halt_time"] == pd.Timestamp(f"{self.DAY} 10:30", tz=ET)
        # Flattened at the NEXT bar's open, not the strategy's intended exit.
        assert kept.loc[0, "exit_time"] == pd.Timestamp(f"{self.DAY} 10:35", tz=ET)
        assert kept.loc[0, "exit_reason"] == "loss_limit_flatten"

    def test_shallow_open_loss_does_not_halt(self):
        trades = price_trades(pd.DataFrame([
            self.trade("10:00", "15:55", exit_=5000.0, reason="session_end")
        ]))
        bars = self.bars({"10:30": 4960.0})  # about -$204, inside the limit
        kept, halts = enforce_daily_loss_limit(trades, bars, limit=300.0)
        assert len(halts) == 0
        assert kept.loc[0, "exit_reason"] == "session_end"

    def test_short_position_marks_the_other_way(self):
        trades = price_trades(pd.DataFrame([
            self.trade("10:00", "15:55", direction="short", exit_=5000.0,
                       reason="session_end")
        ]))
        # Short filled at 4999.75; at 5065 the open loss is -$326.25.
        kept, halts = enforce_daily_loss_limit(
            trades, self.bars({"10:30": 5065.0}), limit=300.0
        )
        assert len(halts) == 1
        assert kept.loc[0, "exit_reason"] == "loss_limit_flatten"

    def test_realised_losses_bring_the_halt_forward(self):
        """A prior realised loss means a smaller open loss is enough."""
        trades = price_trades(pd.DataFrame([
            self.trade("09:45", "09:55", exit_=4960.0, reason="stop"),
            self.trade("10:00", "15:55", exit_=5000.0, reason="session_end"),
        ]))
        kept, halts = enforce_daily_loss_limit(
            trades, self.bars({"10:30": 4978.0}), limit=300.0
        )
        assert len(halts) == 1
        assert kept.loc[1, "exit_reason"] == "loss_limit_flatten"

    def test_later_trades_are_blocked_after_a_halt(self):
        trades = price_trades(pd.DataFrame([
            self.trade("10:00", "11:00", exit_=5000.0, reason="session_end"),
            self.trade("11:30", "12:00", exit_=5010.0, reason="target"),
            self.trade("12:30", "13:00", exit_=5010.0, reason="target"),
        ]))
        kept, halts = enforce_daily_loss_limit(
            trades, self.bars({"10:30": 4935.0}), limit=300.0
        )
        assert len(kept) == 1
        assert int(halts.iloc[0]["trades_blocked"]) == 2

    def test_forced_exit_repriced_at_the_next_bar_open(self):
        trades = price_trades(pd.DataFrame([
            self.trade("10:00", "15:55", exit_=5000.0, reason="session_end")
        ]), costs=HISTORICAL)
        bars = self.bars({"10:30": 4935.0, "10:35": 4930.0})
        kept, _ = enforce_daily_loss_limit(trades, bars, costs=HISTORICAL, limit=300.0)
        # Exits at the 10:35 open of 4930, less one tick of slippage on the sell.
        assert kept.loc[0, "exit_price"] == pytest.approx(4930.0)
        assert kept.loc[0, "exit_fill"] == pytest.approx(4929.75)
        expected = (4929.75 - 5000.25) * 5.0 - HISTORICAL.commission_round_turn()
        assert expected == pytest.approx(-355.0)
        assert kept.loc[0, "net_pnl"] == pytest.approx(expected)

    def test_breach_on_the_final_bar_closes_at_that_bar(self):
        trades = price_trades(pd.DataFrame([
            self.trade("15:00", "15:55", exit_=5000.0, reason="session_end")
        ]))
        bars = self.bars({"15:50": 4935.0})
        kept, halts = enforce_daily_loss_limit(trades, bars, limit=300.0)
        assert len(halts) == 1
        assert kept.loc[0, "exit_time"] == pd.Timestamp(f"{self.DAY} 15:55", tz=ET)

    def test_adverse_mark_halts_where_close_mark_does_not(self):
        """An intrabar dip that recovers by the close still hits real equity."""
        idx = pd.date_range(f"{self.DAY} 09:30", f"{self.DAY} 15:55",
                            freq="5min", tz=ET)
        px = pd.Series(5000.0, index=idx, dtype=float)
        bars = pd.DataFrame(
            {"open": px, "high": px, "low": px.copy(), "close": px, "volume": 1},
            index=idx,
        )
        bars.loc[f"{self.DAY} 10:30", "low"] = 4930.0

        trades = price_trades(pd.DataFrame([
            self.trade("10:00", "15:55", exit_=5000.0, reason="session_end")
        ]))
        _, close_halts = enforce_daily_loss_limit(trades, bars, limit=300.0,
                                                  mark="close")
        _, adverse_halts = enforce_daily_loss_limit(trades, bars, limit=300.0,
                                                    mark="adverse")
        assert len(close_halts) == 0
        assert len(adverse_halts) == 1

    def test_invalid_mark_rejected(self):
        trades = price_trades(pd.DataFrame([self.trade("10:00", "11:00")]))
        with pytest.raises(ValueError, match="mark must be"):
            enforce_daily_loss_limit(trades, self.bars({}), mark="midpoint")

    def test_halt_does_not_leak_into_the_next_session(self):
        day2 = "2025-07-15"
        rows = [
            self.trade("10:00", "15:55", exit_=5000.0, reason="session_end"),
            {
                "entry_time": pd.Timestamp(f"{day2} 10:00", tz=ET),
                "exit_time": pd.Timestamp(f"{day2} 10:30", tz=ET),
                "direction": "long",
                "entry_price": 5000.0,
                "exit_price": 5010.0,
                "exit_reason": "target",
            },
        ]
        idx2 = pd.date_range(f"{day2} 09:30", f"{day2} 15:55", freq="5min", tz=ET)
        px2 = pd.Series(5000.0, index=idx2, dtype=float)
        bars2 = pd.DataFrame(
            {"open": px2, "high": px2, "low": px2, "close": px2, "volume": 1},
            index=idx2,
        )
        bars = pd.concat([self.bars({"10:30": 4935.0}), bars2])
        kept, halts = enforce_daily_loss_limit(
            price_trades(pd.DataFrame(rows)), bars, limit=300.0
        )
        assert len(halts) == 1
        assert halts.iloc[0]["session_date"] == date(2025, 7, 14)
        assert len(kept) == 2  # the next session trades normally

    def test_empty_input(self):
        kept, halts = enforce_daily_loss_limit(pd.DataFrame(), pd.DataFrame(),
                                               limit=300.0)
        assert kept.empty and halts.empty


class TestDrawdownAndSharpe:
    def test_max_drawdown(self):
        curve = pd.Series([100.0, 150.0, 60.0, 90.0, 40.0])
        assert max_drawdown(curve) == pytest.approx(-110.0)  # 150 -> 40

    def test_drawdown_measures_from_zero_when_never_profitable(self):
        curve = pd.Series([-10.0, -25.0, -15.0])
        assert max_drawdown(curve) == pytest.approx(-25.0)

    def test_sharpe_matches_the_formula(self):
        daily = pd.Series([10.0, -5.0, 20.0, 15.0])
        expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(daily) == pytest.approx(expected)

    def test_sharpe_undefined_for_constant_or_single_day(self):
        assert np.isnan(sharpe_ratio(pd.Series([5.0])))
        assert np.isnan(sharpe_ratio(pd.Series([5.0, 5.0, 5.0])))


class TestBuildTrades:
    def _bars(self, index, opens):
        return pd.DataFrame(
            {"open": opens, "high": opens, "low": opens, "close": opens}, index=index
        )

    def test_pairs_entry_with_exit(self):
        idx = pd.date_range("2025-07-14 09:45", periods=4, freq="5min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [False, True, False, False],
                "entry_short": [False, False, False, False],
                "exit_long": [False, False, False, True],
                "exit_short": [False, False, False, False],
                "exit_price": [None, None, None, 5010.0],
                "exit_reason": [None, None, None, "target"],
            },
            index=idx,
        )
        bars = self._bars(idx, [5000.0, 5001.0, 5002.0, 5003.0])
        trades = build_trades(signals, bars)
        assert len(trades) == 1
        assert trades.loc[0, "entry_price"] == 5001.0  # the bar's open
        assert trades.loc[0, "exit_price"] == 5010.0  # the strategy's level
        assert trades.loc[0, "exit_reason"] == "target"

    def test_unclosed_entry_is_dropped(self):
        idx = pd.date_range("2025-07-14 09:45", periods=3, freq="5min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [False, True, False],
                "entry_short": [False, False, False],
                "exit_long": [False, False, False],
                "exit_short": [False, False, False],
            },
            index=idx,
        )
        trades = build_trades(signals, self._bars(idx, [1.0, 2.0, 3.0]))
        assert trades.empty

    def test_second_entry_while_open_is_ignored(self):
        idx = pd.date_range("2025-07-14 09:45", periods=5, freq="5min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [True, True, False, False, False],
                "entry_short": [False, False, False, False, False],
                "exit_long": [False, False, False, True, False],
                "exit_short": [False, False, False, False, False],
                "exit_price": [None, None, None, 5010.0, None],
                "exit_reason": [None, None, None, "target", None],
            },
            index=idx,
        )
        trades = build_trades(signals, self._bars(idx, [5000.0] * 5))
        assert len(trades) == 1


class TestEntryPriceColumn:
    """A strategy entering on a resting order publishes its own fill price.

    Added for ORB-2, whose stop order fills at its own level rather than at the
    open of the bar that reached it. The fallback must stay exactly as it was
    for every strategy that publishes no such column.
    """

    def _bars(self, index, opens):
        return pd.DataFrame(
            {"open": opens, "high": opens, "low": opens, "close": opens}, index=index
        )

    def _signals(self, idx, **extra):
        base = {
            "entry_long": [False, True, False, False],
            "entry_short": [False, False, False, False],
            "exit_long": [False, False, False, True],
            "exit_short": [False, False, False, False],
            "exit_price": [None, None, None, 5010.0],
            "exit_reason": [None, None, None, "target"],
        }
        base.update(extra)
        return pd.DataFrame(base, index=idx)

    def test_published_entry_price_is_used(self):
        idx = pd.date_range("2025-07-14 09:45", periods=4, freq="5min", tz=ET)
        signals = self._signals(idx, entry_price=[None, 5007.5, None, None])
        trades = build_trades(signals, self._bars(idx, [5000.0, 5001.0, 5002.0, 5003.0]))
        assert trades.loc[0, "entry_price"] == 5007.5

    def test_absent_column_still_fills_at_the_open(self):
        idx = pd.date_range("2025-07-14 09:45", periods=4, freq="5min", tz=ET)
        trades = build_trades(
            self._signals(idx), self._bars(idx, [5000.0, 5001.0, 5002.0, 5003.0])
        )
        assert trades.loc[0, "entry_price"] == 5001.0

    def test_null_entry_price_falls_back_to_the_open(self):
        idx = pd.date_range("2025-07-14 09:45", periods=4, freq="5min", tz=ET)
        signals = self._signals(idx, entry_price=[None, None, None, None])
        trades = build_trades(signals, self._bars(idx, [5000.0, 5001.0, 5002.0, 5003.0]))
        assert trades.loc[0, "entry_price"] == 5001.0


class TestSameBarEntryAndExit:
    """A trade opened and closed inside one bar must survive.

    ORB-2 can fill a resting stop and be stopped out in the same minute. The
    engine used to drop such a trade: the exit was consumed while no position
    was open, then the entry opened and never closed.
    """

    def _bars(self, index, opens):
        return pd.DataFrame(
            {"open": opens, "high": opens, "low": opens, "close": opens}, index=index
        )

    def test_entry_and_exit_on_one_bar_produces_a_trade(self):
        idx = pd.date_range("2025-07-14 09:45", periods=3, freq="1min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [False, True, False],
                "entry_short": [False, False, False],
                "exit_long": [False, True, False],
                "exit_short": [False, False, False],
                "entry_price": [None, 5003.0, None],
                "exit_price": [None, 4993.0, None],
                "exit_reason": [None, "stop", None],
            },
            index=idx,
        )
        trades = build_trades(signals, self._bars(idx, [5000.0, 5001.0, 5002.0]))
        assert len(trades) == 1
        row = trades.loc[0]
        assert row["entry_time"] == row["exit_time"] == idx[1]
        assert row["entry_price"] == 5003.0
        assert row["exit_price"] == 4993.0
        assert row["exit_reason"] == "stop"

    def test_short_side_mirrors(self):
        idx = pd.date_range("2025-07-14 09:45", periods=3, freq="1min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [False, False, False],
                "entry_short": [False, True, False],
                "exit_long": [False, False, False],
                "exit_short": [False, True, False],
                "entry_price": [None, 4997.0, None],
                "exit_price": [None, 5007.0, None],
                "exit_reason": [None, "stop", None],
            },
            index=idx,
        )
        trades = build_trades(signals, self._bars(idx, [5000.0] * 3))
        assert len(trades) == 1
        assert trades.loc[0, "direction"] == "short"

    def test_an_unrelated_exit_flag_does_not_close_a_new_entry(self):
        """A long entry must not be closed by a short exit mark on the bar."""
        idx = pd.date_range("2025-07-14 09:45", periods=3, freq="1min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [False, True, False],
                "entry_short": [False, False, False],
                "exit_long": [False, False, True],
                "exit_short": [False, True, False],
                "exit_price": [None, 4993.0, 5010.0],
                "exit_reason": [None, "stop", "target"],
            },
            index=idx,
        )
        trades = build_trades(signals, self._bars(idx, [5000.0] * 3))
        assert len(trades) == 1
        assert trades.loc[0, "exit_time"] == idx[2]
        assert trades.loc[0, "exit_reason"] == "target"

    def test_unclosed_entry_is_still_dropped(self):
        """The old guarantee must survive the new one."""
        idx = pd.date_range("2025-07-14 09:45", periods=3, freq="1min", tz=ET)
        signals = pd.DataFrame(
            {
                "entry_long": [False, True, False],
                "entry_short": [False, False, False],
                "exit_long": [False, False, False],
                "exit_short": [False, False, False],
            },
            index=idx,
        )
        assert build_trades(signals, self._bars(idx, [1.0, 2.0, 3.0])).empty


# ---------------------------------------------------------------------------
# Rule 5b: the trailing drawdown halt, and the guard pipeline both runners use
# ---------------------------------------------------------------------------


class TestTrailingDrawdownHalt:
    """Moved here from the entry 5 runner's tests; the semantics are unchanged.

    Day-level: a session that opens already past the $1,500 trail is dropped
    whole. The halt has no lock-in, but a halted day takes no trades, so the
    balance cannot move and in practice the halt never releases.
    """

    def _trades(self, pnls, start=date(2025, 7, 14)):
        from datetime import timedelta

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


class TestInternalGuards:
    """``apply_internal_guards``: the daily loss limit, then the trailing halt.

    One pipeline for every runner, so a generated verdict and a hand-built one
    are scored on the same basis. Costs are zero and size is one contract so a
    trade's net P&L is exactly five times its point move.
    """

    FREE = CostModel(commission_per_side=0.0, slippage_ticks=0.0)
    START = date(2025, 7, 14)

    def _bars(self, days: int) -> pd.DataFrame:
        from datetime import timedelta

        frames = []
        for i in range(days):
            day = self.START + timedelta(days=i)
            idx = pd.date_range(f"{day} 09:30", f"{day} 10:35", freq="1min", tz=ET)
            frames.append(pd.DataFrame(
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                 "volume": 100, "instrument_id": 1}, index=idx))
        return pd.concat(frames)

    def _priced(self, per_day: list[list[float]]) -> pd.DataFrame:
        """``per_day[i]`` lists that session's trade P&Ls, taken in sequence."""
        from datetime import timedelta

        rows = []
        for i, pnls in enumerate(per_day):
            day = self.START + timedelta(days=i)
            for j, pnl in enumerate(pnls):
                entry = pd.Timestamp(f"{day} 09:46", tz=ET) + timedelta(minutes=10 * j)
                rows.append({
                    "entry_time": entry,
                    "exit_time": entry + timedelta(minutes=4),
                    "direction": "long", "entry_price": 100.0,
                    "exit_price": 100.0 + pnl / 5.0, "exit_reason": "target",
                })
        return price_trades(pd.DataFrame(rows), MES, self.FREE, 1)

    def test_returns_the_stream_and_both_halt_logs(self):
        priced = self._priced([[100.0], [-50.0]])
        trades, loss_halts, dd_halts = apply_internal_guards(
            priced, self._bars(2), MES, self.FREE, 1)
        assert len(trades) == 2
        assert loss_halts.empty and dd_halts.empty
        assert list(dd_halts.columns) == ["session_date", "balance", "peak", "drawdown"]

    def test_trailing_halt_drops_sessions_past_the_trail(self):
        priced = self._priced([[-800.0], [-800.0], [500.0]])
        trades, loss_halts, dd_halts = apply_internal_guards(
            priced, self._bars(3), MES, self.FREE, 1)
        assert list(trades["net_pnl"]) == [-800.0, -800.0]
        assert loss_halts.empty
        assert len(dd_halts) == 1
        assert dd_halts.iloc[0]["drawdown"] == pytest.approx(1600.0)

    def test_the_halt_can_be_switched_off_for_a_comparable_basis(self):
        priced = self._priced([[-800.0], [-800.0], [500.0]])
        trades, _, dd_halts = apply_internal_guards(
            priced, self._bars(3), MES, self.FREE, 1, trailing_halt=False)
        assert len(trades) == 3
        assert dd_halts.empty
        assert list(dd_halts.columns) == ["session_date", "balance", "peak", "drawdown"]

    def test_daily_loss_limit_runs_before_the_trailing_halt(self):
        """The halt sees the loss-limited stream, not the raw one.

        Day 1 loses $450 and then would have made $2,000; the daily limit
        blocks the second trade, so the day closes at -$450. Day 2 loses
        $1,100, which puts the account $1,550 under its peak: day 3 is halted.
        Had the halt seen the raw +$1,550 day 1, day 3 would have traded.
        """
        priced = self._priced([[-450.0, 2000.0], [-1100.0], [100.0]])
        trades, loss_halts, dd_halts = apply_internal_guards(
            priced, self._bars(3), MES, self.FREE, 1)
        assert list(trades["net_pnl"]) == [-450.0, -1100.0]
        assert len(loss_halts) == 1
        assert int(loss_halts.iloc[0]["trades_blocked"]) == 1
        assert len(dd_halts) == 1

    def test_empty_input(self):
        empty = price_trades(pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price",
            "exit_price", "exit_reason"]), MES, self.FREE, 1)
        trades, loss_halts, dd_halts = apply_internal_guards(
            empty, self._bars(1), MES, self.FREE, 1)
        assert trades.empty and loss_halts.empty and dd_halts.empty
