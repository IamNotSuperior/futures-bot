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
    MES, MNQ, ContractSpec, CostModel, build_trades,
    enforce_daily_loss_limit, price_trades,
)
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


@pytest.fixture
def priced(raw_trades) -> pd.DataFrame:
    return price_trades(raw_trades, spec=MES, costs=CostModel(), contracts=1)


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
        assert CostModel(commission_per_side=1.25).commission_round_turn() == 2.50
        assert CostModel(commission_per_side=1.25).commission_round_turn(2) == 5.00


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
                "exit_price": 4940.0,  # -60 pts = -$300 gross
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
        assert breaches.iloc[0]["worst_running_pnl"] <= -300
        assert breaches.iloc[0]["closing_pnl"] > 0


class TestDailyLossEnforcement:
    """Trading stops for the day once realised P&L reaches the limit."""

    def _day(self, day: str, pnls: list[float]) -> pd.DataFrame:
        """Trades on one session engineered to realise the given P&L values.

        Each is a long of `pnl/5 + 1` points before costs, so after the $5.00
        round-turn cost the net is exactly `pnl`.
        """
        rows = []
        for i, pnl in enumerate(pnls):
            move = pnl / 5.0 + 1.0  # +1 point covers commission + slippage
            rows.append(
                {
                    "entry_time": pd.Timestamp(f"{day} {10 + i}:00", tz=ET),
                    "exit_time": pd.Timestamp(f"{day} {10 + i}:30", tz=ET),
                    "direction": "long",
                    "entry_price": 5000.0,
                    "exit_price": 5000.0 + move,
                    "exit_reason": "stop" if pnl < 0 else "target",
                }
            )
        return price_trades(pd.DataFrame(rows))

    def test_fixture_realises_the_intended_pnl(self):
        priced = self._day("2025-07-14", [-100.0, -250.0, 50.0])
        assert list(priced["net_pnl"].round(2)) == [-100.0, -250.0, 50.0]

    def test_trades_after_the_breach_are_dropped(self):
        priced = self._day("2025-07-14", [-100.0, -250.0, 50.0, -20.0])
        kept, halts = enforce_daily_loss_limit(priced, limit=300.0)
        # -100 then -350 breaches; the breaching trade stands, the rest go.
        assert len(kept) == 2
        assert list(kept["net_pnl"].round(2)) == [-100.0, -250.0]
        assert len(halts) == 1
        assert halts.iloc[0]["realised_at_halt"] == pytest.approx(-350.0)
        assert halts.iloc[0]["trades_blocked"] == 2

    def test_day_that_never_breaches_is_untouched(self):
        priced = self._day("2025-07-14", [-100.0, -150.0, 40.0])
        kept, halts = enforce_daily_loss_limit(priced, limit=300.0)
        assert len(kept) == 3
        assert len(halts) == 0

    def test_breach_exactly_at_the_limit_halts(self):
        priced = self._day("2025-07-14", [-300.0, 100.0])
        kept, halts = enforce_daily_loss_limit(priced, limit=300.0)
        assert len(kept) == 1
        assert len(halts) == 1

    def test_halt_does_not_leak_into_the_next_session(self):
        a = self._day("2025-07-14", [-400.0, 100.0])
        b = self._day("2025-07-15", [75.0, 25.0])
        kept, halts = enforce_daily_loss_limit(
            pd.concat([a, b], ignore_index=True), limit=300.0
        )
        assert len(halts) == 1
        assert len(kept) == 3  # one from the halted day, both from the next
        assert kept[kept["session_date"] == date(2025, 7, 15)].shape[0] == 2

    def test_enforcement_cannot_worsen_a_day(self):
        priced = self._day("2025-07-14", [-400.0, -100.0, -100.0])
        kept, _ = enforce_daily_loss_limit(priced, limit=300.0)
        assert kept["net_pnl"].sum() >= priced["net_pnl"].sum()

    def test_run_backtest_flag_toggles_enforcement(self):
        priced = self._day("2025-07-14", [-400.0, 100.0])
        kept, halts = enforce_daily_loss_limit(priced, limit=300.0)
        assert len(kept) == 1 and len(halts) == 1

    def test_empty_input(self):
        kept, halts = enforce_daily_loss_limit(pd.DataFrame(), limit=300.0)
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
