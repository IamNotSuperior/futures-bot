"""Tests for the parameter scan: the split must be clean, and the scan must
reproduce what the standalone backtest produces for the same parameters.
"""

from datetime import date, time

import pandas as pd
import pytest

import scan
from engine import MES, CostModel, build_trades, enforce_daily_loss_limit, price_trades
from orb import ORBParams, OpeningRangeBreakout, resample_bars
from test_orb import OPENING, bars_from_closes, flat

ET = "America/New_York"
PARQUET = scan.PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
needs_data = pytest.mark.skipif(
    not PARQUET.exists(), reason="cached bars not present"
)


class TestSplitDefinition:
    def test_windows_do_not_overlap(self):
        assert scan.IS_END < scan.OOS_START

    def test_windows_are_adjacent_with_no_gap(self):
        assert scan.OOS_START == scan.IS_END + pd.Timedelta(days=1)

    def test_windows_are_ordered(self):
        assert scan.IS_START < scan.IS_END
        assert scan.OOS_START < scan.OOS_END

    def test_in_sample_precedes_out_of_sample_in_time(self):
        """The split is chronological, not random: no future data in-sample."""
        assert scan.IS_END < scan.OOS_START
        assert scan.IS_START < scan.OOS_START


class TestSliceByDate:
    def _frame(self, days):
        idx = pd.DatetimeIndex(
            [pd.Timestamp(f"{d} 10:00", tz=ET) for d in days]
        )
        return pd.DataFrame({"x": range(len(days))}, index=idx)

    def test_bounds_are_inclusive(self):
        f = self._frame(["2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02"])
        got = scan.slice_by_date(f, date(2025, 12, 31), date(2026, 1, 1))
        assert [ts.date() for ts in got.index] == [date(2025, 12, 31), date(2026, 1, 1)]

    def test_the_two_windows_partition_without_overlap(self):
        days = ["2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02"]
        f = self._frame(days)
        is_part = scan.slice_by_date(f, scan.IS_START, scan.IS_END)
        oos_part = scan.slice_by_date(f, scan.OOS_START, scan.OOS_END)
        is_dates = {ts.date() for ts in is_part.index}
        oos_dates = {ts.date() for ts in oos_part.index}
        assert is_dates & oos_dates == set()
        assert len(is_part) + len(oos_part) == len(f)
        assert max(is_dates) < min(oos_dates)


class TestNoLeakage:
    """Generating over the full span then slicing must equal generating on the
    slice alone. If it did not, out-of-sample bars would be influencing
    in-sample signals."""

    def _three_sessions(self):
        frames = []
        for day, bump in (("2025-07-14", 0.0), ("2025-07-15", 5.0), ("2025-07-16", -3.0)):
            closes = [c + bump for c in OPENING] + flat(102.5 + bump, 4) \
                + [103.0 + bump] + flat(103.0 + bump, 40)
            frames.append(bars_from_closes(closes, day=day))
        return pd.concat(frames)

    def test_sliced_signals_match_signals_generated_on_the_slice(self):
        bars = self._three_sessions()
        bars5 = resample_bars(bars, 5)
        strat = OpeningRangeBreakout(ORBParams())

        full = strat.generate_signals_resampled(bars5)
        sliced = scan.slice_by_date(full, date(2025, 7, 14), date(2025, 7, 15))

        early_bars5 = scan.slice_by_date(bars5, date(2025, 7, 14), date(2025, 7, 15))
        direct = strat.generate_signals_resampled(early_bars5)

        cols = ["entry_long", "entry_short", "exit_long", "exit_short"]
        pd.testing.assert_frame_equal(sliced[cols], direct[cols])

    def test_trades_stay_inside_their_window(self):
        bars = self._three_sessions()
        bars5 = resample_bars(bars, 5)
        strat = OpeningRangeBreakout(ORBParams())
        signals = strat.generate_signals_resampled(bars5)

        summary_in = scan.evaluate(
            signals, bars5, CostModel(), date(2025, 7, 14), date(2025, 7, 15), "is"
        )
        win_signals = scan.slice_by_date(signals, date(2025, 7, 14), date(2025, 7, 15))
        win_bars = scan.slice_by_date(bars5, date(2025, 7, 14), date(2025, 7, 15))
        trades = price_trades(build_trades(win_signals, win_bars))

        assert summary_in["is_trades"] == len(trades)
        for ts in list(trades["entry_time"]) + list(trades["exit_time"]):
            assert date(2025, 7, 14) <= ts.date() <= date(2025, 7, 15)

    def test_window_results_are_independent_of_the_other_window(self):
        """Adding a later session must not change the earlier window's result."""
        bars = self._three_sessions()
        bars5 = resample_bars(bars, 5)
        strat = OpeningRangeBreakout(ORBParams())

        full = strat.generate_signals_resampled(bars5)
        with_all = scan.evaluate(
            full, bars5, CostModel(), date(2025, 7, 14), date(2025, 7, 15), "is"
        )

        two_only = scan.slice_by_date(bars5, date(2025, 7, 14), date(2025, 7, 15))
        partial = strat.generate_signals_resampled(two_only)
        with_two = scan.evaluate(
            partial, two_only, CostModel(), date(2025, 7, 14), date(2025, 7, 15), "is"
        )
        # Compared as Series so NaN (an undefined Sharpe on a short window)
        # counts as equal to NaN rather than failing dict equality.
        pd.testing.assert_series_equal(pd.Series(with_all), pd.Series(with_two))


class TestGrid:
    def test_grid_size(self):
        assert len(scan.parameter_grid()) == 192

    def test_grid_covers_every_combination_once(self):
        keys = {
            (p.opening_range_minutes, p.trade_window_end, p.stop_multiple,
             p.target_multiple)
            for p in scan.parameter_grid()
        }
        assert len(keys) == 192

    def test_defaults_are_in_the_grid(self):
        default = ORBParams()
        assert any(
            p.opening_range_minutes == default.opening_range_minutes
            and p.trade_window_end == default.trade_window_end
            and p.stop_multiple == default.stop_multiple
            and p.target_multiple == default.target_multiple
            for p in scan.parameter_grid()
        )


@pytest.fixture(scope="module")
def context():
    """Real bars, loaded once for the whole module."""
    import loader
    bars = loader.load_bars(PARQUET)
    return {
        "bars5": resample_bars(bars, 5),
        "roll_dates": loader.detect_roll_dates(bars),
        "early_closes": loader.detect_early_close_dates(bars),
    }


@needs_data
class TestReproducesBaseline:
    """The scan's evaluation path must agree with a direct backtest."""

    WINDOW = (date(2025, 1, 1), date(2025, 3, 31))


    def test_scan_path_matches_direct_backtest(self, context):
        start, end = self.WINDOW
        params = ORBParams()  # the documented defaults
        strat = OpeningRangeBreakout(
            params,
            roll_dates=context["roll_dates"],
            early_close_dates=context["early_closes"],
        )
        costs = CostModel()

        # Scan path: generate once over everything, then slice.
        scanned = scan.evaluate(
            strat.generate_signals_resampled(context["bars5"]),
            context["bars5"], costs, start, end, "is",
        )

        # Direct path: restrict the bars first, then generate.
        direct_bars = scan.slice_by_date(context["bars5"], start, end)
        direct_signals = strat.generate_signals_resampled(direct_bars)
        direct_trades = price_trades(
            build_trades(direct_signals, direct_bars), MES, costs, 1
        )
        direct_trades, halts = enforce_daily_loss_limit(
            direct_trades, direct_bars, MES, costs, 1
        )

        assert scanned["is_trades"] == len(direct_trades)
        assert scanned["is_net_pnl"] == pytest.approx(direct_trades["net_pnl"].sum())
        assert scanned["is_loss_limit_halts"] == len(halts)

    def test_default_params_produce_trades(self, context):
        start, end = self.WINDOW
        strat = OpeningRangeBreakout(
            ORBParams(),
            roll_dates=context["roll_dates"],
            early_close_dates=context["early_closes"],
        )
        got = scan.evaluate(
            strat.generate_signals_resampled(context["bars5"]),
            context["bars5"], CostModel(), start, end, "is",
        )
        assert got["is_trades"] > 0
