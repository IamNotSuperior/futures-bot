"""Tests for entry 10's runner: the frozen parameter sets and the reproduction check.

The reproduction check is the entry's ninth pre-registered test: the new code,
with the flatten at 09:25, costs at $1.25 and entry 6's range-based sizing,
must reproduce entry 6's 1x/ON stream on MES and entry 7's on MNQ trade for
trade. These tests pin the parameter sets and the comparison itself; the
comparison against the saved streams is run by the script, not here.
"""

from datetime import time

import pandas as pd
import pytest

import rules
from engine import MES, MNQ
from london import LondonParams
from run_entry10 import (
    CRITICAL_Z, ENTRY6_PARAMS, ENTRY10_PARAMS, INSTRUMENTS, MIN_DEPARTURE_POINTS,
    MIN_PASS_PROBABILITY, MIN_PROFITABLE_FOLDS, REPRODUCTION_COSTS, attribution,
    compare_streams, evaluate_criteria, truncate_for_benchmark, verdict_block,
    verdict_status,
)

ET = "America/New_York"


class TestFrozenParameterSets:
    def test_entry_six_params_are_the_module_defaults_plus_point_value(self):
        """The reproduction must run entry 6's rule, not entry 10's."""
        for name, spec in (("MES", MES), ("MNQ", MNQ)):
            p = ENTRY6_PARAMS(spec.point_value)
            assert p == LondonParams(target_multiple=1.0, use_trend_filter=True,
                                     point_value=spec.point_value)
            assert p.flatten_time == time(9, 25)
            assert p.size_on == "range"
            assert p.early_close_flatten_time is None

    def test_entry_ten_params_are_the_frozen_specification(self):
        for name, spec in (("MES", MES), ("MNQ", MNQ)):
            p = ENTRY10_PARAMS(spec.point_value)
            assert p.flatten_time == time(15, 55)
            assert p.early_close_flatten_time == time(12, 55)
            assert p.size_on == "stop"
            assert p.target_multiple == 1.0
            assert p.use_trend_filter is True
            assert p.point_value == spec.point_value
            assert p.risk_dollars == 200.0

    def test_entry_ten_differs_from_entry_six_only_in_flatten_and_sizing(self):
        six = ENTRY6_PARAMS(MES.point_value)
        ten = ENTRY10_PARAMS(MES.point_value)
        changed = {f for f in six.__dataclass_fields__
                   if getattr(six, f) != getattr(ten, f)}
        assert changed == {"flatten_time", "early_close_flatten_time", "size_on"}

    def test_reproduction_costs_are_the_historical_two_tick_base_case(self):
        assert REPRODUCTION_COSTS.commission_per_side == rules.ASSUMED_COMMISSION_PER_SIDE
        assert REPRODUCTION_COSTS.slippage_ticks == 2.0

    def test_both_instruments_and_only_those(self):
        assert set(INSTRUMENTS) == {"MES", "MNQ"}
        assert INSTRUMENTS["MES"]["spec"] is MES
        assert INSTRUMENTS["MNQ"]["spec"] is MNQ


def stream(rows):
    return pd.DataFrame([{
        "entry_time": pd.Timestamp(f"2025-07-{d:02d} 03:05", tz=ET),
        "exit_time": pd.Timestamp(f"2025-07-{d:02d} 09:25", tz=ET),
        "direction": "long", "entry_price": 100.0, "exit_price": 100.0 + move,
        "exit_reason": reason, "contracts": n, "net_pnl": pnl,
    } for d, move, reason, n, pnl in rows])


class TestCompareStreams:
    def test_identical_streams_have_no_differences(self):
        a = stream([(14, 2.0, "target", 2, 10.0), (15, -1.0, "stop", 1, -7.5)])
        assert compare_streams(a, a.copy()) == []

    def test_a_saved_stream_read_back_from_csv_still_matches(self, tmp_path):
        """Timestamps come back as strings spanning EST and EDT."""
        a = stream([(14, 2.0, "target", 2, 10.0)])
        path = tmp_path / "s.csv"
        a.to_csv(path, index=False)
        assert compare_streams(a, pd.read_csv(path)) == []

    def test_a_different_trade_count_is_reported_first(self):
        a = stream([(14, 2.0, "target", 2, 10.0), (15, -1.0, "stop", 1, -7.5)])
        b = stream([(14, 2.0, "target", 2, 10.0)])
        diffs = compare_streams(a, b)
        assert diffs and "2 trades" in diffs[0] and "1 trades" in diffs[0]

    def test_a_changed_exit_is_named_with_its_row(self):
        a = stream([(14, 2.0, "target", 2, 10.0), (15, -1.0, "stop", 1, -7.5)])
        b = a.copy()
        b.loc[1, "exit_reason"] = "flatten_0925"
        diffs = compare_streams(a, b)
        assert any("exit_reason" in d and "2025-07-15" in d for d in diffs)

    def test_a_changed_size_or_pnl_is_caught(self):
        a = stream([(14, 2.0, "target", 2, 10.0)])
        b = a.copy()
        b.loc[0, "contracts"] = 3
        b.loc[0, "net_pnl"] = 15.0
        diffs = compare_streams(a, b)
        assert any("contracts" in d for d in diffs)
        assert any("net_pnl" in d for d in diffs)

    def test_float_noise_is_not_a_difference(self):
        a = stream([(14, 2.0, "target", 2, 10.0)])
        b = a.copy()
        b.loc[0, "net_pnl"] = 10.0 + 1e-9
        assert compare_streams(a, b) == []


# ---------------------------------------------------------------------------
# The 15:55 run: the pieces that decide something
# ---------------------------------------------------------------------------

from datetime import date  # noqa: E402

D1, D2 = date(2025, 7, 16), date(2025, 7, 17)


def minute_bars(day: date, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(f"{day} {start}", f"{day} {end}", freq="1min", tz=ET)
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1}, index=idx)


class TestBenchmarkTruncation:
    """The bootstrap's horizon is entry to the flatten. On an early-close day
    the flatten is 12:55, so the bars it may resample stop there."""

    def test_early_close_dates_are_cut_at_the_early_flatten_inclusive(self):
        bars = pd.concat([minute_bars(D1, "03:00", "16:05"),
                          minute_bars(D2, "03:00", "16:05")])
        out = truncate_for_benchmark(bars, {D2}, time(12, 55))
        d2 = out[out.index.date == D2]
        assert d2.index.max().time() == time(12, 55)
        assert (d2.index.time <= time(12, 55)).all()

    def test_normal_days_are_untouched(self):
        bars = pd.concat([minute_bars(D1, "03:00", "16:05"),
                          minute_bars(D2, "03:00", "16:05")])
        out = truncate_for_benchmark(bars, {D2}, time(12, 55))
        d1 = out[out.index.date == D1]
        pd.testing.assert_frame_equal(d1, bars[bars.index.date == D1])

    def test_no_early_closes_is_the_identity(self):
        bars = minute_bars(D1, "03:00", "16:05")
        pd.testing.assert_frame_equal(truncate_for_benchmark(bars, set(), time(12, 55)), bars)


def _stats(**over):
    base = dict(trades=600, net_pnl=1_000.0, mean_per_trade=1.67, n_resolved=600,
                profitable_years=5,
                pass_probability=0.40, payout_probability=0.5, sharpe=0.5,
                profit_factor=1.1, max_drawdown=-2_000.0, max_daily_loss=-300.0,
                blowups=1, avg_duration_min=300.0, microscalp_pct=0.0,
                targets=360, stops=240, flattens=50, loss_limit_exits=0,
                observed_share=0.60, mean_win=150.0, mean_loss=190.0,
                realised_break_even=0.5588, sizing={1: 10, 2: 20},
                risk_median=165.0, risk_mean=160.0, risk_max=200.0,
                overshoot_median=1.0, overshoot_mean=1.5, overshoot_max=20.0,
                years=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
                year_pnl={y: 100.0 for y in range(2020, 2027)})
    base.update(over)
    return base


class TestKillCriteria:
    """Entry 10's four criteria, each at its pre-registered line."""

    def test_the_lines_are_the_entrys(self):
        assert MIN_PROFITABLE_FOLDS == 4
        assert MIN_PASS_PROBABILITY == 0.35
        assert CRITICAL_Z == 2.5
        assert MIN_DEPARTURE_POINTS == 2.0

    def test_all_four_pass(self):
        # 0.60 against 0.54 on 600 resolved: z = 0.06 / 0.02034 = 2.95.
        crit = evaluate_criteria(_stats(), observed=0.60, benchmark=0.54, n_resolved=600)
        assert [c.name for c in crit] == ["rule_13", "pass_probability", "z", "departure"]
        assert all(c.passed for c in crit)
        assert crit[2].value == pytest.approx(2.95, abs=0.01)
        assert crit[3].value == pytest.approx(6.0)

    def test_rule_13_needs_both_halves(self):
        crit = evaluate_criteria(_stats(profitable_years=4, net_pnl=-1.0),
                                 observed=0.60, benchmark=0.54, n_resolved=600)
        assert crit[0].passed is False
        crit = evaluate_criteria(_stats(profitable_years=3, net_pnl=500.0),
                                 observed=0.60, benchmark=0.54, n_resolved=600)
        assert crit[0].passed is False

    def test_pass_probability_line_is_inclusive_at_35(self):
        below = evaluate_criteria(_stats(pass_probability=0.349), 0.60, 0.54, 600)
        at = evaluate_criteria(_stats(pass_probability=0.35), 0.60, 0.54, 600)
        assert below[1].passed is False and at[1].passed is True

    def test_a_small_departure_fails_z_and_the_floor(self):
        # 0.555 against 0.54: 1.5 points, z = 0.74.
        crit = evaluate_criteria(_stats(), observed=0.555, benchmark=0.54, n_resolved=600)
        assert crit[2].passed is False and crit[3].passed is False

    def test_a_clean_z_on_a_huge_sample_still_needs_two_points(self):
        # 1.5 points on 40,000 resolved: z = 6.0, but the floor says no.
        crit = evaluate_criteria(_stats(), observed=0.555, benchmark=0.54, n_resolved=40_000)
        assert crit[2].passed is True and crit[3].passed is False

    def test_an_effect_the_size_of_entry_sixs_fails(self):
        """+3.96 points on 650 resolved against 0.54: z = 2.03 < 2.5."""
        crit = evaluate_criteria(_stats(), observed=0.5796, benchmark=0.54, n_resolved=650)
        assert crit[2].passed is False


class TestVerdictStatus:
    def _all(self, passed):
        return evaluate_criteria(_stats(), 0.60 if passed else 0.55, 0.54, 600)

    def test_accepted_only_when_every_criterion_passes_on_both(self):
        assert verdict_status({"MES": self._all(True), "MNQ": self._all(True)}) == "ACCEPTED"

    def test_one_failure_on_one_instrument_rejects(self):
        assert verdict_status({"MES": self._all(True), "MNQ": self._all(False)}) == "REJECTED"
        assert verdict_status({"MES": self._all(False), "MNQ": self._all(True)}) == "REJECTED"

    def test_both_instruments_are_required(self):
        with pytest.raises(ValueError, match="MNQ"):
            verdict_status({"MES": self._all(True)})


class TestAttribution:
    """Criterion 5: how much of any result is the extended hold, how much the
    sizing change. A = 15:55/stop, B = 09:25/stop, C = 09:25/range."""

    def test_extended_hold_is_a_minus_b_and_sizing_is_b_minus_c(self):
        a = stream([(14, 2.0, "target", 2, 10.0), (15, -1.0, "stop", 1, -5.0)])
        b = stream([(14, 0.5, "flatten_0925", 2, 4.0), (15, -1.2, "flatten_0925", 1, -6.0)])
        c = stream([(14, 0.5, "flatten_0925", 3, 4.0), (15, -1.2, "flatten_0925", 1, -6.0),
                    (16, 0.4, "flatten_0925", 1, 2.0)])
        got = attribution(a, b, c)
        assert got["a_net"] == pytest.approx(5.0)
        assert got["b_net"] == pytest.approx(-2.0)
        assert got["c_net"] == pytest.approx(0.0)
        assert got["extended_hold"] == pytest.approx(7.0)
        assert got["sizing_change"] == pytest.approx(-2.0)
        assert (got["a_trades"], got["b_trades"], got["c_trades"]) == (2, 2, 3)

    def test_the_two_parts_sum_to_the_whole_change_from_entry_six(self):
        a = stream([(14, 2.0, "target", 2, 10.0)])
        b = stream([(14, 0.5, "flatten_0925", 2, 4.0)])
        c = stream([(14, 0.5, "flatten_0925", 3, 6.0)])
        got = attribution(a, b, c)
        assert got["extended_hold"] + got["sizing_change"] == pytest.approx(got["a_net"] - got["c_net"])


def _results(status_pass=True):
    obs = 0.60 if status_pass else 0.55
    out = {}
    for name in ("MES", "MNQ"):
        crit = evaluate_criteria(_stats(), obs, 0.54, 600)
        out[name] = {
            "criteria": crit,
            "base": {"standard": _stats(), "comparable": _stats(trades=650, blowups=4),
                     "dd_halts": 3, "loss_halts": 0},
            "sens": {1.0: {"standard": _stats(net_pnl=2_000.0), "comparable": _stats()},
                     3.0: {"standard": _stats(net_pnl=-500.0), "comparable": _stats()}},
            "bench": {"target_share": 0.54, "n_trades": 600, "resolved_fraction": 0.93,
                      "replications": 1000, "per_trade_dispersion": 0.1},
            "observed": {"share": obs, "targets": 360, "stops": 240, "flattens": 50},
            "z": crit[2].value, "departure": crit[3].value,
            "attribution": {"a_net": 1_000.0, "b_net": -200.0, "c_net": -300.0,
                            "extended_hold": 1_200.0, "sizing_change": 100.0,
                            "a_trades": 600, "b_trades": 600, "c_trades": 650},
        }
    return out


class TestVerdictBlock:
    def test_rejected_block_carries_both_instruments_and_both_bases(self):
        block = verdict_block(_results(False), "REJECTED", "2026-09-11")
        assert block.startswith("### Verdict: REJECTED")
        assert "#### MES" in block and "#### MNQ" in block
        assert "| Sessions blocked by the trailing halt | 3 |" in block
        assert "halt off" in block.lower()
        assert "FAIL" in block and "PASS" in block
        assert "1 tick" in block and "3 ticks" in block

    def test_accepted_block_leads_with_the_caveat(self):
        block = verdict_block(_results(True), "ACCEPTED", "2026-09-11")
        first_content = block.split("\n")[2]
        assert "chosen after seeing" in first_content
        assert "03:00" in first_content and "slippage" in first_content

    def test_the_block_states_the_attribution_in_those_terms(self):
        block = verdict_block(_results(False), "REJECTED", "2026-09-11")
        assert "extended hold" in block.lower()
        assert "sizing change" in block.lower()
