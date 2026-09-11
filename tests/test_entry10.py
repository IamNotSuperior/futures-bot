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
    ENTRY6_PARAMS, ENTRY10_PARAMS, INSTRUMENTS, REPRODUCTION_COSTS,
    compare_streams,
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
