"""The rule 6 / rule 7 regression check.

Rule 6's 30-second floor means rule 7's share of profit from trades held five
seconds or less must read 0.00% in any correct run. The existing flags fire
at the 30% warning line; a strategy that produced *one* three-second trade
would sail under them and the only sign would be a small non-zero figure in
a table nobody reads. This check names any hold under the floor as a
regression, on every surface a trade stream reaches: the metrics dict, the
printed report, the generated verdict block, and the Discord embed field.
"""

from datetime import date

import pandas as pd
import pytest

import rules
from engine import MES, CostModel, price_trades
from metrics import compute_metrics, format_report

ET = "America/New_York"


def trades(holds_seconds: list[int]) -> pd.DataFrame:
    """One long trade per entry, held for the given seconds, all winners."""
    rows = []
    for i, held in enumerate(holds_seconds):
        entry = pd.Timestamp(f"2025-07-{14 + i:02d} 10:00:00", tz=ET)
        rows.append({"entry_time": entry, "exit_time": entry + pd.Timedelta(seconds=held),
                     "direction": "long", "entry_price": 5000.0, "exit_price": 5004.0,
                     "exit_reason": "target"})
    return price_trades(pd.DataFrame(rows), MES, CostModel(), 1)


class TestMetric:
    def test_a_clean_stream_has_no_regression(self):
        m = compute_metrics(trades([60, 600, 3600]))
        assert m["min_hold_violation_count"] == 0
        assert m["microscalp_trade_count"] == 0
        assert m["hold_regression"] is False

    def test_a_hold_under_the_floor_is_a_regression_even_when_over_five_seconds(self):
        m = compute_metrics(trades([60, 20, 3600]))
        assert m["min_hold_violation_count"] == 1
        assert m["microscalp_trade_count"] == 0
        assert m["hold_regression"] is True

    def test_a_five_second_hold_counts_on_both_measures(self):
        m = compute_metrics(trades([60, rules.MICROSCALP_SECONDS, 3600]))
        assert m["min_hold_violation_count"] == 1
        assert m["microscalp_trade_count"] == 1
        assert m["hold_regression"] is True

    def test_exactly_the_floor_is_not_a_violation(self):
        m = compute_metrics(trades([rules.MIN_HOLD_SECONDS]))
        assert m["min_hold_violation_count"] == 0
        assert m["hold_regression"] is False

    def test_the_thresholds_are_read_from_rules(self):
        import inspect

        import metrics

        source = inspect.getsource(metrics)
        assert "< 30" not in source and "<= 5" not in source


class TestReport:
    def test_a_clean_report_says_so_once(self):
        text = format_report(compute_metrics(trades([60, 600])))
        assert "REGRESSION" not in text
        assert "hold regression" in text.lower()

    def test_a_regression_is_loud_and_names_the_counts(self):
        text = format_report(compute_metrics(trades([60, 3, 20])))
        assert "REGRESSION" in text
        assert "2 trade(s) held under 30s" in text
        assert "1 held <= 5s" in text


class TestGeneratedVerdict:
    def _result(self, metrics):
        import run_generated

        return run_generated.WalkforwardResult(
            name="x", folds=pd.DataFrame({"test_net_pnl": [1.0] * 7}),
            trades=pd.DataFrame({"net_pnl": [10.0]}), metrics=metrics,
            pass_probability=0.5, payout_probability=0.6, blowups=0,
            profitable_folds=7, accepted=True, reasons=[], contracts=1,
            span=(date(2020, 1, 1), date(2026, 8, 31)), dd_halts=0)

    def test_the_line_reads_the_metrics_and_defaults_to_intact(self):
        from run_generated import hold_regression_line

        assert "intact" in hold_regression_line({}).lower()
        line = hold_regression_line({"min_hold_violation_count": 2,
                                     "microscalp_trade_count": 1,
                                     "microscalp_profit_pct": 12.5})
        assert line.startswith("**REGRESSION")
        assert "2" in line and "12.50%" in line

    def test_the_block_carries_the_line(self):
        import run_generated

        clean = run_generated.verdict_block(self._result(
            {"sharpe": 1.0, "profit_factor": 1.5, "max_drawdown": -100.0,
             "max_daily_loss": -50.0, "avg_duration_seconds": 600.0,
             "microscalp_profit_pct": 0.0, "min_hold_violation_count": 0,
             "microscalp_trade_count": 0}), 9)
        assert "Rule 6/7 regression check" in clean
        assert "REGRESSION" not in clean

        bad = run_generated.verdict_block(self._result(
            {"sharpe": 1.0, "profit_factor": 1.5, "max_drawdown": -100.0,
             "max_daily_loss": -50.0, "avg_duration_seconds": 600.0,
             "microscalp_profit_pct": 40.0, "min_hold_violation_count": 3,
             "microscalp_trade_count": 3}), 9)
        assert "**REGRESSION" in bad

    def test_an_accepted_verdict_with_a_regression_is_not_accepted(self):
        """A stream that broke the floor cannot be evidence of anything."""
        from run_generated import decide

        m = {"min_hold_violation_count": 1, "microscalp_trade_count": 0}
        accepted, reasons = decide(profitable_folds=7, net_pnl=1000.0, metrics=m)
        assert accepted is False
        assert any("regression" in r.lower() for r in reasons)

    def test_a_clean_stream_still_decides_on_rule_13(self):
        from run_generated import decide

        accepted, reasons = decide(profitable_folds=7, net_pnl=1000.0,
                                   metrics={"min_hold_violation_count": 0,
                                            "microscalp_trade_count": 0})
        assert accepted is True and reasons == []
        accepted, reasons = decide(profitable_folds=3, net_pnl=1000.0,
                                   metrics={"min_hold_violation_count": 0,
                                            "microscalp_trade_count": 0})
        assert accepted is False
