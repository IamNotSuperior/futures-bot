"""Tests for the research bot's work functions.

The Discord layer is a thin shell, so what is worth testing is here: argument
validation, the refusals a user will actually hit, and that the read-only
commands render. ``run_backtest`` over real bars is exercised by the smoke run
rather than the suite - it loads 2.58M bars and would dominate the runtime.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BOTS = str(PROJECT_ROOT / "bots")
if _BOTS not in sys.path:
    sys.path.insert(0, _BOTS)

import runners  # noqa: E402
from runners import WorkError  # noqa: E402


class TestStrategyResolution:
    def test_known_strategies_are_registry_names(self):
        names = runners.known_strategies()
        assert set(names) <= set(runners.registry().names())
        assert "orb" in names

    def test_unknown_strategy_is_a_work_error(self):
        with pytest.raises(WorkError, match="no runnable configuration"):
            runners._runner("does_not_exist")

    def test_the_error_lists_what_is_runnable(self):
        with pytest.raises(WorkError, match="orb"):
            runners._runner("does_not_exist")

    def test_manual_discretionary_says_it_is_not_code(self):
        """It is in the registry but has no class; the message should say so."""
        with pytest.raises(WorkError, match="not code"):
            runners._runner("manual_discretionary")

    def test_every_runner_is_in_the_registry(self):
        registry_names = set(runners.registry().names())
        assert set(runners.RUNNERS) <= registry_names

    def test_every_runner_builds_a_real_class(self):
        reg = runners.registry()
        for name in runners.RUNNERS:
            assert reg.get(name).class_path, f"{name} has no class_path"


class TestDateParsing:
    def test_parses_an_iso_date(self):
        from datetime import date
        assert runners.parse_day("2024-09-01", "start") == date(2024, 9, 1)

    @pytest.mark.parametrize("bad", ["01/09/2024", "2024-9-1x", "yesterday", ""])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(WorkError, match="must look like"):
            runners.parse_day(bad, "start")

    def test_backtest_rejects_a_reversed_range(self):
        with pytest.raises(WorkError, match="is after end"):
            runners.run_backtest("orb", "2026-01-01", "2020-01-01")

    def test_backtest_validates_the_strategy_before_loading_bars(self):
        """A typo should fail instantly, not after a 2.58M-bar load."""
        with pytest.raises(WorkError, match="no runnable configuration"):
            runners.run_backtest("nope", "2024-01-01", "2024-02-01")


class TestWalkforward:
    def test_renders_the_saved_orb_table(self):
        text = runners.run_walkforward("orb")
        assert "walk-forward" in text
        for year in ("2020", "2026"):
            assert year in text
        assert "folds profitable" in text
        assert "orb_walkforward_slip1.csv" in text

    def test_says_so_when_a_strategy_has_no_walkforward(self):
        with pytest.raises(WorkError, match="no walk-forward run"):
            runners.run_walkforward("orb_flat_1030")

    def test_unknown_strategy(self):
        with pytest.raises(WorkError):
            runners.run_walkforward("nope")


class TestEvalsim:
    def test_orb_pass_probability_renders(self):
        fields = runners.run_evalsim("orb", paths=2_000)
        assert "Pass probability" in fields
        assert "Expected attempts" in fields
        assert fields["Pass probability"].endswith("%")

    def test_reports_its_source_and_when_it_was_produced(self):
        fields = runners.run_evalsim("orb", paths=1_000)
        assert "orb_oos_stream_slip1.csv" in fields["Source"]
        assert "produced" in fields["Source"]

    def test_unknown_strategy(self):
        with pytest.raises(WorkError):
            runners.run_evalsim("nope")


class TestHypotheses:
    def test_listing_shows_every_entry_with_its_status(self):
        text = runners.hypothesis_text()
        for n in (1, 2, 3, 4, 5):
            assert f"{n}." in text
        assert "REJECTED" in text and "PROPOSED" in text

    def test_one_entry_shows_title_status_and_verdict(self):
        text = runners.hypothesis_text(1)
        assert "Entry 1" in text
        assert "REJECTED" in text
        assert "b8e9411" in text

    def test_an_open_entry_has_no_verdict_commit(self):
        assert "verdict commit" not in runners.hypothesis_text(3)

    def test_unknown_entry_number(self):
        with pytest.raises(WorkError, match="no entry 99"):
            runners.hypothesis_text(99)

    def test_output_fits_in_a_discord_message(self):
        for n in [None, 1, 2, 3, 4, 5]:
            assert len(runners.hypothesis_text(n)) <= 1900


class TestStatus:
    def test_shows_the_registry(self):
        text = runners.status_text()
        for name in ("orb", "orb2", "orb_flat_1030"):
            assert name in text

    def test_reports_agreement_with_the_log(self):
        assert "agrees with hypotheses.md" in runners.status_text()

    def test_fits_in_a_discord_message(self):
        assert len(runners.status_text()) <= 1900
