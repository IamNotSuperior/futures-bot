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

    def test_every_registry_entry_with_a_class_is_runnable(self):
        """The other direction, and the one that catches a new strategy.

        ``test_every_runner_is_in_the_registry`` stops a runner naming a
        strategy that does not exist. This stops the opposite and far more
        likely mistake: a strategy added to ``registry.yaml`` with a
        ``class_path`` and no entry in ``RUNNERS``, which is invisible until
        someone types ``/walkforward <name>`` in Discord and gets "no runnable
        configuration" for a strategy that plainly exists.

        Entries with ``class_path: null`` are exempt - ``manual_discretionary``
        is a log entry, not code, and has nothing to run.
        """
        import submissions

        reg = runners.registry()
        # Generated strategies are runnable through run_walkforward_generated,
        # which computes rather than reading a saved table, so they are not in
        # RUNNERS by design.
        missing = [r.name for r in reg.all()
                   if r.class_path and r.name not in runners.RUNNERS
                   and not submissions.is_generated(r.name)]
        assert not missing, (
            f"registry entries with a class but no runner: {missing}. "
            f"Add each to RUNNERS in bots/runners.py - with build=None if it "
            f"cannot be replayed through the scalar-contracts engine - so the "
            f"bot can answer for it."
        )

    def test_generated_strategies_are_answerable_without_being_in_RUNNERS(self):
        """The other half of the coverage claim: they route to the run path."""
        import submissions

        reg = runners.registry()
        generated = [r.name for r in reg.all() if submissions.is_generated(r.name)]
        if not generated:
            pytest.skip("no generated strategy in the registry")
        for name in generated:
            assert name in runners.known_strategies()
            with pytest.raises(WorkError, match="generated strategy"):
                runners.run_walkforward(name)

    def test_saved_outputs_named_by_a_runner_exist(self):
        """Every CSV a runner points at is on disk, where results exist.

        These commands read saved output rather than recomputing, so a typo in
        a filename is a runtime error in Discord and nothing else.

        ``backtests/results/*`` is gitignored - the outputs are regenerable but
        slow - so on a fresh clone there is nothing to check and this skips.
        It is a check on *this* tree's ability to serve the commands, not a
        claim about the repository.
        """
        present = [(n, lab, csv) for n, run in runners.RUNNERS.items()
                   for lab, csv in (("walkforward", run.walkforward_csv),
                                    ("oos", run.oos_csv))
                   if csv is not None]
        if not any((runners.RESULTS / csv).exists() for _, _, csv in present):
            pytest.skip("backtests/results/ is empty (gitignored); nothing to check")
        missing = [f"{n} {lab}: {csv}" for n, lab, csv in present
                   if not (runners.RESULTS / csv).exists()]
        assert not missing, (
            f"runners name files that are not in backtests/results/: {missing}"
        )

    def test_per_session_sized_strategies_refuse_backtest(self):
        """Entry 6 and 7 size per session; a uniform-size replay is not them."""
        for name in ("london_1x", "london_2x", "london_mnq_replication"):
            assert runners.RUNNERS[name].build is None
            with pytest.raises(WorkError, match="sizes per session"):
                runners.run_backtest(name, "2024-01-01", "2024-12-31")

    def test_london_runners_point_at_the_two_tick_base_case(self):
        """Entry 6's pre-registered base case is 2 ticks, not the optimistic 1.

        The slip1 files exist and are the sensitivity arm; reporting them
        would show numbers that appear in no verdict in the log.
        """
        for name in ("london_1x", "london_2x"):
            run = runners.RUNNERS[name]
            assert "slip2" in run.walkforward_csv
            assert "slip2" in run.oos_csv


def _require(name: str, attr: str = "walkforward_csv"):
    """The saved file for ``name``, or skip. See the note on gitignore above."""
    csv = getattr(runners.RUNNERS[name], attr)
    path = runners.RESULTS / csv
    if not path.exists():
        pytest.skip(f"{csv} not present; regenerate with "
                    f"run_london.py/run_entry7.py --folds-only")
    return path


class TestLondonSavedOutputs:
    """The saved fold tables must still say what the verdicts say.

    These are regression tests on the *numbers in the log*. entry 6 and 7 are
    rejected and their verdicts are never revised, so a fold table that stops
    reproducing -$7,702 / 1 of 7 means the regeneration path changed what a
    fold means - which would quietly make the bot report something no verdict
    ever said.
    """

    @pytest.mark.parametrize("name,net,profitable", [
        ("london_1x", -7702, 1),
        ("london_2x", -20781, 2),
        ("london_mnq_replication", 116, 4),
    ])
    def test_fold_table_reproduces_the_recorded_verdict(self, name, net, profitable):
        import pandas as pd

        folds = pd.read_csv(_require(name))
        assert len(folds) == 7, "entry 6 and 7 both use 2020-2026"
        assert round(folds["test_net_pnl"].sum()) == pytest.approx(net, abs=2)
        assert int((folds["test_net_pnl"] > 0).sum()) == profitable

    @pytest.mark.parametrize("name", ["london_1x", "london_2x",
                                      "london_mnq_replication"])
    def test_walkforward_renders(self, name):
        _require(name)
        text = runners.run_walkforward(name)
        assert "folds profitable" in text
        assert "2020" in text and "2026" in text

    @pytest.mark.parametrize("name", ["london_1x", "london_2x",
                                      "london_mnq_replication"])
    def test_evalsim_runs_off_the_saved_stream(self, name):
        _require(name, "oos_csv")
        fields = runners.run_evalsim(name, paths=2000)
        assert fields["Source"].startswith(runners.RUNNERS[name].oos_csv)
        assert "%" in fields["Pass probability"]


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
        _require("orb")
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
        _require("orb", "oos_csv")
        fields = runners.run_evalsim("orb", paths=2_000)
        assert "Pass probability" in fields
        assert "Expected attempts" in fields
        assert fields["Pass probability"].endswith("%")

    def test_reports_its_source_and_when_it_was_produced(self):
        _require("orb", "oos_csv")
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
