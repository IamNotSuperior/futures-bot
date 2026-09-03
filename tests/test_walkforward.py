"""Tests for the walk-forward fold construction.

The property that matters: in every fold, all training data strictly precedes
all test data, and no test year is ever used to select the parameters that are
then scored on it.
"""

from datetime import date

import pytest

import walkforward as wf


class TestFoldConstruction:
    def test_folds_exist(self):
        assert len(wf.build_folds()) == 7

    def test_training_always_precedes_testing(self):
        for fold in wf.build_folds():
            assert fold.train_end < fold.test_start

    def test_training_window_never_contains_the_test_year(self):
        for fold in wf.build_folds():
            assert fold.train_end.year < fold.test_year
            assert fold.test_start.year == fold.test_year

    def test_training_windows_expand(self):
        folds = wf.build_folds()
        for earlier, later in zip(folds, folds[1:]):
            assert later.train_end > earlier.train_end
            assert later.train_start == earlier.train_start

    def test_test_windows_are_disjoint(self):
        folds = wf.build_folds()
        for earlier, later in zip(folds, folds[1:]):
            assert earlier.test_end < later.test_start

    def test_test_windows_tile_the_years_after_the_first(self):
        years = [f.test_year for f in wf.build_folds()]
        assert years == sorted(years)
        assert years == list(range(years[0], years[-1] + 1))

    def test_no_fold_tests_beyond_the_data(self):
        for fold in wf.build_folds():
            assert fold.test_end <= wf.DATA_END

    def test_short_first_year_is_still_used_if_long_enough(self):
        """2019 starts in May, which clears the 200-day minimum."""
        first = wf.build_folds()[0]
        assert first.test_year == 2020
        assert (first.train_end - first.train_start).days >= wf.MIN_TRAIN_DAYS

    def test_insufficient_history_drops_the_fold(self):
        folds = wf.build_folds(data_start=date(2019, 12, 1), data_end=date(2021, 12, 31))
        assert [f.test_year for f in folds] == [2021]

    def test_partial_final_year_is_truncated_to_the_data(self):
        last = wf.build_folds()[-1]
        assert last.test_year == 2026
        assert last.test_end == wf.DATA_END


class TestNoLookahead:
    def test_every_training_day_is_before_every_test_day(self):
        """The property stated as a date comparison, fold by fold."""
        for fold in wf.build_folds():
            assert fold.train_start <= fold.train_end
            assert fold.test_start <= fold.test_end
            assert fold.train_end < fold.test_start

    def test_a_fold_never_trains_on_a_later_fold_test_year(self):
        folds = wf.build_folds()
        for fold in folds:
            for other in folds:
                if other.test_year >= fold.test_year:
                    # A later (or same) fold's test window must not be inside
                    # this fold's training window.
                    assert not (
                        fold.train_start <= other.test_start <= fold.train_end
                    )


class TestSelectors:
    """The pass-probability selector added for entry 4.

    The training-Sharpe rule must keep behaving exactly as it did, because
    entries 1 and 2 were evaluated under it and their recorded results have to
    stay reproducible.
    """

    def _block(self, rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_registry_exposes_both_rules(self):
        assert set(wf.SELECTORS) == {"train_sharpe", "pass_probability"}

    def test_default_is_still_train_sharpe(self):
        selector, needs = wf.SELECTORS["train_sharpe"]
        assert selector is wf.select_by_train_sharpe
        assert needs is False

    def test_train_sharpe_picks_the_highest(self):
        block = self._block([
            {"train_trades": 50, "train_sharpe": 0.1, "tag": "a"},
            {"train_trades": 50, "train_sharpe": 0.9, "tag": "b"},
            {"train_trades": 50, "train_sharpe": 0.4, "tag": "c"},
        ])
        chosen, pool, eligible = wf.select_by_train_sharpe(block, 30)
        assert chosen["tag"] == "b"
        assert (pool, eligible) == (3, 3)

    def test_eligibility_floor_filters(self):
        block = self._block([
            {"train_trades": 10, "train_sharpe": 9.0, "tag": "thin"},
            {"train_trades": 50, "train_sharpe": 0.2, "tag": "thick"},
        ])
        chosen, pool, eligible = wf.select_by_train_sharpe(block, 30)
        assert chosen["tag"] == "thick"
        assert (pool, eligible) == (1, 1)

    def test_unreachable_floor_falls_back_to_everything(self):
        """The trap the handoff records: it must be visible, not silent."""
        block = self._block([
            {"train_trades": 5, "train_sharpe": 0.2, "tag": "a"},
            {"train_trades": 7, "train_sharpe": 0.8, "tag": "b"},
        ])
        chosen, pool, eligible = wf.select_by_train_sharpe(block, 1000)
        assert chosen["tag"] == "b"
        assert pool == 2 and eligible == 0

    def test_pass_probability_picks_the_highest(self):
        block = self._block([
            {"train_trades": 50, "train_pass_probability": 0.10,
             "train_net_pnl": 900.0, "train_max_drawdown": 100.0, "tag": "a"},
            {"train_trades": 50, "train_pass_probability": 0.30,
             "train_net_pnl": 100.0, "train_max_drawdown": 500.0, "tag": "b"},
        ])
        chosen, _, _ = wf.select_by_pass_probability(block, 30)
        assert chosen["tag"] == "b"

    def test_ties_break_on_net_pnl(self):
        block = self._block([
            {"train_trades": 50, "train_pass_probability": 0.0,
             "train_net_pnl": -500.0, "train_max_drawdown": 10.0, "tag": "a"},
            {"train_trades": 50, "train_pass_probability": 0.0,
             "train_net_pnl": 250.0, "train_max_drawdown": 900.0, "tag": "b"},
        ])
        chosen, _, _ = wf.select_by_pass_probability(block, 30)
        assert chosen["tag"] == "b"

    def test_then_on_drawdown(self):
        block = self._block([
            {"train_trades": 50, "train_pass_probability": 0.0,
             "train_net_pnl": 100.0, "train_max_drawdown": 900.0, "tag": "a"},
            {"train_trades": 50, "train_pass_probability": 0.0,
             "train_net_pnl": 100.0, "train_max_drawdown": 50.0, "tag": "b"},
        ])
        chosen, _, _ = wf.select_by_pass_probability(block, 30)
        assert chosen["tag"] == "b"

    def test_final_tie_break_is_grid_order(self):
        """All three keys equal: the first candidate in grid order wins."""
        block = self._block([
            {"train_trades": 50, "train_pass_probability": 0.0,
             "train_net_pnl": 100.0, "train_max_drawdown": 50.0, "tag": "first"},
            {"train_trades": 50, "train_pass_probability": 0.0,
             "train_net_pnl": 100.0, "train_max_drawdown": 50.0, "tag": "second"},
        ])
        chosen, _, _ = wf.select_by_pass_probability(block, 30)
        assert chosen["tag"] == "first"

    def test_single_candidate_selection_is_a_no_op(self):
        """Entry 4 has no grid: selection must simply return the one row."""
        block = self._block([
            {"train_trades": 4, "train_pass_probability": 0.0,
             "train_net_pnl": -10.0, "train_max_drawdown": 5.0, "tag": "only"},
        ])
        for selector, _ in wf.SELECTORS.values():
            if selector is wf.select_by_train_sharpe:
                block = block.assign(train_sharpe=-2.0)
            chosen, pool, _ = selector(block, 30)
            assert chosen["tag"] == "only"
            assert pool == 1

    def test_unknown_selection_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown selection"):
            wf.run_walkforward(None, set(), set(), None, selection="nope")
