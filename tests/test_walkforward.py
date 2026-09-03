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
