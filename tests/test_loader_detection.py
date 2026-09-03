"""Tests for the roll-date and early-close detectors in data/loader.py.

These run on synthetic bars so they do not need the cached parquet, but the
shapes mirror what Databento returns for a continuous series.
"""

from datetime import date

import pandas as pd
import pytest

import loader

ET = "America/New_York"


def session_bars(day: str, instrument_id: int, last_bar: str = "15:59") -> pd.DataFrame:
    """One session of 1-minute bars from 09:30 to ``last_bar`` inclusive."""
    index = pd.date_range(f"{day} 09:30", f"{day} {last_bar}", freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "instrument_id": instrument_id,
        },
        index=index,
    )


class TestDetectRollDates:
    def test_finds_the_session_where_the_contract_changes(self):
        bars = pd.concat(
            [
                session_bars("2025-06-16", 111),
                session_bars("2025-06-17", 111),
                session_bars("2025-06-18", 222),
                session_bars("2025-06-19", 222),
            ]
        )
        assert loader.detect_roll_dates(bars) == {date(2025, 6, 18)}

    def test_first_session_is_never_a_roll(self):
        bars = pd.concat([session_bars("2025-06-16", 111), session_bars("2025-06-17", 111)])
        assert loader.detect_roll_dates(bars) == set()

    def test_multiple_rolls(self):
        bars = pd.concat(
            [
                session_bars("2025-03-17", 1),
                session_bars("2025-06-16", 2),
                session_bars("2025-09-15", 3),
            ]
        )
        assert loader.detect_roll_dates(bars) == {date(2025, 6, 16), date(2025, 9, 15)}

    def test_stray_bars_do_not_trigger_a_roll(self):
        """A handful of outgoing-contract bars must not outvote the session."""
        day = session_bars("2025-06-17", 111)
        day.iloc[:3, day.columns.get_loc("instrument_id")] = 999
        bars = pd.concat([session_bars("2025-06-16", 111), day])
        assert loader.detect_roll_dates(bars) == set()

    def test_requires_instrument_id(self):
        bars = session_bars("2025-06-16", 111).drop(columns=["instrument_id"])
        with pytest.raises(ValueError, match="instrument_id"):
            loader.detect_roll_dates(bars)


class TestDetectEarlyCloseDates:
    def test_finds_a_short_session(self):
        bars = pd.concat(
            [
                session_bars("2025-07-02", 1),
                session_bars("2025-07-03", 1, last_bar="12:59"),
                session_bars("2025-07-07", 1),
            ]
        )
        assert loader.detect_early_close_dates(bars) == {date(2025, 7, 3)}

    def test_full_sessions_are_not_flagged(self):
        bars = pd.concat([session_bars("2025-07-01", 1), session_bars("2025-07-02", 1)])
        assert loader.detect_early_close_dates(bars) == set()

    def test_large_input_is_not_swallowed_by_the_empty_check(self):
        """Regression: a column-less frame reports .empty on any row count.

        The detector used to build `pd.DataFrame(index=...)`, which has zero
        columns and is therefore always `.empty`, so it returned no dates no
        matter how much data it was given.
        """
        bars = pd.concat(
            [session_bars(f"2025-07-{d:02d}", 1) for d in range(1, 11)]
            + [session_bars("2025-07-11", 1, last_bar="12:59")]
        )
        assert len(bars) > 3000
        assert loader.detect_early_close_dates(bars) == {date(2025, 7, 11)}

    def test_session_with_no_rth_bars_is_not_an_early_close(self):
        """A full closure has no RTH bars at all; that is not a short session."""
        overnight = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
             "instrument_id": 1},
            index=pd.date_range("2025-12-25 18:00", "2025-12-25 23:59", freq="1min", tz=ET),
        )
        bars = pd.concat([session_bars("2025-12-24", 1, last_bar="12:59"), overnight])
        assert loader.detect_early_close_dates(bars) == {date(2025, 12, 24)}
