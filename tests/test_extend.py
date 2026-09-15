"""Tests for data/extend.py, the forward-data pull.

Forward MES bars accumulate in their own rolling file. The 2019-05..2026-08
cache is the sample eleven verdicts were scored on and is never written by
this script; the forward file holds only days no verdict has seen. Every
test runs on synthetic bars and a fake Databento client, so nothing here
touches the network or spends anything.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import extend

UTC_MINUTE = pd.Timedelta(minutes=1)


def bars(start: str, end: str) -> pd.DataFrame:
    """One-minute bars, UTC-indexed like a Databento parquet, ``end`` inclusive."""
    index = pd.date_range(start, end, freq="1min", tz="UTC", name="ts_event")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
         "instrument_id": 1, "symbol": "MES.v.0"},
        index=index,
    )


class FakeClient:
    """Records every Databento call; returns canned answers."""

    def __init__(self, cost: float = 0.05, available_end: str = "2026-09-14T19:00:00Z",
                 pulled: pd.DataFrame | None = None):
        self.calls: list[str] = []
        outer = self
        pulled = bars("2026-09-01 00:00", "2026-09-01 00:05") if pulled is None else pulled

        class Metadata:
            def get_cost(self, **kw):
                outer.calls.append("get_cost")
                return cost

            def get_billable_size(self, **kw):
                outer.calls.append("get_billable_size")
                return 1_000

            def get_dataset_range(self, dataset):
                outer.calls.append("get_dataset_range")
                return {"start": "2019-05-06T00:00:00Z", "end": available_end}

        class Timeseries:
            def get_range(self, **kw):
                outer.calls.append("get_range")
                outer.last_range = kw

                class Store:
                    def to_df(self):
                        return pulled

                return Store()

        self.metadata = Metadata()
        self.timeseries = Timeseries()


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A synthetic cache ending 2026-08-31 and an absent forward file."""
    cache = tmp_path / "cache.parquet"
    bars("2026-08-31 23:50", "2026-08-31 23:59").to_parquet(cache)
    forward = tmp_path / "forward.parquet"
    monkeypatch.setattr(extend, "CACHE", cache)
    monkeypatch.setattr(extend, "FORWARD", forward)
    return {"cache": cache, "forward": forward}


# --- last_bar_date ---------------------------------------------------------


class TestLastBarDate:
    def test_reads_the_date_of_the_last_bar(self, tmp_path):
        path = tmp_path / "x.parquet"
        bars("2026-09-01 00:00", "2026-09-13 23:59").to_parquet(path)
        assert extend.last_bar_date(path) == date(2026, 9, 13)

    def test_missing_file_is_none(self, tmp_path):
        assert extend.last_bar_date(tmp_path / "absent.parquet") is None


# --- default_span ----------------------------------------------------------


class TestDefaultSpan:
    def test_starts_the_day_after_the_forward_file(self):
        start, _ = extend.default_span(cache_last=date(2026, 8, 31),
                                       forward_last=date(2026, 9, 13),
                                       available_end=pd.Timestamp("2026-10-02T19:00Z"))
        assert start == date(2026, 9, 14)

    def test_falls_back_to_the_day_after_the_cache(self):
        start, _ = extend.default_span(cache_last=date(2026, 8, 31), forward_last=None,
                                       available_end=pd.Timestamp("2026-09-14T19:00Z"))
        assert start == date(2026, 9, 1)

    def test_ends_at_the_date_databento_has_data_to(self):
        _, end = extend.default_span(cache_last=date(2026, 8, 31), forward_last=None,
                                     available_end=pd.Timestamp("2026-09-14T19:00Z"))
        assert end == date(2026, 9, 14)


# --- validate_span ---------------------------------------------------------


class TestValidateSpan:
    CACHE_LAST = date(2026, 8, 31)

    def test_refuses_a_start_inside_the_cache(self):
        with pytest.raises(ValueError, match="cache"):
            extend.validate_span(date(2026, 8, 31), date(2026, 9, 14),
                                 cache_last=self.CACHE_LAST, forward_last=None)

    def test_refuses_a_gap_after_the_forward_file(self):
        with pytest.raises(ValueError, match="gap"):
            extend.validate_span(date(2026, 9, 20), date(2026, 9, 30),
                                 cache_last=self.CACHE_LAST, forward_last=date(2026, 9, 13))

    def test_refuses_a_gap_after_the_cache_when_no_forward_file_exists(self):
        with pytest.raises(ValueError, match="gap"):
            extend.validate_span(date(2026, 9, 3), date(2026, 9, 14),
                                 cache_last=self.CACHE_LAST, forward_last=None)

    def test_refuses_an_end_not_after_the_start(self):
        with pytest.raises(ValueError, match="end"):
            extend.validate_span(date(2026, 9, 1), date(2026, 9, 1),
                                 cache_last=self.CACHE_LAST, forward_last=None)

    def test_accepts_a_re_pull_that_overlaps_the_forward_file(self):
        extend.validate_span(date(2026, 9, 10), date(2026, 9, 30),
                             cache_last=self.CACHE_LAST, forward_last=date(2026, 9, 13))

    def test_accepts_the_first_forward_day(self):
        extend.validate_span(date(2026, 9, 1), date(2026, 9, 14),
                             cache_last=self.CACHE_LAST, forward_last=None)


# --- merge -----------------------------------------------------------------


class TestMerge:
    def test_creates_from_nothing(self):
        new = bars("2026-09-01 00:00", "2026-09-01 00:02")
        out = extend.merge(new, existing=None)
        pd.testing.assert_frame_equal(out, new)

    def test_keeps_the_existing_row_on_a_shared_timestamp_and_sorts(self):
        existing = bars("2026-09-01 00:01", "2026-09-01 00:03")
        existing["close"] = 5.0
        new = bars("2026-09-01 00:00", "2026-09-01 00:02")
        out = extend.merge(new, existing)
        assert list(out.index) == list(pd.date_range("2026-09-01 00:00", "2026-09-01 00:03",
                                                     freq="1min", tz="UTC"))
        assert out.loc[pd.Timestamp("2026-09-01 00:01Z"), "close"] == 5.0
        assert out.loc[pd.Timestamp("2026-09-01 00:00Z"), "close"] == 1.0
        assert not out.index.duplicated().any()


# --- the command -----------------------------------------------------------


class TestRun:
    def test_estimate_never_calls_get_range(self, paths):
        client = FakeClient()
        assert extend.run(["--estimate"], client) == 0
        assert "get_range" not in client.calls
        assert not paths["forward"].exists()

    def test_estimate_defaults_to_the_first_unseen_day_through_the_available_end(self, paths, capsys):
        client = FakeClient(available_end="2026-09-14T19:00:00Z")
        extend.run(["--estimate"], client)
        out = capsys.readouterr().out
        assert "2026-09-01 -> 2026-09-14" in out

    def test_nothing_new_to_pull_is_reported_and_costs_nothing(self, paths, capsys):
        """Forward file current through the last complete day: not an error."""
        bars("2026-09-01 00:00", "2026-09-13 23:59").to_parquet(paths["forward"])
        client = FakeClient(available_end="2026-09-14T19:00:00Z")
        assert extend.run(["--pull"], client) == 0
        assert client.calls == ["get_dataset_range"]
        out = capsys.readouterr().out
        assert "nothing to pull" in out.lower()
        assert "2026-09-13" in out

    def test_pull_above_the_cap_downloads_nothing(self, paths):
        client = FakeClient(cost=0.50)
        assert extend.run(["--pull", "--max-cost", "0.10"], client) == 1
        assert "get_range" not in client.calls
        assert not paths["forward"].exists()

    def test_pull_writes_the_forward_file_and_never_touches_the_cache(self, paths):
        cache_bytes = paths["cache"].read_bytes()
        pulled = bars("2026-09-01 00:00", "2026-09-13 23:59")
        client = FakeClient(cost=0.05, pulled=pulled)
        assert extend.run(["--pull", "--max-cost", "0.10"], client) == 0
        assert client.last_range["start"] == "2026-09-01"
        assert client.last_range["end"] == "2026-09-14"
        assert paths["cache"].read_bytes() == cache_bytes
        assert extend.last_bar_date(paths["forward"]) == date(2026, 9, 13)

    def test_second_pull_appends_from_the_day_after_the_forward_file(self, paths):
        bars("2026-09-01 00:00", "2026-09-13 23:59").to_parquet(paths["forward"])
        pulled = bars("2026-09-14 00:00", "2026-09-30 23:59")
        client = FakeClient(available_end="2026-10-01T19:00:00Z", pulled=pulled)
        assert extend.run(["--pull"], client) == 0
        assert client.last_range["start"] == "2026-09-14"
        merged = pd.read_parquet(paths["forward"])
        assert merged.index.min() == pd.Timestamp("2026-09-01 00:00Z")
        assert merged.index.max() == pd.Timestamp("2026-09-30 23:59Z")
        assert not merged.index.duplicated().any()

    def test_explicit_start_inside_the_cache_is_refused_before_any_spend(self, paths):
        client = FakeClient()
        assert extend.run(["--pull", "--start", "2026-08-31"], client) == 1
        assert client.calls == ["get_dataset_range"] or "get_range" not in client.calls
        assert "get_cost" not in client.calls

    def test_merge_file_folds_a_local_pull_in_without_the_api(self, paths, tmp_path):
        local = tmp_path / "mes_v_0_ohlcv_1m_2026-09_2026-09.parquet"
        bars("2026-09-01 00:00", "2026-09-13 23:59").to_parquet(local)
        client = FakeClient()
        assert extend.run(["--merge-file", str(local)], client) == 0
        assert client.calls == []
        assert extend.last_bar_date(paths["forward"]) == date(2026, 9, 13)

    def test_merge_file_refuses_bars_the_cache_already_holds(self, paths, tmp_path):
        local = tmp_path / "overlap.parquet"
        bars("2026-08-31 23:58", "2026-09-01 00:05").to_parquet(local)
        client = FakeClient()
        assert extend.run(["--merge-file", str(local)], client) == 1
        assert not paths["forward"].exists()

    def test_requires_an_action(self, paths):
        with pytest.raises(SystemExit):
            extend.run([], FakeClient())


class TestNames:
    def test_forward_file_is_not_the_cache(self):
        assert extend.FORWARD != extend.CACHE
        assert "forward" in extend.FORWARD.name
        assert extend.CACHE.name == "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
