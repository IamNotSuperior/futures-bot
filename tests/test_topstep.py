"""data/topstep.py — the TopstepX/ProjectX repository as a second, read-only bar
source, and the cross-check that says how well it agrees with the Databento
cache. Databento remains the only backtest source; this module never writes
to the cache and never replaces it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import rules
import topstep


def write_topstep_csv(directory: Path, symbol: str, rows: pd.DataFrame,
                      timeframe: str = "1min", start: str = "20260120", end: str = "20260415") -> Path:
    (directory / symbol).mkdir(parents=True, exist_ok=True)
    path = directory / symbol / f"{symbol}_{timeframe}_{start}_{end}.csv"
    rows.to_csv(path, index=False)
    return path


def utc_rows(start: str, n: int, base: float = 6000.0) -> pd.DataFrame:
    """Rows in the repository's format: naive UTC ISO timestamps, OHLCV."""
    idx = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame({
        "datetime": idx.strftime("%Y-%m-%dT%H:%M:%S"),
        "open": base, "high": base + 1, "low": base - 1, "close": base + 0.5,
        "volume": 10,
    })


def write_cache(path: Path, start: str, n: int, base: float = 6000.0) -> Path:
    """A Databento-shaped parquet: UTC DatetimeIndex named ts_event."""
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC", name="ts_event")
    pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base + 0.5,
                  "volume": 10, "instrument_id": 1, "symbol": "MES.v.0"}, index=idx).to_parquet(path)
    return path


class TestLoader:
    def test_loads_bars_in_the_cache_loaders_shape(self, tmp_path):
        write_topstep_csv(tmp_path, "MES", utc_rows("2026-03-02 14:30", 3))
        bars = topstep.load_topstep_bars("MES", directory=tmp_path)
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
        assert bars.index.name == "ts_event_et"
        assert str(bars.index.tz) == rules.ET
        assert bars.index[0] == pd.Timestamp("2026-03-02 09:30", tz=rules.ET)
        assert len(bars) == 3

    def test_symbol_is_validated_against_the_allowlist(self, tmp_path):
        write_topstep_csv(tmp_path, "NQ", utc_rows("2026-03-02 14:30", 3))
        with pytest.raises(ValueError, match="MES"):
            topstep.load_topstep_bars("NQ", directory=tmp_path)
        with pytest.raises(ValueError):
            topstep.load_topstep_bars("mes/../NQ", directory=tmp_path)

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MES_1min"):
            topstep.load_topstep_bars("MES", directory=tmp_path)

    def test_timeframe_selects_the_file(self, tmp_path):
        write_topstep_csv(tmp_path, "MES", utc_rows("2026-03-02 14:30", 2), timeframe="5min")
        assert len(topstep.load_topstep_bars("MES", timeframe="5min", directory=tmp_path)) == 2
        with pytest.raises(ValueError):
            topstep.load_topstep_bars("MES", timeframe="../x", directory=tmp_path)

    def test_default_directory_comes_from_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOPSTEP_DATA_DIR", str(tmp_path))
        assert topstep.topstep_dir() == tmp_path
        monkeypatch.delenv("TOPSTEP_DATA_DIR")
        assert topstep.topstep_dir().name == "cme-futures-ohlc"


class TestCompare:
    def test_identical_feeds_agree_completely(self, tmp_path):
        write_topstep_csv(tmp_path, "MES", utc_rows("2026-03-02 14:30", 60))
        cache = write_cache(tmp_path / "cache.parquet", "2026-03-02 14:30", 60)
        report = topstep.compare("MES", cache_path=cache, directory=tmp_path)
        assert report["symbol"] == "MES"
        assert report["topstep_rows"] == 60
        assert report["common_rows"] == 60
        assert report["columns"]["close"]["identical_pct"] == 100.0
        assert report["columns"]["close"]["max_abs_diff"] == 0.0
        assert report["volume_identical_pct"] == 100.0
        assert report["differing_days"] == []
        assert report["rth_count_mismatches"] == []
        assert report["labelled_by_open_minute"] is True

    def test_an_offset_day_is_reported_as_differing(self, tmp_path):
        rows = pd.concat([utc_rows("2026-03-16 14:30", 30, base=6050.0),
                          utc_rows("2026-03-18 14:30", 30, base=6000.0)])
        write_topstep_csv(tmp_path, "MES", rows)
        idx = pd.date_range("2026-03-16 14:30", periods=30, freq="1min", tz="UTC").append(
            pd.date_range("2026-03-18 14:30", periods=30, freq="1min", tz="UTC"))
        idx.name = "ts_event"
        pd.DataFrame({"open": 6000.0, "high": 6001.0, "low": 5999.0, "close": 6000.5, "volume": 10,
                      "instrument_id": 1, "symbol": "MES.v.0"}, index=idx).to_parquet(tmp_path / "cache.parquet")
        report = topstep.compare("MES", cache_path=tmp_path / "cache.parquet", directory=tmp_path)
        assert report["columns"]["close"]["identical_pct"] == 50.0
        assert [d["date"] for d in report["differing_days"]] == ["2026-03-16"]
        assert report["differing_days"][0]["mean_diff"] == 50.0

    def test_report_is_json_serialisable_and_names_the_files(self, tmp_path):
        write_topstep_csv(tmp_path, "MES", utc_rows("2026-03-02 14:30", 5))
        cache = write_cache(tmp_path / "cache.parquet", "2026-03-02 14:30", 5)
        report = topstep.compare("MES", cache_path=cache, directory=tmp_path)
        json.dumps(report, allow_nan=False)
        assert report["topstep_file"].startswith("MES_1min_")
        assert report["cache_file"] == "cache.parquet"
        assert "computed" in report

    def test_write_report_lands_in_results(self, tmp_path):
        write_topstep_csv(tmp_path, "MES", utc_rows("2026-03-02 14:30", 5))
        cache = write_cache(tmp_path / "cache.parquet", "2026-03-02 14:30", 5)
        out = topstep.write_report("MES", cache_path=cache, directory=tmp_path,
                                  results=tmp_path / "results")
        assert out == tmp_path / "results" / "data_check_mes.json"
        assert json.loads(out.read_text(encoding="utf-8"))["symbol"] == "MES"

    def test_cli_compare_writes_the_report(self, tmp_path, monkeypatch):
        write_topstep_csv(tmp_path, "MES", utc_rows("2026-03-02 14:30", 5))
        cache = write_cache(tmp_path / "cache.parquet", "2026-03-02 14:30", 5)
        monkeypatch.setattr(topstep, "RESULTS", tmp_path / "results")
        assert topstep.main(["--compare", "--symbol", "MES", "--cache", str(cache),
                             "--dir", str(tmp_path)]) == 0
        assert (tmp_path / "results" / "data_check_mes.json").exists()

    def test_module_never_writes_the_cache_or_the_forward_file(self):
        import inspect

        source = inspect.getsource(topstep)
        assert "to_parquet" not in source
        assert "extend" not in source
