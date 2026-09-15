"""Tests for data/roll.py, the calendar roll.

Databento's ``.v.0`` is volume-rolled, and the gateway's volume counts only
trades filled on the ProjectX platform, so a volume roll cannot be
reproduced from it. The substitute is the CME equity-index convention: the
Thursday eight days before expiry, no back-adjustment. See
``docs/superpowers/specs/2026-09-15-projectx-bar-source-design.md`` §5.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import loader
import roll


def test_expiry_is_the_third_friday_of_a_quarterly_month():
    assert roll.expiry(2025, 9) == date(2025, 9, 19)
    assert roll.expiry(2025, 12) == date(2025, 12, 19)
    assert roll.expiry(2026, 3) == date(2026, 3, 20)
    assert roll.expiry(2026, 6) == date(2026, 6, 19)


def test_a_non_quarterly_month_has_no_expiry():
    with pytest.raises(ValueError):
        roll.expiry(2026, 7)


def test_the_roll_lands_on_the_thursday_eight_days_before_expiry():
    for year, month in ((2025, 9), (2025, 12), (2026, 3), (2026, 6)):
        rolled = roll.roll_date(year, month)
        assert (roll.expiry(year, month) - rolled).days == 8
        assert rolled.weekday() == 3  # Thursday


def test_the_front_contract_changes_on_the_roll_date_not_at_expiry():
    assert roll.active_contract(date(2025, 9, 10)) == (2025, 9)
    assert roll.active_contract(date(2025, 9, 11)) == (2025, 12)
    # Expiry week still belongs to the December contract.
    assert roll.active_contract(date(2025, 9, 19)) == (2025, 12)


def test_contract_ids_use_the_gateway_format():
    assert roll.contract_id("MES", 2025, 9) == "CON.F.US.MES.U25"
    assert roll.contract_id("MNQ", 2025, 12) == "CON.F.US.MNQ.Z25"


def session_bars(contract: str, day: str) -> pd.DataFrame:
    """One RTH hour of bars for a contract, UTC-indexed, ET session ``day``."""
    index = pd.date_range(f"{day} 14:30", periods=60, freq="1min",
                          tz="UTC", name="ts_event")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
         "instrument_id": abs(hash(contract)) % 10**6, "symbol": contract},
        index=index,
    )


def test_stitching_keeps_only_the_contract_active_on_each_session():
    """Both contracts trade on both days; only the active one survives."""
    before, after = "2025-09-10", "2025-09-11"
    frames = {
        "CON.F.US.MES.U25": pd.concat([session_bars("CON.F.US.MES.U25", before),
                                       session_bars("CON.F.US.MES.U25", after)]),
        "CON.F.US.MES.Z25": pd.concat([session_bars("CON.F.US.MES.Z25", before),
                                       session_bars("CON.F.US.MES.Z25", after)]),
    }

    stitched = roll.stitch(frames, root="MES")

    by_day = {str(d): set(g["symbol"])
              for d, g in stitched.groupby(stitched.index.tz_convert(loader.ET).date)}
    assert by_day[before] == {"CON.F.US.MES.U25"}
    assert by_day[after] == {"CON.F.US.MES.Z25"}
    assert not stitched.index.has_duplicates
    assert stitched.index.is_monotonic_increasing


def test_a_stitched_series_reports_its_roll_to_detect_roll_dates():
    """The whole point of synthesising instrument_id: rule 2's roll guard."""
    frames = {
        "CON.F.US.MES.U25": pd.concat([session_bars("CON.F.US.MES.U25", d)
                                       for d in ("2025-09-09", "2025-09-10")]),
        "CON.F.US.MES.Z25": pd.concat([session_bars("CON.F.US.MES.Z25", d)
                                       for d in ("2025-09-11", "2025-09-12")]),
    }

    stitched = roll.stitch(frames, root="MES")

    assert loader.detect_roll_dates(stitched) == {date(2025, 9, 11)}


class StubClient:
    """A client whose two endpoints answer from canned frames."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.requested: list[str] = []

    def resolve_contracts(self, root: str) -> list[str]:
        return list(self.frames)

    def retrieve_bars(self, contract_id, start, end, page_limit=None):
        self.requested.append(contract_id)
        return self.frames[contract_id]


def test_a_pull_stitches_every_resolved_contract():
    frames = {
        "CON.F.US.MES.U25": pd.concat([session_bars("CON.F.US.MES.U25", d)
                                       for d in ("2025-09-09", "2025-09-10")]),
        "CON.F.US.MES.Z25": pd.concat([session_bars("CON.F.US.MES.Z25", d)
                                       for d in ("2025-09-11", "2025-09-12")]),
    }
    client = StubClient(frames)

    stitched, depth = roll.pull(
        client, "MES",
        pd.Timestamp("2025-09-01", tz="UTC"),
        pd.Timestamp("2025-09-30", tz="UTC"),
    )

    assert client.requested == list(frames)
    assert loader.detect_roll_dates(stitched) == {date(2025, 9, 11)}
    assert depth["CON.F.US.MES.U25"] == pd.Timestamp("2025-09-09 14:30", tz="UTC")


def test_a_contract_the_gateway_serves_nothing_for_is_reported_as_empty():
    """The depth report is the measurement; a blank must not vanish."""
    frames = {
        "CON.F.US.MES.U25": session_bars("CON.F.US.MES.U25", "2025-09-09"),
        "CON.F.US.MES.M25": session_bars("CON.F.US.MES.M25", "2025-09-09").iloc[:0],
    }
    client = StubClient(frames)

    _, depth = roll.pull(
        client, "MES",
        pd.Timestamp("2025-09-01", tz="UTC"),
        pd.Timestamp("2025-09-30", tz="UTC"),
    )

    assert depth["CON.F.US.MES.M25"] is None


def test_the_output_filename_cannot_be_mistaken_for_a_databento_file():
    path = roll.output_path("MES",
                            pd.Timestamp("2025-09-01", tz="UTC"),
                            pd.Timestamp("2025-11-30", tz="UTC"))

    assert path.name == "mes_projectx_ohlcv_1m_2025-09_2025-11.parquet"
    assert "v_0" not in path.name
