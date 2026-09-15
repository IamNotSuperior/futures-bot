"""Tests for data/projectx.py, the ProjectX Gateway bar client.

Every test drives a fake HTTP session, so nothing here touches the network,
authenticates, or spends a subscription. The gateway has no sandbox, so the
offline surface is the whole testable surface until credentials exist.
"""

from __future__ import annotations

import pandas as pd
import pytest

import loader
import projectx


def minute_bars(start: str, count: int) -> list[dict]:
    """Gateway-shaped bars: six fields, ISO timestamps, newest last."""
    index = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    return [
        {"t": ts.isoformat(), "o": 100.0 + i, "h": 101.0 + i,
         "l": 99.0 + i, "c": 100.5 + i, "v": 10 + i}
        for i, ts in enumerate(index)
    ]


class FakeGateway:
    """Serves a canned bar list the way the gateway is documented to.

    ``endTime`` is treated as **inclusive** and the newest ``limit`` bars in
    range are returned. Inclusivity is undocumented, so the client must not
    depend on it either way: this fake repeats the boundary bar on every
    page, and a client that fails to drop it will loop or duplicate.
    """

    def __init__(self, bars: list[dict]):
        self.bars = bars
        self.pages = 0

    def __call__(self, body: dict) -> FakeResponse:
        self.pages += 1
        end = pd.Timestamp(body["endTime"])
        start = pd.Timestamp(body["startTime"])
        in_range = [b for b in self.bars
                    if start <= pd.Timestamp(b["t"]) <= end]
        return FakeResponse({
            "success": True,
            "errorCode": 0,
            "bars": in_range[-body["limit"]:],
        })


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self.payload


class FakeSession:
    """Records every request; returns canned answers keyed by URL suffix."""

    def __init__(self, answers: dict[str, object] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.answers = answers or {}

    def post(self, url: str, json: dict | None = None, headers: dict | None = None):
        self.calls.append((url, json or {}))
        for suffix, answer in self.answers.items():
            if url.endswith(suffix):
                if callable(answer):
                    return answer(json or {})
                return answer
        return FakeResponse({"success": True})


def test_login_posts_the_credentials_and_returns_the_session_token():
    session = FakeSession({
        "/api/Auth/loginKey": FakeResponse(
            {"success": True, "errorCode": 0, "token": "jwt-123"}
        ),
    })
    client = projectx.TopstepXClient("user", "key", session=session)

    token = client.login()

    assert token == "jwt-123"
    url, body = session.calls[0]
    assert url == "https://api.topstepx.com/api/Auth/loginKey"
    assert body == {"userName": "user", "apiKey": "key"}


def test_a_rejected_login_raises_without_echoing_the_key():
    """A credential must never reach a log or a traceback."""
    session = FakeSession({
        "/api/Auth/loginKey": FakeResponse(
            {"success": False, "errorCode": 3, "errorMessage": "Invalid credentials"}
        ),
    })
    client = projectx.TopstepXClient("user", "sup3r-secret", session=session)

    with pytest.raises(projectx.AuthenticationError) as excinfo:
        client.login()

    assert "sup3r-secret" not in str(excinfo.value)


def test_root_outside_the_allowlist_is_refused_before_any_request():
    """Rule 1: MES and MNQ only, and a refusal costs no network call."""
    session = FakeSession()
    client = projectx.TopstepXClient("user", "key", session=session)

    with pytest.raises(projectx.InstrumentNotAllowed):
        client.resolve_contracts("ES")

    assert session.calls == []


def logged_in(answers: dict) -> projectx.TopstepXClient:
    """A client that has already exchanged its key for a token."""
    answers = {
        "/api/Auth/loginKey": FakeResponse(
            {"success": True, "errorCode": 0, "token": "jwt"}
        ),
        **answers,
    }
    client = projectx.TopstepXClient("user", "key", session=FakeSession(answers))
    client.login()
    return client


def test_resolve_contracts_returns_the_gateway_ids_newest_first():
    client = logged_in({
        "/api/Contract/search": FakeResponse({
            "success": True,
            "errorCode": 0,
            "contracts": [
                {"id": "CON.F.US.MES.Z25", "name": "MESZ25"},
                {"id": "CON.F.US.MES.U25", "name": "MESU25"},
            ],
        }),
    })

    assert client.resolve_contracts("MES") == [
        "CON.F.US.MES.Z25",
        "CON.F.US.MES.U25",
    ]


def test_a_foreign_root_in_the_search_response_is_dropped():
    """The search is fuzzy; 'MES' can return contracts we may not trade."""
    client = logged_in({
        "/api/Contract/search": FakeResponse({
            "success": True,
            "errorCode": 0,
            "contracts": [
                {"id": "CON.F.US.MES.Z25", "name": "MESZ25"},
                {"id": "CON.F.US.EP.Z25", "name": "ESZ25"},
            ],
        }),
    })

    assert client.resolve_contracts("MES") == ["CON.F.US.MES.Z25"]


def test_pages_assemble_in_order_without_duplicating_the_boundary_bar():
    """Seven bars at a page size of three: four requests, seven unique rows."""
    gateway = FakeGateway(minute_bars("2026-09-01 14:30", 7))
    client = logged_in({"/api/History/retrieveBars": gateway})

    frame = client.retrieve_bars(
        "CON.F.US.MES.Z25",
        pd.Timestamp("2026-09-01 00:00", tz="UTC"),
        pd.Timestamp("2026-09-01 23:59", tz="UTC"),
        page_limit=3,
    )

    assert len(frame) == 7
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    assert gateway.pages == 4


def test_an_empty_page_terminates_pagination():
    gateway = FakeGateway([])
    client = logged_in({"/api/History/retrieveBars": gateway})

    frame = client.retrieve_bars(
        "CON.F.US.MES.Z25",
        pd.Timestamp("2026-09-01 00:00", tz="UTC"),
        pd.Timestamp("2026-09-01 23:59", tz="UTC"),
        page_limit=3,
    )

    assert frame.empty
    assert gateway.pages == 1
    assert list(frame.columns) == list(projectx.BAR_COLUMNS)


def test_bars_normalise_to_the_repo_schema_and_survive_a_parquet_round_trip(tmp_path):
    """loader.load_bars must consume the output with no special casing."""
    gateway = FakeGateway(minute_bars("2026-09-01 14:30", 5))
    client = logged_in({"/api/History/retrieveBars": gateway})

    frame = client.retrieve_bars(
        "CON.F.US.MES.Z25",
        pd.Timestamp("2026-09-01 00:00", tz="UTC"),
        pd.Timestamp("2026-09-01 23:59", tz="UTC"),
    )

    assert list(frame.columns) == list(projectx.BAR_COLUMNS)
    assert str(frame.index.tz) == "UTC"
    assert frame["volume"].iloc[0] == 10

    path = tmp_path / "round_trip.parquet"
    frame.to_parquet(path)
    reloaded = loader.load_bars(path)
    assert len(reloaded) == 5
    assert str(reloaded.index.tz) == "America/New_York"


def test_every_bar_carries_the_contract_it_was_fetched_from():
    """Roll detection needs instrument_id; per-contract fetching knows it."""
    gateway = FakeGateway(minute_bars("2026-09-01 14:30", 3))
    client = logged_in({"/api/History/retrieveBars": gateway})

    frame = client.retrieve_bars(
        "CON.F.US.MES.Z25",
        pd.Timestamp("2026-09-01 00:00", tz="UTC"),
        pd.Timestamp("2026-09-01 23:59", tz="UTC"),
    )

    assert set(frame["symbol"]) == {"CON.F.US.MES.Z25"}
    assert frame["instrument_id"].nunique() == 1


def test_the_volume_source_is_recorded_on_the_frame():
    """The gateway's volume is platform-internal, not CME's. Say so."""
    gateway = FakeGateway(minute_bars("2026-09-01 14:30", 3))
    client = logged_in({"/api/History/retrieveBars": gateway})

    frame = client.retrieve_bars(
        "CON.F.US.MES.Z25",
        pd.Timestamp("2026-09-01 00:00", tz="UTC"),
        pd.Timestamp("2026-09-01 23:59", tz="UTC"),
    )

    assert frame.attrs["volume_source"] == "projectx_platform"


class Throttled:
    """Returns 429 for the first ``fails`` calls, then a real answer."""

    def __init__(self, fails: int):
        self.fails = fails
        self.calls = 0

    def __call__(self, body: dict) -> FakeResponse:
        self.calls += 1
        if self.calls <= self.fails:
            return FakeResponse({}, status_code=429)
        return FakeResponse({"success": True, "errorCode": 0, "bars": []})


def test_a_rate_limited_request_is_retried_with_growing_backoff():
    gateway = Throttled(fails=2)
    slept: list[float] = []
    client = projectx.TopstepXClient(
        "user", "key",
        session=FakeSession({
            "/api/Auth/loginKey": FakeResponse(
                {"success": True, "errorCode": 0, "token": "jwt"}
            ),
            "/api/History/retrieveBars": gateway,
        }),
        sleep=slept.append,
    )
    client.login()

    frame = client.retrieve_bars(
        "CON.F.US.MES.Z25",
        pd.Timestamp("2026-09-01 00:00", tz="UTC"),
        pd.Timestamp("2026-09-01 23:59", tz="UTC"),
    )

    assert frame.empty
    assert gateway.calls == 3
    assert slept == sorted(slept) and len(slept) == 2


def test_persistent_rate_limiting_raises_rather_than_looping_forever():
    gateway = Throttled(fails=99)
    client = projectx.TopstepXClient(
        "user", "key",
        session=FakeSession({
            "/api/Auth/loginKey": FakeResponse(
                {"success": True, "errorCode": 0, "token": "jwt"}
            ),
            "/api/History/retrieveBars": gateway,
        }),
        sleep=lambda _: None,
    )
    client.login()

    with pytest.raises(projectx.RateLimited):
        client.retrieve_bars(
            "CON.F.US.MES.Z25",
            pd.Timestamp("2026-09-01 00:00", tz="UTC"),
            pd.Timestamp("2026-09-01 23:59", tz="UTC"),
        )

    assert gateway.calls == projectx.MAX_RETRIES


def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TOPSTEPX_USERNAME", "trader1")
    monkeypatch.setenv("TOPSTEPX_API_KEY", "abc123")

    assert loader.get_topstepx_credentials() == ("trader1", "abc123")


def test_a_missing_credential_names_the_key_but_never_its_value(monkeypatch):
    monkeypatch.setenv("TOPSTEPX_USERNAME", "trader1")
    monkeypatch.setenv("TOPSTEPX_API_KEY", "   ")

    with pytest.raises(loader.MissingCredentialError) as excinfo:
        loader.get_topstepx_credentials()

    assert "TOPSTEPX_API_KEY" in str(excinfo.value)
    assert "trader1" not in str(excinfo.value)


def test_a_gateway_filename_marks_the_volume_as_platform_internal():
    """`attrs` does not survive parquet, so the filename carries provenance."""
    import validate

    assert validate.volume_source(
        "mes_projectx_ohlcv_1m_2026-07_2026-09.parquet"
    ) == projectx.PLATFORM_VOLUME


def test_a_databento_filename_is_reported_as_exchange_volume():
    import validate

    assert validate.volume_source(
        "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
    ) == validate.EXCHANGE_VOLUME
