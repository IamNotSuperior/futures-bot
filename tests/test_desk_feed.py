"""The bar feed: webhook parsing, replay ordering, and the lookahead guard.

The replay feed is what makes every other desk test meaningful - it is the only
way to exercise the live path without TradingView. So the properties tested
here are the ones that would make a replay lie: bars out of order, a bucket
published before its last minute arrived, or a wall clock leaking into a
historical run.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

import feed as feed_mod
import rules


def payload(ts="2026-08-24 10:05", symbol="MES1!", o=6800.0, h=6805.0,
            l=6799.0, c=6803.0, v=120.0) -> dict:
    epoch_ms = int(pd.Timestamp(ts, tz=rules.ET).timestamp() * 1000)
    return {"symbol": symbol, "time": epoch_ms, "open": o, "high": h,
            "low": l, "close": c, "volume": v}


def bars_frame(start="2026-08-24 09:30", minutes=30, price=6800.0) -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(start, tz=rules.ET),
                          periods=minutes, freq="1min")
    return pd.DataFrame({
        "open": [price + i for i in range(minutes)],
        "high": [price + i + 2 for i in range(minutes)],
        "low": [price + i - 2 for i in range(minutes)],
        "close": [price + i + 1 for i in range(minutes)],
        "volume": [100.0] * minutes,
    }, index=index)


# -- payload parsing --------------------------------------------------------

def test_valid_payload_parses_to_et():
    bar = feed_mod.Bar.from_payload(payload())
    assert bar.symbol == "MES"
    assert bar.timestamp == pd.Timestamp("2026-08-24 10:05", tz=rules.ET)
    assert bar.close == 6803.0


def test_contract_qualified_tickers_reduce_to_the_root():
    for ticker in ("MES1!", "MESZ2026", "MNQ1!", "mnq1!"):
        assert feed_mod.Bar.from_payload(payload(symbol=ticker)).symbol in ("MES", "MNQ")


def test_symbols_outside_rule_1_are_rejected():
    for ticker in ("ES1!", "NQ1!", "SPY", "CL1!"):
        with pytest.raises(feed_mod.BarRejected, match="not permitted"):
            feed_mod.Bar.from_payload(payload(symbol=ticker))


def test_ohlc_integrity_is_checked():
    with pytest.raises(feed_mod.BarRejected, match="integrity"):
        feed_mod.Bar.from_payload(payload(o=6800, h=6790, l=6780, c=6785))
    with pytest.raises(feed_mod.BarRejected, match="integrity"):
        feed_mod.Bar.from_payload(payload(o=6800, h=6810, l=6805, c=6803))


def test_missing_fields_are_rejected():
    body = payload()
    del body["close"]
    with pytest.raises(feed_mod.BarRejected, match="missing"):
        feed_mod.Bar.from_payload(body)


def test_unparseable_time_is_rejected():
    with pytest.raises(feed_mod.BarRejected, match="unparseable"):
        feed_mod.Bar.from_payload({**payload(), "time": "yesterday-ish"})


def test_stale_bars_are_detectable():
    bar = feed_mod.Bar.from_payload(payload("2026-08-24 10:05"))
    assert bar.is_stale(pd.Timestamp("2026-08-24 10:30", tz=rules.ET))
    assert not bar.is_stale(pd.Timestamp("2026-08-24 10:06", tz=rules.ET))


# -- the webhook endpoint ---------------------------------------------------

def test_endpoint_accepts_a_bar_and_calls_back():
    from fastapi.testclient import TestClient

    seen: list[feed_mod.Bar] = []
    client = TestClient(feed_mod.build_webhook_app(seen.append))
    response = client.post("/bar", json=payload())
    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert len(seen) == 1 and seen[0].symbol == "MES"


def test_endpoint_rejects_a_bad_bar_without_calling_back():
    from fastapi.testclient import TestClient

    seen: list[feed_mod.Bar] = []
    client = TestClient(feed_mod.build_webhook_app(seen.append))
    assert client.post("/bar", json=payload(symbol="ES1!")).status_code == 422
    assert client.post("/bar", content=b"not json").status_code == 400
    assert seen == []


def test_endpoint_enforces_a_token_when_configured():
    from fastapi.testclient import TestClient

    seen: list[feed_mod.Bar] = []
    client = TestClient(feed_mod.build_webhook_app(seen.append, token="s3cret"))
    assert client.post("/bar", json=payload()).status_code == 401
    assert client.post("/bar?token=wrong", json=payload()).status_code == 401
    assert client.post("/bar?token=s3cret", json=payload()).status_code == 200
    assert client.post("/bar", json=payload(),
                       headers={"X-Desk-Token": "s3cret"}).status_code == 200
    assert len(seen) == 2


def test_health_reports_counts():
    from fastapi.testclient import TestClient

    client = TestClient(feed_mod.build_webhook_app(lambda b: None))
    client.post("/bar", json=payload())
    client.post("/bar", json=payload(symbol="ES1!"))
    body = client.get("/health").json()
    assert body == {"status": "ok", "accepted": 1, "rejected": 1,
                    "last_bar_at": "2026-08-24T10:05:00-04:00"}


def test_health_is_behind_the_token_too():
    """An open /health on a public tunnel is free reconnaissance.

    The counts are not secrets, but they confirm the service exists and show
    whether it is actively receiving bars. Every legitimate reader of this
    endpoint can hold the token.
    """
    from fastapi.testclient import TestClient

    client = TestClient(feed_mod.build_webhook_app(lambda b: None, token="s3cret"))
    assert client.get("/health").status_code == 401
    assert client.get("/health?token=wrong").status_code == 401
    assert client.get("/health?token=s3cret").status_code == 200
    assert client.get("/health",
                      headers={"X-Desk-Token": "s3cret"}).status_code == 200


def test_no_endpoint_is_open_when_a_token_is_set():
    """Whole-surface check: adding a route must not add an unauthenticated one."""
    from fastapi.testclient import TestClient

    app = feed_mod.build_webhook_app(lambda b: None, token="s3cret")
    client = TestClient(app)
    paths = {r.path for r in app.routes if hasattr(r, "path")
             and not r.path.startswith("/openapi")}
    for path in paths:
        for method, call in (("GET", client.get), ("POST", client.post)):
            response = call(path)
            assert response.status_code != 200, (
                f"{method} {path} answered without a token"
            )


# -- replay -----------------------------------------------------------------

def test_replay_emits_every_bar_in_order():
    replay = feed_mod.ReplayFeed(bars_frame(minutes=30), speed=0)
    bars = list(replay.iter_bars())
    assert len(bars) == 30
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)
    assert bars[0].timestamp == pd.Timestamp("2026-08-24 09:30", tz=rules.ET)


def test_replay_preserves_ohlcv_exactly():
    frame = bars_frame(minutes=5)
    for bar, (ts, row) in zip(feed_mod.ReplayFeed(frame, speed=0).iter_bars(),
                              frame.iterrows()):
        assert bar.timestamp == ts
        assert (bar.open, bar.high, bar.low, bar.close) == (
            row["open"], row["high"], row["low"], row["close"])


def test_replay_rejects_an_instrument_outside_rule_1():
    with pytest.raises(rules.RuleViolation):
        feed_mod.ReplayFeed(bars_frame(), symbol="ES")


def test_replay_sorts_unsorted_input():
    frame = bars_frame(minutes=10).sample(frac=1.0, random_state=0)
    bars = list(feed_mod.ReplayFeed(frame, speed=0).iter_bars())
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)


def test_replay_queue_ends_with_a_sentinel():
    async def go():
        queue: asyncio.Queue = asyncio.Queue()
        replay = feed_mod.ReplayFeed(bars_frame(minutes=4), speed=0)
        await replay.run(queue)
        return [queue.get_nowait() for _ in range(queue.qsize())]

    items = asyncio.run(go())
    assert len(items) == 5
    assert items[-1] is None


def test_drain_stops_at_the_sentinel():
    async def go():
        queue: asyncio.Queue = asyncio.Queue()
        await feed_mod.ReplayFeed(bars_frame(minutes=3), speed=0).run(queue)
        return [bar async for bar in feed_mod.drain(queue)]

    assert len(asyncio.run(go())) == 3


def test_replay_speed_controls_the_delay():
    """speed=60 means one bar-minute per wall-clock second."""
    import time as wall

    async def go(speed):
        queue: asyncio.Queue = asyncio.Queue()
        await feed_mod.ReplayFeed(bars_frame(minutes=2), speed=speed).run(queue)

    start = wall.perf_counter()
    asyncio.run(go(6000))          # 0.01s per bar
    elapsed = wall.perf_counter() - start
    assert 0.005 < elapsed < 1.0   # generous: asserts the sleep exists at all


def test_replay_from_parquet_window_is_inclusive(tmp_path):
    frame = pd.concat([bars_frame("2026-08-24 09:30", 5),
                       bars_frame("2026-08-25 09:30", 5),
                       bars_frame("2026-08-26 09:30", 5)])
    path = tmp_path / "bars.parquet"
    frame.to_parquet(path)

    replay = feed_mod.ReplayFeed.from_parquet(path, "2026-08-24", "2026-08-25",
                                              speed=0)
    days = {b.timestamp.date() for b in replay.iter_bars()}
    assert days == {pd.Timestamp("2026-08-24").date(),
                    pd.Timestamp("2026-08-25").date()}


def test_replay_from_parquet_refuses_an_empty_window(tmp_path):
    path = tmp_path / "bars.parquet"
    bars_frame("2026-08-24 09:30", 5).to_parquet(path)
    with pytest.raises(ValueError, match="no cached bars"):
        feed_mod.ReplayFeed.from_parquet(path, "2026-09-01", "2026-09-02")


# -- aggregation and lookahead ---------------------------------------------

def test_bucket_is_published_only_after_its_last_minute():
    """The 09:30 5-minute bucket must not appear until 09:35 arrives.

    Publishing at 09:34 would hand the desk a bucket whose high and low were
    still moving. That is lookahead arriving through the feed.
    """
    agg = feed_mod.BarAggregator(5)
    bars = list(feed_mod.ReplayFeed(bars_frame("2026-08-24 09:30", 6),
                                    speed=0).iter_bars())
    published = [agg.push(b) for b in bars]
    assert published[:5] == [None] * 5
    assert published[5] is not None
    assert published[5].timestamp == pd.Timestamp("2026-08-24 09:30", tz=rules.ET)


def test_bucket_aggregates_ohlcv_correctly():
    agg = feed_mod.BarAggregator(5)
    bars = list(feed_mod.ReplayFeed(bars_frame("2026-08-24 09:30", 6),
                                    speed=0).iter_bars())
    for bar in bars:
        bucket = agg.push(bar)
    first_five = bars[:5]
    assert bucket.open == first_five[0].open
    assert bucket.close == first_five[-1].close
    assert bucket.high == max(b.high for b in first_five)
    assert bucket.low == min(b.low for b in first_five)
    assert bucket.volume == sum(b.volume for b in first_five)


def test_flush_emits_the_open_bucket():
    agg = feed_mod.BarAggregator(5)
    for bar in feed_mod.ReplayFeed(bars_frame("2026-08-24 09:30", 3),
                                   speed=0).iter_bars():
        agg.push(bar)
    flushed = agg.flush()
    assert flushed is not None
    assert flushed.timestamp == pd.Timestamp("2026-08-24 09:30", tz=rules.ET)
    assert agg.flush() is None


def test_out_of_order_bars_are_dropped_not_merged():
    agg = feed_mod.BarAggregator(5)
    bars = list(feed_mod.ReplayFeed(bars_frame("2026-08-24 09:30", 12),
                                    speed=0).iter_bars())
    for bar in bars[:7]:
        agg.push(bar)
    assert agg.push(bars[0]) is None      # a 09:30 bar arriving after 09:35


def test_one_minute_aggregator_is_a_pass_through_after_the_first():
    agg = feed_mod.BarAggregator(1)
    bars = list(feed_mod.ReplayFeed(bars_frame("2026-08-24 09:30", 3),
                                    speed=0).iter_bars())
    out = [agg.push(b) for b in bars]
    # Each bucket is published when the next bar arrives, so a 1-minute
    # aggregator lags by exactly one bar. This is why desk.py drives orb2 from
    # raw bars instead of putting a 1-minute aggregator in front of it.
    assert out[0] is None
    assert out[1].timestamp == bars[0].timestamp
