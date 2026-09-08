"""Bar feeds. Two producers, one consumer, and no way to tell them apart.

The desk loop reads :class:`Bar` objects off an :class:`asyncio.Queue` and does
not know whether they arrived from a TradingView webhook or from a parquet
replay. That is deliberate: a replay that took a different path through the
code would test the replay rather than the desk.

Two producers:

* :class:`WebhookFeed` - a FastAPI app with one POST endpoint, fed by the
  TradingView alert that ``pine/bar_feed.pine`` fires on every bar close.
* :class:`ReplayFeed` - cached parquet bars, emitted in order at a configurable
  speed-up. ``--replay`` uses it.

Timestamp convention
--------------------
Bars are **left-closed and labelled by their opening minute**, matching
``data/loader.py`` and the backtest engine. A bar labelled ``15:25`` covers
15:25:00-15:25:59. The 5-minute bucket labelled ``15:25`` therefore covers
15:25-15:29 and is only complete once the ``15:29`` one-minute bar has arrived
- which is what :class:`BarAggregator` exists to determine. Reading a bucket
early is lookahead, and it is silent.

Naive timestamps are interpreted as ET by ``rules.to_et``. TradingView is
configured to send epoch milliseconds, which is unambiguous; the parser rejects
anything it cannot place on a timeline.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("data", "journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import rules  # noqa: E402

# FastAPI must be imported at module level, not inside build_webhook_app.
# `from __future__ import annotations` turns every annotation into a string,
# and FastAPI resolves those against the *module* globals - so a `Request`
# bound only as a function local is unresolvable, and FastAPI silently falls
# back to treating the parameter as a query field. The symptom is a 422 on
# every valid POST, blaming a missing query parameter named "request".
try:
    from fastapi import FastAPI, Header, HTTPException, Query, Request
except ImportError:  # pragma: no cover - replay-only use needs no web server
    FastAPI = Header = HTTPException = Query = Request = None

log = logging.getLogger("desk.feed")

#: Bars older than this when they arrive are logged as stale. TradingView
#: alerts can be delayed; a delayed bar is still tradeable, a very old one is a
#: reconnect replaying history and should not fire a signal.
STALE_AFTER = pd.Timedelta(minutes=3)


class BarRejected(ValueError):
    """A payload was not a usable bar. The message says why."""


@dataclass(frozen=True)
class Bar:
    """One completed OHLCV bar, timestamped by its opening minute in ET."""

    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_payload(cls, payload: dict) -> "Bar":
        """Parse and validate one webhook payload.

        Raises :class:`BarRejected` rather than returning a partly-formed bar:
        a feed that silently repairs bad data is a feed that hides an alert
        misconfiguration until it costs something.
        """
        missing = {"symbol", "time", "open", "high", "low", "close"} - set(payload)
        if missing:
            raise BarRejected(f"payload missing {sorted(missing)}")

        symbol = str(payload["symbol"]).strip().upper()
        # TradingView sends contract-qualified tickers ("MES1!", "MESZ2026").
        # Rule 1's allowlist is on the root, so reduce before checking.
        root = symbol[:3]
        if not rules.is_allowed_instrument(root):
            raise BarRejected(
                f"symbol {symbol!r} (root {root!r}) is not permitted; rule 1 "
                f"allows {sorted(rules.ALLOWED_INSTRUMENTS)}"
            )

        raw_time = payload["time"]
        try:
            if isinstance(raw_time, (int, float)):
                # TradingView's {{timenow}} and bar time are epoch milliseconds.
                ts = pd.Timestamp(int(raw_time), unit="ms", tz="UTC")
            else:
                ts = pd.Timestamp(str(raw_time))
        except (ValueError, TypeError) as exc:
            raise BarRejected(f"unparseable time {raw_time!r}: {exc}") from None
        if ts is pd.NaT or pd.isna(ts):
            raise BarRejected(f"unparseable time {raw_time!r}")
        timestamp = rules.to_et(ts).floor("1min")

        try:
            o, h, l, c = (float(payload[k]) for k in ("open", "high", "low", "close"))
            v = float(payload.get("volume") or 0.0)
        except (ValueError, TypeError) as exc:
            raise BarRejected(f"non-numeric OHLCV: {exc}") from None

        # OHLC integrity. data/validate.py applies the same check to cached
        # bars; a live feed deserves it more, not less.
        if h < max(o, c) or l > min(o, c) or h < l:
            raise BarRejected(
                f"OHLC integrity violation at {timestamp}: "
                f"o={o} h={h} l={l} c={c}"
            )
        if v < 0:
            raise BarRejected(f"negative volume {v} at {timestamp}")

        return cls(root, timestamp, o, h, l, c, v)

    def is_stale(self, now: pd.Timestamp) -> bool:
        return (rules.to_et(now) - self.timestamp) > STALE_AFTER


class BarAggregator:
    """One-minute bars in, completed N-minute bars out.

    A bucket is emitted only when a bar belonging to a *later* bucket arrives,
    so a bucket is never published before its last minute has been seen. This
    is the mechanism that keeps the desk free of lookahead, and it is why the
    aggregator - not the caller - decides when a bucket is done.

    The consequence worth knowing: the 15:25 five-minute bucket is published
    when the 15:30 one-minute bar arrives, not when 15:29 does. A session's
    final bucket is flushed explicitly by :meth:`flush`.
    """

    def __init__(self, minutes: int = 5) -> None:
        if minutes < 1:
            raise ValueError(f"minutes must be >= 1, got {minutes}")
        self.minutes = int(minutes)
        self._bucket: pd.Timestamp | None = None
        self._bars: list[Bar] = []

    def _label(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(f"{self.minutes}min")

    def _combine(self) -> Bar:
        bars = self._bars
        return Bar(
            symbol=bars[0].symbol,
            timestamp=self._bucket,
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
        )

    def push(self, bar: Bar) -> Bar | None:
        """Add a one-minute bar. Returns a completed bucket, or None."""
        label = self._label(bar.timestamp)
        finished: Bar | None = None
        if self._bucket is not None and label != self._bucket:
            if label < self._bucket:
                # Out-of-order arrival. Dropping is safer than rewriting a
                # bucket the desk may already have acted on.
                log.warning("dropping out-of-order bar %s (bucket %s open)",
                            bar.timestamp, self._bucket)
                return None
            finished = self._combine()
        if finished is not None or self._bucket is None:
            self._bucket = label
            self._bars = []
        self._bars.append(bar)
        return finished

    def flush(self) -> Bar | None:
        """Emit the open bucket, if any. Used at a session boundary."""
        if not self._bars:
            return None
        out = self._combine()
        self._bucket, self._bars = None, []
        return out


class ReplayFeed:
    """Cached parquet bars, replayed in order at ``speed`` times real time.

    ``speed=60`` turns one bar-minute into one wall-clock second, so a full
    RTH session takes about six and a half minutes. ``speed=0`` disables the
    sleep entirely, which is what the tests use.

    The replay's ``now`` is the **bar's** timestamp, not the wall clock. Every
    guard in the desk takes an injected ``now`` for exactly this reason: a
    replay of 2026-08-24 must be judged against that day's session clock, and
    a wall-clock guard would block every bar of it as after-hours.
    """

    def __init__(self, bars: pd.DataFrame, symbol: str = "MES",
                 speed: float = 60.0) -> None:
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError("replay bars must be indexed by timestamp")
        self.bars = bars.sort_index()
        self.symbol = rules.require_allowed_instrument(symbol)
        self.speed = float(speed)

    @classmethod
    def from_parquet(cls, path: Path, start, end, symbol: str = "MES",
                     speed: float = 60.0) -> "ReplayFeed":
        """Load ``[start, end]`` inclusive of both session dates."""
        import loader  # noqa: PLC0415 - data/ is on sys.path above

        bars = loader.load_bars(Path(path))
        lo = rules.to_et(pd.Timestamp(start)).normalize()
        hi = rules.to_et(pd.Timestamp(end)).normalize() + pd.Timedelta(days=1)
        window = bars.loc[(bars.index >= lo) & (bars.index < hi)]
        if window.empty:
            raise ValueError(
                f"no cached bars between {lo.date()} and {hi.date() - pd.Timedelta(days=1)}"
                f"; the parquet covers {bars.index[0].date()} to {bars.index[-1].date()}"
            )
        return cls(window, symbol=symbol, speed=speed)

    def __len__(self) -> int:
        return len(self.bars)

    def iter_bars(self) -> Iterable[Bar]:
        """Synchronous iteration, for tests and for the summary run."""
        for ts, row in self.bars.iterrows():
            yield Bar(
                symbol=self.symbol,
                timestamp=rules.to_et(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0) or 0.0),
            )

    async def run(self, queue: "asyncio.Queue[Bar | None]") -> None:
        """Push every bar onto ``queue``, then a ``None`` sentinel."""
        delay = 0.0 if self.speed <= 0 else 60.0 / self.speed
        for bar in self.iter_bars():
            await queue.put(bar)
            if delay:
                await asyncio.sleep(delay)
        await queue.put(None)


def build_webhook_app(on_bar: Callable[[Bar], None], token: str | None = None):
    """A FastAPI app exposing ``POST /bar`` and ``GET /health``.

    ``on_bar`` is called once per accepted bar. It must not block for long -
    the desk passes a queue-put, and the strategy work happens on the consumer
    side.

    ``token``, when set, is required as ``?token=`` or an ``X-Desk-Token``
    header. The receiver binds to localhost by default, so this is a guard
    against another process on the same machine rather than against the
    internet; TradingView cannot reach localhost without a tunnel, and if one
    is opened the token becomes the only thing standing in front of the feed.
    """
    if FastAPI is None:
        raise RuntimeError(
            "fastapi is not installed; `pip install -r requirements.txt`. "
            "Replay does not need it - only the live webhook does."
        )

    app = FastAPI(title="desk bar feed", docs_url=None, redoc_url=None)
    app.state.accepted = 0
    app.state.rejected = 0
    app.state.last_bar_at = None

    def _authorise(supplied: str | None) -> None:
        if token and supplied != token:
            raise HTTPException(status_code=401, detail="bad or missing token")

    @app.get("/health")
    async def health(
        token_q: str | None = Query(default=None, alias="token"),
        x_desk_token: str | None = Header(default=None),
    ) -> dict:
        # Behind the same token as /bar. The counts and last-bar time are not
        # secrets, but on a public tunnel an open endpoint confirms the service
        # exists and shows whether it is actively receiving - which is free
        # reconnaissance for anyone who finds the hostname. There is no reader
        # of this endpoint that cannot also hold the token.
        _authorise(token_q or x_desk_token)
        last = app.state.last_bar_at
        return {
            "status": "ok",
            "accepted": app.state.accepted,
            "rejected": app.state.rejected,
            "last_bar_at": last.isoformat() if last is not None else None,
        }

    @app.post("/bar")
    async def receive_bar(
        request: Request,
        token_q: str | None = Query(default=None, alias="token"),
        x_desk_token: str | None = Header(default=None),
    ) -> dict:
        _authorise(token_q or x_desk_token)
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - any malformed body is one failure
            app.state.rejected += 1
            raise HTTPException(status_code=400, detail="body is not JSON") from None
        if not isinstance(payload, dict):
            app.state.rejected += 1
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        try:
            bar = Bar.from_payload(payload)
        except BarRejected as exc:
            app.state.rejected += 1
            log.warning("rejected bar: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from None

        app.state.accepted += 1
        app.state.last_bar_at = bar.timestamp
        on_bar(bar)
        return {"accepted": True, "timestamp": bar.timestamp.isoformat()}

    return app


class WebhookFeed:
    """Runs :func:`build_webhook_app` under uvicorn, pushing onto a queue."""

    def __init__(self, queue: "asyncio.Queue[Bar | None]", host: str = "127.0.0.1",
                 port: int = 8787, token: str | None = None) -> None:
        self.queue = queue
        self.host = host
        self.port = int(port)
        self.token = token
        self._loop: asyncio.AbstractEventLoop | None = None
        self.app = build_webhook_app(self._enqueue, token=token)

    def _enqueue(self, bar: Bar) -> None:
        # Called from the request handler, which runs on the same loop.
        self.queue.put_nowait(bar)

    async def run(self) -> None:
        import uvicorn  # noqa: PLC0415

        self._loop = asyncio.get_running_loop()
        config = uvicorn.Config(self.app, host=self.host, port=self.port,
                                log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        log.info("webhook listening on http://%s:%d/bar", self.host, self.port)
        await self.server.serve()

    def stop(self) -> None:
        """Ask uvicorn to exit cleanly.

        Cancelling the serving task instead works, but uvicorn's lifespan
        handler logs the CancelledError as a full traceback - which, on a
        deliberate fatal exit, buries the one line the operator needs under
        twenty lines of noise. ``should_exit`` lets ``serve()`` return on its
        own.
        """
        server = getattr(self, "server", None)
        if server is not None:
            server.should_exit = True


async def drain(queue: "asyncio.Queue[Bar | None]") -> AsyncIterator[Bar]:
    """Yield bars until the ``None`` sentinel."""
    while True:
        bar = await queue.get()
        if bar is None:
            return
        yield bar
