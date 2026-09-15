"""ProjectX Gateway bar client for MES/MNQ intraday data (second source).

See ``docs/superpowers/specs/2026-09-15-projectx-bar-source-design.md`` for
the design and for what the replacement of Databento gives up.

The gateway has no sandbox, so nothing here can be exercised end to end
without a paid subscription and real credentials. The HTTP session is
injected so the whole module is testable offline.
"""

from __future__ import annotations

import time
import zlib

import pandas as pd
import requests

BASE_URL = "https://api.topstepx.com"

#: ``unit`` enum on retrieveBars: 1=second, 2=minute, 3=hour, 4=day.
UNIT_MINUTE = 2

#: The gateway's documented ceiling on bars per request.
PAGE_LIMIT = 20_000

#: What the gateway's ``v`` field counts. Not CME's tape: the ProjectX SDK
#: documents it as trades executed on the ProjectX platform only.
PLATFORM_VOLUME = "projectx_platform"

#: Attempts per request before a 429 becomes an error rather than a wait.
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0

#: Column order the repo's parquets use, so `loader.load_bars` needs no
#: special casing for a gateway-sourced file.
BAR_COLUMNS = ("open", "high", "low", "close", "volume", "instrument_id", "symbol")

#: CLAUDE.md rule 1: MES and MNQ only.
ALLOWED_ROOTS = frozenset({"MES", "MNQ"})


class InstrumentNotAllowed(ValueError):
    """Raised when a symbol root is outside the rule 1 allowlist."""


class AuthenticationError(RuntimeError):
    """Raised when the gateway refuses a login. Never carries the key."""


class RateLimited(RuntimeError):
    """Raised when the gateway is still returning 429 after every retry."""


class TopstepXClient:
    """Authenticated access to the gateway's contract and history endpoints."""

    def __init__(self, username: str, api_key: str, session=None,
                 base_url: str = BASE_URL, sleep=time.sleep):
        self.username = username
        self.api_key = api_key
        self.session = session if session is not None else requests.Session()
        self.base_url = base_url.rstrip("/")
        self.sleep = sleep
        self.token: str | None = None

    def login(self) -> str:
        """Exchange the API key for a session token.

        The field names come from community SDKs: the gateway's own auth page
        returns HTTP 403 to documentation retrieval, so they are confirmed
        against a real response on the first live run rather than trusted.
        """
        payload = self._post(
            "/api/Auth/loginKey",
            {"userName": self.username, "apiKey": self.api_key},
            authenticated=False,
        )
        if not payload.get("success") or not payload.get("token"):
            # The message is the gateway's, never the credential's.
            raise AuthenticationError(
                "Gateway refused the login for user "
                f"{self.username!r}: {payload.get('errorMessage') or 'no reason given'} "
                f"(errorCode {payload.get('errorCode')})."
            )
        token = str(payload["token"])
        self.token = token
        return token

    def _post(self, path: str, body: dict, authenticated: bool = True) -> dict:
        """POST with backoff on 429.

        The gateway publishes no rate-limit numbers beyond "fair use", so the
        only safe assumption is that a pull deep enough to be worth making
        will hit one. Retries are bounded: a client that retries forever
        turns a throttle into a hang.
        """
        headers = {"Content-Type": "application/json"}
        if authenticated and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = self.base_url + path

        for attempt in range(MAX_RETRIES):
            response = self.session.post(url, json=body, headers=headers)
            if getattr(response, "status_code", 200) != 429:
                return response.json()
            if attempt < MAX_RETRIES - 1:
                self.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

        raise RateLimited(
            f"{path} still rate-limited after {MAX_RETRIES} attempts."
        )

    def resolve_contracts(self, root: str) -> list[str]:
        """Contract ids for a symbol root, newest first.

        The allowlist is checked before any request is issued, so a rejected
        symbol costs no network call and no rate-limit budget.
        """
        if root not in ALLOWED_ROOTS:
            raise InstrumentNotAllowed(
                f"{root!r} is not tradable in this project. "
                f"Allowed roots: {', '.join(sorted(ALLOWED_ROOTS))}."
            )
        payload = self._post("/api/Contract/search",
                             {"searchText": root, "live": False})
        contracts = payload.get("contracts") or []
        # The search is a text match, so it can return a different product
        # whose symbol happens to contain the root. Keep only contracts whose
        # own symbol starts with the root asked for: MESZ25 yes, ESZ25 no.
        return [c["id"] for c in contracts
                if str(c.get("name", "")).startswith(root)]

    def retrieve_bars(self, contract_id: str, start: pd.Timestamp,
                      end: pd.Timestamp,
                      page_limit: int = PAGE_LIMIT) -> pd.DataFrame:
        """One contract's 1-minute bars over ``[start, end]``, UTC-indexed.

        The gateway caps a response at 20,000 bars, so the span is walked
        backwards a page at a time: each request ends where the previous
        page began, and the walk stops when a page is empty or adds nothing
        new. Whether ``endTime`` is inclusive is undocumented, so bars are
        collected into a dict keyed by timestamp and a repeated boundary bar
        is dropped rather than assumed away.

        The earliest timestamp in the result is the deepest history the
        gateway served for this contract - the figure the design records
        rather than guesses at.
        """
        collected: dict[pd.Timestamp, dict] = {}
        cursor = pd.Timestamp(end)

        while True:
            payload = self._post("/api/History/retrieveBars", {
                "contractId": contract_id,
                "live": False,
                "startTime": pd.Timestamp(start).isoformat(),
                "endTime": cursor.isoformat(),
                "unit": UNIT_MINUTE,
                "unitNumber": 1,
                "limit": page_limit,
                "includePartialBar": False,
            })
            page = payload.get("bars") or []
            if not page:
                break

            fresh = 0
            earliest = cursor
            for bar in page:
                ts = pd.Timestamp(bar["t"])
                ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
                earliest = min(earliest, ts)
                if ts not in collected:
                    collected[ts] = bar
                    fresh += 1

            # A page that is all boundary repeats means the gateway has no
            # more history in range; without this the walk cannot terminate.
            if fresh == 0 or earliest <= pd.Timestamp(start):
                break
            cursor = earliest

        return self._frame(collected, contract_id)

    @staticmethod
    def _frame(collected: dict[pd.Timestamp, dict], contract_id: str) -> pd.DataFrame:
        """Gateway bars to the repo's schema, with provenance attached."""
        index = pd.DatetimeIndex(sorted(collected), tz="UTC", name="ts_event")
        rows = [collected[ts] for ts in index]
        frame = pd.DataFrame(
            {
                "open": [float(r["o"]) for r in rows],
                "high": [float(r["h"]) for r in rows],
                "low": [float(r["l"]) for r in rows],
                "close": [float(r["c"]) for r in rows],
                "volume": [r["v"] for r in rows],
                "instrument_id": instrument_id(contract_id),
                "symbol": contract_id,
            },
            index=index,
            columns=list(BAR_COLUMNS),
        )
        # Rule 7 of the design: the gateway's volume counts trades filled on
        # the ProjectX platform, not CME's. Anything that scores it should be
        # able to see that without reading this module.
        frame.attrs["volume_source"] = PLATFORM_VOLUME
        return frame


def instrument_id(contract_id: str) -> int:
    """A stable integer per contract id.

    ``loader.detect_roll_dates`` keys on ``instrument_id`` and raises without
    it. Databento supplies one; the gateway does not, but fetching per
    contract means the value is known rather than inferred. CRC32 is used
    only because it is deterministic across runs and processes - two pulls of
    the same contract must agree, or a roll appears where none happened.
    """
    return zlib.crc32(contract_id.encode("utf-8"))
