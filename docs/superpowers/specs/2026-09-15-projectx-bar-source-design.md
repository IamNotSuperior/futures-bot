# ProjectX Gateway as a second bar source — design

**Date:** 2026-09-15. **Status:** built and tested offline. Landed as a
**second, read-only source**. The backtest source is unchanged.

## Purpose

A live ProjectX Gateway (TopstepX) client for MES/MNQ 1-minute bars,
alongside Databento. It talks to the gateway's REST API directly, which is
what distinguishes it from `data/topstep.py` (§8).

**It is not a backtest source and nothing reads it by default.**

## The decision trail

This design was written to *replace* Databento, on the operator's direction,
reaffirmed three times after the objections below were put in front of them.
While it was being built, a parallel session recorded the opposite decision
in `docs/HANDOFF.md` §3.14, committed at `55e050a`: **"Databento stays the
only backtest source."**

The operator resolved the contradiction: **keep both, neither swapped.** The
client lands; `backtests/walkforward.py:42` and `bots/runners.py:49` keep
pointing at the Databento cache; §3.14 stands.

The objections that made the replacement wrong are recorded in §3.14 and are
not restated here. Two of them corrected figures reported during this
design's own research, and the corrections are the ones to keep:

| Claim made during design | Measured in §3.14 |
|---|---|
| Minute-bar depth "~8 weeks", from an unexecuted spike issue | **About three months**, measured over 84,075 MES bars |
| Gateway bars "won't be tick-identical" to Databento | Timestamps and RTH bar counts match **exactly**; close identical on **96.1%** of bars, the residual being three days at the March→June roll — a roll-method difference, not a feed-quality one |

The one concern that survives intact is the volume column (§6).

## 1. What is explicitly not changed

- **No `PARQUET` constant moves.** No runner, no walk-forward and no verdict
  reads gateway data.
- **`data/mes_v_0_ohlcv_1m_*.parquet` and the forward file are not written
  to, moved or deleted.** Entry 13 is pre-registered against the forward
  file with a no-peek rule until about September 2028.
- **No recorded verdict is revised** (rule 12).
- **`data/fetch.py` and `data/extend.py` are untouched.**

## 2. What the API serves

From primary documentation, 2026-09-15:

| Fact | Source |
|---|---|
| `POST https://api.topstepx.com/api/History/retrieveBars`, body `{contractId, live, startTime, endTime, unit, unitNumber, limit, includePartialBar}`; `unit` 1=sec 2=min 3=hour 4=day 5=week 6=month | gateway.docs.projectx.com |
| Response bars carry six fields: `t, o, h, l, c, v` | gateway.docs.projectx.com |
| Maximum 20,000 bars per request | gateway.docs.projectx.com |
| $29/month, $14.50 for Topstep traders; active Topstep account required; **no sandbox** | help.topstep.com |
| Volume is "trades executed through the ProjectX platform only, not full exchange volume from CME" | project-x-py SDK docs |

Bars are served per `contractId` (`CON.F.US.MES.Z25`); there is no `.v.0`
equivalent and no instrument id on a bar. §4 turns that into an advantage.

## 3. `data/projectx.py` — the client

Three calls: `POST /api/Auth/loginKey` `{userName, apiKey}` for a JWT;
`POST /api/Contract/search` to resolve a root, with the rule 1 allowlist
checked **before any request is issued** so a rejected symbol costs no
network call; `POST /api/History/retrieveBars` paginated backwards from
`endTime` in pages of at most 20,000, stopping when a page is empty or adds
nothing new.

Whether `endTime` is inclusive is undocumented, so bars are collected into a
dict keyed by timestamp and a repeated boundary bar is dropped rather than
assumed away. Without that, an inclusive `endTime` makes the walk fail to
terminate.

`MAX_RETRIES` bounded backoff on HTTP 429: the gateway publishes no
rate-limit numbers beyond "fair use", and a client that retries forever
turns a throttle into a hang.

**The auth field names are unverified.** The gateway's auth documentation
page returns HTTP 403 to retrieval, so they come from community SDKs and are
confirmed against a real response on the first live run.

## 4. Schema, and the synthesised instrument id

Gateway `t, o, h, l, c, v` maps to `open, high, low, close, volume` on a UTC
`DatetimeIndex`, so `loader.load_bars` consumes the output unmodified.

`instrument_id` is synthesised as a stable CRC32 per contract id. Because
bars are fetched per contract, the column records a fact known at fetch time
rather than one inferred from a vendor's stitching, and
`loader.detect_roll_dates` — which raises without it, and which the
no-entry-on-roll-date rule depends on — works unmodified. CRC32 only for
determinism: two pulls of the same contract must agree, or a roll appears
where none happened.

## 5. `data/roll.py` — continuous series by calendar roll

A volume roll cannot be reproduced from a feed whose volume is not the
market's, so the substitute is the CME equity-index convention: quarterlies
expiring the third Friday, rolling the Thursday eight days earlier, **no
back-adjustment** — matching `.v.0`, which is not back-adjusted either.

Bars are kept only where their contract is active on that bar's **ET session
date**; a roll that moved with UTC would land mid-session for part of the
year.

`roll.pull` returns the stitched frame **and a depth map** — each contract
id to the earliest bar the gateway actually served, or `None` where it
served nothing. A contract that returns nothing is reported rather than
dropped, because the absence is the finding.

## 6. Provenance, and the volume

Output goes to `data/<root>_projectx_ohlcv_1m_<span>.parquet`. A filename
that cannot be mistaken for a Databento file is the cheapest guard against a
later session comparing two epochs by accident.

`frame.attrs["volume_source"]` is set at construction, but **`.attrs` does
not survive `to_parquet`**, so `validate.volume_source(path)` reads
provenance off the filename and the data-quality report prints it. The
gateway's volume counts fills on one platform; a volume-based signal scored
on it measures the broker's flow, not the market's.

## 7. Credentials

`.env`, as `TOPSTEPX_USERNAME` and `TOPSTEPX_API_KEY`, read by
`loader.get_topstepx_credentials()`, raising `MissingCredentialError`
without echoing either value. The names keep the vendor, not the gateway:
the account being authenticated is a TopstepX account.

`config.json` from the upstream script is deliberately not adopted — this
repo has one gitignored place a secret can live, and a second one is how a
key reaches GitHub.

## 8. Relationship to `data/topstep.py`

Two modules, one character apart, doing different things:

| | `data/topstep.py` (committed, `55e050a`) | `data/projectx.py` (this design) |
|---|---|---|
| Source | CSV clone of `axb0306/cme-futures-ohlc` | Live gateway REST API |
| Needs credentials | No | Yes, paid subscription |
| Role | Read-only cross-check of the cache, feeds the viewer's data panel | Second bar source; nothing reads it by default |

The gateway module was renamed from `topstepx.py` to `projectx.py` for this
reason: `topstep.py` is committed and referenced by §3.14 and the viewer.

## 9. Testing

26 tests, mocked HTTP, no network: allowlist refusal before any request;
page assembly across the cap with the boundary bar dropped; empty-page
termination; normalisation surviving a parquet round trip through
`loader.load_bars`; synthesised `instrument_id` driving `detect_roll_dates`
to the calendar roll date; credential errors that name the variable and
never the value; bounded 429 backoff; expiry and roll arithmetic; stitching;
the depth map reporting an empty contract as `None`; the filename marker.

## 10. Open until credentials exist

There is no sandbox, so nothing runs end to end until `.env` carries a real
key. Open items:

1. the auth request and response field names (§3);
2. the real minute-bar depth per contract — §3.14 measures the *CSV
   repository* at about three months; what the live API serves is a
   different question;
3. whether expired contracts are served at all at `unit=2`. If they are not,
   §5 has nothing to roll between and the deliverable is a current-contract
   fetcher;
4. the actual rate limits.
