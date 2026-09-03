# Handoff

Written 2026-09-03. For a session starting fresh on this repository.

This file holds only what is **not** already in `CLAUDE.md` (the hard rules and
the firm/internal limit table), `README.md` (structure and the phased plan), or
`research/hypotheses.md` (what has been tested and rejected, and why). Read
those three first; this is the delta.

---

## 1. Current state

**Branch** `master`. **Working tree clean.** **365 tests pass** —
`venv\Scripts\python.exe -m pytest tests\ -q`, about 35 seconds.

**Nothing is pending.** No untracked files, no half-finished refactor, no
failing test, no branch other than `master`.

### Recent commits, newest first

The commit that last updated this file is not listed below — a file cannot
contain its own hash. `git log --oneline | head -3` gives the true tip.

| Hash | What |
|---|---|
| `ef1aee2` | Per-trade candlestick chart for visual verification |
| `5cb7a14` | Add `docs/HANDOFF.md` |
| `c6d9fd0` | Track the journal in git; add the CME holiday calendar |
| `f1996c6` | Add the manual-trading discipline layer (`journal/`) |
| `a9c6bb3` | Monte Carlo eval simulator; record ORB's pass probability |
| `c087f38` | Re-run both walk-forwards with trailing drawdown |
| `bb93072` | Track EOD trailing drawdown in the engine and `review.py` |
| `4a14a83` | Retarget limits to the Lucid 50K Pro eval, FIRM/INTERNAL split |
| `a1c8095` | Record hypothesis 2's verdict commit hash |
| `a788e2b` | Hypothesis 2 rejected — no tradeable EOD rebalance drift |
| `43aa4e1` | Pre-register hypothesis 2 (spec frozen here) |
| `885a67e` | Log ORB as rejected; require a hypothesis before any code |

### Files on disk that are not in git

`.gitignore` excludes these deliberately. They are **expensive or slow to
recreate** — do not delete them casually.

- `data/mes_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 40 MB, 2.58M bars,
  2019-05-05 to 2026-08-31. **This is the dataset everything uses.**
  Re-pulling it costs real money (see §5).
- `backtests/results/*` — scan CSVs, walk-forward outputs, heatmaps, trade
  charts. Regenerable, but the ORB walk-forward takes ~55 minutes.

The earlier two-year cache (`..._2024-09_2026-08.parquet`) was **deleted** once
its contents were merged into the file above. `data/extend.py` used to read it
as its merge target, so `EXISTING` now points at the merged file — re-running
`extend.py --pull` is idempotent, and `merge()` raises rather than writing a
file that would hold only the historical span under a name claiming the full
one.

`journal/trades.jsonl` **is tracked** (see §2) but **does not exist yet** — no
paper trades have been logged. It appears on the first `pretrade.py` ALLOW.

---

## 2. What was just finished

Two changes in `c6d9fd0`:

**The journal is tracked in git, and `close.py` commits it automatically.**
`journal/*.jsonl` was previously ignored. It is now tracked on purpose: each
`close.py` run stages and commits the journal file with a message naming the
ticket, instrument, exit reason and net P&L. An outcome edited afterwards then
shows up as a rewrite of a committed file rather than a silent change to an
untracked one. `--no-commit` opts out. A git failure is reported and swallowed,
never raised — the trade record is already on disk before the commit is
attempted and must never be lost to a git problem.

**`data/cme_calendar.py` supplies future holiday dates.** The cached bars end
2026-08-31, so `loader.detect_early_close_dates` cannot see any half-day after
that. The calendar covers 2026-09-01 to 2027-12-31. `pretrade.py` unions three
sources — the calendar, the half-days detected in the bar data, and the
`--early-close` flag — and now also blocks entirely on full closures
(Christmas, New Year's, Good Friday, July 4 observed), where no session exists.

The case that motivated it: a 12:55 entry on **2026-11-27** (the day after
Thanksgiving) is now blocked by the calendar with no flag. `test_cme_calendar.py`
asserts both that it blocks *and* that the same entry would have been allowed
without the calendar.

---

## 3. Open items

Three known gaps, in priority order. None is a bug; all are places where the
model is knowingly approximate.

### 3.1 Verify the holiday dates before trading them — highest priority

**`data/cme_calendar.py` was constructed from the standard US market holiday
rules, not read from an exchange feed.** Check every date against
<https://www.cmegroup.com/tools-information/holiday-calendar.html> before
relying on it.

A wrong date fails in the dangerous direction: a half-day recorded as a normal
day allows a late entry that should be blocked. This matters within weeks —
**2026-11-26 and 2026-11-27** are the first dates in the table that a live
paper-trading run would actually reach.

### 3.2 The 13:00 versus 13:15 early-close approximation

CME equity index closes at **13:15** ET on some half-days — the day after
Thanksgiving and Christmas Eve among them — and 13:00 on others. This was
visible directly in the bar data: those sessions have 225 RTH bars (missing
13:15 onward) rather than 210 (missing 13:00 onward).

`rules.EARLY_SESSION_CLOSE` models a single 13:00 close for all of them. That
pulls the entry cutoff back from 13:05 to 12:50 on 13:15 days, blocking trades
that would have been legal.

**This is deliberate and should not be "fixed" casually.** Over-blocking is the
safe error for a risk control, and the 12:50 cutoff is what makes the
2026-11-27 test case block. Making it exact means giving `rules.py` a per-date
close time and updating `flatten_deadline` / `entry_deadline` to take it —
worth doing eventually, but it loosens a limit, so it needs stating as such.

### 3.3 The `NEEDS_VERIFICATION` dates in late 2027

`cme_calendar.NEEDS_VERIFICATION` flags three dates whose observance rules are
genuinely ambiguous, all clustered at the end of 2027:

- **2027-12-23** — Christmas Eve treatment when Dec 25 falls on a Saturday
- **2027-12-24** — whether the exchange closes the Friday for a Saturday Christmas
- **2027-12-31** — New Year's Day 2028 falls on a Saturday; US equity markets
  do not usually close the preceding Friday, so this is modelled as a normal
  session

`coverage_warning()` surfaces these, and also warns when a date is past
`COVERAGE_END` (2027-12-31) so a stale table cannot answer confidently about a
day it knows nothing about. **Extend the calendar before 2028.**

---

## 4. What the operator is doing next

**Manual paper trading on TradingView**, gated by `journal/pretrade.py`. Not
automated execution — the human places every trade by hand; this repo enforces
the rules and keeps the record.

The loop, per trade:

```bash
python journal/pretrade.py --instrument MES --direction long \
    --entry 6800.25 --stop 6795.00 --contracts 2 \
    --thesis "failed breakdown, reclaimed VWAP"
```
```bash
python journal/close.py --ticket <id> --exit 6812.50 --reason target
```
```bash
python journal/review.py
```

**No trade is placed without a logged ticket.** A BLOCK writes nothing, so a
blocked idea leaves no ticket to close.

### The gate, pre-registered in `research/hypotheses.md` entry 3

Before a **$115 Lucid evaluation** is purchased, all three must hold:

1. **60+ rule-clean paper trades**
2. **Positive expectancy per trade** after commission and 1 tick slippage/side
3. **`eval_sim` pass probability above 50%**

**Kill criterion: any rule violation resets the clean-trade count to zero.**
Not a deduction — a reset. Detected violations are a late entry, a hold under
30 seconds, size above 5 contracts, and a day breaching the $400 internal daily
loss limit. `journal/review.py` computes all four and prints a READY / NOT
READY verdict; it is not a manual check.

Entry 3 also records, before any trades, that **no counterparty has been
identified** for discretionary trading, and that passing the gate licenses one
$115 attempt rather than establishing an edge. A fresh session should not
soften that framing.

---

## 5. What a fresh session would get wrong

The rest of this file. Each of these has already cost time at least once.

### Imports: flat modules, no packages

There are **no `__init__.py` files and no package imports.** Modules are
imported by bare name — `import rules`, `import store`, `from engine import
...` — and the directories are put on `sys.path` by each entry point and by
`tests/conftest.py` (which adds `data`, `strategies`, `backtests`, `journal`).

`from strategies.rules import ...` **will fail.** Follow the existing pattern.

### Two limit sets, and the guards read only one

`strategies/rules.py` defines `FIRM` (Lucid's ceilings — what ends the account)
and `INTERNAL` (what the code enforces). **Nothing enforces against `FIRM`.**
Module-level aliases (`DAILY_LOSS_LIMIT`, `POSITION_CAP`, `TRAILING_DD_STOP`,
`WORST_DAY_FLAG_PCT`) all point at `INTERNAL`, and `tests/test_rules.py`
asserts each internal value is strictly tighter than its firm counterpart.

The one exception is deliberate: **`eval_sim` uses the firm's $2,000 trailing
line**, because it asks "would this account have survived", not "would the
guard have stopped us".

### Never restate a threshold

Every limit lives in `rules.py` and is imported. This is not style — it has
already caused a real bug: `enforce_daily_loss_limit` carried a hardcoded
`limit=300.0` default, so callers passing `rules.DAILY_LOSS_LIMIT` got $400
while callers taking the default silently kept $300. Defaults now resolve to
`None` and fall back to the rules module. Do the same for anything new.

### Backtest numbers from before `4a14a83` are not comparable

That commit moved the daily loss limit $300 → $400 and the position cap 2 → 5.
Any figure quoted in a commit message or an earlier hypothesis entry from
before it was produced under the old limits. The hypothesis entries note this
where it matters.

### Both strategies are rejected — do not "improve" them

`research/hypotheses.md` records ORB (entry 1) and LETF end-of-day rebalance
drift (entry 2) as **REJECTED**, with walk-forward evidence. `CLAUDE.md` rule
12 requires a hypothesis entry, written before any code, for anything new.

The tempting move — "ORB nearly worked, try a wider grid / different exits" —
is explicitly forbidden by entry 1's *Next* section, and entry 2's says the
same about a different entry time or horizon. That search is how a clean
negative becomes a fitted positive.

### Do not propose order-placement or automation code yet

**Do not propose order-placement or automation code until EITHER (a) the
60-trade paper-trading gate is met, for automating discretionary trading, OR
(b) a pre-registered hypothesis has passed walk-forward validation with kill
criteria stated in advance, for automating a systematic strategy.**

Neither condition currently holds. The journal has no trades yet, and both
hypotheses on the books are rejected. `CLAUDE.md`'s scope note says the same
thing more narrowly; this is the operative version.

The failure mode is offering to "wire it up to Tradovate" because the
scaffolding exists and looks ready. It is ready in the sense that the guards
work; it is not ready in the sense that there is nothing with demonstrated
positive expectancy to automate.

### Bar labelling: left-closed, labelled by opening minute

A 5-minute bar labelled `15:25` covers 15:25–15:29 and **closes at 15:29:59**.
So "the price as of 15:30" is the close of the `15:25` bar, not the `15:30`
bar. This distinction is the entire timing spec of hypothesis 2 and is easy to
get backwards.

### Naive timestamps are read as ET, not UTC

`rules.to_et` interprets a naive timestamp as America/New_York, matching the
bar index from `loader.load_bars`. Passing a naive **UTC** timestamp will be
misread by five hours with no error. Localise before calling.

Working in ET wall-clock terms is what makes the rules DST-correct: 16:20 ET is
21:20 UTC in winter and 20:20 UTC in summer, and the tz database resolves that.

### Signals are session-independent — exploit it, it is tested

Each session's ORB logic uses only bars within that session. So generating
signals once over the whole history and slicing by date gives **exactly** the
same result as generating per window. `tests/test_scan.py` asserts this
directly. The scans rely on it; regenerating per window is roughly four times
slower for no benefit.

### The ORB walk-forward takes ~55 minutes

192 combinations × 7 folds over 7.3 years. Run it in the background and do
something else. The EOD-rebalance walk-forward is 12 combinations and takes
~75 seconds.

If only the stitched out-of-sample stream is needed, rebuild it from the saved
fold summary (`backtests/results/orb_walkforward_slip1.csv`) via
`walkforward.oos_trade_stream` — 7 signal generations, about two minutes.

### Data costs real money

Databento spend so far is about **$12** — $2.56 (MES.c.0, later discarded),
$2.58 (MES.v.0 two years), $6.84 (MES.v.0 back to 2019 launch), plus a
half-cent probe. `data/fetch.py` and `data/extend.py` both estimate first and
refuse above a cap. **Always run `--estimate` and show the number before
pulling.** Metadata calls are free; timeseries calls are not.

Use `MES.v.0` (volume-rolled continuous), never `MES.c.0`: the front-month
series has **no RTH bars at all on the 8 quarterly expiry days**, because the
expiring contract stops at the 09:30 open. Bare `MES` does not resolve.

### `venv` and pinned versions

Python 3.14.7, virtualenv at `venv/`. Invoke it explicitly:
`venv\Scripts\python.exe`. **`plotly` is pinned below 6** because vectorbt
registers themes using the `scattermapbox` trace type that plotly 6 renamed;
unpinning it breaks `import vectorbt`.

`vectorbt` is installed but effectively unused. The backtest engine is plain
pandas, deliberately: ORB computes exact intrabar stop/target fills, and
`Portfolio.from_signals` would re-derive fills from the price series instead of
using them. Do not "modernise" the engine onto vectorbt without re-reading that
reasoning in `backtests/engine.py`.

### Windows encoding will destroy files

`pathlib.Path.write_text(s)` defaults to **cp1252** on this machine and raises
on any non-ASCII character — *after* truncating the file. This destroyed
`CLAUDE.md` once (recovered from git). **Always pass `encoding="utf-8"`** when
writing, and `encoding="utf-8"` when reading files containing en-dashes or
symbols.

Related: the Bash tool here is Git Bash. Heredocs containing nested quotes and
apostrophes break in ways that are tedious to debug — write a patch script to a
file and run it instead.

### matplotlib parses `$` as math

Any label containing paired dollar signs gets silently italicised as mathtext.
`backtests/plot_trade.py` sets `plt.rcParams["text.parse_math"] = False`. Any
new chart with currency labels needs the same.

---

## 6. Next direction — ORB-2

The operator intends to build an **ORB-based bot targeting the Lucid 50K Pro
evaluation**. Stated shape:

- Active from **09:30 ET**
- **One trade per day maximum**
- Parameters selected for **eval pass probability** (`backtests/eval_sim.py`)
  rather than Sharpe

**No code for ORB-2 exists.** The hypothesis entry must be written and frozen
in `research/hypotheses.md` before any is written — `CLAUDE.md` rule 12.

### This is a new hypothesis, not a re-test of entry 1

Entry 1's prohibition stands: ORB as specified there is rejected, and its grid
must not be re-run wider or with different exits. ORB-2 is a **separate entry
with its own mechanism statement, counterparty claim, grid and kill criteria**,
pre-registered before code the same way entry 2 was. It does not inherit entry
1's verdict and it does not get to reuse entry 1's search.

The distinction is worth holding onto, because "new hypothesis" is also exactly
what a re-test would call itself. What makes ORB-2 legitimate is that its
objective genuinely differs — see below — not that it is ORB with fresh
paperwork. If the entry ends up proposing the same signal, the same grid and
the same objective, it is a re-test and entry 1 forbids it.

### Three things the entry will have to confront

**One trade per day is a real change to the code, not a parameter.**
`strategies/orb.py` currently allows one trade *per direction* per session, so
up to two. ORB-2's constraint is stricter and needs implementing and testing.

**Optimising for pass probability is still optimising.** Switching the
objective from Sharpe to `eval_sim` pass probability does not escape
overfitting — it changes what is overfitted to. The walk-forward discipline
applies unchanged: select on prior folds only, score each test year once. Note
also that `eval_sim` draws days i.i.d., so it under-represents clustered losing
streaks and its pass probability is **optimistic**; selecting on an optimistic
metric compounds that.

**The baseline is discouraging and should be stated in the entry.** ORB at the
default parameters returned a 7.67% pass probability over 20,000 paths, with a
57.6% blow-up rate and about 13 attempts (~$1,499) expected per pass. A
different objective function does not change the underlying distribution — mean
day −$1.50 against a $120.50 standard deviation. ORB-2's entry should say what
it expects to change about that, in advance.

### Longer-term intent: copying trades across multiple funded accounts

The operator intends eventually to copy the same trade across several funded
accounts.

**This multiplies outcomes in both directions. It is not risk reduction and
must not be described as diversification.** Identical trades on N accounts are
perfectly correlated: the same losing day draws down every account
simultaneously, and a trailing-drawdown breach terminates all of them on the
same date. N accounts running one strategy is one bet at N times the size, with
N times the fees — not N independent bets.

The only thing it diversifies is the *evaluation attempt*, and only when
accounts are started at different times on different price paths. Once they are
funded and trading in lockstep, the correlation is 1.

---

## 7. Command reference

```bash
venv\Scripts\python.exe -m pytest tests\ -q                    # 365 tests, ~35s
venv\Scripts\python.exe backtests\run_orb.py                   # ORB baseline
venv\Scripts\python.exe backtests\run_eod.py                   # hypothesis 2, ~75s
venv\Scripts\python.exe backtests\walkforward.py               # ORB, ~55 min
venv\Scripts\python.exe backtests\review.py <trades.csv>       # limits + drawdown
venv\Scripts\python.exe backtests\eval_sim.py <trades.csv>     # pass probability
venv\Scripts\python.exe backtests\plot_trade.py --date 2026-08-25
venv\Scripts\python.exe data\validate.py data\<file>.parquet   # data quality
venv\Scripts\python.exe data\fetch.py --estimate               # cost, no spend
```

Journal, in the order they are used:

```bash
venv\Scripts\python.exe journal\pretrade.py --instrument MES --direction long --entry 6800 --stop 6795 --contracts 1 --thesis "..."
venv\Scripts\python.exe journal\close.py --ticket <id> --exit 6812 --reason target
venv\Scripts\python.exe journal\review.py
```

`pretrade.py` accepts `--now` to override the clock and `--journal` to point at
a different file; both exist for testing and are safe to use.
