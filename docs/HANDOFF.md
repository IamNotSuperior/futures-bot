# Handoff

Rewritten 2026-09-05. For a session starting fresh on this repository.

This file holds only what is **not** already in `CLAUDE.md` (the hard rules and
the firm/internal limit table), `README.md` (structure, how to run things, the
research bot), or `research/hypotheses.md` (what has been tested and rejected,
and why). Read those three first; this is the delta.

---

## 1. Current state

**Branch** `master`. **Working tree clean.** **587 tests pass** —
`venv\Scripts\python.exe -m pytest tests\ -q`, about 40 seconds.

**Nothing is pending.** No failing test, no half-finished refactor, no branch
other than `master`.

**Seven hypothesis entries. Six rejected, one open and untouched.**

| # | Idea | Status |
|---|---|---|
| 1 | Opening Range Breakout | REJECTED |
| 2 | Leveraged ETF end-of-day rebalance drift | REJECTED |
| 3 | Manual discretionary trading | **PROPOSED — still open, zero trades logged** |
| 4 | ORB-2, fixed bracket with a daily trend filter | REJECTED |
| 5 | ORB flat by 10:30 | REJECTED |
| 6 | London breakout of the overnight range | REJECTED |
| 7 | Replication of entry 6's non-null finding, on MNQ | REJECTED (not replicated) |

**No strategy has ever reached `paper` status in the registry.** Everything
downstream of that — the desk bot, execution, multi-account fan-out — is gated
on it and none of it is built. See §6.

### Files on disk that are not in git

`.gitignore` excludes these deliberately. They are **expensive or slow to
recreate** — do not delete them casually.

- `data/mes_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 40 MB, 2.58M bars,
  2019-05-05 to 2026-08-31.
- `data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 42.6 MB, 2.58M bars, same
  span. Pulled for entry 7 at **$9.4256**.
- `backtests/results/*` — scan CSVs, walk-forward outputs, trade streams,
  report text files, heatmaps, trade charts. Regenerable, but the ORB
  walk-forward takes ~55 minutes and the entry 6 run about 20.
- `.env` — `DATABENTO_API_KEY` and `DISCORD_TOKEN`. Ignored at `.gitignore:2`;
  `.env.example` is the tracked template.

`journal/trades.jsonl` **is tracked** but **does not exist yet** — no paper
trades have been logged. It appears on the first `pretrade.py` ALLOW.

**Databento spend is about $21.43** — roughly $12 on MES across three pulls, and
$9.43 on MNQ. `data/fetch.py` estimates first and refuses above a cap; the cap
is a `--max-cost` argument now, defaulting to $10.

---

## 2. What exists that is worth knowing about

**The strategy registry** (`strategies/registry.py` + `registry.yaml`) is the
machine-readable record of where every idea stands, and `Registry.promote` is
the only way a status changes. It takes **two arguments and no override** — a
test asserts that signature by introspection so a `force=` cannot be added
quietly. `rejected` is terminal, because the log's standing rule is that a
verdict is never revised. `testing -> paper` requires an ACCEPTED walk-forward
in `hypotheses.md`; `paper -> live` calls `journal/review.readiness`, which is
the same function the CLI prints from. `verify()` cross-checks the registry
against the log and `/status` prints the result.

**The research bot** (`bots/research.py`) is live and read-only. Launch it with
`start_bot.bat`. Commands, the registry, and the design are documented in
`README.md`. Nothing it exposes promotes a strategy, writes to the journal, or
places an order.

**The evidence pipeline**, in the order it is actually used:

> video or screenshots → a hypothesis entry in `research/hypotheses.md` with a
> stated mechanism, a named counterparty and kill criteria decided **before**
> any code → freeze it in a commit → build → `/walkforward` → verdict written
> into the entry before anyone sees numbers → verdict commit hash recorded in a
> one-line follow-up commit.

That loop is the product. Six of seven entries died in it, which is the point.

---

## 3. Open items

**3.1 Verify the CME holiday dates before trading them — still the highest
priority.** `data/cme_calendar.py` was constructed from standard US market
holiday rules, not read from an exchange feed. Check every date against
<https://www.cmegroup.com/tools-information/holiday-calendar.html>. A wrong date
fails in the dangerous direction: a half-day recorded as normal allows a late
entry that should be blocked. **2026-11-26 and 2026-11-27** are the first dates
a live paper run would reach.

**3.2 The 13:00 versus 13:15 early-close approximation.** `rules.EARLY_SESSION_CLOSE`
models a single 13:00 close where CME equity index closes 13:15 on some
half-days. Over-blocking is the safe error; fixing it loosens a limit and needs
stating as such.

**3.3 The `NEEDS_VERIFICATION` dates in late 2027** (2027-12-23/24/31). Extend
the calendar before 2028.

**3.4 `enforce_trailing_drawdown_halt` lives in `backtests/run_orb_flat.py`,
not the engine.** `engine.py` has `enforce_daily_loss_limit` and no trailing
equivalent. Rule 9 wants it as shared runtime logic; promote it when a second
caller needs it.

**3.5 Entry 3's gate has never been started.** Sixty rule-clean paper trades,
positive expectancy, `eval_sim` pass probability above 50%. Zero trades logged.

---

## 4. What the operator is doing next

Nothing is running. The research bot is available for querying the existing
work. **The manual paper-trading loop (entry 3) is the only path currently open
that could move a strategy toward `paper` status**, and it has not been started.

---

## 5. What a fresh session would get wrong

The rest of this file. Each of these has already cost time at least once.

### Imports: flat modules, no packages

There are **no `__init__.py` files and no package imports.** Modules are
imported by bare name — `import rules`, `import store`, `from engine import ...`
— and directories are put on `sys.path` by each entry point and by
`tests/conftest.py`.

`from strategies.rules import ...` **will fail.** Follow the existing pattern.

### There are two modules called `review`, and import order decides which you get

`backtests/review.py` and `journal/review.py` both exist, and `journal/review.py`
imports the backtests one **by bare name**. Whichever directory is earlier on
`sys.path` wins. `strategies/registry.py` needs the journal one and cannot get
it by name, so it loads it by file path under a distinct module name and builds
`sys.path` in reverse to keep `backtests` ahead of `journal`. If you add a third
consumer, do the same rather than reordering the path globally.

### Two limit sets, and the guards read only one

`strategies/rules.py` defines `FIRM` (Lucid's ceilings) and `INTERNAL` (what the
code enforces). **Nothing enforces against `FIRM`.** The one deliberate
exception is `eval_sim`, which uses the firm's $2,000 trailing line because it
asks "would this account have survived", not "would the guard have stopped us".

### Never restate a threshold

Every limit lives in `rules.py` and is imported. This has already caused a real
bug once (`enforce_daily_loss_limit` carried a hardcoded `limit=300.0` default).
The same discipline is why `journal/review.readiness` was extracted: the CLI,
the registry and the bot all read one definition of the 60-trade, positive-
expectancy and 50% gates.

### Hardcoded contract specs are the same bug wearing a different hat

`run_london.apply_daily_limit` hardcoded `MES`. Entry 7 ran MNQ through it and
marked positions at **$5.00 a point instead of $2.00**, inflating open P&L two
and a half times and firing 61 spurious daily-loss exits. Those exits removed
would-be *stops* from the bracket and pushed the statistic under test from
57.26% to 70.2% — **z = 4.69, "REPLICATED"** — before the bug was found.

**A silent unit mismatch that inflates exactly the number you are measuring, in
the favourable direction.** It was caught only because the exit-reason table
showed a `loss_limit_flatten` row that the MES column did not have. Any function
touching an instrument must take the spec as an argument.

### `a / (a + b)` is not a valid random-walk benchmark under a time limit

Entry 6 measured its target share against `a / (a + b)`, the driftless
probability of touching one barrier before the other. **That formula assumes an
unbounded horizon.** With a forced flatten, the farther barrier is systematically
under-reached, and the distortion scales with how far the target sits — it
manufactured an apparent twelve-point "shortfall" in entry 6's 2× arm that said
nothing about the signal.

Entry 7 replaced it with a **de-meaned bootstrap** (`backtests/bootstrap_benchmark.py`):
resample the trade's own 1-minute `(high, low, close)` changes with replacement
over the same horizon against the same barriers, drift removed. That correction
alone dropped entry 6's headline from **+5.59 points / z = 2.06** to
**+3.96 / z = 1.47** — below significance, before MNQ was consulted.

If you use a barrier-touch benchmark, state whether the holding period is long
enough for the closed form to apply. It usually is not.

### The flatten's sign is geometry, not market behaviour

A trade surviving to its forced flatten is **conditioned into the band between
its stop and its target**, so its expected value is that band's midpoint. This
has now been confirmed four times:

| Entry | Bracket | Band midpoint | Flatten mean |
|---|---|---|---|
| 4 | stop 10, target 18 | +4 pts | **+$39.91** |
| 5 | same, 45-minute hold | +4 pts, little dispersion | **+$2.64** |
| 6, 1× arm | stop = range + overshoot, target = range | negative | **−$21.72** |
| 7, MNQ | same | negative | **−$7.53** |

**Where a bracket is asymmetric, the unresolved population is not neutral.** In
entry 6 this was the whole result: the 1×/ON bracket alone netted −$481 across
341 resolved trades — the best breakout bracket this project has measured — and
345 flattens turned it into a −$7,702 loss. Do not read a positive flatten mean
as an edge, and do not design a bracket without asking what its midpoint is.

### Per-trade sizing does not fit the engine's scalar `contracts`

`engine.price_trades` takes one `contracts` integer for the whole trade list.
Entry 6 sizes per session from the range height, so it does not fit.

**The workaround, which needs no engine change:** P&L is exactly linear in
contracts *within* a trade — `net_pnl = contracts × (net_points × point_value −
2.5)` — so price at one contract and scale each row by its own size. The daily
loss limit is then applied **per contract-size group**, which is exact *only*
because entry 6 takes at most one trade a day, so no day mixes sizes and
grouping by size cannot split a day. A strategy taking two trades a day at
different sizes would break that and needs a real engine change.

Related: **the size-linearity assertion that guards entries 4 and 5 does not
apply to entry 6.** Those two can be rescaled after the fact; entry 6 cannot.

### `london_date` is not `rules.session_date`, and conflating them is silent

`rules.session_date` is the calendar date of a timestamp. **Entry 6's overnight
range window is not:** bars from 19:00 ET onward belong to the *next* day's
London session, so a Monday trade is measured against a range that began Sunday
at the Globex reopen.

`london_date()` in `strategies/london.py` is the only place that mapping lives,
and `tests/test_london.py` asserts both that it differs from `session_date` and
that moving the *prior evening's* bars moves the Monday range. The daily loss
limit still groups by calendar date, which is correct for the trade — only the
range construction crosses the boundary. Do not assume the two agree.

### rules.py has no concept of an overnight session

`is_entry_allowed` returns True at 03:00 ET — but because 03:00 is numerically
before a cutoff designed for the *afternoon*, not because the module models an
overnight session. It has one RTH session per calendar day. The guards are
satisfied by accident of arithmetic. A future overnight strategy trading later
in the day could expose that.

### Backtest numbers from before `4a14a83` are not comparable

That commit moved the daily loss limit $300 → $400 and the position cap 2 → 5.

### Every breakout entry is rejected — do not "improve" them

Entries 1, 4, 5, 6 and 7 have tested breakout continuation across two sessions,
two instruments, three signal definitions, four holding periods and two
benchmarks. **All rejected.** Each entry's `Next` section forbids the obvious
follow-up, and entry 7 closes the London family explicitly.

**Entry 1's condition has never been met and is the only route back:** establish
the counterparty claim independently of backtest results — order-flow evidence
about who takes the other side of a range break and under what constraint. That
is a data purchase, not a backtest, and should be priced before it is started.

### Do not propose order-placement or automation code yet

**Not until EITHER (a) the 60-trade paper-trading gate is met, for automating
discretionary trading, OR (b) a pre-registered hypothesis has passed
walk-forward validation with kill criteria stated in advance.** Neither holds.
The registry enforces this in code; `Registry.promote` will refuse.

The failure mode is offering to wire up a broker because the scaffolding exists
and looks ready. It is ready in the sense that the guards work; it is not ready
in the sense that there is nothing with demonstrated positive expectancy.

### Bar labelling: left-closed, labelled by opening minute

A 5-minute bar labelled `15:25` covers 15:25–15:29 and **closes at 15:29:59**.
So "the price as of 15:30" is the close of the `15:25` bar. A corollary that has
bitten test fixtures repeatedly: **to make a 5-minute candle close at a value,
set the *last* one-minute bar in its bucket**, not the bar at the label.

### Naive timestamps are read as ET, not UTC

`rules.to_et` interprets a naive timestamp as America/New_York. Passing a naive
**UTC** timestamp will be misread by five hours with no error.

### Signals are session-independent — exploit it, it is tested

Generating signals once over the whole history and slicing by date gives exactly
the same result as generating per window. `tests/test_scan.py` and
`tests/test_orb2.py` assert this. Regenerating per window is roughly four times
slower for no benefit.

### Long runs, and where the time actually goes

The ORB walk-forward is ~55 minutes; entry 6's four-arm run about 20; entry 7's
bootstrap about 15. Run them in the background.

**One performance trap, already hit:** `bootstrap_benchmark._by_day` pre-slices
the bars once. The first version scanned the whole 1.7M-row frame inside the
per-trade loop — O(trades × bars), about 1.2 billion row comparisons — and
turned a minute of work into an hour with no output to show for it.

### Data costs real money

`MES.v.0` and `MNQ.v.0` — volume-rolled continuous, never `.c.0`: the
front-month series has **no RTH bars at all on the 8 quarterly expiry days**.
Bare `MES`/`MNQ` does not resolve. **Always run `--estimate` and show the number
before pulling.** Metadata calls are free; timeseries calls are not.

### `venv` and pinned versions

Python 3.14.7, virtualenv at `venv/`. Invoke it explicitly:
`venv\Scripts\python.exe`. **`plotly` is pinned below 6** because vectorbt
registers themes using a trace type plotly 6 renamed. `vectorbt` is installed
but effectively unused — the engine is plain pandas, deliberately, because ORB
computes exact intrabar stop/target fills.

### Windows encoding and shell escaping will destroy files

`pathlib.Path.write_text(s)` defaults to **cp1252** and raises on any non-ASCII
character *after* truncating the file. **Always pass `encoding="utf-8"`.**

**Backslash escapes are worse.** Writing `bots\research.py` or
`strategies\registry.py` through a shell heredoc into Python turns `\r` into a
carriage return — silently producing `botsesearch.py`. This has happened three
times. **Write files with the editor tool, or write a script to a file and run
it; do not pass Windows paths through a shell heredoc into Python string
literals.**

### matplotlib parses `$` as math

Any label containing paired dollar signs gets silently italicised. Set
`plt.rcParams["text.parse_math"] = False` on any new chart with currency labels.

---

## 6. Next direction — the desk bot and multi-account fan-out

**Neither is built. Both are gated, and the gate is not met.**

The operator intends:

**A desk bot** — a 24/7 live scanner that posts tickets to Discord. Paper mode
first. Execution behind a **`BrokerAdapter` interface** with two
implementations: **`Paper`** (fills simulated locally, no money) and
**`TradersPost`** (real routing). The adapter boundary exists so the paper and
live paths cannot diverge and so a strategy cannot reach a broker without
passing through the same guards.

**Multi-account fan-out** — copying the same trade across several funded
accounts.

### The gate

**Both wait on a strategy reaching `paper` status in the registry. As of now
none has**, and `Registry.promote` will refuse to put one there without an
ACCEPTED walk-forward recorded in `hypotheses.md`. Six of the seven entries are
rejected and the seventh has no trades.

This is not a formality to route around. The scaffolding is genuinely ready —
the guards work, the registry gates, the bot runs — and that readiness is
exactly what makes it tempting to build execution for a strategy that has none
of the evidence.

### On multi-account fan-out, recorded in four entries and repeated here

**Copying identical trades across N funded accounts multiplies outcomes in both
directions. It is not diversification and must never be described as such.**
The same losing day draws down every account simultaneously, and a
trailing-drawdown breach terminates all of them on the same date. **N accounts
running one strategy is one bet at N times the size, with N times the fees** —
not N independent bets.

The only thing it diversifies is the *evaluation attempt*, and only while
accounts are started at different times on different price paths. Once funded
and trading in lockstep, the correlation is 1.

### What would actually unblock this

One of:

1. **Entry 3's gate met** — 60 rule-clean journaled paper trades, positive
   expectancy after costs, `eval_sim` pass probability above 50%. This is the
   only currently-open path and it has not been started.
2. **A new pre-registered hypothesis that passes walk-forward** — profitable in
   a majority of yearly folds, positive total P&L after commission and
   slippage, surviving at 2 ticks. Six entries have tried; none has.

Anything else is building execution for a strategy that does not exist.

---

## 7. Command reference

```powershell
venv\Scripts\python.exe -m pytest tests\ -q                    # 587 tests, ~40s
venv\Scripts\python.exe strategies\registry.py                 # where everything stands
.\start_bot.bat                                                # research bot, own window

venv\Scripts\python.exe backtests\run_orb.py                   # ORB baseline
venv\Scripts\python.exe backtests\run_eod.py                   # entry 2, ~75s
venv\Scripts\python.exe backtests\walkforward.py               # ORB, ~55 min
venv\Scripts\python.exe backtests\run_orb2.py                  # entry 4, --from-cache to re-report
venv\Scripts\python.exe backtests\run_orb_flat.py              # entry 5
venv\Scripts\python.exe backtests\run_london.py                # entry 6, 2 ticks base
venv\Scripts\python.exe backtests\run_entry7.py                # entry 7 replication

venv\Scripts\python.exe backtests\review.py <trades.csv>       # limits + drawdown
venv\Scripts\python.exe backtests\eval_sim.py <trades.csv>     # pass probability
venv\Scripts\python.exe data\validate.py data\<file>.parquet   # data quality
venv\Scripts\python.exe data\fetch.py --estimate --symbol MNQ.v.0 --start 2019-05-01 --end 2026-09-01
```

Journal, in the order they are used:

```powershell
venv\Scripts\python.exe journal\pretrade.py --instrument MES --direction long --entry 6800 --stop 6795 --contracts 1 --thesis "..."
venv\Scripts\python.exe journal\close.py --ticket <id> --exit 6812 --reason target
venv\Scripts\python.exe journal\review.py
```

`pretrade.py` accepts `--now` to override the clock and `--journal` to point at
a different file; both exist for testing and are safe to use.
