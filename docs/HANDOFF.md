# Handoff

Rewritten 2026-09-09. For a session starting fresh on this repository.

This file holds only what is **not** already in `CLAUDE.md` (the hard rules and
the firm/internal limit table), `README.md` (structure, how to run things, the
bots), or `research/hypotheses.md` (what has been tested and rejected, and
why). Read those three first; this is the delta.

---

## 1. Current state

**Branch** `master`, tracking `origin/master` at
<https://github.com/IamNotSuperior/futures-bot> (private). **Nothing unpushed.**
**1,111 tests pass**, 2 skipped (a symlink test the OS refuses, and one
that needs an orphaned generated module on disk, of which there is none) —

```powershell
venv\Scripts\python.exe -m pytest tests\ -q
```

about two minutes. Both generated strategies and their generated tests are
tracked and pass, so the plain command is the whole contract now; the
`/submit` pipeline runs a subset of it — the base suite plus the submitting
strategy's own generated test (§5).

**Working tree is clean.**

**Ten hypothesis entries. Nine rejected, one open and untouched (entry 3).**
No strategy has ever reached `paper`.

| # | Idea | Status |
|---|---|---|
| 1 | Opening Range Breakout | REJECTED |
| 2 | Leveraged ETF end-of-day rebalance drift | REJECTED |
| 3 | Manual discretionary trading | **PROPOSED — open, zero trades logged** |
| 4 | ORB-2, fixed bracket with a daily trend filter | REJECTED |
| 5 | ORB flat by 10:30 | REJECTED |
| 6 | London breakout of the overnight range | REJECTED |
| 7 | Replication of entry 6's non-null finding, on MNQ | REJECTED (not replicated) |
| 8 | `orborb_flat_1030` — entry 5 re-submitted through `/submit` | REJECTED (pipeline validation, see §2) |
| 9 | `orb_full_day_test` — entry 4's OFF arm re-submitted through `/submit` | REJECTED (pipeline validation, see §2) |
| 10 | `london_full_day` — entry 6's 1×/ON held to 15:55, MES and MNQ | REJECTED (both instruments; verdict `82006bf`, see §2) |

`venv\Scripts\python.exe strategies\registry.py` prints the registry; it agrees
with the log.

### Files on disk that are not in git

`.gitignore` excludes these deliberately. The first two are **expensive to
recreate** — do not delete them casually.

- `data/mes_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 40.3 MB, 2.58M bars,
  2019-05-05 to 2026-08-31.
- `data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 44.6 MB, same span. Pulled
  for entry 7 at **$9.4256**.
- `backtests/results/*` — 70 files: scan CSVs, walk-forward fold tables, trade
  streams, reports, charts. Regenerable; the ORB walk-forward takes ~55 minutes,
  entry 6 about 20. The London and entry 7 fold tables rebuild in seconds with
  `--folds-only`.
- `.env` — five keys: `DATABENTO_API_KEY`, `DISCORD_TOKEN`,
  `DESK_WEBHOOK_TOKEN`, `DESK_OWNER_ID`, `ANTHROPIC_API_KEY`. All set. Ignored
  at `.gitignore:2`; `.env.example` documents each.
- `journal/desk_state.json`, `journal/tunnel.log`, `journal/live_bars/*.csv` —
  desk runtime. `live_bars/` holds one file per session date the desk has seen;
  `/read` reads only today's, so a stale file is inert.

`journal/trades.jsonl` **is tracked and does not exist** — no manual paper
trade has ever been logged, by CLI or by `/read`. `journal/decisions.jsonl`
(desk ticket button presses) does not exist either. Both appear on first use.

**Databento spend is about $21.43.** `data/fetch.py` estimates first and refuses
above `--max-cost`, default $10.

---

## 2. What exists that is worth knowing about

### The `/submit` pipeline — validated on one known verdict, one pending

Strategies now arrive through Discord. `bots/submissions.py` is the pipeline;
`bots/generate.py` calls the Claude API (`claude-opus-5`, streaming, adaptive
thinking); `bots/sandbox.py` is what generated code may write and contain;
`bots/submit_view.py` is the modal and approval buttons;
`backtests/run_generated.py` is the walk-forward that actually runs.

The ordering is the product, and must not be rearranged:

1. **Pre-register first.** The operator's mechanism, counterparty and kill
   criteria go into `hypotheses.md` as PROPOSED, in their own words, and are
   **committed before the API is called** (rule 12). The model's draft is
   appended under its own heading, labelled as not the pre-registration.
2. Generate → sandbox-write → suite → register at `proposed` (`Registry.add`,
   which has **no status argument** — a test asserts the signature, as for
   `promote`).
3. Approve (owner only) → `testing`, implementation frozen in a commit.
   Reject → REJECTED, terminal, entry stays.
4. `/walkforward` runs the folds and **writes and commits the verdict before
   the embed posts** — `record_verdict` returns the hash the embed prints. A
   REJECTED verdict promotes the record to `rejected`; an ACCEPTED one
   deliberately does *not* promote to `paper`, because `paper` unblocks the
   desk and reaching it should be an act, not a background job's side effect.

**Retry.** A name whose attempt died before review is re-submitted against the
same entry with a dated `### Attempt N` note; the pre-registration is never
rewritten. A name that reached the registry is taken.

**Where it stands: validated on two known verdicts.** Entry 8 re-submitted
entry 5's description as a plumbing test and **reproduced entry 5's signal to
within 2 trades in seven years** on the corrected span — 476 against entry 5's
478, both REJECTED, 0 of 7 folds either way. Entry 8's diagnostic and addendum
carry the full accounting. Entry 9 re-submitted entry 4's filter-OFF arm; its
third attempt passed the suite, was approved, and landed within about 6% of
entry 4 at 2 ticks — **−$8,450 over 503 trades against entry 4's −$8,970 over
504**, 2 of 7 folds against 3, 8 evaluations blown against 9. REJECTED. The
pipeline can be trusted on a new idea.

**Since 2026-09-11 a generated verdict is scored under both internal guards**
— the daily loss limit and the $1,500 trailing halt (§3.5) — and reports the
same signals with the halt off alongside, as the comparable basis, because a
halt that fires in year one leaves the fold test with nothing to count. Rule
13 is decided on the guarded stream. Entries 8 and 9 carry dated addenda with
both bases. Under the halt entry 9 (4 contracts) stops trading on 2020-05-29
after 27 trades, as entry 5 did in 2020; entry 8 (1 contract) takes until
2022-11-17 and 214 trades to lose the same $1,500. Neither resumes.

### Entry 10 — REJECTED on both instruments, 2026-09-11 (verdict `82006bf`)

Entry 6's 1×/ON London breakout held to **15:55 ET** (12:55 on early-close
days) instead of 09:25, on **both MES and MNQ**, with the log's corrected
realised-stop sizing. It overrode entry 7's closure of the London family by
operator direction and said so; it was informed by entry 6's post-hoc
exit-reason finding and said that too, which is why its bar sat above the
effect that motivated it: pooled pass probability ≥ 35%, z ≥ 2.5 against the
bootstrap benchmark recomputed for the 15:55 horizon, departure ≥ +2 points,
plus rule 13, at 2 ticks, every line on both instruments. Frozen at `fd0c5c8`
before any code; reproduction of entries 6 and 7 trade for trade at `3ff81fe`;
verdict written into the entry and committed before anyone read it.

**Result.** MES failed all four criteria (0 of 7 folds, −$1,298; 1.29% pass;
z +0.35; +1.89 points). MNQ passed the pass probability (50.82%) and the
departure (+5.47) and failed rule 13 (1 of 7, +$1,239) and z (+1.16). Three
findings, all in the entry's 2026-09-11 addendum:

- **The exit was not hiding an edge.** Like-for-like on the unhalted streams,
  the extended hold is worth $2,129 on MES and $165 on MNQ, and the trades the
  US session resolved did so at the driftless rate — 52.5% on MES, 51.5% on
  MNQ — pulling the pooled target share under break-even. Prediction 2 held.
- **The trailing halt left the z test almost no sample.** It ended both guarded
  streams in year two (552 and 282 sessions blocked), leaving 87 and 112
  resolved trades where the entry's power arithmetic assumed ~650 and ~420, so
  z = 2.5 needed +13 points. A property of the pre-registration, not a reason
  to revise anything; the next entry that combines a halt with a share test
  must say which stream the test runs on.
- **On MNQ the corrected sizing rule cost $1,755 against entry 6's**, because
  MNQ's overshoot is four times MES's and the rule prices it.

**The London family is closed again, on the exit as well as the entry.** Entry
1's condition — order-flow evidence about the counterparty — remains the only
route back for any breakout idea.

**Code that exists.** `strategies/london.py` gained three parameters whose
defaults are entry 6's, so entries 6 and 7 are untouched: `flatten_time` labels
its exit by time (`flatten_1555`), `early_close_flatten_time` (rule 2's
deadline is the backstop when unset), and `size_on="stop"`.
`backtests/run_entry10.py --reproduce` is the reproduction check;
`--run` is the 15:55 test (both guards as the two engine functions
`apply_internal_guards` composes, applied per size group; halt-OFF alongside;
1 and 3 ticks as sensitivities; the bootstrap at the 15:55 horizon with
early-close days truncated at 12:55; the verdict block to
`backtests/results/entry10_verdict.md`). Streams and folds are in
`backtests/results/entry10_*`; the bot's `london_full_day` runner reads the
MES base case.

**The sandbox is two layers.** `safe_write` confines writes to
`strategies/generated/` and `tests/generated/`, resolving before comparing so
`..`, absolute paths and symlinks are refused. That does not contain the code,
which pytest imports and executes — so `screen_source` AST-screens imports
(only pandas/numpy/base/rules/pytest and stdlib maths; a generated *test* may
additionally import its own strategy, bare or dotted, and no other),
`eval`/`exec`/`open`/`__import__` and dunder reflection, and `run_tests`
executes in a subprocess with every secret stripped from the environment. The
screen guards against mistakes and drift; **it is not a security boundary
against an adversary.** The scrubbed environment is what makes a miss
survivable.

### The desk bot — shadow mode, and its existence is not evidence the gate was met

`bots/desk.py` was built 2026-09-06 as a deliberate pre-gate exception to test
plumbing. The only strategy it runs is **`orb2`, which entry 4 REJECTED** — a
shadow ticket must never be mistakable for a result. `shadow:` in
`registry.yaml` is a list of names, not a status; `Registry.promote` never
reads it. Tickets go to `journal/shadow_trades.jsonl`, never `trades.jsonl`, so
entry 3's count is untouched. `PaperAdapter` is the only adapter — a test
asserts `BrokerAdapter.__subclasses__()` has exactly one member — and
`broker.require_live_eligible` checks registry status before the account flag,
so a flag flipped by accident still refuses.

The desk reuses rather than reimplements: `tickets.evaluate_signal` calls
`pretrade.evaluate` and `tests/test_desk_rules.py` asserts the decisions match
check-for-check. Ticket embeds carry **Execute / Don't trade** buttons, owner
only; presses go to `journal/decisions.jsonl` as the operator's record and are
**reported by `review.py`, not counted toward entry 3** — a click on a
rejected strategy's signal is not a discretionary trade. Execute is disabled
and refused for anything below `paper`/`live`.

Discord is required (`DESK_CHANNEL_NAME`, default `general`); a missing
channel or token is fatal at startup, never a silent stdout fallback. The feed
server starts from `setup_hook`, exactly once, whatever Discord's reconnects
do. Bars arrive from a TradingView alert (`pine/bar_feed.pine`) through a
Cloudflare tunnel (`bots/tunnel.py`), which refuses to start without
`DESK_WEBHOOK_TOKEN`. The desk mirrors each bar to `journal/live_bars/` for
`/read`, which runs in the other process.

### The research bot

`bots/research.py`: `/backtest`, `/walkforward`, `/evalsim`, `/hypotheses`,
`/status`, `/read`, `/submit`. `/read` describes a chart with no bias label,
score or opinion (by test), and its **Log long / Log short** buttons write to
the manual journal through `pretrade.evaluate` — those **do** count toward
entry 3, because the operator picks the direction and writes the thesis, which
is exactly the mechanism entry 3 pre-registered. Guards re-run at submit time.
`/walkforward` runs the folds for a generated strategy and reads saved output
for everything else, because those verdicts are frozen.

### The registry and the log

`Registry.promote` takes two arguments and no override; `rejected` is
terminal; `testing → paper` requires an ACCEPTED walk-forward in the log;
`paper → live` calls `journal/review.readiness`. `verify()` cross-checks the
registry against the log and is how a real drift bug was caught this session.

**The evidence pipeline** is unchanged and is the product: entry with
mechanism, counterparty and kill criteria → committed → build → walk-forward →
verdict frozen before anyone sees numbers → hash recorded in a follow-up. Seven
of nine entries died in it.

---

## 3. Open items

**3.1 Verify the CME holiday dates before trading them — still the highest
priority.** `data/cme_calendar.py` was constructed from standard US holiday
rules, not an exchange feed. Check every date against
<https://www.cmegroup.com/tools-information/holiday-calendar.html>. A wrong
date fails in the dangerous direction: a half-day recorded as normal permits a
late entry that should be blocked. **2026-11-26 and 2026-11-27** are the first
dates a live paper run reaches.

**3.2 Closed, 2026-09-11: MES commission on a 50K LucidPro evaluation is
$0.50 a side, $1.00 a round turn**, confirmed by Lucid support
(support.lucidtrading.com article 11508978), not the $1.25 every verdict
assumed. `rules.COMMISSION_PER_SIDE` carries it with the source beside it;
`rules.ASSUMED_COMMISSION_PER_SIDE` keeps the $1.25 by name so any earlier
run reproduces with `--commission 1.25`. Entries 1, 2, 4, 6 and 7 were
re-priced from their saved streams (`backtests/reprice.py`, exact; parameter
selection in 1 and 2 held at what $1.25 chose) and entries 5, 8 and 9 re-run
in full with both guards re-decided. Each carries a dated addendum.

**No verdict changes. One kill criterion would have resolved differently:**
entry 2's median fold Sharpe, +0.201 → +0.523 against its 0.30 line — and
entry 2 still dies on its pre-cost mechanism test, which is the one that
matters. The closest any entry now sits to a line is entry 4's pooled pass
probability, 16.11% → 22.70% against 25% at 1 tick (12.55% at the 2-tick
base case). Entry 6's 1×/ON bracket alone turns from −$481 to +$896 while
the arm still loses $5,387 on its flattens. MNQ (entry 7) was re-priced at
the same $0.50 on the assumption Lucid's micro rate is common to MES and MNQ;
**only MES was confirmed.** Break-even figures in the entry 4 and 5 runners
are now computed from the cost model (38.21% at 1 tick, 40.00% at 2); the
frozen 2-tick reports had printed the 1-tick constant.

**3.3 The 13:00 versus 13:15 early-close approximation.**
`rules.EARLY_SESSION_CLOSE` models a single 13:00 close where CME equity index
closes 13:15 on some half-days. Over-blocking is the safe error; fixing it
loosens a limit and must be stated as such.

**3.4 The `NEEDS_VERIFICATION` dates in late 2027.** Extend the calendar
before 2028.

**3.5 Closed, 2026-09-11.** `enforce_trailing_drawdown_halt` now lives in
`backtests/engine.py` behind `engine.apply_internal_guards` — the daily loss
limit, then the trailing halt on the loss-limited stream — which both
`run_orb_flat.py` and `run_generated.py` call. `tests/test_guard_parity.py`
holds them to it: a synthetic stream that trips the halt comes out identical
through either runner, and neither may define a halt of its own. Entry 5's
reports came out **byte-identical** before and after the move at 1 and 2
ticks. Generated verdicts now score on the guarded stream with the halt-OFF
stream alongside (§2).

**3.6 Closed.** Entry 9's third attempt passed its suite, was approved, and
was REJECTED by `/walkforward` on 2026-09-09 (verdict commit `29178fd`).
Nothing left to retry; see §2 for what it validated.

**3.7 Entry 3's gate has never been started.** Sixty rule-clean paper trades,
positive expectancy, `eval_sim` pass probability above 50%. Zero trades logged,
and `/read`'s buttons now make logging one click — which makes the discipline
of only logging trades actually taken matter more, not less.

**3.8 `rules.py` has no session-open guard.** A signal at 08:00 passes every
check; nothing today can produce one;
`test_premarket_signal_is_not_blocked_by_rules_py` pins it.

**3.9 The Discord user id in commit `9dc95c2`.** `tests/test_submissions.py`
once hard-coded the operator's real `DESK_OWNER_ID`; the tip uses a fake id.
The real one remains in that commit's history, which was already pushed. A
Discord snowflake is not a credential — it authorises nothing and is visible
to anyone sharing a server. **It stays; do not rewrite history for it.**

**3.10 Verified, 2026-09-11: API and automated order placement is permitted
on Lucid evaluation accounts**, via Rithmic or Tradovate, under Lucid's Other
Trading Activities policy (support.lucidtrading.com article 11404728). This
was the account-side question that gated the execution layer — whether an
automated desk would be allowed at all — and it closes in the permissive
direction. **Nothing else about the gate moves:** no execution code exists or
is to be written until a strategy reaches `paper` (§5, §6), and
`Registry.promote` still refuses without an ACCEPTED walk-forward in the log.

**3.11 Slippage is now the dominant cost term, and it is an assumption.** At
the confirmed $0.50 a side, commission is $1.00 of a round turn. One tick a
side of slippage is $2.50 on MES — 71% of the $3.50 total — and rule 13's
two-tick survival bar is $5.00 of $6.00, 83%. Every verdict's margin now
rests mainly on a number nobody has measured. It can only be measured on a
live account, fill by fill against the signal level; TradingView paper fills
are optimistic and the journal cannot see it. Until then the two-tick bar is
the hedge, not a measurement.

---

## 4. Open directions

None is started. In the order the operator raised them:

**A portfolio layer in the desk.** Today each runner is its own book: one
position at a time per strategy, the daily budget checked per signal. A
portfolio layer would hold **one position across strategies**, draw on a
**shared daily budget**, and check **consistency at the account level** (rule
8) rather than per stream. This is real engine work — `engine.price_trades`
takes one scalar `contracts` per trade list (§5) — and it changes what a
"blocked" signal means, so it needs a design before code.

**A selector diagnostic on the seven rejected OOS streams.** Seven
out-of-sample trade streams exist in `backtests/results/`. A diagnostic that
asks whether *any* selection rule across them — by regime, by day, by
volatility — would have produced a positive stream is worth one run,
**pre-registered as a diagnostic with no verdict**, because a selector fitted
to seven known-negative streams is the textbook way to manufacture an edge.

**The commission number is settled (§3.2): $0.50 a side.** What remains an
assumption is slippage, now the larger part of every round turn and
measurable only on a live account (§3.11). A portfolio of near-break-even
strategies is still not worth assembling on an unmeasured slippage figure.

**The desk bot's gate (§6) is unchanged** — no strategy is at `paper`, and
nothing above relaxes that.

---

## 5. What a fresh session would get wrong

Each of these has already cost time at least once. The first five are new this
session.

### `truststore` recurses forever on Python 3.14 / Windows — and the fix must never become `verify=False`

`anthropic` 1.x installs `truststore` to read the Windows certificate store.
On this machine its `_set_ssl_context_verify_mode` recurses into `ssl.py`'s
`verify_mode` property without terminating, so **every SDK request dies as
`APIConnectionError`** while the same request through `urllib` returns 200.
`httpx2` alone fails the same way; that is how it was isolated.

`generate._http_client()` supplies a certifi-backed `ssl.create_default_context`
— `CERT_REQUIRED`, hostname checking on — so the handshake never reaches
truststore. **Verification is not weakened; a CA bundle is added, not a check
removed.** "Fix the TLS error" is exactly the change that tends to arrive as
`verify=False`, so `test_tls_verification_is_not_weakened` walks the
function's AST and fails on a `False` `verify`/`check_hostname` keyword or a
`CERT_NONE` reference — AST rather than substring, because the docstring
deliberately names the thing it forbids.

### `run_tests` runs the base suite plus the submitting strategy's own test — nothing else

`tests/generated/` holds every submission's generated test. Running `tests/`
wholesale meant one failed attempt's leftover file would fail **every later
submission**, and did fail the operator's plain `pytest`. `test_command`
passes `--ignore=tests/generated` and adds the submitting strategy's own file
back explicitly. Judge a submission on the base suite and its own test. When
you run the suite yourself, use the same contract (§1) or expect entry 9's
untracked test to be red.

### The bot commits the *whole* of `hypotheses.md` — never leave it dirty while the bot runs

Every bot write to the log ends in `git add` of the file, so an uncommitted
edit already in the tree rides into the bot's next commit under a message
that describes something else. That happened once: a diagnostic on entry 8
was committed under "Pre-register entry 9". Not data loss — misattribution, in
a file whose value is that its history means what it says.
`require_clean_log()` now runs before `pre_register`, `approve`, `reject` and
`record_verdict` and refuses with the reason, rendered in Discord as a message.
**If you edit `hypotheses.md` by hand, commit before using `/submit`.**
`registry.yaml` is deliberately *not* guarded: `register_proposed` leaves the
bot's own registry change uncommitted until `approve`, and a guard there would
refuse the bot's own work.

### The scored span is imported from `walkforward.py`, not restated

`run_generated.SCORE_START` is `walkforward.build_folds()[0].test_start`
(2020-01-01). Signals are generated over the **full** history so a 50-day EMA
is seeded by January 2020; the scored trades start at that date. A test
asserts the constant equals walkforward's and that no literal `date(` restates
it. This is the same discipline as "never restate a threshold", applied to a
date.

### The 110-trade gap was the span, not the strategy

Entry 8's first run reported 588 trades against entry 5's 478 and looked like
a rule interpreted differently. It was not. On identical bars the two
strategies differ on **four sessions in seven years**, and the generated one
trades fewer, never more (three triggers on the 10:29 bar it declines to act
on at 10:30; one bar spanning both stops it refuses to resolve). The 110 was
2019: `run_generated.py` scored from 2019-05-05 and entry 5 from 2020. On entry
5's own window the numbers are 478 and 476. **Contract size does not change a
trade count** — 476 at both 1 and 4 contracts; sizing scales P&L. Before
attributing a count discrepancy to logic, check the span and the halts.

### Slippage is applied in one place: `engine.price_trades`

`PaperAdapter` once slipped the entry a tick *and* `price_trades` slipped both
fills again — $1.25 per contract of phantom cost, found by hand-checking one
ticket against the engine. The adapter reports the signal level; the cost
model owns costs. A future live adapter reports a *real* fill and will need a
zero-slippage cost model, or the same double-count returns.

### `from __future__ import annotations` breaks a FastAPI `Request` imported inside a function

String annotations resolve against module globals. A `Request` bound only as a
function local is unresolvable, FastAPI silently treats the parameter as a
query field, and every valid POST returns **422 blaming a missing query
parameter named `request`**. Import FastAPI symbols at module level.

### Imports: flat modules, no packages

There are **no `__init__.py` files and no package imports.** Modules are
imported by bare name — `import rules`, `import store`, `from engine import ...`
— and directories are put on `sys.path` by each entry point and by
`tests/conftest.py`, which now also adds `strategies/generated`.

`from strategies.rules import ...` **will fail.** Follow the existing pattern.
A submission may not be named after an existing module (`sandbox.RESERVED_NAMES`):
a generated `rules.py` would shadow the guard module.

### There are two modules called `review`, and import order decides which you get

`backtests/review.py` and `journal/review.py` both exist, and `journal/review.py`
imports the backtests one **by bare name**. Whichever directory is earlier on
`sys.path` wins. `strategies/registry.py` loads the journal one by file path
under a distinct module name (`_journal_review()`); tests that need it do the
same. If you add a third consumer, do that rather than reordering the path.

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
expectancy and 50% gates. Rule 13's acceptance test lives in
`run_generated.py` and nowhere else.

The same discipline now covers a cost. `rules.COMMISSION_PER_SIDE` ($0.50,
Lucid's confirmed rate, source recorded beside it) is what `engine.CostModel`
and every runner's `--commission` default read; a test scans the runners for
a restated default. `rules.ASSUMED_COMMISSION_PER_SIDE` ($1.25) exists only
so a verdict scored before 2026-09-11 can be reproduced — `--commission 1.25`
on any runner — and is never a default. `backtests/reprice.py` re-prices a
saved stream between the two exactly, recovering each trade's size from the
commission it carried.

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
Entry 6 sizes per session from the range height, so it does not fit — which is
why `london_*` runners carry `build=None` and `/backtest` refuses them.

**The workaround, which needs no engine change:** P&L is exactly linear in
contracts *within* a trade — `net_pnl = contracts × (net_points × point_value −
2.5)` — so price at one contract and scale each row by its own size. The daily
loss limit is then applied **per contract-size group**, which is exact *only*
because entry 6 takes at most one trade a day. A strategy taking two trades a
day at different sizes would break that and needs a real engine change — as
would the portfolio layer in §4.

Related: **the size-linearity assertion that guards entries 4 and 5 does not
apply to entry 6.** Those two can be rescaled after the fact; entry 6 cannot.

### `london_date` is not `rules.session_date`, and conflating them is silent

`rules.session_date` is the calendar date of a timestamp. **Entry 6's overnight
range window is not:** bars from 19:00 ET onward belong to the *next* day's
London session, so a Monday trade is measured against a range that began Sunday
at the Globex reopen. `london_date()` in `strategies/london.py` is the only
place that mapping lives. The same trap bit `run_london.fold_frame` when read
back from CSV: the fold year must be the **ET** year, parsed with `utc=True`
and converted, or a 31 December evening entry lands in the next year's fold.

### rules.py has no concept of an overnight session

`is_entry_allowed` returns True at 03:00 ET — but because 03:00 is numerically
before a cutoff designed for the *afternoon*, not because the module models an
overnight session. It has one RTH session per calendar day. The guards are
satisfied by accident of arithmetic. See §3.8.

### Backtest numbers from before `4a14a83` are not comparable

That commit moved the daily loss limit $300 → $400 and the position cap 2 → 5.

### Every breakout entry is rejected — do not "improve" them

Entries 1, 4, 5, 6, 7 and 8 have tested breakout continuation across two
sessions, two instruments, three signal definitions, four holding periods, two
benchmarks and now two implementations. **All rejected.** Each entry's `Next`
section forbids the obvious follow-up, and entry 7 closes the London family.

**Entry 10 reopens the London family by operator direction, on the exit rather
than the entry**, and records the override and the upward bias of its prior in
its own text. That is the one sanctioned exception, and it earned it by raising
its bar above the effect that motivated it. It does not license anything else
in this family.

**Entry 1's condition has never been met and is the only route back:** establish
the counterparty claim independently of backtest results — order-flow evidence
about who takes the other side of a range break and under what constraint. That
is a data purchase, not a backtest, and should be priced before it is started.

### Do not propose order-placement or automation code yet

**Not until EITHER (a) the 60-trade paper-trading gate is met, OR (b) a
pre-registered hypothesis has passed walk-forward with kill criteria stated in
advance.** Neither holds. The registry enforces this in code; `Registry.promote`
will refuse, and `Registry.add` cannot create anything above `proposed`.

The failure mode is offering to wire up a broker because the scaffolding exists
and looks ready. It is ready in the sense that the guards work; it is not ready
in the sense that there is nothing with demonstrated positive expectancy. The
desk bot's existence in shadow mode makes this temptation stronger, not weaker.

### Bar labelling: left-closed, labelled by opening minute

A 5-minute bar labelled `15:25` covers 15:25–15:29 and **closes at 15:29:59**.
So "the price as of 15:30" is the close of the `15:25` bar. To make a 5-minute
candle close at a value, **set the *last* one-minute bar in its bucket**, not
the bar at the label. `feed.BarAggregator` publishes a bucket only when a bar
from the *next* bucket arrives, which is what keeps the desk free of lookahead.

### Naive timestamps are read as ET, not UTC

`rules.to_et` interprets a naive timestamp as America/New_York. Passing a naive
**UTC** timestamp will be misread by five hours with no error. The Pine feed
sends epoch milliseconds for this reason.

### Signals are session-independent — exploit it, it is tested

Generating signals once over the whole history and slicing by date gives exactly
the same result as generating per window. `tests/test_scan.py` and
`tests/test_orb2.py` assert this; the desk relies on it, regenerating over the
accumulated session each bar.

### Long runs, and where the time actually goes

The ORB walk-forward is ~55 minutes; entry 6's four-arm run about 20; entry 7's
bootstrap about 15; a generated strategy's walk-forward two to three. The desk
loads the 40 MB parquet twice at startup (trend EMA, then replay) — about a
minute. Run long things in the background.

**One performance trap, already hit:** `bootstrap_benchmark._by_day` pre-slices
the bars once. The first version scanned the whole 1.7M-row frame inside the
per-trade loop and turned a minute of work into an hour with no output.

### Data costs real money

`MES.v.0` and `MNQ.v.0` — volume-rolled continuous, never `.c.0`: the
front-month series has **no RTH bars at all on the 8 quarterly expiry days**.
Bare `MES`/`MNQ` does not resolve. **Always run `--estimate` and show the number
before pulling.** Metadata calls are free; timeseries calls are not. The Claude
API costs money too: `/submit` is one streaming call per attempt.

### `venv` and pinned versions

Python 3.14.7, virtualenv at `venv/`. Invoke it explicitly:
`venv\Scripts\python.exe`. **`plotly` is pinned below 6** because vectorbt
registers themes using a trace type plotly 6 renamed. `vectorbt` is installed
but effectively unused — the engine is plain pandas, deliberately. `fastapi`,
`uvicorn`, `httpx` (the desk webhook and its TestClient) and `anthropic`
(`/submit`) were added this session and are in `requirements.txt`.

### Windows encoding and shell escaping will destroy files

`pathlib.Path.write_text(s)` defaults to **cp1252** and raises on any non-ASCII
character *after* truncating the file. **Always pass `encoding="utf-8"`.**
`json.dumps` escapes non-ASCII by default, which hides this until a real
non-ASCII value arrives — `desk_state.py` writes with `ensure_ascii=False` so
the explicit encoding is load-bearing.

**Backslash escapes are worse.** Writing `bots\research.py` through a shell
heredoc into Python turns `\r` into a carriage return — silently producing
`botsesearch.py`. A heredoc also mangled a 150-line append this session.
**Write files with the editor tool, or write a script to a file and run it.**
Also: `-c` snippets are in cp1252 on the console — an em dash in a label
renders as `?`; the string itself is fine.

### matplotlib parses `$` as math

Any label containing paired dollar signs gets silently italicised. Set
`plt.rcParams["text.parse_math"] = False` on any new chart with currency labels.

### Pine: `"#"` number formats emit invalid JSON

`str.tostring(x, "#.##")` renders zero as an empty string and values below 1
as `.5` — both unparseable, and a zero-volume overnight bar is routine. Use
`"0.##########"`. `pine/bar_feed.pine` sends `time` (the bar's *opening*
minute), not `time_close`; swapping them shifts every bar forward a minute.

---

## 6. The gate — desk bot and multi-account fan-out

**The desk exists in shadow mode (§2). The gate is unchanged and not met.**

Both the desk trading a strategy and multi-account fan-out **wait on a strategy
reaching `paper` status in the registry. None has.** `Registry.promote` will
not put one there without an ACCEPTED walk-forward in `hypotheses.md`, and
`/walkforward`'s ACCEPTED path stops at `testing` on purpose.

**On multi-account fan-out, recorded in four entries and repeated here:**
copying identical trades across N funded accounts multiplies outcomes in both
directions. It is not diversification. The same losing day draws down every
account simultaneously, and a trailing-drawdown breach terminates all of them
on the same date. **N accounts running one strategy is one bet at N times the
size, with N times the fees.** The only thing it diversifies is the
*evaluation attempt*, and only while accounts start at different times.

### What would actually unblock this

1. **Entry 3's gate met** — 60 rule-clean journaled paper trades, positive
   expectancy after costs, `eval_sim` pass probability above 50%. `/read` makes
   logging one click; it does not make the trades any better.
2. **A new pre-registered hypothesis that passes walk-forward** — a majority of
   yearly folds, positive total after commission and slippage, at 2 ticks.
   `/submit` now takes one from Discord to a verdict in about twenty minutes.
   Seven have tried; none has passed. The commission is now Lucid's confirmed
   $0.50 a side (§3.2); the cost term still resting on an assumption is
   slippage (§3.11).

---

## 7. Runtime and command reference

**Both bots launch from `.bat` files** in the project root, each in its own
console window; both read `.env`:

```powershell
.\start_bot.bat        # research bot: /backtest /walkforward /evalsim /hypotheses /status /read /submit
.\start_desk.bat       # desk bot (shadow) + Cloudflare tunnel in a second window
```

**The desk needs a fresh tunnel URL pasted into TradingView on every restart.**
Without `DESK_TUNNEL_HOSTNAME` the tunnel is a *quick* tunnel with a random
`*.trycloudflare.com` hostname that changes each launch; the "Desk Tunnel"
window prints the full paste-ready URL (hostname + `/bar?token=…`) on start.
Alert: condition **Bar feed → Any alert() function call**, **Once Per Bar
Close**, message empty. A named tunnel (stable hostname) needs a domain on
Cloudflare and is not set up.

`DESK_OWNER_ID` and `ANTHROPIC_API_KEY` are set in `.env`; only that Discord
user can press any button, and `/submit` reaches the API.

```powershell
venv\Scripts\python.exe -m pytest tests\ --ignore=tests\generated tests\generated\test_orborb_flat_1030.py -q   # 963, ~65s
venv\Scripts\python.exe strategies\registry.py                 # where everything stands
venv\Scripts\python.exe journal\review.py                      # entry 3 gate + desk decisions

venv\Scripts\python.exe bots\desk.py --replay --start 2026-08-10 --end 2026-08-14 --speed 0   # replay, no Discord, no tunnel
venv\Scripts\python.exe bots\desk.py --no-discord              # live webhook, console only
venv\Scripts\python.exe bots\tunnel.py                         # tunnel alone, prints the URL

venv\Scripts\python.exe backtests\walkforward.py               # ORB, ~55 min
venv\Scripts\python.exe backtests\run_london.py --folds-only   # rebuild fold CSVs in seconds
venv\Scripts\python.exe backtests\run_entry7.py --folds-only
venv\Scripts\python.exe backtests\run_generated.py <name> --class-path <module:Class>   # what /walkforward runs
venv\Scripts\python.exe backtests\run_orb2.py                  # entry 4, --from-cache to re-report
venv\Scripts\python.exe backtests\run_orb_flat.py              # entry 5

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
