# Handoff

Rewritten 2026-09-13 and finalised 2026-09-14, at the end of the session
that closed entries 11 and 12, verified the CME calendar, added the
hold-regression check and set up the tooling (§2, *Tooling*). Updated later
on 2026-09-14 for the forward MES file and entry 13. For a session starting
fresh on this repository.

This file holds only what is **not** already in `CLAUDE.md` (the hard rules and
the firm/internal limit table), `README.md` (structure, how to run things, the
bots), or `research/hypotheses.md` (what has been tested and rejected, and
why). Read those three first; this is the delta.

---

## 1. Current state

**Branch** `master`, tracking `origin/master` at
<https://github.com/IamNotSuperior/futures-bot> (private). **Nothing unpushed.**
**1,222 tests pass**, 2 skipped (a symlink test the OS refuses, and one that
needs an orphaned generated module on disk, of which there is none) —

```powershell
venv\Scripts\python.exe -m pytest tests\ -q
```

about two minutes. The plain command is the whole contract; the `/submit`
pipeline runs a subset of it — the base suite plus the submitting strategy's
own generated test (§5). **Working tree is clean.**

### The research

**Thirteen hypothesis entries. Eleven rejected. Entry 3 is open with zero
trades logged. Entry 13 is frozen and waits on forward data until about
September 2028.** No strategy has ever reached `paper`.

| # | Idea | Status |
|---|---|---|
| 1 | Opening Range Breakout | REJECTED |
| 2 | Leveraged ETF end-of-day rebalance drift | REJECTED — on its mechanism test |
| 3 | Manual discretionary trading | **PROPOSED — open, zero trades logged** |
| 4 | ORB-2, fixed bracket with a daily trend filter | REJECTED |
| 5 | ORB flat by 10:30 | REJECTED |
| 6 | London breakout of the overnight range | REJECTED |
| 7 | Replication of entry 6's non-null finding, on MNQ | REJECTED (not replicated) |
| 8 | `orborb_flat_1030` — entry 5 re-submitted through `/submit` | REJECTED (pipeline validation) |
| 9 | `orb_full_day_test` — entry 4's OFF arm re-submitted through `/submit` | REJECTED (pipeline validation) |
| 10 | `london_full_day` — entry 6's 1×/ON held to 15:55, MES and MNQ | REJECTED, both instruments (verdict `82006bf`) |
| 11 | `tom_intraday` — turn-of-month flows, long 09:30–15:55 on T-1..T+3 | REJECTED (verdict `2bf4fc4`) — the near-miss in this log |
| 12 | `tom_intraday_mnq` — entry 11 replicated on MNQ at one contract | REJECTED at Part A (verdict `1b304bb`); Part B (forward MES) never run |
| 13 | Turn-of-month forward test on MES, mechanism-only, no registry record yet | **PROPOSED — frozen at `6c65c89`; runs once at 96 forward window sessions (~Sep 2028); no runner written** |

**The breakout family — entries 1, 4, 5, 6, 7, 8, 9 and 10 — is closed on entry
and on exit.** Eight entries tested breakout continuation across two sessions,
two instruments, three signal definitions, five holding periods, two benchmarks
and two implementations. Entry 10 took the one question entry 6 left open, the
exit, and answered it: the trades the US session resolves resolve at the
driftless rate, and the departure shrinks. **Entry 1's order-flow condition is
the only route back** — evidence about who takes the other side of a range
break and under what constraint. That is a data purchase, not a backtest, and
it is priced before it is started or not started.

**Entries 11 and 12 are one marginal result measured twice, not two.** The
intraday turn-of-month excess is about 0.11–0.12 control standard deviations a
day, one-sided p 0.03–0.04, concentrated on T+2, on MES and again on MNQ over
the same 314 sessions. MES and MNQ same-day returns correlate 0.927 and MNQ's
window effect conditioned on MES is zero, so entry 12 added no independent
evidence; its entry records that as a pre-registration design error. **For a
calendar effect the sample is days, not instruments.** The only route left is
a new entry on forward MES data, with the power stated (96 forward window
sessions ≈ 20–25% power against the observed effect; ~330 ≈ 50%). At one
contract the stop arm's comparable stream was 7 of 7 folds and pass 79.7%,
with a $2,406 worst drawdown the $1,500 internal halt cannot carry; that
geometry is the starting point for any account criteria on this mechanism.
**Entry 13 is that new entry**, frozen 2026-09-14 as mechanism-only (§4a).

**Entry 2 died on its mechanism test, and still does.** At the corrected
commission it passes two of its three criteria (4 of 7 folds profitable; median
fold Sharpe +0.523 against a 0.30 line) and fails the third: the effect size
does not rise with the threshold — Spearman −1.000 on pre-cost returns, the
exact opposite of the prediction. That criterion does not depend on costs and
is the one that tests the claimed mechanism. It is the right one to have died
on, and the log says so in the entry's 2026-09-11 addendum.

**Every verdict is now scored at $0.50 a side** — `rules.COMMISSION_PER_SIDE`,
confirmed by Lucid support (article 11508978). The $1.25 every verdict before
2026-09-11 carried is `rules.ASSUMED_COMMISSION_PER_SIDE`, and any earlier run
reproduces with `--commission 1.25` on its runner. The correction moved every
entry's margin and flipped no verdict (§3.2, §5). **Slippage is now the
dominant cost term and is unmeasured:** $2.50 of a $3.50 round turn at one tick
on MES, $5.00 of $6.00 at rule 13's two-tick bar. It can only be measured on a
live account, fill by fill (§3.10).

`venv\Scripts\python.exe strategies\registry.py` prints the registry; it agrees
with the log.

### Files on disk that are not in git

`.gitignore` excludes these deliberately. The first two are **expensive to
recreate** — do not delete them casually.

- `data/mes_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 40.3 MB, 2.58M bars,
  2019-05-05 to 2026-08-31.
- `data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet` — 44.6 MB, same span. Pulled
  for entry 7 at **$9.4256**.
- `data/mes_v_0_ohlcv_1m_forward.parquet` — the rolling **forward** MES file,
  sessions after 2026-08-31 only, which no verdict has seen. Started
  2026-09-14 with 2026-09-01 → 2026-09-14 (12,180 bars, 0.20 MB, **$0.0445**);
  holds the September turn-of-month window (T+1..T+3, 1–3 Sep) and nothing
  more. `data/extend.py --pull` appends the next slice (§4a, §7); the
  2019–2026-08 cache is never written. `mes_v_0_ohlcv_1m_2026-09_2026-09.parquet`
  is that first pull as `fetch.py` wrote it, already merged, kept only as the
  raw copy.
- `backtests/results/*` — 126 files: scan CSVs, walk-forward fold tables, trade
  streams, reports, charts, 26 `entry10_*` files and 24 `entry11_*`/`entry12_*`
  files (session-return populations, both arms at both cost levels on both
  bases, folds, verdict blocks). All regenerable: the ORB walk-forward takes
  ~55 minutes, entry 6 about 20, entry 10's full run about an hour (the two
  bootstraps are most of it), entries 11 and 12 about a minute each. The London and entry 7 fold tables rebuild in
  seconds with `--folds-only`. **Generated and hand-built streams in this
  directory are now priced at $0.50**; `--commission 1.25` regenerates the
  $1.25 ones the frozen verdicts cite.
- `.env` — five keys: `DATABENTO_API_KEY`, `DISCORD_TOKEN`,
  `DESK_WEBHOOK_TOKEN`, `DESK_OWNER_ID`, `ANTHROPIC_API_KEY`. All set. Ignored
  at `.gitignore:2`; `.env.example` documents each.
- `journal/desk_state.json`, `journal/tunnel.log`, `journal/live_bars/*.csv` —
  desk runtime. `live_bars/` holds one file per session date the desk has seen;
  `/read` reads only today's, so a stale file is inert.

`journal/trades.jsonl` **is tracked and does not exist** — no manual paper
trade has ever been logged, by CLI or by `/read`. `journal/decisions.jsonl`
(desk ticket button presses) does not exist either. Both appear on first use.

**Databento spend is about $21.47.** `data/fetch.py` estimates first and refuses
above `--max-cost`, default $10; `data/extend.py` does the same for the forward
file, default $15, and a forward month costs about $0.11.

---

## 2. What exists that is worth knowing about

### Infrastructure, one paragraph each

**The desk bot is in shadow mode.** `bots/desk.py` was built 2026-09-06 as a
deliberate pre-gate exception to test plumbing. The only strategy it runs is
**`orb2`, which entry 4 REJECTED** — a shadow ticket must never be mistakable
for a result. `shadow:` in `registry.yaml` is a list of names, not a status;
`Registry.promote` never reads it. Tickets go to `journal/shadow_trades.jsonl`,
never `trades.jsonl`, so entry 3's count is untouched. Ticket embeds carry
**Execute / Don't trade** buttons, owner only; presses go to
`journal/decisions.jsonl` as the operator's record and are **reported by
`review.py`, not counted toward entry 3**. Execute is disabled and refused for
anything below `paper`/`live`. The desk reuses rather than reimplements:
`tickets.evaluate_signal` calls `pretrade.evaluate` and `tests/test_desk_rules.py`
asserts the decisions match check-for-check. Discord is required
(`DESK_CHANNEL_NAME`, default `general`); a missing channel or token is fatal
at startup. Bars arrive from a TradingView alert (`pine/bar_feed.pine`) through
a Cloudflare tunnel (`bots/tunnel.py`), which refuses to start without
`DESK_WEBHOOK_TOKEN`; the desk mirrors each bar to `journal/live_bars/` for
`/read`, which runs in the other process.

**The research bot** — `bots/research.py`: `/status`, `/hypotheses`,
`/walkforward`, `/evalsim`, `/read`, `/submit`, plus `/backtest`. `/read`
describes a chart with no bias label, score or opinion (by test), and its
**Log long / Log short** buttons write to the manual journal through
`pretrade.evaluate` — those **do** count toward entry 3, because the operator
picks the direction and writes the thesis, which is exactly the mechanism entry
3 pre-registered. `/walkforward` runs the folds for a generated strategy and
reads saved output for everything else, because those verdicts are frozen.
`/evalsim` reports the pass probability and, since 2026-09-11, the probability
of touching Lucid's $52,100 payout line.

**`/submit` is validated against two known verdicts, to within 1–2 trades.**
Entry 8 re-submitted entry 5's description and reproduced its signal to within
2 trades in seven years (476 against 478, 0 of 7 folds either way). Entry 9
re-submitted entry 4's filter-OFF arm and landed within 1 trade of it (503
against 504) and about 6% in P&L at 2 ticks. Both REJECTED, both as expected.
The pipeline can be trusted on a new idea.

**Video intake is a chat protocol, not code.** The `watch` Claude Code plugin
(installed 2026-09-12; yt-dlp, ffmpeg and Deno via winget; transcript mode
by default, captions only, no Whisper key) turns a pasted YouTube URL into a
timestamped transcript in the session. The protocol, fixed with the operator
on 2026-09-12: the operator pastes a URL; the session runs `/watch` in
transcript mode; it reports first whether the video names a counterparty and
whether the rule falls in a closed family, then drafts *Mechanism* and
*Signal definition* labelled as a draft; the operator writes the counterparty
paragraph, kill criteria and prediction in their own words (rule 12); sizing
comes from the control standard deviation before the entry is frozen; then
`/submit` or a hand-built runner. **Wiring `/watch` output into the `/submit`
modal is deliberately not built**; the chat protocol keeps the operator's
words as the pre-registration by construction. The filter for what to send:
forced or constrained counterparties only — options expiry and dealer
hedging, index rebalances, fund month-end marks, economic-release
positioning, auction mechanics. No breakout, opening-range, London or
indicator-crossover content; that family is closed and those rules have no
counterparty. First video through the protocol (2026-09-13, a TradingLab
supply-and-demand retest rule): no counterparty, breakout-continuation
family, discretionary rules a machine cannot apply — no entry, by design.
Second (2026-09-14, The Moving Average, "Fibonacci Retracement explained in
under 5 minutes"): pullback-continuation entry in the 0.5–0.618 zone on a
discretionary reversal signal, RSI divergence as confirmation, subjective
targets; its only stated mechanism is "psychological levels" and ratios
"seen in nature" — no counterparty, indicator content, closed family; no
entry, by design. Both videos so far were rejected at the first question,
so the modal pre-fill stays deferred.
Nothing in the repository reads a video; the plugin's `~/.config/watch/.env`
and the session's shell PATH are the only state, and a fresh desktop-app
launch is needed for its shells to see the winget PATH entries.

**The execution adapter is NOT built and is gated in code.** `PaperAdapter` is
the only adapter — a test asserts `BrokerAdapter.__subclasses__()` has exactly
one member — and `broker.require_live_eligible` checks registry status before
the account flag, so a flag flipped by accident still refuses. `Registry.promote`
will not reach `paper` without an ACCEPTED walk-forward in the log, and
`Registry.add` cannot create anything above `proposed`. **Lucid confirmed on
2026-09-11 that API and automated order placement is permitted on evaluation
accounts via Rithmic or Tradovate** (Other Trading Activities policy, article
11404728) **and that MES commission is $0.50 a side** (article 11508978). The
account side is therefore open; the evidence side is not, and nothing about
the gate moves (§6).

### The `/submit` pipeline

`bots/submissions.py` is the pipeline; `bots/generate.py` calls the Claude API
(`claude-opus-5`, streaming, adaptive thinking); `bots/sandbox.py` is what
generated code may write and contain; `bots/submit_view.py` is the modal and
approval buttons; `backtests/run_generated.py` is the walk-forward that
actually runs.

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

**A generated verdict is scored under both internal guards** — the daily loss
limit and the $1,500 trailing halt, via `engine.apply_internal_guards` — and
reports the same signals with the halt off alongside, as the comparable basis,
because a halt that fires in year one leaves the fold test with nothing to
count. Rule 13 is decided on the guarded stream; evaluation blow-ups are read
on the comparable one. `--commission` exists on the CLI so a pre-correction
verdict is one flag away; the bot never passes it.

**The sandbox is two layers.** `safe_write` confines writes to
`strategies/generated/` and `tests/generated/`, resolving before comparing so
`..`, absolute paths and symlinks are refused. That does not contain the code,
which pytest imports and executes — so `screen_source` AST-screens imports
(only pandas/numpy/base/rules/pytest and stdlib maths; a generated *test* may
additionally import its own strategy, and no other), `eval`/`exec`/`open`/
`__import__` and dunder reflection, and `run_tests` executes in a subprocess
with every secret stripped from the environment. The screen guards against
mistakes and drift; **it is not a security boundary against an adversary.**
The scrubbed environment is what makes a miss survivable.

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
findings, all in the entry's 2026-09-11 addendum and in §5:

- **The exit was not hiding an edge.** Like-for-like on the unhalted streams,
  the extended hold is worth $2,129 on MES and $165 on MNQ, and the trades the
  US session resolved did so at the driftless rate — 52.5% on MES, 51.5% on
  MNQ — pulling the pooled target share under break-even. Prediction 2 held.
- **The trailing halt left the z test almost no sample.** It ended both guarded
  streams in year two (552 and 282 sessions blocked), leaving 87 and 112
  resolved trades where the entry's power arithmetic assumed ~650 and ~420, so
  z = 2.5 needed +13 points.
- **On MNQ the corrected sizing rule cost $1,755 against entry 6's**, because
  MNQ's overshoot is four times MES's and the rule prices it.

**Code.** `strategies/london.py` gained three parameters whose defaults are
entry 6's, so entries 6 and 7 are untouched: `flatten_time` labels its exit by
time (`flatten_1555`), `early_close_flatten_time` (rule 2's deadline is the
backstop when unset), and `size_on="stop"`. `backtests/run_entry10.py
--reproduce` is the reproduction check and `--run` the 15:55 test (both guards
as the two engine functions `apply_internal_guards` composes, applied per size
group; halt-OFF alongside; 1 and 3 ticks as sensitivities; the bootstrap at the
15:55 horizon with early-close days truncated at 12:55; the verdict block to
`backtests/results/entry10_verdict.md`). The bot's `london_full_day` runner
reads the MES base case.

### Entries 11 and 12 — turn-of-month, REJECTED twice, one sample (verdicts `2bf4fc4`, `1b304bb`)

**Entry 11** (frozen `a6d9af1`): long MES at the 09:30 open on T-1, T+1, T+2,
T+3 by the cash trading calendar, flat at the 15:55 open, against a control
of every eligible non-window session; 4 contracts; a signal arm with no stop
and a stop arm at 15 points; $0.50, 1 tick base, 2 ticks sensitivity; kill
criteria pre-registered by the operator plus rule 13. **Result:** pooled
excess +4.95 points pre-cost, Welch t 1.94 against 2.0 — failed by
six-hundredths; window over control in 5 of 7 years; T+2 carried the effect
post hoc; the 4-contract stop arm blew ten evaluations on a +$14,369
halt-OFF stream and the $1,500 halt ended the guarded stream after eleven
trades. Two code lessons in §5 (the population must be built over the whole
file and sliced; the halt leaves a pooled test little sample).

**Entry 12** (frozen `b632028`): entry 11's test on MNQ at one contract, the
stop derived at run time as 0.366 of the instrument's control standard
deviation (entry 11's 15 / 40.99), plus a pre-registered forward-MES Part B
at a 96-session trigger with its low power stated. **Result:** MNQ
reproduced entry 11's shape — +0.112 control sd against +0.121, t 1.75, 6 of
7 years, T+2 largest — and failed the same line; the one-contract stop arm's
halt-OFF stream was 7 of 7 folds, +$11,650, pass 79.7%, worst drawdown
$2,406, and the guarded stream died to the halt in 2022. **Post hoc, labelled
as such:** MES and MNQ same-day returns correlate 0.927 and MNQ's window
effect conditioned on MES is −0.005 sd. The replication replicated nothing;
the entry records that as a design error in its own pre-registration (§5).
Part B was never run.

**Code.** `strategies/tom.py` holds the cash trading calendar
(`trading_days`, `label_window_days`, `cash_half_days` — CME sessions less
the cash-closed holiday sessions, half-days kept in the count and skipped as
trade days), `session_calendar`/`session_returns`, and `TurnOfMonth` with
`TOMParams(stop_points)`. `backtests/run_entry11.py` and `run_entry12.py`
are the runners (`--reproduce` before `--run`; entry 12 takes `--part A|B`
and `--parquet`); `research/power_check_tom.py --parquet` is the calendar-
only power check. The bot's `tom_intraday` and `tom_intraday_mnq` runners
read the saved stop-arm base-case files.

### Tooling, 2026-09-12 to 2026-09-14

**Claude Code plugins installed at user scope:** `superpowers`,
`security-guidance`, `frontend-design`, `context7`, `commit-commands`,
`watch` (video intake, §2 above), `pyright-lsp` and `hookify`. Community
trading plugins were reviewed and declined — they are strategy generators and
optimisers, the process rule 12 exists to prevent. A Composio `connect-apps`
MCP server exists in `~/.mcp.json` from a separate session (§3.13); nothing
here uses it.

**Pyright.** `pyrightconfig.json` at the project root gives the language
server the flat-module `extraPaths`, the venv, basic mode, and silences the
pandas-driven categories (argument, attribute, call) that its stubs make into
noise. Baseline: 169 errors across 93 files, down from 927 without the
config; every one sampled was a false positive (a `None` check Pyright does
not carry through a comprehension, a Discord channel guarded at startup,
FastAPI parameters typed optional by the framework). Nothing gates on it. Run
`pyright` from the root to see the list.

**The live backtest view, 2026-09-15.** A read-only page that follows a
walk-forward while it runs. `backtests/live.py` gives a runner a `LiveRun`
that writes one JSON line per event (start, stage, trades, folds, done or
error) to `backtests/results/live/<name>.jsonl`, gitignored output, with CSV
copies of the priced streams and the fold table beside it; `NullLive` is the
default and writes nothing. `run_generated.py`, `run_entry11.py` and
`run_entry12.py` take `--live`, and the bot's `/walkforward` writes the
stream when `FUTURES_LIVE=1` is in its environment. `bots/liveview.py`
serves the page on **port 8790** (the desk feed has 8787): stage timeline
with elapsed times, the equity curve drawn trade by trade *from the finished
stream* once it exists, the fold table, the outcome pill. Every route is
GET, the module imports no runner, and a test asserts both. `.claude/launch.json`
opens it in the desktop app's browser pane. **A walk-forward here is not
incremental** — signals, pricing and folds are each one call — so the stages
are the run's real granularity and the curve is labelled a replay. Design
note: `docs/superpowers/specs/2026-09-15-live-backtest-view-design.md`. Two
things it caught on its first real run: Starlette's JSON response refuses
NaN (the halted fold years carry NaN Sharpe), and a re-run under the same
name truncates the file, so the page resets on the run's start time.
**Saved verdicts replay in the same page** (later the same day): the picker's
second group lists every registry name; selecting one shows the registry
status, the runner note, the file names and their date in place of stages,
then the same equity replay and fold table read from `backtests/results/`.
The file mapping is `runners.RUNNERS` (the verdict's own arm and cost
level, per entry) plus the generated runner's fixed naming; the viewer
imports that table and calls no runner function, by test. A name whose
files are not on disk is listed as unavailable rather than hidden. Entry 1's
fold loop remains out of scope.

**Hookify rules**, in `.claude/hookify.*.local.md`, tracked in git and read
live on every tool call — no restart:

- `block-tls-weakening` **blocks** a `.py` edit outside `tests/` that adds
  `verify=False`, `check_hostname=False` or `CERT_NONE`. It fired on its
  first day against a scratchpad script that quoted those literals in prose,
  which is the intended behaviour: write such a script under a non-`.py`
  name, or keep the literals out of it.
- `warn-restated-threshold` **warns** on a numeric assignment to a name
  `rules.py` owns, a literal non-zero `commission_per_side`, or a three-digit
  `limit=`, outside `rules.py` and `tests/`. Aliases to `rules.*` and the
  zero-cost reproduction model do not trip it.
- `warn-commit-needs-approval` **warns** on every `git commit` / `git push`
  that the operator approves each one in conversation.

Two things about the engine worth knowing before writing a fourth: it
compiles every pattern case-insensitively, so an uppercase constant name
needs a scoped `(?-i:...)` group or it matches lowercase keyword arguments;
and Edit puts text in `new_string` while Write puts it in `content`, so a
rule must use the `content` field to cover both. Rules match on any file
path under the current directory's rules, including scratch files outside
the repo. The "refuse `/submit` while the log is dirty" rule is deliberately
not a hook: hookify cannot see git state and `/submit` is a Discord command;
`submissions.require_clean_log()` enforces it in code. `python3` on this
machine resolves to the real 3.14.7, which is what lets the hooks run at
all; the `watch` skill's note that it is the Store stub is wrong here.

**The operator's session protocol.** At the end of each completed step, put
up a clickable "Should I do the next step?" question naming the step, rather
than a prose offer; the operator drives one step at a time and answers yes,
no or free text. Ask before commits, pushes and new dependencies; fold that
into the same question when it is what the step needs.

### The rule 6 / rule 7 regression check, 2026-09-13

Rule 6's 30-second floor means rule 7's share must read 0.00% in any correct
run, so a non-zero figure is a bug signal, not a risk warning — but the only
flag used to be the 30% warning line, and one three-second trade would have
passed under it. `metrics.compute_metrics` now carries
`min_hold_violation_count` (holds under the floor) and `hold_regression`
(any such hold, or any hold at or under five seconds); `format_report` prints
a `[REGRESSION]` line; `run_generated.hold_regression_line` puts the same
sentence in every generated verdict block and in the `/walkforward` embed;
and `run_generated.decide` **rejects** a stream that carries one, whatever
its P&L, because it was not produced under rule 6. The journal
(`review.hold_violations`, which resets entry 3's count) and the desk's
end-of-day summary already reported theirs. `tests/test_hold_regression.py`
holds all of it.

### The registry and the log

`Registry.promote` takes two arguments and no override; `rejected` is
terminal; `testing → paper` requires an ACCEPTED walk-forward in the log;
`paper → live` calls `journal/review.readiness`. `verify()` cross-checks the
registry against the log and is how a real drift bug was caught once.

**The evidence pipeline** is unchanged and is the product: entry with
mechanism, counterparty and kill criteria → committed → build → walk-forward →
verdict frozen before anyone sees numbers → hash recorded in a follow-up.
Eleven of thirteen entries died in it; one is open and one waits on data. `backtests/reprice.py` re-prices any saved stream
between two commissions exactly, recovering each trade's size from the
commission it carried; `backtests/eval_sim.py` reports the $52,100 payout
probability alongside the pass probability.

---

## 3. Open items

**3.1 Closed, 2026-09-12: the CME holiday dates are verified.** Every date
in `data/cme_calendar.py` was checked against CME's Globex schedule service
for E-mini S&P 500 (`services/trading-hours-by-product`, product 133), per
holiday range through 2028-01-02. All 2026 dates were right. Three 2027 dates
were wrong in the safe direction and are corrected: 2027-07-02 and 2027-12-23
are regular sessions, 2027-07-05 is a 13:00 ET halt rather than a closure.
The module carries the record (`VERIFIED_ON`, `VERIFIED_SOURCE`,
`VERIFIED_DATES`, `PUBLISHED_CLOSE_ET`) and a test holds the model's 13:00
close at or before every published time. CME finalises hours about two weeks
before each holiday; re-check a date in that fortnight.

**3.2 Closed, 2026-09-11: MES commission is $0.50 a side, $1.00 a round
turn** (Lucid support, article 11508978). `rules.COMMISSION_PER_SIDE` carries
it with the source beside it; `engine.CostModel` and every runner's
`--commission` default read it, and a test scans the runners for a restated
default. Entries 1, 2, 4, 6 and 7 were re-priced from their saved streams
(exact; parameter selection in 1 and 2 held at what $1.25 chose) and entries
5, 8 and 9 re-run in full with both guards re-decided; each carries a dated
addendum. **No verdict changed. One criterion would have resolved
differently:** entry 2's median fold Sharpe, and entry 2 still dies on its
mechanism test. The closest any entry sits to a line is entry 4's pooled pass
probability, 22.70% against 25% at 1 tick (12.55% at the 2-tick base case).
Break-even figures in the entry 4 and 5 runners are now computed from the cost
model (38.21% at 1 tick, 40.00% at 2); the frozen 2-tick reports had printed
the 1-tick constant.

**3.3 MNQ's commission is assumed equal to MES's.** Only MES was confirmed.
Entry 7's MNQ stream and entry 10's MNQ arm are priced at $0.50 on that
assumption; if MNQ differs, entry 7's MNQ figure moves by $2 × 438 per $1 of
difference a side, and nothing in any verdict rests on it.

**3.4 The 13:00 versus 13:15 early-close approximation — now measured.**
`rules.EARLY_SESSION_CLOSE` models a single 13:00 close. CME publishes a
12:00 CT (13:00 ET) halt on holiday Globex sessions and a 12:15 CT (13:15 ET)
close on the day after Thanksgiving and Christmas Eve; the per-date figures
are in `cme_calendar.PUBLISHED_CLOSE_ET`. Over-blocking by 15 minutes on
those two days a year is the safe error; fixing it loosens a limit and must
be stated as such.

**3.5 Closed, 2026-09-12: the late-2027 dates are verified** (2027-12-23
regular, 2027-12-24 closed, 2027-12-31 regular). `NEEDS_VERIFICATION` is
empty. The calendar still ends 2027-12-31; extend it before 2028.

**3.6 Closed, 2026-09-11: the trailing halt lives in the engine.**
`enforce_trailing_drawdown_halt` sits in `backtests/engine.py` behind
`engine.apply_internal_guards` — the daily loss limit, then the trailing halt
on the loss-limited stream — which `run_orb_flat.py` and `run_generated.py`
call and `tests/test_guard_parity.py` holds them to. Entry 5's reports came out
byte-identical before and after the move. Entries 8 and 9 were re-scored under
it (both stop trading under the halt; verdicts unchanged).

**3.7 Entry 3's gate has never been started.** Sixty rule-clean paper trades,
positive expectancy after costs, `eval_sim` pass probability above 50%. Zero
trades logged, and `/read`'s buttons make logging one click — which makes the
discipline of only logging trades actually taken matter more, not less (§4b).

**3.8 `rules.py` has no session-open guard.** A signal at 08:00 passes every
check; nothing today can produce one;
`test_premarket_signal_is_not_blocked_by_rules_py` pins it. Entry 10's 03:00
entries pass for the same arithmetic reason (§5).

**3.9 Verified, 2026-09-11: API and automated order placement is permitted on
Lucid evaluation accounts**, via Rithmic or Tradovate (Other Trading Activities
policy, article 11404728). This was the account-side question that gated the
execution layer, and it closes in the permissive direction. **Nothing else
about the gate moves:** no execution code exists or is to be written until a
strategy reaches `paper` (§5, §6).

**3.10 Slippage is the dominant cost term, and it is an assumption.** At $0.50
a side, commission is $1.00 of a round turn; one tick a side is $2.50 on MES,
71% of the $3.50 total, and rule 13's two-tick bar is $5.00 of $6.00. Every
verdict's margin rests mainly on a number nobody has measured. It can only be
measured on a live account, fill by fill against the signal level; TradingView
paper fills are optimistic and the journal cannot see it. Entry 10
pre-registered that its 03:00 fills are the thinnest this project trades and
that no backtest could settle them. Until a live fill exists, the two-tick bar
is the hedge, not a measurement.

**3.12 The trailing halt, not the daily limit, is the binding account
constraint for anything with a real drawdown.** Entry 12's one-contract stop
arm had a worst comparable-stream drawdown of $2,406 against the firm's
$2,000 and the internal $1,500; entry 11's at four contracts, $7,161. Any
future account criterion should be designed against the $1,500 line first,
and the sizing rule (contracts from the control population's daily standard
deviation) written into the entry before it is frozen.

**3.13 `~/.mcp.json` carries a Composio `connect-apps` MCP server** that the
operator set up in a separate Claude Code session on 2026-09-12, with an API
key in it. Nothing in this project uses it and nothing should route through
it; it is noted so a fresh session is not surprised to find an external
connector loaded. The key was also visible in that session's terminal
scrollback.

**3.11 The Discord user id in commit `9dc95c2`.** `tests/test_submissions.py`
once hard-coded the operator's real `DESK_OWNER_ID`; the tip uses a fake id.
The real one remains in that commit's history, which was pushed. A Discord
snowflake is not a credential — it authorises nothing and is visible to anyone
sharing a server. **It stays; do not rewrite history for it.**

---

## 4. Open directions, in priority order

None is started. The operator set the order on 2026-09-12.

**(a) Entry 13 — the forward-data test of the turn-of-month mechanism — is
frozen and waiting on data.** Pre-registered 2026-09-14: skeleton at
`5da0bc5`, operator decisions at `6c65c89`, the diff between them being the
pre-registration. Its terms, in the entry and repeated here so a fresh
session does not reopen them: the sample is forward MES days only, from
`data/mes_v_0_ohlcv_1m_forward.parquet` (the calendar is built over cache
plus forward so the September 2026 boundary is labelled; only forward
sessions are scored; 2026-08-31 is excluded as fitted data); **the entry
runs once, at 96 eligible forward window sessions, about September 2028,
never earlier**; mechanism and counterparty as entry 11; one contract with
entry 12's derived stop from the forward control population; **mechanism-
only** — criteria 1 and 2 (excess above one round turn and Welch t ≥ 2.0;
window over control in a majority of forward years with ≥ 20 sessions)
decide, 3–5 are reported, and ACCEPTED licenses no strategy and no `paper`;
power 17.5% against entry 11's +0.121 sd, recorded beside the trigger;
prediction positive, 0 to +0.12 sd, criterion 1 failing on power; prior
about one in five.

**The no-peek rule is part of the entry.** Between now and the trigger the
only action is the monthly `data/extend.py --pull` (estimate shown and
capped before each spend, about $0.11 a month; a start inside the cache or a
gap after the last bar is refused). It prints bar counts, the span and
session counts and no price. **No one computes, plots or describes a return
on any forward window session before the trigger** — not the operator, not a
session, not `/read` on a forward date. The trigger count comes from
`research/power_check_tom.py --parquet` on the joined calendar, which reads
timestamps only. The runner (`backtests/run_entry13.py`, to assert the
forward file's first scored session is after 2026-08-31 and to refuse below
96) is not written; nothing needs it before 2028 and it is written only on
request. The first pull (2026-09-01 → 2026-09-14, $0.0445) holds three
window sessions.

**(b) Entry 3's 60-trade manual gate, via `/read`.** Sixty rule-clean paper
trades logged through `/read`'s buttons or `journal/pretrade.py`, positive
expectancy after costs, pass probability above 50%. Zero logged so far. This
is the only route to `paper` that does not need a passed walk-forward, and
its kill criterion — any rule violation resets the count to zero — is the
part that does the work.

**(c) Video intake, running as a protocol (§2).** Waits on the operator
sending URLs under the filter. The `/submit` modal pre-fill is deferred until
two or three videos have gone through the protocol and its shape is known.

**(d) Closed, 2026-09-14: `pyright-lsp` and `hookify` are installed and
configured** (§2, *Tooling*). Nothing further to add from the marketplaces
reviewed.

**(e) The portfolio layer and the selector diagnostic — deferred until a
second strategy exists.** The portfolio layer would hold one position across
strategies with a shared daily budget and account-level consistency; it is
real engine work (`engine.price_trades` takes one scalar `contracts`, §5) and
needs a design before code. The selector diagnostic would ask whether any
selection rule across the rejected out-of-sample streams produces a positive
one, pre-registered as a diagnostic with no verdict, because a selector fitted
to known-negative streams is the textbook way to manufacture an edge. Neither
is worth starting with nothing at `paper`.

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

Entries 1, 4, 5, 6, 7, 8, 9 and 10 have tested breakout continuation across two
sessions, two instruments, three signal definitions, five holding periods, two
benchmarks and two implementations. **All rejected.** Each entry's `Next`
section forbids the obvious follow-up.

**Entry 10 was the one sanctioned exception** — it reopened the London family
by operator direction on the exit rather than the entry, recorded the override
and the upward bias of its prior in its own text, and raised its bar above the
effect that motivated it. It did not pay: the trades the US session resolves
resolve at the driftless rate. **The family is now closed on entry and on exit.**
Nothing in it licenses another variant.

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
Lucid permitting API order placement via Rithmic or Tradovate (§3.9) makes
it stronger still, and changes nothing: the gate is evidence, not permission.

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

### A second index future on the same days is not a replication of a calendar effect

Entry 12 replicated entry 11's turn-of-month test on MNQ and reproduced its
shape almost exactly. It proved nothing: MES and MNQ open-to-15:55 returns on
the same sessions correlate 0.927, and MNQ's window effect after conditioning
on MES's same-day return is −0.005 sd, t −0.21. The series was untouched; the
*days* were the same, and a calendar effect lives in the days. Entry 7's
instrument replication was valid because its claim was about barrier
mechanics on the instrument. **Before pre-registering a replication, name
the independent sample**: instruments for instrument-level claims, days for
calendar-level claims. Two correlated near-misses would have read as a
replication had Part A passed.

### A trailing halt truncates the sample a share test runs on — pre-register for it

Entry 10 pre-registered its z test on the guarded stream and sized the bar from
the sample it expected the strategy to produce: roughly 650 resolved trades on
MES and 420 on MNQ, so z = 2.5 needed about +5 and +6 points. The $1,500
trailing halt ended both guarded streams in their second year and left the test
**87 and 112 resolved trades**, where z = 2.5 needed **+13 and +12 points** —
three times any effect this log has measured. The criterion was close to
unpassable before the run started, and nothing in the pre-registration said so.

A pre-registration that combines a trailing halt with a share test must state
**which stream the test runs on** and compute its power **on the sample the
halt will leave**, not on the unhalted count. If the guarded sample is too small
to test, say so in advance and pre-register the share test on the unhalted
stream as a diagnostic, with the verdict still decided on the guarded one.

### Like-for-like attribution needs unhalted streams

Entry 10's criterion 5 asked how much of any result was the extended hold and
how much the sizing change — A (15:55, stop sizing) minus B (09:25, stop sizing)
minus C (09:25, range sizing). Computed on the guarded streams the figures were
wrong in kind: **each stream's halt fired on its own date** (A after 117 MES
trades, B after 185), so the three trade sets differed and A − B mixed
truncation with the effect being attributed.

The like-for-like version lives on the halt-OFF streams, where A and B take
**identical entries at identical sizes** (assert it) and A − B is purely the
exit: +$2,129 on MES, +$165 on MNQ, against −$508 and +$404 on the halted
streams. When two streams are compared trade for trade, compare them before any
guard that can end them on different days. The halted figures are the verdict
basis; they are not the attribution.

### MNQ's overshoot is four times MES's, and stop-based sizing prices it

The London entry fills at the open of the candle after a close outside the
range, so the fill sits beyond the range edge by an overshoot. On MES the
median overshoot is 1.00 point (mean 1.50); on MNQ it is **3.88 points (mean
4.93)**. Entry 6's range-based sizing ignored the overshoot and so priced the
two instruments alike; the log's corrected rule — size off the realised stop,
skip the session when one contract is over budget — prices it, and on MNQ that
cost **$1,755** against entry 6's rule at the same costs (32 sessions skipped,
smaller size where the overshoot is large). Most of the distance between entry
7's +$1,085 and entry 10's −$505 on MNQ is this, not the exit.

Any instrument translation must re-derive sizing from the realised stop on that
instrument. A rule that looks identical on two contracts because it reads the
range is not identical if the fill mechanics differ.

### The commission correction moved every entry and flipped none

Lucid's confirmed $0.50 a side replaced the assumed $1.25 — $1.50 per contract
per round turn, on every trade in the log. Three things about re-scoring it:

- **Re-pricing a saved stream is exact** and needs no bars: commission never
  touches a fill, and each trade's size is recoverable from the commission it
  carried (`backtests/reprice.py`). That is what let entry 6's per-session sizes
  come through row by row.
- **Re-pricing cannot re-decide a guard or a parameter choice.** Where a
  trailing halt decides the stream (entries 5, 8, 9) the runner was re-run in
  full; where the walk-forward selected parameters (entries 1, 2) the selection
  was held at what $1.25 chose and the addendum says so.
- **No verdict flipped.** The only criterion that would have resolved
  differently was entry 2's median fold Sharpe (+0.201 → +0.523 against 0.30),
  and entry 2's mechanism test is on pre-cost returns and did not move. The
  closest any entry sits to a line is entry 4's pooled pass probability, 22.70%
  against 25%. Entry 6's bracket alone turned from −$481 to +$896 while its arm
  still lost $5,387 on flattens — a sub-finding changing sign is not a verdict
  changing.

The frozen 2-tick reports of entries 4 and 5 had printed the 1-tick break-even
constant; the runners now compute break-even from the cost model they are given.

## 6. The gate — desk bot and multi-account fan-out

**The desk exists in shadow mode (§2). The gate is unchanged and not met.**

Both the desk trading a strategy and multi-account fan-out **wait on a strategy
reaching `paper` status in the registry. None has.** `Registry.promote` will
not put one there without an ACCEPTED walk-forward in `hypotheses.md`, and
`/walkforward`'s ACCEPTED path stops at `testing` on purpose. Lucid permitting
API order placement (§3.9) opens the account side and moves nothing here: the
gate is about evidence, and there is none with positive expectancy.

**On multi-account fan-out, recorded in five entries and repeated here:**
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
   `/submit` takes one from Discord to a verdict in about twenty minutes.
   Nine have tried; none has passed. The commission is settled (§3.2); the
   cost term still resting on an assumption is slippage (§3.10). **Entry 13
   does not count here even if ACCEPTED**: it is mechanism-only by its own
   terms, and a tradeable form on that mechanism is a separate entry sized
   against the $1,500 halt.

---

## 7. Runtime and command reference

**Both bots launch from `.bat` files** in the project root, each in its own
console window; both read `.env`:

```powershell
.\start_bot.bat        # research bot: /status /hypotheses /walkforward /evalsim /read /submit /backtest
.\start_desk.bat       # desk bot (shadow) + Cloudflare tunnel in a second window
```

**The desk needs a fresh tunnel URL pasted into TradingView on every restart.**
Without `DESK_TUNNEL_HOSTNAME` the tunnel is a *quick* tunnel with a random
`*.trycloudflare.com` hostname that changes each launch; the "Desk Tunnel"
window prints the full paste-ready URL (hostname + `/bar?token=…`) on start.
Alert: condition **Bar feed → Any alert() function call**, **Once Per Bar
Close**, message empty. A named tunnel (stable hostname) needs a domain on
Cloudflare and is not set up.

**Secrets and ids.** `DESK_OWNER_ID`, `ANTHROPIC_API_KEY` and
`DESK_WEBHOOK_TOKEN` are set in `.env`: only that Discord user can press any
button, `/submit` reaches the API, and the tunnel refuses to start without the
token. The Discord user id in commit `9dc95c2`'s history is a non-secret that
stays (§3.11).

```powershell
venv\Scripts\python.exe -m pytest tests\ -q                     # 1,222 pass, 2 skipped, ~2 min
venv\Scripts\python.exe strategies\registry.py                  # where everything stands
venv\Scripts\python.exe journal\review.py                       # entry 3 gate + desk decisions

venv\Scripts\python.exe bots\desk.py --replay --start 2026-08-10 --end 2026-08-14 --speed 0   # replay, no Discord, no tunnel
venv\Scripts\python.exe bots\desk.py --no-discord               # live webhook, console only
venv\Scripts\python.exe bots\tunnel.py                          # tunnel alone, prints the URL

venv\Scripts\python.exe backtests\walkforward.py                # entry 1, ~55 min
venv\Scripts\python.exe backtests\run_orb2.py --from-cache      # entry 4, re-report from saved CSVs
venv\Scripts\python.exe backtests\run_orb_flat.py               # entry 5, both guards, ~3 min
venv\Scripts\python.exe backtests\run_london.py --folds-only    # entry 6 fold CSVs in seconds
venv\Scripts\python.exe backtests\run_entry7.py --folds-only    # entry 7
venv\Scripts\python.exe backtests\run_generated.py <name> --class-path <module:Class>   # entries 8, 9; what /walkforward runs
venv\Scripts\python.exe backtests\run_entry10.py --reproduce    # entry 10's reproduction check
venv\Scripts\python.exe backtests\run_entry10.py --run          # entry 10's 15:55 test, ~1 h, verdict to results/
venv\Scripts\python.exe research\power_check_tom.py [--parquet data\mnq_...parquet]   # entries 11/12 calendar-only power check
venv\Scripts\python.exe backtests\run_entry11.py --reproduce    # entry 11 (MES, 4 contracts); then --run, ~1 min
venv\Scripts\python.exe backtests\run_entry12.py --part A --reproduce   # entry 12 Part A (MNQ, 1 contract); then --run
venv\Scripts\python.exe backtests\run_entry12.py --part B --parquet <forward MES file> --run   # refuses below 96 forward window sessions
pyright                                                          # type diagnostics, baseline 169 (all sampled false positives)
claude plugin list                                               # the eight plugins; /hookify:list shows the three live rules
venv\Scripts\python.exe backtests\reprice.py backtests\results\<stream>.csv [--old 1.25 --new 0.50] [--halt]
venv\Scripts\python.exe backtests\eval_sim.py backtests\results\<stream>.csv

venv\Scripts\python.exe data\fetch.py --estimate --symbol MNQ.v.0 --start 2019-05-01 --end 2026-09-01
venv\Scripts\python.exe data\extend.py --estimate                 # forward MES: next slice's cost, free; default span = day after last bar on disk -> Databento's available end
venv\Scripts\python.exe data\extend.py --pull [--max-cost 1]      # append it to data\mes_v_0_ohlcv_1m_forward.parquet; ask before running
venv\Scripts\python.exe data\extend.py --merge-file <parquet>     # fold a fetch.py pull into the forward file, no API call
venv\Scripts\python.exe bots\liveview.py                          # live backtest view at http://127.0.0.1:8790, read-only
venv\Scripts\python.exe backtests\run_entry12.py --part A --reproduce --live   # any runner with --live writes the stream the page follows
```

Every runner takes `--commission` (default `rules.COMMISSION_PER_SIDE`,
$0.50); pass `1.25` to reproduce a pre-2026-09-11 verdict. Long runs go in the
background.

Journal, in the order they are used (the one-page desk version, with the
gate, the per-trade order and what does not count, is `docs/PAPER_TRADING.md`;
verified end to end against a scratch journal on 2026-09-15):

```powershell
venv\Scripts\python.exe journal\pretrade.py --instrument MES --direction long --entry 6800 --stop 6795 --contracts 1 --thesis "..."
venv\Scripts\python.exe journal\close.py --ticket <id> --exit 6812 --reason target
venv\Scripts\python.exe journal\review.py
```

`pretrade.py` accepts `--now` to override the clock and `--journal` to point at
a different file; both exist for testing and are safe to use.
