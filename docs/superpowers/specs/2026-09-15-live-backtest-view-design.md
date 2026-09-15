# Live backtest view — design

**Date:** 2026-09-15. **Status:** approved by the operator in conversation;
implementation follows test-first.

## Purpose

Let the operator watch a walk-forward while it runs, in a browser page,
without giving anyone a surface that can start, re-run or re-score a
strategy. The log's verdicts are frozen and the project has declined every
optimiser; this view is a window, not a lever.

## What a run can honestly show

The generated runner and entries 11 and 12 generate every signal in one
call, price the whole stream in one call, then score the folds in one call.
The real granularity is about seven stage transitions. The page therefore
shows:

- the stages advancing, with elapsed time per stage;
- the equity curve drawn trade by trade **from the priced stream the moment
  it exists** (an animation of a finished stream, labelled as such);
- the fold table as the runner produces it;
- the outcome banner on the done event.

It does not pretend trades arrive one at a time during pricing.

## Components

### 1. `backtests/live.py` — the event stream

`LiveRun(name, runner)` appends one JSON object per line to
`backtests/results/live/<name>.jsonl` (already gitignored under
`backtests/results`). Events:

| event | fields | when |
|---|---|---|
| `start` | `t`, `run`, `runner`, `pid` | `LiveRun` opened |
| `stage` | `t`, `message` | every progress message |
| `trades` | `t`, `path`, `basis`, `rows` | runner hands over a priced stream |
| `folds` | `t`, `path`, `rows` | runner hands over a fold table |
| `done` | `t`, `status` | runner finished |
| `error` | `t`, `message` | runner raised |

`LiveRun.progress(fallback)` returns a callable that calls `fallback`
(today's `print`) **and** writes a `stage` event, so console output is
unchanged. `LiveRun.trades(df, basis)` and `LiveRun.folds(df)` write a CSV
copy beside the events (`<name>_<basis>_trades.csv`, `<name>_folds.csv`) and
emit the event; the runner's own result files are untouched. `LiveRun` is a
context manager: exit writes `done` (or `error` on exception, re-raised).
A `NullLive` with the same interface is used when `--live` is off, so
runner code has no branches.

Timestamps are ISO-8601 with offset. Writes are append-only, one line per
`write` call, flushed, so a reader never sees a partial line except the one
being written, which it skips until complete.

### 2. Runner hooks

`backtests/run_generated.py`, `backtests/run_entry11.py`,
`backtests/run_entry12.py` gain `--live` (CLI) and a `live=` parameter on
`run` (default `NullLive()`). Each passes `live.progress(...)` where it
passes `progress` today, calls `live.trades(...)` after the guarded streams
exist (standard and comparable), `live.folds(...)` after the fold table,
and wraps the body in `with live:`. Entry 12's `reproduce` gets the same
so the demonstration run is watchable. `bots/runners.run_walkforward_generated`
and the `/walkforward` command pass a `LiveRun` when the environment
variable `FUTURES_LIVE=1` is set, so Discord-launched runs are visible.

No runner's computation, ordering, output files or verdict text changes. A
test asserts the runners' saved outputs are byte-identical with `--live`
on and off on a synthetic stream.

### 3. `bots/liveview.py` — the viewer

FastAPI app served by uvicorn on **port 8790** (the desk feed uses 8787).
Routes, all `GET`:

| route | returns |
|---|---|
| `/` | the page |
| `/runs` | `[{name, runner, started, status}]` from the live directory |
| `/runs/{name}/events?after=N` | events with index ≥ N, plus the next cursor |
| `/runs/{name}/trades?basis=standard` | `[{i, exit_time, net_pnl, cum_pnl}]` |
| `/runs/{name}/folds` | the fold table rows |
| `/health` | `{"ok": true}` |

`name` must match `^[a-z0-9_]{1,64}$` and is resolved only inside the live
directory; anything else is 404. The page is one inline HTML/JS document:
run picker, stage timeline, SVG equity curve animated at a fixed rate when
a `trades` event arrives (with a note that it is a replay of the finished
stream), fold table, outcome banner. Polls `/events` once a second. No CDN,
no cookies, no POST.

### 4. Launch

`.claude/launch.json` gets a `liveview` configuration
(`venv\Scripts\python.exe bots\liveview.py`, port 8790) so the page opens
in the desktop app's browser pane. The handoff's §7 gets the command line.

### 5. Guards

- Every route is `GET`; a test enumerates the app's routes and fails on
  any other method.
- The viewer imports nothing from the runners and cannot spawn a process.
- Live files are output, not evidence: gitignored, regenerable, and named
  by run so a re-run overwrites rather than appends across runs.

## Testing

`tests/test_live.py`:

- `LiveRun` writes `start`, `stage`, `trades`, `folds`, `done` in order; the
  CSV copies exist; `error` on exception and the exception propagates.
- `progress` wrapper calls the fallback and writes the event.
- `NullLive` accepts every call and writes nothing.
- Viewer via the existing `httpx` test client: `/runs` lists a synthetic
  run; `/events?after=` paginates; `/trades` returns cumulative P&L in exit
  order; `/folds` returns rows; a bad name is 404 and never touches the
  filesystem outside the live directory; every route is GET.
- Runner hooks: no test today runs a runner end to end (the entry 11 and
  12 tests cover the calendar and strategy; `run_generated` is tested by
  its pieces), so the hook test substitutes the heavy pieces inside
  `run_generated.run` — bar loading, strategy building, the guarded
  streams, fold scoring, the simulator — with small fakes, and asserts the
  `LiveRun` receives `start`, the stage messages, `trades` for both bases,
  `folds` and `done` in order, and that the runner's own output files are
  byte-identical with `live` on and off. Entries 11 and 12 get the same
  treatment at the `progress` and `reproduce` seams.

## First demonstration

`venv\Scripts\python.exe backtests\run_entry12.py --part A --reproduce --live`
(about a minute, writes no verdict) with the page open in the pane.

## Out of scope, named

- Entry 1's hour-long fold loop in `backtests/walkforward.py`.
- Any control that starts, stops or parameterises a run.

## Addendum, 2026-09-15 — saved verdicts

Built the same day, after the live view landed. The page's picker gains a
"saved verdicts" group listing every registry name; three `GET` routes under
`/saved` serve the catalogue, the verdict's stream (and its `_nohalt`
sibling as the comparable basis) and its fold table. The name-to-file
mapping is `runners.RUNNERS`, the table the bot's `/walkforward` and
`/evalsim` already read, plus the generated runner's fixed naming, so the
page shows the arm and cost level each verdict quotes and nothing else. The
viewer imports that table and calls no runner function, by test. Names whose
files are absent are listed as unavailable. `live.frame_points` and
`live.frame_rows` are the shared readers behind both the live and saved
paths.
