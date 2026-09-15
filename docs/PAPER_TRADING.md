# Paper trading checklist — entry 3

One page for the desk. Every number below is enforced in `strategies/rules.py`
and checked by the journal; nothing here restates a limit, it only says where
each one bites. The gate itself is pre-registered in `research/hypotheses.md`
entry 3 and cannot be relaxed after the fact.

## The gate

Before any evaluation is bought, all three must hold, computed by
`journal/review.py`, never by hand:

1. **60 or more rule-clean paper trades**, each logged *before* it was placed.
2. **Positive expectancy per trade after costs** — $0.50 commission a side
   plus one tick of slippage a side, the same cost model the backtests use.
3. **`eval_sim` pass probability above 50%** on the logged trades.

**Any rule violation resets the clean-trade count to zero.** The journal
detects four: an entry after the 16:20 ET cutoff, a hold under 30 seconds,
size above 5 contracts, a day breaching the $400 daily loss limit. Not a
deduction — a reset. This is the part of the entry that does the work.

## Before the session

- Start the research bot: `.\start_bot.bat`. Its `/read` describes a chart
  with no bias and its **Log long / Log short** buttons write a ticket
  through the same pre-trade check as the command line.
- `/read` uses live bars only if the desk and tunnel are running with the
  TradingView alert pointed at them (`.\start_desk.bat`, paste the printed
  URL into the alert). With the desk down it falls back to the parquet
  history, which ends 2026-08-31, **and says so with the as-of date** — the
  level ladder is then stale and only the bracket sizing is current.
- Have the TradingView paper account open on MES or MNQ. Nothing else is
  allowed and the check rejects other symbols.

## Per trade, in this order

1. **Decide** direction, entry, stop, size. One contract is enough: a
   5-point MES stop is about $30 of risk, so one bad day at that size is
   nowhere near the $400 budget, and the count cares about discipline, not
   P&L.

2. **Log first.** Either the `/read` button, or:

   ```powershell
   venv\Scripts\python.exe journal\pretrade.py --instrument MES --direction long --entry 6800.25 --stop 6795.00 --contracts 1 --thesis "failed breakdown of the overnight low, reclaimed VWAP, target prior-day high"
   ```

   Add `--early-close` on an exchange half-day (13:00 ET close).

   - **ALLOW** prints the ticket id (twelve hex characters) and the earliest
     time you may close. Note both.
   - **BLOCK** prints every failing check and writes nothing. **Do not place
     the trade.** A blocked idea leaves no ticket and cannot be added later.

3. **Place it on TradingView paper.** If the fill differs from the logged
   entry, remember the real price for step 5.

4. **Do not close before the floor.** The ALLOW output names the exact time
   (entry plus 30 seconds). **The journal will not stop you** — `close.py`
   records an early close, prints `RULE 6 VIOLATION`, and `review.py` flags
   it as the reset. The guard is your hand staying off the button.

5. **Close on TradingView, then close the ticket:**

   ```powershell
   venv\Scripts\python.exe journal\close.py --ticket <id> --exit 6812.50 --reason target
   ```

   Reasons: `target`, `stop`, `manual`, `session_end`, `loss_limit_flatten`,
   `other`. Pass `--entry-price <fill>` if the real fill differed from the
   plan, so slippage is recorded rather than hidden. P&L is priced by the
   engine's cost model, so it sits on the backtest scale.

   **This command commits `journal/trades.jsonl` to git by itself.** It is
   the one automatic commit in the project, by design: the outcome history
   is tamper-evident. `--no-commit` skips it if you would rather batch.

6. **Flat by 16:30 ET, no new entries after 16:20.** Rule 2 and rule 3 apply
   to paper trades exactly as they would to live ones; the journal records a
   late entry as a violation.

## End of day and weekly

```powershell
venv\Scripts\python.exe journal\review.py                 # today + rolling + trailing drawdown
venv\Scripts\python.exe journal\review.py --rolling-only  # the gate figures alone
```

The day review lists every trade with its thesis and the three rule checks;
the rolling review carries the count, expectancy, Sharpe and the $50K
evaluation simulation; the drawdown section marks the internal warn and stop
lines against the peak end-of-day balance.

## The thesis line

Required, one line, and the most useful thing in the journal even if the
gate is never met, because trades grouped by their stated reason show which
setups pay. Write something a reviewer could check against the chart:

- Good: `failed breakdown of the overnight low, reclaimed VWAP, target prior-day high`
- Good: `rejected the opening-range high twice, short below VWAP into the 13:00 lull`
- Not good: `looks bullish`, `momentum`, `gut feel`

A trade you cannot describe in one line contributes nothing but variance.

## What does not count

- A trade placed without a logged ticket does not exist for the count and
  cannot be added afterwards.
- A ticket logged for a trade never taken is a fabrication of the record the
  gate is supposed to measure. Log what you do; do what you log.
- Desk-bot tickets and their Execute / Don't trade presses are a separate
  journal and are reported, not counted.
- Backtest results, `/read` descriptions and shadow tickets are not trades.

## Where the numbers live

| Limit | Value | Enforced by |
|---|---|---|
| Entry cutoff | 16:20 ET | `pretrade.py` blocks; journal flags |
| Force flatten | 16:30 ET | your close; journal flags a hold past it |
| Minimum hold | 30 s | your hand; `close.py` records and resets |
| Position cap | 5 contracts | `pretrade.py` blocks |
| Daily loss | $400 | `pretrade.py` shows budget left; journal flags a breach |
| Trailing drawdown | warn $1,000, stop $1,500 | `review.py` marks the days |

All from `strategies/rules.py`. If a number here ever disagrees with the
code, the code is right and this page is stale.
