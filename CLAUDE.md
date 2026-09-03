# CLAUDE.md — Intraday CME Futures Trading Bot

This file defines hard rules for this project. They apply to all code written here —
strategy code, backtests, and execution/live-trading code alike. Claude (and any
contributor) must follow these exactly. Where a request would violate one of these
rules, flag the conflict instead of silently working around it.

## Hard rules

1. **Instruments: MES and MNQ only.** CME Micro E-mini S&P 500 (MES) and Micro
   E-mini Nasdaq-100 (MNQ). No other symbols, no full-size contracts, no other
   asset classes. Any code that accepts a symbol/instrument parameter must
   validate it against an allowlist of `{"MES", "MNQ"}` and reject anything else.

2. **Intraday only — force-close by 4:30 PM ET, no exceptions.** No position may
   be held overnight. Any position still open must be force-flattened by 4:30 PM
   ET. This must be enforced by a scheduled/monitored process in the execution
   layer (e.g., a hard time check that fires a market-flatten order), not by a
   strategy merely "trying" to exit before then.

3. **No new entries after 4:20 PM ET.** The entry window closes 10 minutes before
   the forced flatten. No new position may be opened after 4:20 PM ET, and no
   existing position may be added to. This is enforced by the same guard layer as
   rule 2 — a strategy signal arriving after the cutoff is rejected, not deferred.

   This cutoff exists so that the minimum-hold rule (rule 6) and the forced
   flatten (rule 2) can never conflict: every position is opened with at least 10
   minutes of runway before the flatten, so no trade can be caught needing to
   close before its 30-second floor has elapsed.

4. **Hard max position: 5 micro contracts.** Net position size, per instrument
   and in aggregate, must never exceed 5 contracts. Order sizing logic must
   clamp/reject any order that would breach this, independent of what a strategy
   signal requests. The firm allows 40; we use 5.

5. **Daily loss limit: $400.** Once realized + open P&L for the trading day hits
   a $400 loss, trading must stop for the rest of that day — no new entries, and
   any open position is flattened. Measured on equity, marked to market each bar,
   not on realized P&L alone. This must be enforced by a risk-management/guard
   component that runs regardless of what any individual strategy does. The
   firm's line is $1,200.

5b. **End-of-day trailing drawdown: warn at $1,000, stop at $1,500.** The account
   carries a drawdown limit that trails the highest *end-of-day* balance ever
   reached — intraday spikes do not raise it, only a close does. Trading stops
   entirely at $1,500 below the peak EOD balance. The firm terminates the account
   at $2,000, so the internal stop leaves $500 of headroom.

   Modelled as a pure trail with no lock-in. If the firm freezes the floor once
   it reaches the starting balance, this is the conservative reading.

6. **Minimum trade duration: 30 seconds.** No position may be closed less than
   30 seconds after it was opened. Exit logic must check elapsed holding time and
   block/delay closes that would violate this.

   The 30-second floor is a deliberate safety buffer, not the prop firm's actual
   threshold — the firm's microscalping test is the portfolio-level measure in
   rule 7, which keys on trades held 5 seconds or less. The floor is set well
   above 5 seconds so that no individual trade can contribute to that measure at
   all.

   Rule 3's entry cutoff removes the normal-operation conflict between this rule
   and the forced flatten. As an absolute backstop, the rule 2 flatten and a rule
   5 daily-loss breach still take precedence over this floor if they somehow
   coincide with a position younger than 30 seconds — but if that ever fires in
   practice it indicates a bug in the entry guard and must be investigated, not
   accepted as normal.

7. **Microscalping check (portfolio level).** Compute the percentage of total
   profit that came from trades held 5 seconds or less, and flag it if it exceeds
   **30%**.

   The prop firm flags microscalping when more than **50%** of total profit comes
   from trades of 5 seconds or less. The 30% threshold here is an intentionally
   stricter internal warning line, so the account never approaches the firm's
   limit unnoticed.

   Given rule 6's 30-second floor, this figure should be 0% in any correct run.
   A non-zero result is therefore also a regression signal that the minimum-hold
   enforcement is not working — treat it as a bug, not just a risk warning.

8. **Consistency check.** Flag (log/alert) if any single trading day's profit
   exceeds **30%** of total cumulative profit across the tracked period. This is a
   reporting/monitoring requirement, not just a note — it must be computed and
   surfaced, not left as a manual check. The firm's line is 40% on funded
   accounts; 30% is the internal warning.

9. **Risk limits live in code, not comments or config alone.** Rules 2–6 (time
   cutoff, entry cutoff, position cap, daily loss limit, trailing drawdown,
   minimum hold time) must be enforced by actual runtime logic that can
   reject/override orders. A comment
   saying "don't exceed 2 contracts" or a config value nobody reads at runtime
   does not satisfy this rule. If a limit is expressed in config, the code must
   actually load and enforce it every time, with no bypass path.

10. **Backtests must include commission and slippage.** No backtest, parameter
    scan, or performance report may run/report figures on a frictionless fill
    model. Every backtest run must apply realistic commission and slippage
    assumptions to every trade.

11. **Backtests must report:**
    - Max daily loss (the worst single day's P&L)
    - Worst day as a percentage of total profit — the largest losing day
      measured against total profit. This is a drawdown-shape metric and is
      **not** the rule 8 consistency check, which measures the *best* day.
      Both are reported; they are different figures and must not be conflated.
    - Best day as a percentage of total profit (the rule 8 consistency check)
    - Average trade duration
    - Percentage of total profit from trades held 5 seconds or less (rule 7),
      flagged when above 30%

    These are required outputs of every backtest run, not optional/nice-to-have
    metrics.

12. **Every strategy gets a hypothesis entry before any code is written.**
    Add the entry to `research/hypotheses.md` first, stating the mechanism and
    naming who is on the other side of the trade. An idea that cannot name its
    counterparty is a pattern, not a hypothesis. The verdict is filled in after
    walk-forward evaluation and is not revised afterwards — a rejected entry
    stays in the log so the same idea cannot be quietly re-tested until it
    passes.

13. **Evidence standard: walk-forward or nothing.** A strategy is accepted only
    if it is profitable in a majority of yearly walk-forward folds, has positive
    total walk-forward P&L after commission and slippage, and survives at 2
    ticks of slippage per side. In-sample results are never evidence, and a
    single in-sample/out-of-sample split is not enough — ORB passed a favourable
    two-year window at +$2,250 and lost $6,073 across seven folds.

## The prop firm account

The target account is a **Lucid 50K Pro evaluation**: $50,000 nominal, $3,000
profit target.

Two sets of numbers exist in this project and must never be merged. They live in
`strategies/rules.py` as `FIRM` and `INTERNAL`, and the guards read `INTERNAL`
only. If code ever stops at a firm number, an internal guard above it has failed.

| Limit | Internal (enforced) | Firm (account ends) | Headroom |
|---|---|---|---|
| Daily loss | $400 | $1,200 | $800 |
| EOD trailing drawdown | warn $1,000, stop $1,500 | $2,000 | $500 |
| Max micro contracts | 5 | 40 | 35 |
| Consistency (single day as % of profit) | 30% | 40% | 10 points |
| Microscalping (% profit from 5s-or-less trades) | 30% | 50% | 20 points |
| Profit target | — | $3,000 | — |

The buffer is the product, not a rounding artefact. A limit hit exactly is a
limit eventually crossed, and an evaluation is lost once, permanently. Any change
that narrows a gap in this table must be deliberate and stated as such —
`rules.buffer_report()` prints it, and `tests/test_rules.py` asserts that every
internal value is strictly tighter than its firm counterpart.

## Project scope note

Do not write strategy or execution code without an explicit request — this
repository starts as scaffolding only. When strategy/execution code is added
later, it must comply with all rules above from the first commit.
