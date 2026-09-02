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

4. **Hard max position: 2 contracts.** Net position size, per instrument and in
   aggregate, must never exceed 2 contracts. Order sizing logic must clamp/reject
   any order that would breach this, independent of what a strategy signal
   requests.

5. **Daily loss limit: $300.** Once realized + open P&L for the trading day hits
   a $300 loss, trading must stop for the rest of that day — no new entries. This
   must be enforced by a risk-management/guard component that runs regardless of
   what any individual strategy does.

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
   exceeds 40% of total cumulative profit across the tracked period. This is a
   reporting/monitoring requirement, not just a note — it must be computed and
   surfaced, not left as a manual check.

9. **Risk limits live in code, not comments or config alone.** Rules 2–6 (time
   cutoff, entry cutoff, position cap, daily loss limit, minimum hold time) must
   be enforced by actual runtime logic that can reject/override orders. A comment
   saying "don't exceed 2 contracts" or a config value nobody reads at runtime
   does not satisfy this rule. If a limit is expressed in config, the code must
   actually load and enforce it every time, with no bypass path.

10. **Backtests must include commission and slippage.** No backtest, parameter
    scan, or performance report may run/report figures on a frictionless fill
    model. Every backtest run must apply realistic commission and slippage
    assumptions to every trade.

11. **Backtests must report:**
    - Max daily loss (the worst single day's P&L)
    - Worst day as a percentage of total profit (the same figure used for the
      rule 8 consistency check, but reported as a standard backtest metric)
    - Average trade duration
    - Percentage of total profit from trades held 5 seconds or less (rule 7),
      flagged when above 30%

    These are required outputs of every backtest run, not optional/nice-to-have
    metrics.

## Project scope note

Do not write strategy or execution code without an explicit request — this
repository starts as scaffolding only. When strategy/execution code is added
later, it must comply with all rules above from the first commit.
