# Strategy hypothesis log

One entry per strategy idea. **The entry is written before any code**, with the
mechanism stated up front — why the edge should exist, and who is on the other
side of the trade. An idea that cannot name its counterparty is a pattern, not a
hypothesis, and patterns are what overfitting is made of.

The verdict is filled in after walk-forward evaluation and is not revised
afterwards. A rejected entry stays in the log with its verdict intact; the point
of the log is to make it expensive to quietly re-test the same idea until it
passes.

## Standard of evidence

An idea is **accepted** only if, on walk-forward with yearly folds
(`backtests/walkforward.py`):

- it is profitable in a majority of folds, **and**
- total walk-forward P&L is positive after commission and slippage, **and**
- it survives at 2 ticks of slippage per side.

Anything else is **rejected**. In-sample results are never evidence.

---

## 1. Opening Range Breakout (ORB) — REJECTED

**Date:** 2026-09-03
**Commit:** `b8e9411` (walk-forward framework and results)
**Code:** `strategies/orb.py`
**Instrument:** MES, 5-minute bars, RTH only

### Mechanism claimed

The first 15 minutes of the RTH session establish a reference range while
overnight news is absorbed and the opening auction imbalance clears. A decisive
break of that range was claimed to indicate that one side of the order book has
been exhausted, with continuation as remaining liquidity is taken.

**Who is on the other side:** the claim was that fading the break is done by
mean-reversion traders and overnight position-holders covering, who are slower
to reprice than the breakout flow. This was never established independently —
it was assumed from the pattern's popularity rather than from any measurement of
order flow. In hindsight this is the entry's weakest point, and it should have
been treated as a warning before any code was written.

### Parameters tested

| Parameter | Grid |
|---|---|
| Opening range | 5, 10, 15, 30 minutes |
| Trading window end | 10:30, 11:30, 12:30 ET |
| Stop | 0.5, 0.75, 1.0, 1.5 × opening-range height |
| Target | 1.0, 1.5, 2.0, 3.0 × stop distance |

192 combinations. Defaults: 15m / 11:30 / 1.0 / 2.0.

### Walk-forward verdict: REJECTED

Seven yearly folds, 2020–2026, parameters chosen on prior years only, 1 tick
slippage per side and $1.25/side commission:

| Test year | Chosen params | Train Sharpe | OOS P&L | OOS Sharpe | OOS PF |
|---|---|---|---|---|---|
| 2020 | 10m/10:30/1.0/3.0 | 0.48 | −$3,685 | −2.18 | 0.71 |
| 2021 | 30m/11:30/0.5/3.0 | −0.78 | −$1,868 | −2.13 | 0.74 |
| 2022 | 5m/10:30/0.75/3.0 | −0.24 | −$1,690 | −1.02 | 0.87 |
| 2023 | 30m/10:30/1.0/3.0 | 0.20 | +$484 | 0.49 | 1.08 |
| 2024 | 30m/10:30/1.5/1.5 | 0.37 | +$90 | 0.07 | 1.01 |
| 2025 | 30m/10:30/1.5/1.5 | 0.31 | +$1,167 | 0.65 | 1.11 |
| 2026 | 30m/10:30/1.5/2.0 | 0.44 | −$571 | −0.58 | 0.91 |

**Total: −$6,072.81 over 1,409 trades. 3 of 7 folds profitable. Median fold
Sharpe −0.585.**

Fails the first two acceptance criteria outright; the 2-tick check was not run
because it can only make a losing result worse.

### What was learned

**A favourable window flattered it badly.** The first backtest ran 2024-09 to
2026-08 and returned +$2,250. That window happens to contain the two best folds
(2025 at +$1,167 and 2024 at +$90) and none of 2020–2022, which lost $7,243
between them. The two-year number was a favourable draw, not a small sample of a
positive edge.

**Parameter ranking transfers, and it does not help.** Pooled Spearman
correlation between training and test Sharpe was **+0.398**, positive in all
seven folds (+0.11 to +0.77). Selection is not the failure — a chosen set lands
at the 55th percentile of its own test year, marginally better than the median.
The strategy is simply unprofitable, and reliably ranking among losers still
returns a loser. Worth recording because the earlier single-split scan reported
a rank correlation of −0.005 and the conclusion drawn then, that in-sample
ranking carries no information, was itself a small-sample artifact.

**Costs dominate.** At the defaults over 2024–2026, gross P&L was $5,300 against
$3,050 of commission and slippage: friction consumed 58% of gross. A strategy
whose best in-sample profit factor is ~1.3 has no room for an extra tick.

### Next

Do not re-test ORB with a wider grid or different exits. The grid was already
192 combinations across four parameters; widening it searches harder for noise.
Any future breakout idea must first establish the counterparty claim
independently of backtest results.

### Addendum, 2026-09-03 — measured against the Lucid 50K Pro eval

Added after the verdict; the verdict is unchanged. This quantifies what
"rejected" costs in account terms, under the real firm limits (daily loss now
$400 internal, and an end-of-day trailing drawdown terminating at $2,000).

Re-run under the $400 daily limit, the walk-forward total moved from −$6,072.81
to **−$5,964.06** over the same 1,409 trades, still 3 of 7 folds profitable.
The looser daily stop changed almost nothing, which is consistent with ORB
rarely having a second trade left to block.

**The trailing drawdown is where it actually dies.**

| | Baseline (default params, 2019–2026) | Walk-forward OOS stream |
|---|---|---|
| Trading days | 1,821 | 1,296 |
| Worst drawdown from peak | $6,483.75 | $8,339.38 |
| $50K evaluations blown | **5** | **4** |
| Evaluations passed | 1 | 1 |

Against a $2,000 trailing line, ORB destroys an account roughly once a year.
The walk-forward stream's first death is 2020-03-18, 46 trading days in.

Monte Carlo on the baseline daily P&L distribution (`backtests/eval_sim.py`,
20,000 paths, 250-day horizon):

- **Pass probability 7.67%** (95% CI 7.30–8.04%)
- Blow-up 57.60%, ran out of time 34.73%
- **Expected 13.04 attempts to pass once — about $1,499 at $115 an attempt**

The mean day is −$1.50 against a standard deviation of $120.50. That is the
whole story in two numbers: a distribution centred fractionally below zero,
with enough daily variance to walk into a $2,000 trailing line long before it
walks into a $3,000 target. Paying $1,499 in expectation to win a funded
account that would then be traded with a negative-expectancy strategy is worse
than not entering.

Recorded because "unprofitable" and "uninsurable" are different failures, and
the second is the one that matters for a prop account. A −$5,964 result over
seven years reads as a slow bleed; on a trailing-drawdown account it is four
dead evaluations.

### Addendum, 2026-09-11 — re-priced at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** Lucid support confirmed on 2026-09-11 that MES commission on a 50K Pro
evaluation is **$0.50 a side, $1.00 a round turn** (support.lucidtrading.com
article 11508978), not the $1.25 a side this entry and every later one
assumed. The verdict was judged at its pre-registered base case and stands;
this records what the same trades are worth at the real rate.

**Method.** The saved out-of-sample stream (`orb_oos_stream_slip1.csv`, the
$400-daily-limit re-run above, 1,409 trades at 1 contract and 1 tick) was
re-priced with `backtests/reprice.py`: commission never touches a fill, so
each trade moves by exactly $1.50. **The walk-forward's parameter choice per
fold is held at what $1.25 selected**; a re-selection at $0.50 might have
chosen differently and is not attempted. One trade in 1,409 is a daily-loss
flatten that fired at $1.25 and stays as fired. Slippage is unchanged.

| Walk-forward OOS stream, 1 tick | As scored, $1.25 | Re-priced, $0.50 |
|---|---|---|
| Net P&L | −$5,964.06 | **−$3,850.56** |
| Mean per trade | −$4.23 | −$2.73 |
| Folds profitable | 3 of 7 | **3 of 7** |
| Median fold Sharpe | −0.774 | −0.631 |
| Profit factor | 0.908 | 0.939 |
| Max drawdown | −$8,339 | −$7,078 |
| Pass probability | 3.33% | 5.12% |
| Payout probability ($52,100) | 10.84% | 14.75% |
| Evaluations blown | 4 | 4 |

Per fold: 2020 −$3,475 → −$3,079; 2021 −$1,868 → −$1,507; 2022 −$1,690 →
−$1,218; 2023 +$484 → +$724; 2024 +$90 → +$350; 2025 +$1,273 → +$1,513;
2026 −$778 → −$634. The same three folds are profitable.

**Would any acceptance criterion have resolved differently? No.** Rule 13
asks for a majority of folds profitable and a positive total: 3 of 7 and
−$3,851. The commission correction is worth $2,113 over seven years and moves
neither. The 2-tick survival check was not run in the verdict and is not
needed here. This entry states no bracket break-even, so none is restated.

---

## 2. Leveraged ETF end-of-day rebalance drift — REJECTED

**Date:** 2026-09-03
**Spec frozen at:** `43aa4e1` (entry written before any code existed)
**Verdict commit:** `a788e2b`
**Code:** `strategies/eod_rebalance.py`
**Instrument:** MES, 5-minute bars, RTH only

### Mechanism claimed

Leveraged and inverse ETFs hold a constant target leverage against their net
assets. Because their exposure is reset daily, any move in the underlying index
changes their effective leverage and forces a trade to restore it. A 3× long
fund that gains on an up day is left under-levered and must buy; on a down day
it is over-levered and must sell. Inverse funds trade the same direction, for
the mirrored reason. The required notional scales with both fund AUM and the
size of the day's move, and the trade is concentrated in the last ~30 minutes so
the fund can mark against the official close.

The flow is therefore **mechanical, direction-predictable from information
already public at 3:30, and price-insensitive**.

**Who is on the other side:** the fund itself is a forced, price-insensitive
trader — it must complete the rebalance regardless of the price it receives,
because tracking error against its stated leverage is the one thing it cannot
accept. The economic cost is borne by the fund's own shareholders as tracking
drag, which they accept as the price of a daily-reset leveraged product. The
immediate counterparty is whoever supplies liquidity into that imbalance, and
they demand compensation in the form of price impact. The claimed edge is
anticipating that impact.

This is a materially stronger counterparty claim than ORB's, because the
constraint is contractual rather than behavioural: the fund is not choosing to
trade badly, it is mandated to trade at a specific time in a specific direction
regardless of price. That is exactly the "forced flow" property ORB lacked.

### Prediction

On days where the MES return from the RTH open to 3:30 PM ET exceeds a
threshold in magnitude, the 3:35→4:00 PM return continues in the same direction
more often than chance, with effect size increasing in the magnitude of the
day's move.

### Pre-registered test

Fixed before any data is touched. Nothing below may be changed after results are
seen; a change means a new entry, not an edit to this one.

**Signal.** Return from the RTH open (open of the 09:30 bar) to the price as of
15:30:00 ET — the close of the 5-minute bar labelled 15:25, since bars are
labelled by opening minute and closed left. Using the bar labelled 15:30 would
mean measuring through 15:34:59, which is a different (and later) quantity.

**Entry.** Open of the bar labelled 15:35, in the direction of the signal, only
when `|signal| >= threshold`. One trade per session, at most.

**Exit.** Close of the bar labelled 15:55 (15:59:59, the RTH close), or the stop
if hit first.

**Grid.** 4 thresholds × 3 stop settings = **12 combinations**.

| Parameter | Values |
|---|---|
| Threshold on \|open→15:30\| return | 0.5%, 0.75%, 1.0%, 1.5% |
| Stop distance from entry | 0.25%, 0.5%, none (time exit only) |

The grid is deliberately 16× smaller than ORB's 192. Fewer knobs is less surface
for noise to be fitted to.

**Evaluation.** `backtests/walkforward.py` unchanged — seven yearly folds,
2020–2026, expanding training window, parameters chosen on prior years only,
$1.25/side commission and 1 tick/side slippage, then repeated at 2 ticks.

**Effect-size measurement, defined now so it cannot be defined to taste later.**
Separately from trading P&L, and measured on the pooled **out-of-sample** test
windows only:

```
E(θ) = mean[ sign(r_open→15:30) × r_15:35→16:00 ]  over days with |r_open→15:30| ≥ θ
H(θ) = fraction of those days where the two returns share a sign
```

`E` in basis points, before costs. "Effect size increases with threshold" means
E(θ) is non-decreasing across the four thresholds — operationally, Spearman
correlation between θ and E(θ) is positive **and** E(1.5%) > E(0.5%). `H(θ)` is
reported against the 50% null.

Splitting the statistical question (does the drift exist?) from the trading
question (does it survive costs?) matters here: the drift could be real and
still untradeable, and those two outcomes call for different next steps.

### Kill criteria — decided now

Any **one** of these kills the hypothesis:

1. Fewer than 4 of 7 folds profitable out-of-sample.
2. Median fold out-of-sample Sharpe below 0.3.
3. Effect size does not increase with threshold, as defined above.

No appeal, no re-grid, no "but with a different exit". A kill is recorded here
and the idea is closed.

### Prior expectations, recorded before results

**Decay.** The rebalance effect was documented publicly by around 2010 and is
now well known. The expectation is that it has been substantially arbitraged
away since roughly 2015, so **early folds should be stronger than late folds**.
If late folds are stronger, that is a flag to investigate for a bug or a
confound — not a result to celebrate.

**Two problems with testing that expectation on this data, worth stating up
front:**

*The sample is entirely post-decay.* MES launched 2019-05-06, so the earliest
fold tests 2020. If the effect decayed by ~2015, every fold sits in the decayed
regime and the base rate may already be near zero. This test can measure whether
anything remains; it cannot observe the effect in its documented era. A clean
decay curve would need ES rather than MES, which is a different (and more
expensive) data pull.

*2020 confounds decay with volatility.* The earliest fold contains the March 2020
crash. Far more days clear every threshold in a high-volatility year, and larger
moves mean larger rebalance notionals. So "early folds stronger" is exactly what
a pure volatility effect would also produce, with no decay involved. Any
early-vs-late reading must be checked against per-fold realised volatility and
trade counts before being attributed to decay.

**Scale scepticism.** Leveraged S&P 500 ETF AUM is small relative to ES/MES
daily notional volume. The forced flow is real, but it may be small enough that
its price impact is inside the bid-ask spread. Costs here are $5.00 per round
turn against a 25-minute holding period.

### Sample-size guard

At the 1.5% threshold, qualifying days will be rare — possibly single digits per
test year. That is precisely where the "effect increases with threshold" test is
weakest, and a strong-looking E(1.5%) on eight trades is not evidence.

Pre-registered: report n per threshold per fold; any threshold with fewer than
**30 pooled out-of-sample trades** is reported as *insufficient evidence* and
counts as neither a pass nor a fail of criterion 3.

**First step after approval, before any strategy code:** count how many sessions
clear each threshold per year. This looks only at the signal distribution, never
at the 15:35→16:00 outcome, so it is a power check rather than peeking. If the
top thresholds cannot reach ~30 out-of-sample trades, the grid should be revised
*now*, before any outcome has been observed.

### Power check result (run before any strategy code)

`research/power_check_eod.py`, 2026-09-03. Signal distribution only. 1,810 of
1,890 sessions are eligible after excluding 64 early closes (no 15:25 bar), 29
roll days, and any session missing the required bars.

| Year | Sessions | ≥0.50% | ≥0.75% | ≥1.00% | ≥1.50% |
|---|---|---|---|---|---|
| 2019 *(train only)* | 164 | 48 | 28 | 15 | 3 |
| 2020 | 250 | 122 | 95 | 63 | 35 |
| 2021 | 251 | 85 | 50 | 24 | 7 |
| 2022 | 248 | 165 | 129 | 97 | 44 |
| 2023 | 244 | 112 | 69 | 33 | 8 |
| 2024 | 246 | 74 | 40 | 19 | 5 |
| 2025 | 243 | 114 | 57 | 35 | 16 |
| 2026 | 164 | 64 | 30 | 16 | 6 |
| **Pooled OOS (2020-2026)** | **1,646** | **736** | **470** | **287** | **121** |

All four thresholds clear the 30-trade pooled floor, so criterion 3 is testable
across the whole grid and the grid stands unrevised.

**Selection minimum, fixed here.** Per-fold counts at the 1.5% threshold are
thin (min 5, median 8 per test year) even though the pooled total is adequate.
ORB's 100-trade eligibility minimum is unreachable for a strategy taking at most
one trade a session — the 2020 fold trains on 2019 alone, which has 48 qualifying
days at 0.5% and 3 at 1.5% — and an unreachable filter silently falls back to
selecting from the unfiltered pool. Selection therefore requires **30 training
trades**, and any fold whose chosen set produces fewer than **20 test trades**
has its Sharpe reported as low-confidence. This is an implementation detail the
spec above left open; it is fixed now, before any outcome has been observed, and
it does not alter the kill criteria.

### Rule compatibility

Entry 15:35 is inside the 16:20 cutoff; exit 15:59 precedes the 16:30 forced
flatten; the ~24-minute hold clears the 30-second floor with room to spare. A
0.5% stop on MES near 6,800 is roughly 34 points, about $170 on one contract, so
the $300 daily loss limit should bind only on a gap through the stop. With at
most one trade per session, the limit has little to cut — the same structural
reason it barely bound for ORB.

### Walk-forward verdict: REJECTED

Seven yearly folds, 2020–2026, 12 parameter sets, selection on prior years only,
$1.25/side commission and 1 tick/side slippage.

| Test year | Chosen (thr/stop) | Train Shp | OOS P&L | OOS Shp | OOS PF | OOS n |
|---|---|---|---|---|---|---|
| 2020 | 0.50% / 0.50% | −3.26 | +$114 | 0.20 | 1.04 | 122 |
| 2021 | 1.00% / 0.25% | 1.14 | −$746 | −11.82 | 0.23 | 24 |
| 2022 | 1.00% / 0.50% | −0.52 | +$102 | 0.24 | 1.04 | 97 |
| 2023 | 1.50% / 0.25% | 0.27 | −$109 | −5.39 | 0.44 | 8 |
| 2024 | 1.50% / 0.25% | −0.01 | +$43 | 2.85 | 1.59 | 5 |
| 2025 | 1.50% / 0.25% | 0.09 | +$118 | 0.94 | 1.17 | 16 |
| 2026 | 1.50% / 0.25% | 0.27 | −$170 | −6.48 | 0.37 | 6 |

**Total: −$648.05 over 278 trades.**

#### Kill criteria

| # | Criterion | Result | |
|---|---|---|---|
| 1 | ≥ 4 of 7 folds profitable | 4 of 7 | **PASS** |
| 2 | Median fold OOS Sharpe ≥ 0.30 | +0.201 | **FAIL** |
| 3 | Effect size increases with threshold | Spearman −1.000 | **FAIL** |

Two of three failed. **Rejected.**

#### Pooled out-of-sample effect size

Raw returns before costs, 2020–2026. All four thresholds cleared the 30-trade
floor, so criterion 3 was fully testable.

| Threshold | n | E (bps) | H (%) | t-stat |
|---|---|---|---|---|
| 0.50% | 736 | −0.64 | 48.1 | −0.39 |
| 0.75% | 470 | −1.37 | 49.6 | −0.56 |
| 1.00% | 287 | −1.59 | 50.9 | −0.42 |
| 1.50% | 121 | −4.11 | 54.5 | −0.50 |

The drift is **absent, and what little sign there is runs backwards**. E is
negative at every threshold and becomes *more* negative as the threshold rises —
a perfect −1.000 rank correlation, the exact opposite of the prediction. No
t-statistic exceeds 0.6 in magnitude, so the honest reading is that E is
indistinguishable from zero everywhere and the monotonic ordering is itself
noise.

The hit rate H tells the same story from another angle: it rises with threshold
(48.1% → 54.5%) while E falls. Direction continues slightly more often on big
days, but the continuations are smaller than the reversals. A strategy can be
right more than half the time and still lose, and here it does.

#### Early vs late, volatility-normalised

| Year | n | E (bps) | vol (bps) | E/vol | Fold P&L |
|---|---|---|---|---|---|
| 2020 | 122 | −2.19 | 69.41 | −0.032 | +$114 |
| 2021 | 85 | −2.38 | 22.16 | −0.107 | −$746 |
| 2022 | 165 | +3.11 | 33.13 | +0.094 | +$102 |
| 2023 | 112 | +1.86 | 19.02 | +0.098 | −$109 |
| 2024 | 74 | −2.47 | 19.23 | −0.128 | +$43 |
| 2025 | 114 | −3.27 | 23.67 | −0.138 | +$118 |
| 2026 | 64 | −2.62 | 14.98 | −0.175 | −$170 |

Early folds (2020–2022) mean E/vol −0.0150; late folds (2023–2026) −0.0859.
Early exceeds late, which is the direction the pre-registered decay expectation
predicted — so no investigation flag is raised. But this must not be read as
confirmation: both halves are negative, the year-to-year sign flips twice, and
the effect being compared is not statistically distinguishable from zero in
either half. A decay from "no effect" to "no effect" is not evidence of decay.
The 2020 volatility confound recorded in advance did not need to be untangled,
because there was no positive effect in 2020 to attribute to anything.

### What was learned

**The mechanism is real; the tradeable residue is not.** Nothing here disputes
that leveraged funds must rebalance, or that the flow is mechanical and
direction-predictable. What the data says is that by 2020 the flow was already
fully absorbed at the five-minute horizon — which is exactly what the
pre-registered scepticism predicted for a sample that begins entirely
post-decay. Recording that expectation in advance is what makes this a clean
negative result rather than a puzzle.

**Selection actively hurt, and the fold table shows why.** Pooled rank
correlation between training and test Sharpe was **−0.457**, negative in five of
seven folds, and the chosen set landed at the **39th percentile** of its own test
year — worse than picking at random. From 2023 onward selection locked onto
threshold 1.50% / stop 0.25%, which produced test-year samples of 8, 5, 16 and 6
trades. Four of the seven folds are flagged low-confidence for this reason. A
fold Sharpe of +2.85 on five trades (2024) is not a measurement, and the median
fold Sharpe that criterion 2 turns on is partly built from such numbers. The
pre-registered 20-trade low-confidence flag did its job: it made this visible
rather than letting +2.85 read as a success.

**Criterion 1 passed and should be distrusted.** 4 of 7 folds profitable looks
like a near miss. It is not: total P&L is negative, the profitable folds are
small (+$114, +$102, +$43, +$118) and the losing ones are large (−$746, −$170,
−$109). Counting folds ignores magnitude, and a majority of small wins against a
minority of large losses is a losing strategy. Worth noting as a weakness in the
criterion itself for future entries — a fold-count test should be paired with a
magnitude test.

### Next

Do not re-test with a longer horizon, a different entry time, or ES data unless
a *new* entry is written first. The tempting next move — "the flow is real, so
try 15:50 entry or a 3-minute horizon" — is precisely the search that turns a
clean negative into a fitted positive. The effect is negative at every threshold
tested; there is no seam here to widen.

If the idea is ever revisited, the one thing that would justify it is
independent evidence of the flow's *size* relative to ES volume on a given day —
measuring the cause directly rather than inferring it from price. That is a data
problem (fund AUM and daily creation/redemption), not a backtest problem.

### Addendum, 2026-09-11 — re-priced at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 assumed (Lucid support,
2026-09-11, article 11508978; see entry 1's addendum of the same date for the
method). The saved stream `eod_oos_stream_slip1.csv` — 278 trades, 1
contract, 1 tick — was re-priced exactly, each trade moving by $1.50. **The
walk-forward's threshold and stop choice per fold is held at what $1.25
selected.** No daily-loss exit is in the stream.

| Walk-forward OOS stream, 1 tick | As scored, $1.25 | Re-priced, $0.50 |
|---|---|---|
| Net P&L | −$648.05 | **−$231.05** |
| Mean per trade | −$2.33 | −$0.83 |
| Folds profitable | 4 of 7 | 4 of 7 |
| **Median fold Sharpe** | **+0.201** | **+0.523** |
| Profit factor | 0.918 | 0.970 |
| Max drawdown | −$1,397 | −$1,206 |
| Pass probability | 0.24% | 0.66% |
| Payout probability ($52,100) | 2.51% | 4.70% |
| Evaluations blown | 0 | 0 |

Per fold: 2020 +$114 → +$297; 2021 −$746 → −$710; 2022 +$102 → +$248; 2023
−$109 → −$97; 2024 +$43 → +$50; 2025 +$118 → +$142; 2026 −$170 → −$161. The
as-scored median of +0.201 reproduces the verdict's figure exactly, which is
the check on the method.

**One kill criterion would have resolved differently.** Criterion 2 — median
fold out-of-sample Sharpe at or above 0.30 — failed at +0.201 and **would have
passed at +0.523**. With trades this small, $1.50 a round turn is most of the
per-trade expectancy, and the fold Sharpes move with it. Criterion 1 (4 of 7
folds) passed on both. **Criterion 3 — the effect size rising with the
threshold — is measured on raw returns before costs and does not move:
Spearman −1.000, the exact opposite of the prediction.** Any one criterion
kills, so REJECTED stands, and it stands on the criterion that tests the
mechanism rather than the one that tests the cost model. That is the right
one to have died on: at the real commission this entry was a clean negative
on its mechanism and an accident of costs on its Sharpe. This entry states no
bracket break-even.

---

## 3. Manual discretionary trading — PROPOSED

**Date:** 2026-09-03
**Spec frozen at:** the commit adding this entry
**Code:** `journal/pretrade.py`, `journal/close.py`, `journal/review.py`
**Instrument:** MES / MNQ, manual entry on TradingView paper

### Mechanism claimed

A human reading order flow, levels and context in real time makes decisions a
fixed rule set cannot express: which failed breakout is worth fading today,
when a range is genuinely exhausted, when to stand aside entirely. The claim is
that this judgement has positive expectancy after costs.

**Who is on the other side:** this is the entry's weak point and it is recorded
as such, before any trades. ORB was rejected partly because its counterparty
claim was assumed rather than established, and "I am a better reader of the
tape than whoever takes my fill" is exactly that same assumption in a more
flattering form. The honest position is that no counterparty has been
identified. Discretion is not a mechanism; it is a container that might hold
one.

This entry therefore does not test whether discretionary trading works in
general. It tests one narrower and answerable question: **does this operator,
following these rules, produce a positive-expectancy record over enough trades
to distinguish it from noise?** A pass licenses buying one evaluation. It does
not establish an edge, and the entry should not later be read as though it had.

### Pre-registered gate

Fixed now. Nothing below may be relaxed after seeing results; a change gets a
dated addendum, never an edit.

Before any Lucid evaluation is purchased, **all three** must hold:

1. **60 or more rule-clean paper trades.** Logged through
   `journal/pretrade.py`, closed through `journal/close.py`. A trade placed
   without a logged ticket does not exist for this count and cannot be added
   afterwards.
2. **Positive expectancy per trade** after commission and 1 tick of slippage
   per side — the same cost model the backtests use, applied by the same
   function.
3. **Simulator pass probability above 50%**, from `backtests/eval_sim.py` run
   on the logged trades, against the firm's $3,000 target and $2,000
   end-of-day trailing drawdown.

### Kill criterion

**Any rule violation resets the clean-trade count to zero.** Not a warning, not
a deduction — a reset. Detected violations are: an entry after the session's
cutoff, a hold under 30 seconds, size above the 5-contract internal cap, and a
day breaching the $400 internal daily loss limit.

The reset is deliberately harsh, and it is the only part of this entry that
does real work. The gate's other three conditions are measurements; this one is
the discipline. A rule broken once under pressure is a rule that will be broken
again at size, and the count exists to make that expensive now rather than
later.

`journal/review.py` computes and displays all four figures. It is not a manual
check.

### Recorded expectations, before any trades

**The base rate is poor and the sample will be small.** 60 trades is enough to
notice a large effect and nowhere near enough to confirm a small one. A
positive expectancy over 60 trades is consistent with a genuine edge and also
consistent with an ordinary run of luck. Passing this gate is permission to
risk one $115 attempt, not evidence of a strategy.

**Paper is easier than live.** Fills are optimistic, and the psychological cost
of a real drawdown is absent. Whatever expectancy appears here should be
expected to degrade with money at stake.

**The thesis log is the part most likely to be useful.** Even if the gate is
never passed, grouping outcomes by the stated reasoning may show that some
setups pay and others reliably do not. That is a finding worth having
independently of the verdict, and it is the reason `pretrade.py` blocks a trade
with no thesis: a trade you cannot describe in one line cannot be reviewed
later, so it contributes nothing but P&L variance.

### Verdict

Not yet run. To be filled in when the gate is either met or reset, with the
commit hash recorded in a follow-up commit.

---

## 4. ORB-2, fixed-bracket trend-filtered opening-range break — REJECTED

**Date:** 2026-09-03
**Spec frozen at:** the commit adding this entry
**Code:** not yet written
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** MES, 15-minute opening range, 1-minute bars for fills, RTH only
**Provenance:** transcribed from a Pine Script supplied to the operator. See
*Provenance and its cost*, below — this is the entry's largest weakness.

### Relationship to entry 1

This is a new hypothesis with its own signal definition, objective and kill
criteria, pre-registered here before any code exists. It is **not** a re-test of
entry 1 and does not reopen entry 1's verdict.

Entry 1 is rejected and its prohibition stands: **its grid must not be re-run
wider, and its exits must not be varied.** ORB-2 differs from it on every axis
that prohibition covers:

| | Entry 1 | ORB-2 |
|---|---|---|
| Trigger | 5-minute close beyond range | Resting stop order 1.0 pt beyond range |
| Opening range | 5/10/15/30 min, searched | Fixed: the 09:30–09:45 candle |
| Direction | Both, one trade per direction | One, chosen by a daily trend filter |
| Stop / target | Multiples of range height, searched | Fixed: 10.0 pts / 18.0 pts |
| Day filter | None | Skip if range height > 10 pts |
| Fills | 5-minute bars | 1-minute bars |
| Parameters searched | 192 combinations | **None** |
| Selection | Highest training Sharpe | **No selection step exists** |

The handoff's warning is recorded and accepted: *"new hypothesis" is also
exactly what a re-test would call itself.* What makes this one different is not
the paperwork but that **there is no search** — there is a single fixed
configuration and one pre-registered A/B comparison. If ORB-2's results are ever
used to argue entry 1 deserves another look, that is a violation of entry 1's
Next section regardless of what ORB-2 returns.

### Mechanism claimed

**No new edge is claimed for the breakout itself.**

Entry 1 claimed that a decisive break of the opening range indicates one side of
the book has been exhausted, with continuation as remaining liquidity is taken.
That claim was **never established** — it was assumed from the pattern's
popularity rather than measured from order flow, and entry 1 records this as its
weakest point in hindsight. **Nothing since has established it.** No order flow
has been measured and no counterparty has been identified.

**Who is on the other side:** unknown, and unchanged from entry 1. Under this
project's own standard, an idea that cannot name its counterparty is a pattern,
not a hypothesis — so on the mechanism test ORB-2 fails exactly as entry 1 did.

The one genuinely new claim is narrower and is the only thing the A/B test
below is designed to isolate:

> **The trend-filter claim.** Breakouts taken *against* the prevailing daily
> trend fail more often than breakouts taken *with* it, by enough to pay for the
> trades the filter discards.

Even this has no identified counterparty. The nearest thing to a mechanism is
that a break against the daily trend is more likely to be a liquidity sweep into
resting orders that then reverts, while a break with the trend is more likely to
attract continuation flow. **That is a story, not a measurement**, and it is
recorded as such. The A/B test measures whether the filter pays; it does not
establish why, and a positive result would not license claiming it does.

**The honest prior is that neither the breakout nor the filter has an edge, and
a null result is expected.** This is written down before the run so a marginal
positive cannot later be read as confirmation.

### Provenance and its cost

The specification below arrived as a finished Pine Script with tuned constants:
a 10-point stop, an 18-point target, a 1.0-point entry offset, a 10-point range
ceiling, a 50-period EMA. **Those numbers came from somewhere, and this project
does not know where.**

This matters more than it first appears, and it cuts against the entry:

**A borrowed configuration is not an unsearched one.** Having no grid of our own
removes *our* selection overfitting. It does not remove the original author's.
Somebody chose 10 and 18 rather than 12 and 20, and 50 rather than 20 or 200.
Whatever search produced them is invisible, of unknown size, and probably ran on
overlapping data — most published intraday ES/MES scripts are tuned on recent
years, which are precisely this project's test years.

**This is in one respect worse than entry 1.** Entry 1's overfitting was
measurable: 192 combinations, a known grid, a computable rank correlation
between training and test. Here the search size is unknown and unbounded, so the
usual correction cannot be applied even in principle. **A clean out-of-sample
result on inherited parameters is weaker evidence than the same result on
parameters we fitted ourselves and can discount.**

Recorded now so that a good result is read with this discount already applied
rather than discovered afterwards.

### Signal definition

Fixed here. Nothing below may be changed after results are seen; a change means
a new entry, not an edit to this one.

**Opening range.** The high and low of the single 15-minute candle covering
**09:30–09:45 ET**, resampled from 1-minute data, labelled by opening minute and
left-closed, sessions resampled independently so no candle straddles the
overnight break or the open.

**Trend filter.** A **50-period EMA of completed daily closes**. The most recent
input is **yesterday's** close; today's forming bar is never an input.
Seeded with a 50-day simple moving average of the first 50 completed daily
closes, then the standard recursion with α = 2/51. This makes the value used at
09:45 today knowable at yesterday's close — **the source script repaints, this
does not.**

**Direction, decided once at 09:45.** Compare the **open of the 09:45 candle**
with the EMA:

- open **above** EMA → a **buy stop** at `range_high + 1.0`
- open **below** EMA → a **sell stop** at `range_low − 1.0`
- open exactly equal to the EMA → **no trade** (pre-registered so the boundary
  is not decided later)

One direction only, **one trade per day maximum**, no re-entry after an exit.

**Day filter.** **Skip the session entirely if the opening-range height exceeds
10.0 points**, because a fixed 10-point stop would then sit inside the opening
range, where an ordinary retrace back through the range stops the trade out for
reasons unrelated to the signal.

**Bracket.** From the fill price: **stop 10.0 points**, **target 18.0 points**.
Both fixed, neither scaled to volatility.

**Fills.** Entry, stop and target are all resolved on **1-minute bars**.

- A buy stop fills when a 1-minute bar's high reaches the level; the fill price
  is the stop level, **or the bar's open if the bar opened through it** (gaps
  fill at the open, never at the better price). Sell stops mirror this.
- The conservative rule from entry 1 carries over unchanged: **if a single
  1-minute bar touches both the stop and the target, the stop is assumed
  filled.**

**Timing.** An unfilled entry stop is **cancelled at 11:30 ET**. An open position
is **flattened at 15:55 ET**.

**No entries on roll days.** Prices are not back-adjusted across the contract
boundary in MES.v.0, so a position held over it sees an artificial jump.

**Sizing. 4 MES contracts**, which is $200 of risk at the 10-point stop. **Also
run at 1 contract** — see *Why the 1-contract run is not a second test*, below.

**Costs and limits as entry 1:** $1.25/side commission, 1 tick/side slippage,
repeated at 2 ticks; RTH only; internal limits from `strategies/rules.py`
(16:20 entry cutoff, 16:30 forced flatten, 5-contract cap, $400 daily loss
limit, 30-second minimum hold).

**No parameter grid. Nothing is selected.** There is no training step, so folds
2020–2026 are seven disjoint out-of-sample years rather than a walk-forward, and
no eligibility filter or tie-break rule is needed because no ranking is ever
performed.

### The arithmetic this fixes in advance

Because the bracket is fixed in points and costs are per-contract, several
figures are determined before any data is touched. They are recorded here so
that the results can be checked against them rather than interpreted freely.

**Per contract, after $1.25/side commission and 1 tick/side slippage:**

| | Gross points | Net per contract |
|---|---|---|
| Target hit | +18.0 | **+$85.00** |
| Stop hit | −10.0 | **−$55.00** |

**Break-even hit rate: 39.3%** (55 ÷ 140). Frictionless it would be 35.7%
(10 ÷ 28), so **costs raise the required hit rate by 3.6 percentage points** and
cut the effective reward-to-risk from 1.80 to 1.55. Time-based exits at 15:55
land between these two outcomes and are excluded from this identity; the realised
break-even will differ once they are counted, and both figures are reported.

**At 4 contracts:** a target is +$340, a stop is −$220.

- The **$400 internal daily loss limit cannot bind**, because the day's only
  trade risks $220. This is structural, not a coincidence, and it means rule 5
  does no work here.
- **4 of the 5-contract internal cap is used**, leaving no room to add — which
  the one-trade-per-day rule forbids anyway.
- Against the firm's $2,000 trailing line, **nine consecutive full stop-outs is
  −$1,980, twenty dollars short; the tenth ends the account.** Five consecutive
  reach the internal $1,000 warning line.

**The 30-second minimum hold cannot be verified at this resolution.** Entry and
stop can both fall inside the same 1-minute bar, and a backtest on 1-minute data
cannot see whether the exit came 8 seconds or 50 seconds after the fill. The
live guard enforces rule 6; the backtest simply cannot measure it. **Report the
count of trades whose entry and exit share a 1-minute bar** — that is the
population at risk, and if it is large the live strategy will behave differently
from the backtest.

### Why the 1-contract run is not a second test

`backtests/engine.py` computes `net_pnl = contracts × (net_points × 5 − 2.5)`.
Commission and slippage are both strictly per-contract, so **P&L is exactly
linear in size**: the 1-contract series is the 4-contract series divided by four,
trade for trade.

Therefore hit rate, per-contract expectancy, profit factor, fold-level sign and
rank correlation are **identical at both sizes**, and the 1-contract run
contributes no independent evidence about the strategy. The only thing it changes
is the **geometry against fixed-dollar limits** — a $3,000 target and a $2,000
trailing drawdown that do not scale with position size.

The one condition that would break the linearity is the $400 daily loss limit
binding at 4 contracts but not at 1. As shown above, it cannot bind with a single
$220 trade, so linearity holds exactly. **If the two runs disagree on anything
other than eval geometry, that is a bug, not a finding.** This is a cheap and
worthwhile assertion for the test suite.

Recorded consequence: at 1 contract, reaching a $3,000 target requires roughly
four times as many winning days inside the same 250-day horizon, so the
1-contract configuration is expected to **time out rather than pass**, almost
regardless of edge. It is a risk-geometry reference, not an evaluation
candidate. The kill criteria therefore key on the 4-contract run, as specified.

### The pre-registered comparison: filter ON versus OFF

**This is the only variation in the entry, and it exists to isolate the
trend-filter claim.**

- **ON** — the specification above. Direction chosen by the EMA at 09:45.
- **OFF** — identical in every other respect, but **both** brackets are placed at
  09:45: a buy stop at `range_high + 1.0` and a sell stop at `range_low − 1.0`.
  **First fill wins**; the opposite stop is cancelled on that fill. Still one
  trade per day, still no re-entry, still cancelled unfilled at 11:30.

**Ambiguity rule, fixed now.** If a single 1-minute bar reaches *both* entry
stops, the resolution is unobservable at this resolution. Consistent with the
stop-first convention — which resolves an ambiguous bar pessimistically — **the
direction that produces the worse outcome for that day is assumed to have filled
first.** The count of such days is reported, so it is visible whether the
convention mattered.

**Measurement.** Pooled out-of-sample 2020–2026, per contract:

```
E_on  = mean net P&L per trade, filter ON
E_off = mean net P&L per trade, filter OFF
dE    = E_on - E_off
```

Report `dE` with its standard error and t-statistic, alongside hit rate and trade
count for each arm.

**Filter claim verdict.** The trend-filter claim is **rejected unless
`dE` exceeds $5.00 per contract** — one full round turn. Beating OFF by less than
the cost of a trade is not evidence that the filter earns its exclusions.

### Prediction on record

Falsifiable, stated before the run, so that being wrong is visible.

**On the trend filter — the main claim:**

1. **The filter raises the hit rate by less than 3 percentage points.** ON minus
   OFF, pooled out-of-sample.
2. **`dE` does not exceed $5.00 per contract**, so the filter claim is rejected
   by its own criterion.
3. **Whatever advantage ON shows is concentrated in long trades**, and this is
   the confound that matters. MES rose over 2020–2026, so a 50-day EMA filter
   resolves to "be long" on a clear majority of sessions — expected to be
   **60–75% of qualifying days**. A long-biased rule tested on a rising index
   captures drift, not trend-following skill. **The diagnostic, pre-registered
   now: compare ON's long trades against OFF's long trades only.** If ON's
   advantage disappears in that like-for-like comparison, the filter is a
   long-only switch and the trend claim is unsupported. Report the ON long/short
   split so the base rate is visible.

**On the strategy overall:**

4. **Trade count is lowest in 2020 and 2022.** The 10-point range ceiling is a
   volatility filter in disguise: on high-volatility days the 09:30–09:45 range
   routinely exceeds 10 points, so those sessions are skipped. This is worth
   flagging beyond the trade count, because it means the strategy
   **systematically excludes the days on which breakout continuation is most
   often claimed to work.** If the skipped days would have been the profitable
   ones, the ceiling is not a safety filter but the thing removing the edge —
   report the skipped-day count per year and their opening-range distribution.
5. **The realised hit rate falls below 39.3%**, so per-trade expectancy is
   negative before the eval question is even reached.
6. **Pooled out-of-sample `eval_sim` pass probability does not clear 25%** at 4
   contracts, killing the entry on criterion 1.

**What would falsify the pessimism:** a hit rate durably above 39.3% across a
majority of the seven years, with `dE` above $5.00 per contract *and* surviving
the long-only diagnostic in prediction 3. That is a high bar and it is meant to
be.

### A methodological difference that must not be misread

**Moving fill checking from 5-minute bars to 1-minute bars makes ORB-2's P&L
numbers not directly comparable to entry 1's, in a direction that flatters
ORB-2.**

Entry 1 resolved stops and targets on the same 5-minute bars it generated
signals from, and assumed the stop filled whenever one bar contained both
levels. At 1-minute resolution, far fewer bars contain both, so the pessimistic
assumption fires much less often. Some of any improvement ORB-2 shows over entry
1 will come from this alone and has nothing to do with the trend filter or the
fixed bracket.

The 1-minute model is the more accurate one and is the right choice. But the
comparison it invites is invalid, and the temptation to make it is exactly how a
rejected strategy gets quietly resurrected. **ORB-2 is judged against its own
kill criteria below, not against entry 1's P&L.**

### Pre-registered tests

1. **Yearly out-of-sample folds 2020–2026**, seven of them. Data spans
   2019-05-06 to 2026-08-31; 2026 is a partial year ending 2026-08-31. Because
   nothing is selected, every fold is out-of-sample and no training window
   exists. 2019 is additionally reportable once the EMA has seeded (roughly
   mid-July 2019) and is *not* part of the kill criteria, which stay on
   2020–2026 as specified.
2. **The 2026 fold reported separately and alongside all seven.** Separately
   because it is eight months rather than twelve and is the most recent regime;
   alongside because a single recent fold is not evidence and must not be read
   as the headline.
3. **`eval_sim` pass probability, per fold and pooled out-of-sample**, at 4
   contracts and at 1, against the firm's $3,000 target and $2,000 end-of-day
   trailing drawdown — the deliberate exception where the firm number is used,
   because the question is whether the account survives, not whether the internal
   guard fires.
4. **Trailing-drawdown blow-up count across the seven years** at 4 contracts, on
   the stitched out-of-sample stream. Entry 1's comparable figures are 5 on the
   baseline and 4 on the walk-forward stream.
5. **Trade count per fold**, with any fold under **20 trades** flagged
   low-confidence. Entry 2's experience is the reason: four of its seven folds
   fell under that line, and a fold Sharpe of +2.85 on five trades is not a
   measurement. Report alongside it the count of days skipped by the 10-point
   range ceiling and the count of days where the entry stop never filled.
6. **Both arms, ON and OFF**, through every test above, plus `dE`, its standard
   error, the long/short split, and the long-only diagnostic from prediction 3.
7. **Exit-reason breakdown** — target, stop, 15:55 flatten — per year and per
   arm, because the fixed bracket makes the time-exit share the main thing the
   break-even identity above does not capture.
8. **Count of trades entering and exiting within one 1-minute bar**, per the
   rule 6 measurability gap noted above.
9. Repeat at **2 ticks of slippage per side**, per the project standard of
   evidence.

### Kill criteria — decided now

Any **one** of these kills the hypothesis. All are measured on the **4-contract**
run, pooled out-of-sample 2020–2026.

1. **Pooled out-of-sample `eval_sim` pass probability below 25%.**
2. **More than 1 evaluation blown across the seven years.**
3. **Fewer than 4 of 7 folds profitable, or total P&L negative.** Survival
   requires **both**: at least 4 profitable folds **and** positive total P&L.

Separately, and not a kill for the strategy as a whole:

4. **The trend-filter claim is rejected if `dE` does not exceed $5.00 per
   contract** — ON must beat OFF by more than one round turn. If ON survives
   criteria 1–3 while the filter claim is rejected, the honest reading is that
   the *breakout bracket* is carrying the result and the filter is decoration.

   **This criterion is deliberately non-fatal to the strategy as a whole**, and
   it carries a reporting obligation that is not optional: **the verdict must
   state explicitly whether the bracket or the filter is carrying any result**,
   in those terms, whatever the outcome. A verdict that reports a combined
   figure without attributing it to one or the other does not satisfy this
   entry, because the whole purpose of running two arms is to make that
   attribution rather than leave it to be assumed.

No appeal, no re-grid, no "but with a different stop". A kill is recorded here
and the idea is closed.

Criterion 3 is written as a conjunction on purpose. Entry 2 recorded that its own
fold-count criterion passed at 4 of 7 while total P&L was −$648, because counting
folds ignores magnitude and a majority of small wins against a minority of large
losses is still a losing strategy. That weakness is fixed here by requiring the
magnitude test alongside the count.

Criterion 1's threshold sits below the 50% gate entry 3 applies to manual trading
and above entry 1's 7.67% baseline. It is not a standard of profitability — a 25%
pass probability still means about four paid attempts per pass. It is the minimum
at which the question is worth asking again.

### Recorded expectations, before any run

**The baseline is discouraging.** ORB at entry 1's defaults returned 7.67% pass
probability, 57.60% blow-up, and about 13 attempts (~$1,499) per pass, from a
daily distribution with **mean −$1.50 against σ $120.50**. Fixing the bracket and
adding a trend filter changes the shape of that distribution; it does not
manufacture drift that is not there.

**Fixed brackets change the failure mode, not the expectancy.** A 10-point stop
that ignores volatility is too tight on active days and too wide on quiet ones.
The 10-point range ceiling suppresses the first case by skipping the day, which
means the strategy's remaining exposure is concentrated in low-volatility
sessions where an 18-point target is proportionally harder to reach. **Expect a
low target-hit share and a high 15:55-flatten share**; test 7 exists to make that
visible.

**Selection is not a risk here, and that is the entry's one real strength.** With
no grid there is no in-sample ranking to transfer, so entry 1's +0.398 and entry
2's −0.457 rank correlations have no analogue. Every one of the seven years is a
genuine out-of-sample observation of a fixed rule. **That strength is bounded by
the provenance problem above** — the parameters were selected, just not by us, on
data we cannot see and probably overlapping our test years.

**Roll gaps contaminate the EMA slightly.** The daily close series is the
unadjusted MES.v.0 continuous contract, so each roll injects a small artificial
step that propagates through the 50-day EMA for roughly fifty sessions. Using the
unadjusted series is consistent with the rest of the project and is the right
choice; the effect is expected to be small relative to the EMA-to-price distance
that the filter actually keys on. **Report the largest roll gap in the period**
so its size is on record rather than assumed negligible.

### Longer-term intent: copying trades across multiple funded accounts

Recorded here because it changes how any accepted result would be used.

The operator intends eventually to copy the same trade across several funded
accounts. **This multiplies outcomes in both directions. It is not risk reduction
and must not be described as diversification.**

Identical trades on N accounts are perfectly correlated. The same losing day
draws down every account simultaneously, and a trailing-drawdown breach
terminates all of them on the same date. N accounts running one strategy is one
bet at N times the size, with N times the fees — not N independent bets.

The only thing it diversifies is the *evaluation attempt*, and only while
accounts are started at different times on different price paths. Once they are
funded and trading in lockstep, the correlation is 1. A pass probability measured
for one account does not compound across N; the probability that *all* N pass is
not the product of independent draws, and the probability that all N die together
is far higher than independence would suggest.

This interacts badly with the fixed 10-point stop. Nine consecutive stop-outs put
a 4-contract account $20 from the firm's trailing line — **on every copied
account at once, on the same date.**

### Walk-forward verdict: REJECTED

**Date:** 2026-09-03
**Spec frozen at:** `17338c5` (nothing below was decided after seeing results)
**Verdict commit:** `243f9b1`
**Code:** `strategies/orb2.py`, `strategies/trend.py`, `backtests/run_orb2.py`
**Reports:** `backtests/results/orb2_report_slip1.txt`, `..._slip2.txt`

Seven out-of-sample years, 2020–2026. Nothing was selected — the configuration
is fixed, so every fold is a genuine out-of-sample observation of one rule.

#### Fold table, filter ON, 4 contracts, 1 tick slippage

| Test year | Trades | Net P&L | Sharpe | PF | Hit % | Max DD | Pass p | |
|---|---|---|---|---|---|---|---|---|
| 2020 | 73 | −$1,475 | −1.35 | 0.83 | 42.5 | −$2,035 | 6.05% | |
| 2021 | 81 | +$1,210 | 1.04 | 1.15 | 48.1 | −$1,200 | 50.05% | |
| 2022 | 10 | −$1,035 | −9.47 | 0.25 | 10.0 | −$1,035 | 0.00% | low-confidence |
| 2023 | 70 | −$790 | −0.71 | 0.91 | 40.0 | −$1,980 | 13.20% | |
| 2024 | 88 | −$2,030 | −1.44 | 0.83 | 35.2 | −$2,550 | 5.53% | |
| 2025 | 37 | +$565 | 0.94 | 1.14 | 43.2 | −$1,100 | 46.00% | |
| 2026 | 12 | +$860 | 4.02 | 1.73 | 50.0 | −$520 | 93.40% | low-confidence |

**Total −$2,695.00 over 371 trades. 3 of 7 folds profitable. Median fold
Sharpe −0.706.**

#### Fold table, filter OFF, 4 contracts, 1 tick slippage

| Test year | Trades | Net P&L | Sharpe | PF | Hit % | Max DD | |
|---|---|---|---|---|---|---|---|
| 2020 | 100 | −$3,245 | −2.21 | 0.74 | 38.0 | −$3,730 | |
| 2021 | 117 | +$2,000 | 1.17 | 1.18 | 47.0 | −$1,450 | |
| 2022 | 14 | +$260 | 1.30 | 1.23 | 42.9 | −$785 | low-confidence |
| 2023 | 93 | −$690 | −0.47 | 0.94 | 39.8 | −$1,845 | |
| 2024 | 120 | −$1,435 | −0.73 | 0.91 | 36.7 | −$4,535 | |
| 2025 | 47 | −$1,320 | −1.82 | 0.78 | 34.0 | −$2,695 | |
| 2026 | 13 | +$500 | 2.10 | 1.32 | 46.2 | −$660 | low-confidence |

**Total −$3,930.00 over 504 trades. 3 of 7 folds profitable.**

#### Pooled out-of-sample, 4 contracts

| | ON, 1 tick | OFF, 1 tick | ON, 2 ticks | OFF, 2 ticks |
|---|---|---|---|---|
| Trades | 371 | 504 | 371 | 504 |
| Net P&L | −$2,695 | −$3,930 | −$6,405 | −$8,970 |
| Mean per trade, per contract | −$1.82 | −$1.95 | −$4.32 | −$4.45 |
| Hit rate | 40.97% | 40.08% | 40.43% | 39.29% |
| Mean day / sd | −$7.26 / $244 | −$7.80 / $245 | −$17.26 / $244 | −$17.80 / $245 |
| Worst drawdown | $4,740 | $6,210 | $7,765 | $10,115 |
| **Evaluations blown** | **3** | **6** | **6** | **9** |
| **Pass probability** | **16.11%** | 15.86% | **8.40%** | 8.36% |
| Blow-up / timeout | 83.8% / 0.1% | 84.1% / 0.1% | 91.6% / 0.0% | 91.6% / 0.0% |
| Expected attempts | 6.21 | 6.31 | 11.90 | 11.97 |

#### The 2026 fold on its own

Reported separately as pre-registered. Partial year, ends 2026-08-31.
**ON: 12 trades, +$860, Sharpe 4.02, PF 1.73. OFF: 13 trades, +$500, Sharpe
2.10, PF 1.32.** Both are flagged low-confidence at well under 20 trades, and
both sit inside the seven-fold totals above, which are negative. A twelve-trade
year with a Sharpe of 4.02 is not a measurement, and the pre-registered
low-confidence flag exists precisely so this number cannot be read as a
turnaround.

#### Kill criteria

Measured at 4 contracts, pooled out-of-sample 2020–2026, as pre-registered.

| # | Criterion | 1 tick | 2 ticks | |
|---|---|---|---|---|
| 1 | Pooled OOS pass probability ≥ 25% | 16.11% | 8.40% | **FAIL** |
| 2 | Evaluations blown ≤ 1 | 3 | 6 | **FAIL** |
| 3 | ≥ 4 folds profitable **and** P&L > 0 | 3 of 7, −$2,695 | 3 of 7, −$6,405 | **FAIL** |

**All three failed, at both slippage assumptions. Any one is fatal. Rejected.**

Criterion 3's conjunction did no work here — the strategy failed both halves —
but it would have mattered had the folds landed differently, and it is recorded
as having been tested rather than merely carried over.

#### Criterion 4: is the bracket or the filter carrying the result?

Entry 4 obliges this attribution in the verdict whatever the outcome, so it is
stated plainly: **the bracket is carrying the result, the result is negative,
and the filter contributes nothing distinguishable from zero.**

| | ON | OFF |
|---|---|---|
| Trades | 371 | 504 |
| Mean per trade, per contract | −$1.82 | −$1.95 |
| Hit rate | 40.97% | 40.08% |

`dE = E_on − E_off = **+$0.13 per contract**` (SE $4.18, t = +0.03). The
threshold to survive was one round turn, $5.00. **The filter claim is
rejected** — and not narrowly. A t-statistic of 0.03 means the two arms are
indistinguishable; the filter is neither helping nor hurting at the portfolio
level. It removed 133 of 504 trades and moved expectancy by thirteen cents.

The same figure at 2 ticks is **+$0.13 per contract**, identically rejected.

**The long-only diagnostic is the part worth keeping.** As predicted, the filter
is overwhelmingly a long-only switch: **ON is 94.6% long (351 of 371), OFF is
53.4% long (269 of 504).** Comparing like with like:

| | ON longs | OFF longs |
|---|---|---|
| n | 351 | 269 |
| Mean per trade, per contract | −$2.57 | **+$3.40** |

`dE_long = **−$5.97 per contract**` (SE $4.92). **On longs against longs the
filter is worse than no filter at all.** The pre-registered reading applies: the
advantage does not survive like-for-like, so the filter is acting as a long-only
switch rather than a trend filter. The small headline edge it appeared to have
came from changing the long/short mix on a rising index, not from selecting
better breakouts — and once the mix is held fixed, even that reverses.

This is the entry's one genuinely new claim, and it is dead.

#### Predictions scored

Six predictions were recorded before the run. Four held, one was half wrong, and
one was wrong in a way that matters.

**1. Filter raises hit rate by under 3 points — CORRECT.** +0.89 points at
1 tick, +1.15 at 2 ticks.

**2. `dE` does not exceed $5.00 per contract — CORRECT.** +$0.13.

**3. Any ON advantage is concentrated in longs and vanishes like-for-like —
CORRECT, and stronger than predicted.** It does not merely vanish; it reverses
to −$5.97.

**4. Trade count lowest in 2020 and 2022 — HALF WRONG.** 2022 was lowest by a
wide margin (14 trades OFF, 10 ON, against 242 sessions skipped by the range
ceiling). **2020 was not low at all** — 100 trades OFF, among the highest years.
The reasoning behind the prediction was too coarse: 2020's volatility was
concentrated in March and April, so most of the year cleared a 10-point opening
range comfortably, whereas 2022 was persistently elevated and skipped 242
sessions. Volatility's *distribution through the year* drives the ceiling, not
the year's average. The unpredicted low year was 2025 at 47 trades.

**5. Realised hit rate falls below 39.3% — WRONG as stated, and the reason is
instructive.** The pooled hit rate is **40.97% (ON)** and **40.08% (OFF)**,
both *above* the 39.29% break-even, yet both arms lose money. The break-even
identity written into this entry was derived from target-versus-stop outcomes
only, and roughly a quarter of trades exit at the 15:55 flatten instead. On the
population the identity actually governs:

| | Targets | Stops | Target share | Break-even needed |
|---|---|---|---|---|
| ON | 97 | 188 | **34.04%** | 39.29% |
| OFF | 139 | 253 | **35.46%** | 39.29% |

**On the bracket alone the strategy misses break-even by more than five
percentage points**, which is the substance of prediction 5; the metric named in
the prediction was simply the wrong one, because a "win" counted by net P&L
includes small time exits. Stops came in at exactly −$220.00 and targets at
exactly +$340.00 per trade at 4 contracts, matching the pre-computed arithmetic
to the cent. Time exits averaged +$66.10 (ON) and were 64% positive, which
flatters the naive hit rate without covering the bracket's shortfall.

Recorded as a defect in the entry's own instrumentation, not a rescue: a hit
rate quoted against a target/stop break-even must be computed on target/stop
trades. Future entries should define the ratio and the population together.

**6. Pooled pass probability does not clear 25% — CORRECT.** 16.11% at 1 tick,
8.40% at 2.

#### Other pre-registered reporting

**The range ceiling skips more than it trades.** Across 2020–2026, **1,196
sessions were skipped for an opening range wider than 10 points** against 504
traded (OFF). Skipped sessions had a median opening range of **16.75 points**
(mean 19.51, max 88.00); traded sessions a median of **7.75 points**. The
strategy is therefore a low-volatility strategy by construction, and it
systematically excludes the sessions on which breakout continuation is most
often claimed to work. The entry flagged this in advance as the case where the
ceiling would be "the thing removing the edge" rather than a safety filter;
nothing here settles which, because the excluded days were never traded — but
the exclusion is large enough that the question is not marginal.

**Exit reasons**, 4 contracts, 1 tick: ON stop 50.7% / target 26.1% /
session_end 23.2%; OFF stop 50.2% / target 27.6% / session_end 22.2%. The
prediction of a low target share held; the flatten share at ~23% was moderate
rather than the high figure predicted.

**Rule 6 measurability.** **6 of 371 ON trades (1.62%)** and **1 of 504 OFF
trades (0.20%)** entered and exited inside one 1-minute bar and therefore cannot
be checked against the 30-second minimum-hold floor at this resolution. The
population at risk is small but non-zero. It is real: without the engine change
that lets a trade open and close on one bar, these seven trades would have been
silently dropped from the sample — and being same-bar stop-outs, they are
losers, so dropping them would have flattered the result.

**Both entry stops inside one bar** happened on exactly **1 session** across
seven years (OFF arm), resolved pessimistically as pre-registered. The
convention was worth fixing in advance and turned out not to matter.

**Size linearity confirmed on the real streams.** The 1-contract series equals
the 4-contract series divided by four, trade for trade, in both arms — checked
at runtime, not only in the unit tests. The 1-contract run accordingly carries
no independent evidence: it differs only in geometry against fixed-dollar
limits, and there it behaves as predicted, **timing out rather than passing**
(86.1% timeout, 0.03% pass, 0 evaluations blown at 1 tick). At 4 contracts the
same trades blow up 83.8% of the time. The two sizes are the same negative edge
read against a fixed target and a fixed trailing line.

**Roll contamination measured, and the first measurement was wrong.** The
initial figure — 327.50 points on 2020-03-16 — is the crash day's move, not a
roll artefact, and a second attempt keyed on roll *dates* missed that 12 of the
29 rolls are detected on a Sunday Globex reopen with no RTH session, leaving the
contaminated step on the following Monday counted as ordinary. Reading the
contract change off `instrument_id` finds all 29: **median step across a
contract change 38.75 points against 25.25 points on an ordinary session.** The
13.5-point excess enters a 50-day EMA at α = 2/51, so the immediate distortion
is about **0.53 points** — small against the EMA-to-price distance the filter
keys on, as the entry assumed but had not measured.

### What was learned

**A borrowed configuration bought nothing, and the provenance discount was the
right call.** The entry recorded in advance that inherited parameters are not
unsearched parameters, and that a clean out-of-sample result on them would be
weaker evidence than the same result on parameters we fitted and could discount.
That caution cost nothing here, because there was no result to discount: the
configuration loses money in five of seven years and destroys three to six
evaluations depending on slippage.

**The trend filter is the clearest negative in this log.** Entries 1 and 2
rejected mechanisms that might have existed and did not survive costs. This one
is different: the filter was measured directly against its own control, and the
measured effect is thirteen cents per contract with a t-statistic of 0.03. There
is no seam to widen and no ambiguity to revisit. The like-for-like comparison
being *negative* closes it further.

**Fixing the bracket in points made the failure legible.** Because stops and
targets are fixed distances, the arithmetic was determined before any data was
touched, and the realised figures matched it exactly — −$220.00 and +$340.00 per
trade at 4 contracts. That turned a vague question ("is this profitable?") into
an arithmetic one ("does the target share clear 39.29%?"), answered at 34.04%.
Entries that fix their bracket in advance should state the break-even ratio *and
the population it governs* the same way, which this entry did only half of.

**Selection was not the failure, because there was no selection.** Entry 1's
+0.398 rank correlation and entry 2's −0.457 both described selection
transferring or not. Here nothing was selected: every fold is the same rule
observed out of sample, and it lost in five of seven years. That removes the
last available explanation. The rule does not work.

**One trade a day and a volatility ceiling did reduce variance, and it did not
help.** The daily standard deviation is $244 at 4 contracts, against ORB's
$120.50 at one contract — roughly $61 per contract, half of ORB's. The mean day
is −$7.26, or −$1.82 per contract, against ORB's −$1.50. So the pre-registered
mechanism worked exactly as described and produced no benefit: a distribution
with less variance and a slightly worse centre is still a losing distribution,
and at 1 contract it converts blow-ups into timeouts rather than passes (86.1%
timeout) exactly as the entry predicted. Variance reduction cannot rescue
negative drift. That prediction is now measured rather than argued.

### Next

Do not re-test ORB-2 with a different stop, target, offset, range ceiling, EMA
period, or entry window. The configuration was inherited whole and tested whole;
tuning any constant now would be searching a grid that this entry deliberately
did not have, and would convert a clean negative into a fitted positive by the
exact route entries 1 and 2 forbid.

Do not test a third opening-range variant. Three entries in this log now rest on
the same unestablished claim — that a break of an early range predicts
continuation — and all three are rejected. The claim has never been measured
independently of backtest P&L, and until it is, another variant is another
draw from the same empty urn.

The one thing that would justify revisiting any of this is what entry 1 asked
for and never got: **direct evidence about the counterparty.** For a breakout
that means order-flow data showing who is on the other side of a range break and
under what constraint — not another price backtest. That is a data problem, and
an expensive one, and it should be priced before it is started rather than
approached through another parameter set.


### Addendum, 2026-09-04 — what the range ceiling was actually doing

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** This is a diagnostic, not a pre-registered test. It was chosen after
seeing the results above, it has no kill criteria, and nothing in it is evidence
for a strategy. It is recorded because the verdict left one question explicitly
open — the 10-point ceiling skipped 1,196 of 1,719 out-of-sample sessions, and
"nothing here settles which, because the excluded days were never traded" — and
the excluded days can simply be measured.

Filter OFF, 4 contracts, 2020–2026, the configuration otherwise unchanged. The
ceiling is removed by setting it beyond any observed range; nothing else moves.

| | Ceiling ON (entry 4) | Ceiling OFF | …narrow days | …**wide days only** |
|---|---|---|---|---|
| Trades | 504 | 1,689 | 504 | **1,185** |
| Net P&L, 1 tick | −$3,930 | −$26,980 | −$3,930 | **−$23,050** |
| Mean per trade | −$7.80 | −$15.97 | −$7.80 | **−$19.45** |
| Win rate | 40.08% | 37.36% | 40.08% | 36.20% |
| Worst drawdown | $6,210 | $29,375 | $6,210 | $26,470 |
| **Evaluations blown** | **6** | **29** | **6** | **24** |
| Pass probability | 15.86% | 10.73% | 15.86% | 9.09% |

At 2 ticks the no-ceiling variant loses $43,870 and destroys 34 accounts.

**The ceiling is protecting the account, decisively.** The 1,185 excluded-session
trades lose $23,050 on their own, at two and a half times the loss per trade of
the days the strategy does take, and they would have destroyed 24 further
evaluations. Removing it is unambiguously worse on every measure.

#### But not for the reason it was justified on

The ceiling's stated rationale was that a fixed 10-point stop sits inside a wider
opening range, so an ordinary retrace stops the trade out for reasons unrelated
to the signal. The stop-out share is consistent with that — **63.0% on wide days
against 50.2% on narrow ones**. The target share of bracket outcomes is not:

| | Narrow days | Wide days |
|---|---|---|
| Targets / stops | 139 / 253 | 412 / 747 |
| **Target share** | **35.46%** | **35.55%** |
| Break-even required | 39.29% | 39.29% |

**The two are indistinguishable, and both miss break-even by the same margin.**
Bracket expectancy works out at roughly −$21 per bracket trade in each group. So
the ceiling is not selecting sessions where the breakout resolves better; the
breakout resolves equally badly at every opening-range width tested. Whatever
the ceiling is doing, it is not improving the signal.

#### What it is actually selecting for: the flatten

The difference between the two groups is not the bracket, it is how often the
bracket resolves at all.

| | Narrow days | Wide days |
|---|---|---|
| Time exits | 112 of 504 (**22.2%**) | 26 of 1,185 (**2.2%**) |

**The 15:55 flatten is the only exit category with a positive mean.** On entry
4's OFF arm it returns **+$39.91 per trade** against −$220.00 for a stop and
+$340.00 for a target, and it is the only category whose mean is above zero once
frequency is accounted for. Narrow-range sessions produce ten times as many of
them, because on a quiet day the market often fails to travel 10 or 18 points
before the close, while on a wide-range day it almost always resolves one way or
the other.

**So on this strategy the flatten is doing the earning and the bracket is doing
the losing**, and the ceiling helps because it selects sessions where the bracket
frequently never resolves. That is a different mechanism from the one written
into the specification, and it was not visible until the excluded population was
measured.

#### Why this changes nothing about the verdict

It is worth being explicit, because a finding this clean invites being read as a
lead:

- **Every group loses.** Narrow days lose $3,930, wide days lose $23,050. There
  is no subset here that makes money.
- **The ceiling is a good filter for a reason nobody wrote down, and a good
  filter on a losing strategy is still a losing strategy.**
- **This was chosen after seeing the results.** It is one of an unbounded number
  of post-hoc slices, and the fact that it came out clean is not evidence that
  it would survive pre-registration.

Entry 4's Next section stands unchanged: no re-test with a different ceiling, and
no third opening-range variant.

#### Follow-up, same date: the flatten is not an edge either

The obvious next thought — if a position held into the close is the only thing
that earns, perhaps the late session carries something — was tested immediately
as a second diagnostic, and the answer is no. **This closes the lead rather than
opening it.**

**The 15:00 → 15:55 return, unconditional on any signal**, across 1,662 sessions
with a full late session (early closes excluded):

| Session group | n | Mean (points) | Mean $/4 contracts | SE | t |
|---|---|---|---|---|---|
| All sessions | 1,662 | −0.231 | −$4.63 | 0.379 | −0.61 |
| Ceiling skipped (range > 10) | 1,187 | −0.009 | −$0.18 | 0.490 | −0.02 |
| Ceiling took (range ≤ 10) | 459 | −0.326 | −$6.53 | 0.462 | −0.71 |

**There is no late-day drift.** Every group is indistinguishable from zero and
all three point slightly negative. Sign-adjusted to each time-exit trade's own
direction, the same window returns **−0.168 points (−$3.37 per trade, t =
−0.33)** — so the final 55 minutes *subtracts* about 6% from the time exit's
mean rather than producing it.

**Where the time exit's +$56.30 actually comes from: the bracket's own
geometry.** Decomposed, the 15:00→close leg contributes −$3.37 and the
entry-to-15:00 leg +$59.66. And that residual is not an edge — it is an
arithmetic consequence of the bracket. A time exit is by construction a trade
that touched neither the −10 stop nor the +18 target for its whole hold, so its
final price is conditioned to lie inside an **asymmetric band, (−10, +18),
whose midpoint is +4 points.** The observed mean is +$56.30 net, or about
**+3.8 gross points** — sitting just where truncation into that band predicts.

**So the three exit categories are not three findings, they are one identity.**
The stop truncates losses at −10, the target truncates gains at +18, and
whatever survives both must average positive because the surviving band is wider
above than below. It cannot be otherwise, and it would appear on a pure random
walk. Reading the flatten's positive mean as evidence that holding into the
close pays would be reading the bracket's arithmetic as a market effect.

**Composition, not time of day, also explains the exit-time pattern.** Sorting
all 1,689 trades by when they exited, the earliest bucket loses most
(−$130.47 mean, 09:30–10:00) and later buckets turn positive. That is not a
clock effect: **268 of the 319 trades exiting in the first bucket are stops.**
A failed breakout fails fast, so early exits are almost all losses by selection.
Stops average exactly −$220.00 and targets exactly +$340.00 in every bucket,
confirming the fixed bracket behaves identically at every hour.

*(One curiosity, recorded and deliberately not pursued: sign-adjusted, the
15:00→close window is +1.558 points with t = 2.57 on sessions whose trade exited
at target. Those trades were already closed, so this is post-exit drift and
attributes no P&L. Chasing it would be exactly the post-hoc search this log
exists to prevent.)*

#### What this leaves

**Nothing here is a hypothesis, and after the follow-up there is no longer an
obvious candidate to become one.** The ceiling helps for a mechanical reason,
the flatten's positive mean is a truncation artefact, and the late session has
no measurable drift. Entry 4's Next section stands: no re-test with a different
ceiling, and no third opening-range variant.

If any of this is ever pursued it gets a new entry with its own mechanism,
counterparty, kill criteria and grid, written before anything further is run —
not an extension of this one.

### Addendum, 2026-09-11 — payout probability, from the saved streams

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** `backtests/eval_sim.py` now reports, alongside the $3,000 pass
probability, the probability that a path's end-of-day balance touches
**$52,100** — the Lucid 50K account's payout line, $2,100 over the start with
a $500 minimum withdrawal (`rules.FIRM.payout_buffer`, `rules.PAYOUT_BALANCE`)
— before the $2,000 trail ends the account, within the same 250-day horizon
and on the same resampled days. Reaching the line does not end a path, and
because it sits below the target every passing path has touched it first, so
the figure is never below the pass probability. It is a milestone, not a
criterion: nothing in this entry keys on it.

Recomputed with `run_orb2.py --from-cache` from the saved out-of-sample
streams. Every other line of both reports came out identical to the saved
ones; the payout row is the only difference.

| Pooled OOS 2020–2026, 4 contracts | ON, 1 tick | OFF, 1 tick | ON, 2 ticks | OFF, 2 ticks |
|---|---|---|---|---|
| Pass probability (as recorded) | 16.11% | 15.86% | 8.40% | 8.36% |
| **Payout probability** | **27.31%** | **27.04%** | **17.14%** | **16.73%** |

At 1 contract, where the pass probability rounds to zero, the payout line is
touched on 1.05% (ON) and 0.93% (OFF) of paths at 1 tick, 0.18% and 0.16% at 2.

Roughly six in ten of the 4-contract paths that reach $52,100 go on to reach
the target; the other four are ended by the trail, or run out of horizon, in
the $900 between the two lines. That gap is the whole difference between the
two figures, and neither is close to the 25% the kill criterion asked of the
pass probability.

### Addendum, 2026-09-11 — re-priced at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 this entry assumed (Lucid
support, 2026-09-11, article 11508978; method in entry 1's addendum of the
same date). The four saved 4-contract pooled streams were re-priced exactly —
$6.00 a trade at 4 contracts. Nothing is selected in this entry's
walk-forward, so nothing is held fixed by the method; no stream carries a
daily-loss exit.

| Pooled OOS 2020–2026, 4 contracts | ON, 1 tick | OFF, 1 tick | ON, 2 ticks | OFF, 2 ticks |
|---|---|---|---|---|
| Net P&L as scored | −$2,695 | −$3,930 | −$6,405 | −$8,970 |
| **Net P&L re-priced** | **−$469** | **−$906** | **−$4,179** | **−$5,946** |
| Mean per trade, per contract | −$1.82 → −$0.32 | −$1.95 → −$0.45 | −$4.32 → −$2.82 | −$4.45 → −$2.95 |
| Folds profitable | 3 of 7 → 3 of 7 | 3 of 7 → 3 of 7 | 3 of 7 → 3 of 7 | 3 of 7 → 3 of 7 |
| Max drawdown | $4,740 → $4,002 | $6,210 → $4,930 | $7,765 → $5,947 | $10,115 → $7,726 |
| Evaluations blown | 3 → 2 | 6 → 5 | 6 → 4 | 9 → 7 |
| **Pass probability** | **16.11% → 22.70%** | 15.86% → 22.66% | **8.40% → 12.55%** | 8.36% → 12.50% |
| Payout probability | 27.31% → 34.78% | 27.04% → 34.36% | 17.14% → 22.87% | 16.73% → 22.54% |

At 1 contract, 1 tick: ON −$674 → −$117, OFF −$983 → −$227.

**Would any kill criterion have resolved differently? No — but one moved a
long way toward its line.** At 4 contracts, filter ON, as pre-registered:

| # | Criterion | 1 tick | 2 ticks | |
|---|---|---|---|---|
| 1 | Pooled OOS pass probability ≥ 25% | 16.11% → **22.70%** | 8.40% → 12.55% | FAIL |
| 2 | Evaluations blown ≤ 1 | 3 → 2 | 6 → 4 | FAIL |
| 3 | ≥ 4 folds profitable **and** P&L > 0 | 3 of 7, −$469 | 3 of 7, −$4,179 | FAIL |
| 4 | Filter edge dE > $5.00 per contract (non-fatal) | +$0.13 | +$0.13 | REJECTED |

Criterion 1 at 1 tick closes to within 2.3 points of the 25% line; at the
2-tick base case it is half of it. Criterion 4 does not move at all: both
arms pay the same commission per contract per trade, so a uniform rate
change cancels in the ON-minus-OFF difference. The $5.00 threshold was
pre-registered as "one round turn" at $1.25 and 1 tick and is not recomputed
(a round turn is now $3.50); dE clears neither figure.

**Break-even, restated.** The 10/18 bracket at $0.50 and 1 tick nets +$86.50
a winner and −$53.50 a loser per contract, so break-even is **53.5/140 =
38.21%**, not 39.29%. At 2 ticks it is 56/140 = **40.00%**. The frozen
2-tick report printed 39.29% there as well; that was the 1-tick constant, and
the correct figure at $1.25 and 2 ticks was 57.5/140 = 41.07%. The runners
now compute it from the cost model they are given. Realised hit rates on the
bracket were 40.97% (ON) and 40.08% (OFF) at 1 tick — above the corrected
break-even, as they were above the old one, and the arms still lose, for the
reason the 2026-09-04 addendum gives: the flatten population is not part of
the bracket and is where the money goes.


---

## 5. ORB flat by 10:30 — REJECTED

**Date:** 2026-09-05
**Spec frozen at:** the commit adding this entry
**Code:** not yet written
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** MES, 15-minute opening range, 1-minute bars for fills, RTH only
**Source:** transcribed line for line from `orb_flat_1030.pine`, the script the
operator is running.

### This entry is forbidden by entry 4, and is being written anyway

Entry 4's Next section says, in terms: **"Do not test a third opening-range
variant. Three entries in this log now rest on the same unestablished claim —
that a break of an early range predicts continuation — and all three are
rejected."**

**This is a fourth.** The operator has directed it after seeing that verdict.
Recording the conflict here rather than omitting it is the whole function of
this log: an entry that quietly ignores its predecessor's prohibition is how a
rejected idea gets re-run until it passes, and the reader six months from now
needs to see that this one did not sneak past.

**What is genuinely different, and it is not nothing:** every prior entry in
this family varied the *signal* or its *parameters* — the range length, the
stop, the target, the filter, the entry window. This one changes the **holding
period**, and changes it drastically: the position is flat by **10:30**, a
maximum hold of 45 minutes against entry 4's 15:55. That is a structural change
to the strategy's exposure, not another cell of the same grid.

**What is not different:** the signal, the mechanism, and the absence of one.

### Mechanism claimed

**None.** No new edge is claimed, and none has been established at any point in
entries 1, 4, or here. The breakout continuation claim was assumed from the
pattern's popularity in entry 1, was never measured against order flow, and
remains unmeasured.

**Who is on the other side:** unknown. Under this log's own standard that makes
this a pattern rather than a hypothesis, and the entry says so rather than
dressing a shorter hold as a mechanism.

**The honest prior is that it fails, and probably worse per trade than entry 4
did.** The reasoning is below and is specific enough to be wrong.

### Specification, line for line with the Pine

Fixed here. Nothing may change after results are seen.

**Opening range.** High and low of the single 15-minute candle covering
**09:30–09:45 ET**, resampled from 1-minute bars, labelled by opening minute,
left-closed, sessions resampled independently.

**Day filter.** Skip the session entirely if the opening-range height exceeds
**10.0 points**.

**Orders at the 09:45 close.** **Both** rest: a **buy stop at
`range_high + 1.0`** and a **sell stop at `range_low − 1.0`**. The trend filter
is **OFF** and stays off; both sides are always placed.

**First fill wins, OCA at the moment of fill.** The instant one side fills the
other is cancelled — not at the next bar, not at the next 15-minute candle.
Where a single 1-minute bar reaches both levels, the fill is unobservable at
this resolution and is resolved **pessimistically**: the direction producing the
worse outcome for that day is assumed to have filled. The count of such sessions
is reported.

**Bracket.** From the fill: **stop 10.0 points**, **target 18.0 points**, both
fixed in points.

**Fills.** Entry, stop and target resolve on **1-minute bars**. A stop order
fills at its level, or at the bar's open if the bar opened through it. **If one
bar touches both stop and target, the stop is assumed filled.**

**One trade per day**, enforced. No re-entry after an exit.

**10:30 ET, no exceptions.** Any unfilled order is cancelled **and** any open
position is closed **at the open of the 10:30 bar**. The maximum hold is
therefore 45 minutes and the entry window is 09:45–10:30.

**Size.** **4 MES contracts.** $1.25 per contract per side commission, **1 tick
of slippage per side**, repeated at 2 ticks per the project standard.

**Internal guards.** The **$400 daily loss limit**, marked to market, and a
**$1,500 trailing drawdown halt** measured from peak end-of-day equity. Roll
days are skipped. These are `INTERNAL` values; nothing here reads a `FIRM`
number except `eval_sim`, which asks whether the account survived.

### Prediction on record

**No ORB flat-by-10:30 backtest has been run and no partial result inspected.**
What follows is derived from entry 4's published figures and its 2026-09-04
addendum, both already in this log.

**1. The target share of bracket outcomes falls to roughly 25–30%, against
entry 4's 35.5% and a 39.29% break-even.** This is the sharp one. The addendum's
exit-time table shows that resolutions *before 10:30* are markedly stop-heavy:
of the trades exiting in the 09:30 and 10:00 buckets, **647 were stops against
247 targets — a 27.6% target share**, versus 35.5% across the full population. A
failed breakout fails fast; a working one takes longer to travel 18 points than
a failing one takes to travel 10. **Cutting the hold at 10:30 therefore keeps
the stop-heavy fast resolutions and discards the slower target resolutions.**

**2. Per-trade net expectancy is worse than entry 4's −$7.80.** Entry 4's OFF
arm netted −$7.80 per trade at 4 contracts against $20.00 of round-turn
friction, so its *gross* expectancy was about +$12.20 and costs alone made it
negative. With a worse target share and a truncated upside, gross should fall,
while friction is unchanged at $20.00 per round turn.

**3. Most trades end at the 10:30 flatten, and its mean is far smaller than
entry 4's +$39.91.** Forty-five minutes is rarely enough to travel 10 or 18
points. The addendum established that the flatten's positive mean is
**truncation, not drift** — survivors are conditioned into an asymmetric
(−10, +18) band whose midpoint is +4 points. Over 45 minutes most survivors will
sit near zero rather than spread across that band, so the truncation premium
should largely vanish. **This is the prediction most likely to be wrong**, and
it is the one worth watching: if the flatten mean stays large, the truncation
account in the addendum is incomplete.

**4. Fewer trades than entry 4's 504.** The entry window closes an hour earlier.

**5. Pooled seven-year pass probability does not clear 25%**, killing it on
criterion 1.

**What would falsify the pessimism:** a target share at or above 39.29% across a
majority of the seven years with positive total P&L. Predictions 1 and 2 are
specific enough that being wrong about them will be obvious.

### The two-year window is context, not evidence — fixed now

Results are reported for **2024-09-01 to 2026-08-31** alongside the seven years,
as requested. **The verdict turns on the seven-year figures only**, and this is
pre-registered so it cannot be renegotiated once both are visible.

The reason is on the record already: **entry 1's ORB returned +$2,250 over
exactly this two-year window and −$6,073 across seven folds.** That window
contains the two best years and none of 2020–2022. A two-year number that
disagrees with the seven-year number is the expected behaviour of a favourable
draw, not new information.

### Kill criteria — decided now

Any **one** kills the hypothesis. Measured at **4 contracts**, pooled over the
**seven years 2020–2026**.

1. **Pooled seven-year `eval_sim` pass probability below 25%.**
2. **More than 1 evaluation blown across the seven years.**
3. **Fewer than 4 of 7 years profitable, *or* total seven-year P&L negative.**
   Survival requires **both**.

No appeal, no re-grid, no third flatten time.

### Reporting required

Trade count, net P&L, win rate with its standard error, profit factor and max
drawdown, for both windows side by side; P&L and counts by exit reason
(stop / target / 10:30 flatten); the yearly breakdown; evaluations blown over
the seven years at 4 contracts; and `eval_sim` pass probability computed
separately on the two-year and the seven-year daily distributions. Plus the
count of sessions where both entry stops were reached inside one 1-minute bar,
the count of trades entering and exiting inside one bar, and the skipped-session
counts with their opening-range distribution. All repeated at 2 ticks.

### Where the backtest and the Pine still disagree

**1. Fill resolution: 1-minute here, 15-minute in the script.** Between 09:45
and 10:30 the Pine sees three bars; this backtest sees forty-five. For resting
stop orders the two are close, but the Pine's own strategy tester will not
reproduce these numbers and is the coarser model — at 15-minute resolution a bar
containing both stop and target is common, and the stop-first convention then
fires far more often than it should.

**2. The drawdown halt is modelled on end-of-day equity; the Pine marks it
intraday.** `strategy.equity` in Pine includes open profit bar by bar, so the
live script can halt mid-session where this backtest halts only at a daily
boundary. The backtest is therefore mildly *permissive* on this guard.

**3. The halt is implemented in the entry-5 runner, not in `engine.py`.** The
engine has `enforce_daily_loss_limit` but no trailing-drawdown equivalent. Rule
9 wants this as shared runtime logic and it should be promoted to the engine
when a second caller needs it; for now it lives beside the run and is tested
there.

**4. OCA resolution.** The Pine leaves a both-sides-touched bar to the broker's
OCA group; this backtest resolves it pessimistically and reports how often it
mattered.

### Longer-term intent: copying trades across multiple funded accounts

Unchanged and repeated because it governs how any accepted result would be used.
Copying identical trades across N funded accounts **multiplies outcomes in both
directions and is not diversification**: the same losing day draws down every
account at once and a trailing-drawdown breach terminates all of them on the
same date. It is one bet at N times the size with N times the fees. Only the
evaluation *attempt* is diversified, and only while accounts start at different
times on different price paths.

### Verdict: REJECTED

**Date:** 2026-09-05
**Spec frozen at:** `c7c6f01` (nothing below was decided after seeing results)
**Verdict commit:** `d5b0bce`
**Code:** `strategies/orb2.py` (with `flatten_at_next_open`),
`backtests/run_orb_flat.py`
**Reports:** `backtests/results/orb_flat_report_slip1.txt`, `..._slip2.txt`

**Rejected on every basis measured, at 1 and 2 ticks of slippage.**

#### The drawdown halt ends the seven-year run in 2020

This has to come first, because it governs how the rest reads.

The $1,500 trailing halt fires early in 2020 and **never releases**. A halted
account takes no trades, so its balance never recovers, so the halt is
permanent: **433 sessions are blocked and the run stops after 45 trades, all of
them in 2020.** Years 2021–2026 contain no trades at all.

That is faithful to `orb_flat_1030.pine` — its `halted` flag works the same way
— and it is the honest live outcome: **run with its own guards, this strategy
stops trading in year one and does not resume.** But it makes two of the three
kill criteria degenerate, so both bases are reported.

| Seven years, 4 contracts, 1 tick | As frozen (halt ON) | Halt OFF (comparable) |
|---|---|---|
| Trades | 45 | 478 |
| Net P&L | −$1,415 | **−$11,780** |
| Mean per trade | −$31.44 | −$24.64 |
| Win rate | 37.78% | 40.17% |
| Standard error | 7.23 pts | 2.24 pts |
| Profit factor | 0.527 | 0.673 |
| Max drawdown | $1,500 | $12,065 |
| **Target share of bracket** | **0.00%** | **21.71%** |
| Evaluations blown | 0 | **6** |
| Pass probability | 0.00% | 0.22% |
| Years profitable | 0 of 7 | 1 of 7 |

The halt-OFF column is the basis entries 1 and 4 were measured on, and is the
only one comparable with them.

#### Kill criteria

| # | Criterion | Halt ON | Halt OFF | |
|---|---|---|---|---|
| 1 | Pass probability ≥ 25% | 0.00% | 0.22% | **FAIL** |
| 2 | Evaluations blown ≤ 1 | 0 | 6 | **FAIL** (see below) |
| 3 | ≥ 4 of 7 years profitable **and** P&L > 0 | 0 of 7, −$1,415 | 1 of 7, −$11,780 | **FAIL** |

At 2 ticks, halt OFF: **−$16,560, 0 of 7 years profitable, 10 evaluations
blown, pass probability 0.02%.**

**Criterion 2 is defective as written, and it passed on the frozen basis for the
wrong reason.** With the halt active, no evaluation can be blown *because the
internal guard stops trading before the firm's $2,000 line is reached* — that is
precisely what the guard is for. A blow-up count and a permanent halt are
incompatible measurements: the count assumes a stream that keeps trading and
restarts a fresh evaluation after each death, which is how entries 1 and 4
computed theirs. **The entry should have specified which basis criterion 2 was
measured on, and did not.** Recorded as a flaw in this entry's construction. It
changes nothing here — criteria 1 and 3 fail on both bases, and criterion 2
fails outright on the comparable one — but a future entry combining a halt with
a blow-up count must say which it means.

#### P&L by exit reason, seven years, halt OFF

| Exit | n | Mean | Total |
|---|---|---|---|
| 10:30 flatten | **349** | **+$2.64** | +$920 |
| Stop | 101 | −$220.00 | −$22,220 |
| Target | 28 | +$340.00 | +$9,520 |

On the frozen basis the 45 trades break down as **38 flattens (+$3.29 mean), 7
stops, and zero targets** — 18 points inside 45 minutes on a session whose
opening range was under 10 points essentially never happened in that sample.

#### Yearly breakdown, halt OFF, 1 tick

| Year | Trades | Net P&L | Win % | PF | Max DD | Stops | Targets | Flatten |
|---|---|---|---|---|---|---|---|---|
| 2020 | 92 | −$3,670 | 34.8 | 0.47 | −$3,890 | 18 | 2 | 72 |
| 2021 | 112 | −$15 | 48.2 | 1.00 | −$1,125 | 7 | 6 | 99 |
| 2022 | 14 | −$190 | 42.9 | 0.81 | −$730 | 3 | 1 | 10 |
| 2023 | 87 | −$1,820 | 43.7 | 0.75 | −$2,045 | 25 | 5 | 57 |
| 2024 | 117 | −$4,405 | 35.9 | 0.59 | −$4,405 | 33 | 11 | 73 |
| 2025 | 44 | −$1,710 | 34.1 | 0.52 | −$1,800 | 12 | 1 | 31 |
| 2026 | 12 | +$30 | 41.7 | 1.03 | −$510 | 3 | 2 | 7 |

The only profitable year is 2026 at **+$30 on 12 trades**, which is noise.

#### The two-year window, as pre-registered: context, not evidence

**Halt OFF, 1 tick: 91 trades, −$2,820, win rate 39.56% ± 5.13, PF 0.633, 1
evaluation blown, pass probability 0.12%.** The two-year window agrees with the
seven-year one here. That agreement is not itself informative — a favourable
draw was possible and did not occur — and the pre-registered rule that the
verdict turns on seven years stands either way.

#### Predictions scored

All five held. One was understated and one was flagged in advance as the most
likely to be wrong; it held clearly.

**1. Target share falls to roughly 25–30% — CORRECT, and understated.** Actual
**21.71%** against entry 4's 35.5% and a 39.29% break-even. The mechanism
predicted from the addendum was right: resolutions before 10:30 are stop-heavy,
so cutting the hold keeps the fast stops and discards the slower targets. **101
stops against 28 targets.** The magnitude was worse than predicted.

**2. Per-trade expectancy worse than entry 4's −$7.80 — CORRECT.** Actual
**−$24.64**, roughly three times worse. And the decomposition is sharper than
the prediction: friction is $20.00 per round turn, so **gross expectancy is
about −$4.64**. Entry 4's gross was **+$12.20** and only costs made it negative.
**Cutting the hold at 10:30 turned a positive gross expectancy negative.** That
is the single most useful number in this entry.

**3. Most trades flatten, and the flatten's mean falls far below +$39.91 —
CORRECT.** This was flagged in advance as the prediction most likely to be
wrong, because it tested the addendum's truncation account rather than restating
it. **349 of 478 trades (73.0%) end at the flatten, at a mean of +$2.64** —
against +$39.91 over a full session. The truncation premium did not merely
shrink, it nearly vanished.

That is a quantitative confirmation of the 2026-09-04 addendum. The premium
comes from survivors being conditioned into an asymmetric (−10, +18) band, and
its size depends on how far price disperses inside that band before the flatten.
Over 45 minutes there is almost no dispersion, so survivors sit near zero and
the premium collapses. **If the flatten had been a genuine late-day edge rather
than truncation, shortening the hold would not have removed it.**

**4. Fewer trades than entry 4's 504 — CORRECT, but barely: 478.** Cutting a
full hour off the entry window removed only 26 trades, because a resting stop
one point beyond a narrow opening range is almost always reached in the first
45 minutes if it is reached at all. Worth recording: the 09:45–10:30 window
captures nearly all the fills the 09:45–11:30 window did, so the extra hour in
entry 4 was contributing hold time, not entries.

**5. Pass probability does not clear 25% — CORRECT.** 0.22% halt OFF, 0.00% as
frozen.

#### Other pre-registered reporting

**Guards.** Zero daily-loss halts in either window — with one trade a day
risking $220, the $400 limit cannot bind, the same structural result entry 4
recorded. 433 drawdown halts over seven years, 54 over two.

**Measurability.** **Zero** trades entered and exited inside one 1-minute bar,
so rule 6's 30-second floor is fully verifiable on this configuration — unlike
entry 4, where 7 trades were unverifiable. **One** session across seven years had
both entry stops reached inside a single bar, resolved pessimistically.

**The range ceiling** skipped 1,196 of 1,719 sessions, unchanged from entry 4
since neither the range nor the ceiling moved. Skipped sessions had a median
opening range of 16.75 points; traded sessions 7.75.

### What was learned

**Shortening the hold made it worse, and the reason is now measured rather than
argued.** Entry 4's bracket lost money but its gross expectancy was positive;
friction alone sank it. Entry 5 keeps the friction, removes the slower target
resolutions, and keeps the fast stops — and gross expectancy goes negative. **A
10:30 flatten is not a risk control on this strategy, it is an adverse
selection filter.**

**The truncation account survived a real test.** The addendum explained the
flatten's positive mean as an artefact of the (−10, +18) survivor band rather
than a late-day effect. That explanation made a falsifiable prediction —
shorten the hold and the premium should collapse — and the premium fell from
+$39.91 to +$2.64. A genuine time-of-day edge would not behave that way.

**The internal guard works, and working is not the same as helping.** Run as
specified, the $1,500 halt caught the account in 2020 and prevented all six
evaluation blow-ups the unguarded stream would have suffered. It did that by
ending the account's trading life in year one. A guard that saves an evaluation
by stopping a negative-expectancy strategy is doing its job; it is not evidence
the strategy is safe to run.

**Four opening-range entries, four rejections.** Entries 1, 4 and 5 have now
tested the same unestablished continuation claim across three signal
definitions, two exit regimes, four holding periods and two objectives. None has
produced a positive result, and none has ever measured the claim directly.

### Next

**Do not test a fifth opening-range variant.** Entry 4's Next section already
forbade a third; this was a fourth, written with that conflict recorded, and it
failed in the direction predicted. There is no remaining timing, bracket or
filter permutation that this log has any reason to expect to work, and each new
one is another draw against the same absent edge.

The prohibition entry 1 wrote still stands unmet: **any future breakout idea
must first establish the counterparty claim independently of backtest results.**
That means order-flow evidence about who takes the other side of a range break
and under what constraint. It is a data purchase, not a backtest, and it should
be priced before it is started.

If the operator intends to keep running `orb_flat_1030.pine` live, that is a
decision this log cannot make. What it can record is that the script's own
guards stopped the equivalent backtest in 2020, that its unguarded seven-year
result is −$11,780 at 1 tick and −$16,560 at 2, and that its gross expectancy
before costs is negative.

### Addendum, 2026-09-11 — payout probability, both bases

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** The simulator now reports the probability of touching the **$52,100**
payout line before the trail, alongside the pass probability — see entry 4's
addendum of the same date for the definition. `run_orb_flat.py` regenerates
its signals rather than reading a cache, and both reports came out identical
to the saved ones apart from the added row, which is also the check that the
re-run reproduced the verdict.

| Seven years, 4 contracts | Halt ON, 1 tick | Halt OFF, 1 tick | Halt ON, 2 ticks | Halt OFF, 2 ticks |
|---|---|---|---|---|
| Trades | 45 | 478 | 33 | 478 |
| Pass probability (as recorded) | 0.00% | 0.22% | 0.00% | 0.02% |
| **Payout probability** | **0.01%** | **1.29%** | **0.00%** | **0.29%** |

Two-year window, 1 tick: halt ON 0.75% (pass 0.13%), halt OFF 0.83% (pass
0.12%). At 2 ticks: 0.52% and 0.21%.

With a seven-year mean day between −$25 and −$49 on every basis, reaching
$2,100 over the start is a tail event, and the payout figure says the same thing the pass
figure did with one more decimal place. Recorded so the entry carries the
milestone every later verdict reports.

### Addendum, 2026-09-11 — re-run at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 this entry assumed (Lucid
support, 2026-09-11, article 11508978). `run_orb_flat.py` was **re-run in
full** at the new default at 1 and 2 ticks, both guards re-decided at the new
cost — the trailing halt decides this entry's frozen basis, so a re-pricing
of the halted stream would have missed that it fires later. The halt-OFF
totals are exactly the frozen figures plus $6.00 × 478 trades, the check that
only the commission moved.

| Seven years, 4 contracts | Halt ON, $1.25 | **Halt ON, $0.50** | Halt OFF, $1.25 | **Halt OFF, $0.50** |
|---|---|---|---|---|
| **1 tick** — trades | 45 | **46** | 478 | 478 |
| Net P&L | −$1,415 | **−$1,359** | −$11,780 | **−$8,912** |
| Years profitable | 0 of 7 | 0 of 7 | 1 of 7 | 2 of 7 |
| Evaluations blown | 0 | 0 | 6 | 5 |
| Pass probability | 0.00% | 0.00% | 0.22% | 0.85% |
| Payout probability ($52,100) | 0.01% | 0.02% | 1.29% | 3.42% |
| **2 ticks** — trades | 33 | **44** | 478 | 478 |
| Net P&L | −$1,625 | **−$1,511** | −$16,560 | **−$13,692** |
| Years profitable | 0 of 7 | 0 of 7 | 0 of 7 | 0 of 7 |
| Evaluations blown | 0 | 0 | 10 | 8 |
| Pass probability | 0.00% | 0.00% | 0.02% | 0.07% |
| Payout probability ($52,100) | 0.00% | 0.01% | 0.29% | 0.71% |

Two-year window, 1 tick: halt ON 37 → 43 trades, −$1,270 → −$1,217; halt OFF
91 trades, −$2,820 → −$2,274. The target share of bracket outcomes is
unchanged at 21.71% (halt OFF) — commission does not move an exit.

**Would any kill criterion have resolved differently? No.**

| # | Criterion | Halt ON, $0.50 | Halt OFF, $0.50 | |
|---|---|---|---|---|
| 1 | Pass probability ≥ 25% | 0.00% | 0.85% | FAIL |
| 2 | Evaluations blown ≤ 1 | 0 (degenerate, as recorded) | 5 | FAIL |
| 3 | ≥ 4 of 7 years profitable **and** P&L > 0 | 0 of 7, −$1,359 | 2 of 7, −$8,912 | FAIL |

**Break-even, restated.** The 10/18 bracket at $0.50 and 1 tick breaks even
at **38.21%**, not 39.29%; at 2 ticks, **40.00%**. The frozen 2-tick report
printed 39.29% there too — that was the 1-tick constant, and the correct
figure at $1.25 and 2 ticks was 41.07%. `run_orb_flat.py` now computes it
from the cost model it is given, so `--commission 1.25` reproduces the frozen
1-tick report and corrects the 2-tick line. Friction per round turn at 4
contracts is $14.00, not the $20.00 the entry states (1 tick); the
observation that gross expectancy before costs is negative is unaffected.
The halt at 1 tick fires one trade later, at 2 ticks eleven trades later,
and in every case inside 2020.

---

## 6. London open breakout of the overnight range — REJECTED

**Date:** 2026-09-05
**Spec frozen at:** the commit adding this entry
**Code:** not yet written
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** MES, 5-minute bars resampled from 1-minute, 19:00 ET through
09:25 ET
**Source:** YouTube, "London Breakout Strategy the Right Way", demonstrated on
AUDUSD and translated here to MES.

### Mechanism claimed

The overnight range is built between 19:00 and 02:55 ET, covering the Asian
session, when the population trading US index futures is small and liquidity is
thin. At 03:00 ET the European cash session opens and a materially larger
population begins repricing the index. The claim is that the first break of the
overnight range carried by that new volume continues, because the range was
established by too few participants to represent a price the larger population
agrees with.

**Who is on the other side:** overnight participants positioned inside the range
who are stopped out as it breaks. Their stops are the liquidity the breakout
consumes.

**This is a behavioural claim, not a forced flow, and that is the entry's
weakest point.** Nobody is contractually obliged to trade against it. Entry 2's
leveraged-ETF rebalance had a genuinely forced counterparty — a fund that must
trade at a stated time regardless of price — and it still failed. What is
claimed here is that one group of traders is systematically slower or worse
positioned than another, which is the same assumption entry 1 made about
opening-range breakouts and never established.

**This is the fifth breakout-continuation hypothesis in this log, after four
rejections:** entry 1 (ORB, 5-minute close trigger), entry 4 (ORB-2, 15-minute
candle with a trend filter), entry 5 (flat by 10:30), and the 2026-09-04 ceiling
diagnostic, which found the opening-range breakout resolves equally badly at
every range width — 35.46% target share on narrow days against 35.55% on wide.

**Prior for survival: low.** Recorded before the run so a marginal positive
cannot later be read as vindication rather than noise.

**What is genuinely different from entries 1, 4 and 5:** a different session, a
different participant population, a range built over nine hours rather than
fifteen minutes, volatility-scaled stops and targets rather than fixed points,
and risk-based position sizing. It is not a retiming of the same trade. That
does not make the mechanism any better established.

### Signal definition

Fixed here. Nothing may change after results are seen.

**Bars.** 5-minute candles resampled from 1-minute data, labelled by opening
minute, left-closed.

**Overnight range.** High and low of **19:00 ET through 02:55 ET inclusive**.
The window spans midnight: the range for a trade on date D begins at 19:00 on
**D−1**. A Monday trade is measured against a range beginning Sunday 19:00, the
Globex reopen. **This is not the same as `rules.session_date`, and the code must
not assume it is** — see *Session-date compatibility* below.

**Entry window.** 03:00–05:00 ET. The **first 5-minute candle that closes
outside the range** triggers; the position is taken at the **next candle's
open**, since a candle labelled 03:00 is only known once 03:05 arrives.

**First break only. One trade per day. The opposite side is never traded that
day**, whether or not the first trade is still open.

**Trend filter.** A **200-period EMA of 5-minute closes**, computed on the
**continuous 23-hour series** (no session filter), **seeded with a 200-bar
simple mean**, and read at 03:00 from **closes strictly before 03:00** — the
same no-lookahead discipline `strategies/trend.py` enforces for entry 4, and
tested the same way. A long break is taken only if the 03:00 open is above the
EMA, a short only if below. **Pre-registered comparison: filter ON versus OFF.**

**Stop.** The opposite side of the overnight range.

**Target — two arms, both run, each judged independently:**

- **1× arm:** target = 1× range height from entry. **This is the source
  strategy.**
- **2× arm:** target = 2× range height from entry. Same stop, same everything
  else. This tests whether the signal has anything a fairer payoff could keep.

**Neither arm's result licenses a third target multiple.** Two were
pre-registered because the R:R arithmetic below made one of them near-certain to
fail for reasons unrelated to the signal; a third would be a search.

**Sizing.** `contracts = floor($200 / (range_pts × $5))`, clamped to [1, 5].
**Skip the session if `range_pts × $5 > $200`** — an overnight range wider than
40 points. Sizing is computed from the **range height**, not from the realised
stop distance; see *The sizing rule under-states risk* below.

**Flatten at 09:25 ET** if still open. No position is carried into the US cash
open.

**No entries on roll days.**

**Costs.** $1.25 per contract per side commission. **2 ticks of slippage per
side as the base case, reflecting overnight liquidity**, with **1 tick as an
optimistic sensitivity**. This inverts the convention of entries 1–5, where 1
tick was the base and 2 the stress case, and it is deliberate: the 03:00–05:00
window is thinner than RTH and a 1-tick assumption there would flatter the
result.

**Internal limits as entry 1:** the $400 daily loss limit marked to market, the
5-contract position cap, and the 30-second minimum hold. **No trailing-drawdown
halt is applied**, matching entry 1 and entry 4; evaluations blown are counted
on the continuous stream instead. Entry 5 recorded why these two are
incompatible measurements — a permanent halt makes a blow-up count degenerate —
and this entry takes the count.

**Evaluation structure: seven out-of-sample yearly folds, 2020–2026. Nothing is
selected**, so there is no training step and every fold is a genuine
out-of-sample observation of one fixed rule. 2019 is partial from 2019-05-05 and
is reportable as an extra unselected period, outside the kill criteria. **The
2026 fold is reported separately and alongside the seven**, as it is a partial
year ending 2026-08-31.

### Rules compatibility: the first overnight strategy in this project

Checked rather than assumed, because this is the first entry to trade outside
RTH.

`rules.ENTRY_CUTOFF` is 16:20 ET and `rules.FLATTEN_TIME` is 16:30 ET.
`entry_deadline` resolves to 16:20 and `flatten_deadline` to 16:30 on a normal
session. **`rules.is_entry_allowed` returns True at 03:00, 03:05, 04:55 and
05:00 ET, and `must_flatten` returns False at all of them and at 09:25.**

**No rule adjustment is required, and none is made.** But the reason matters:
these entries are permitted because 03:00 is numerically before a cutoff
designed for the *afternoon*, not because `rules.py` models an overnight
session. The module has one RTH session per calendar day and no concept of a
Globex session spanning two dates. The guards are satisfied here by accident of
arithmetic rather than by design, and a future overnight strategy that trades
later in the day could expose that. Recorded so it is not discovered as a
surprise.

### Session-date compatibility

**`rules.session_date` is the calendar date of a timestamp. The overnight range
window is not.** The range for a trade on D runs from 19:00 on D−1, so the range
window and the session date have different definitions and the code must not
assume they agree.

This is coherent for the rest of the machinery: exactly one trade per calendar
day, entered between 03:00 and 05:00 and exited by 09:25, so the daily loss
limit's grouping by `session_date` is correct for the *trade*. Only the range
construction crosses the boundary.

**A test must assert that the 19:00–02:55 range is built from the prior calendar
date's evening bars** — specifically, that a Monday session's range includes
Sunday's 19:00–23:55 bars and that shifting those bars changes the Monday range.

### The arithmetic this fixes in advance

**The 1× arm's reward-to-risk is slightly worse than 1:1.** The stop is the far
side of the range and the target is one range height from the entry. Because the
entry is the open of the candle *after* a close outside the range, it sits beyond
the range edge, so the distance to the stop is the range height **plus the
overshoot** while the distance to the target is exactly the range height.

At the median size of 2 contracts and 2 ticks per side, friction is $7.50 per
contract per round turn. A typical 1× win is about **+$150** against a typical
loss of about **−$180**, putting **break-even near 54.5%**.

**The 2× arm's break-even is near 36.4%.** Target 2× range against a stop of
about 1× range gives a 2:1 payoff; with the same friction, `p × 315 = (1 − p) ×
180` resolves to 36.4%.

**P&L is not linear in contract count here, unlike entries 4 and 5.** Size varies
per session with the range height, so the trade stream cannot be rescaled after
the fact. The size-linearity assertion that guards entries 4 and 5 does not
apply and must not be carried over.

### The sizing rule under-states risk, and that is being kept

`contracts = floor($200 / (range_pts × $5))` computes risk from the range height,
but the realised stop distance is the range height **plus the overshoot** from
the entry to the range edge. **Actual risk therefore exceeds $200 on every
trade** by the overshoot amount.

This is kept as written, matching the source. It is far inside the $400 daily
loss limit, so no guard is threatened. **The overshoot distribution and the
realised risk distribution are both reported**, so the breach is measured rather
than assumed small.

### Prediction on record

Nothing below has been computed. No backtest of this strategy has been run and
no outcome inspected — only the range distribution in the power check.

**The central prediction, and the sharpest thing in this entry.**

Entry 4's opening-range bracket had a target 1.8× its stop, and its measured
target share of bracket outcomes was **35.46% (narrow days) and 35.55% (wide)**.
For a driftless random walk with barriers at −a and +b, the probability of
touching the target first is `a / (a + b)`; for a 1.8:1 bracket that is
**35.7%**.

**Entry 4's breakout was therefore indistinguishable from a driftless random
walk, to within two-tenths of a percentage point, at every range width tested.**
That is the strongest single statement this log contains about breakout
continuation, and it is the baseline the 03:00 volume step has to beat.

So, falsifiably:

1. **The 1× arm's target share lands near 50%** — the random-walk value for a
   1:1 bracket — and specifically **inside 45–53%**, against a break-even of
   54.5%. It therefore loses roughly the friction.
2. **The 2× arm's target share lands near 33.3%** — the random-walk value for a
   2:1 bracket — and specifically **inside 29–36%**, against a break-even of
   36.4%. It also loses roughly the friction, by a smaller margin.
3. **The 03:00 volume step adds nothing measurable to continuation.** Stated
   against the requested benchmark: the London breakout's departure from its own
   random-walk value will be no larger than entry 4's, which was 0.2 points.
   **If the volume step is real, the 1× arm should clear 54.5% and the 2× arm
   36.4%; predicted, neither does.**
4. **Total P&L is negative in both arms at 2 ticks**, and the 1× arm loses more
   in dollar terms than the 2× arm, because a 1:1 payoff punishes a sub-50%
   hit rate harder than a 2:1 payoff punishes a sub-33% one.
5. **The trend filter changes little.** Entry 4 measured its filter at
   `dE = +$0.13` per contract, `t = +0.03`, acting as a long-only switch that was
   *worse* like-for-like at −$5.97. A 200-period EMA on 5-minute bars is far
   faster than a 50-day EMA and may behave differently, but nothing in this log
   suggests a trend filter on a breakout pays for the trades it removes.
6. **Volatility-scaled brackets do not rescue it.** The ceiling diagnostic
   established that breakout quality is invariant to range width. Scaling the
   bracket to the range changes the size of each outcome without changing the
   ratio between them, which is the only thing that would help.

**What would falsify the pessimism:** a target share above 54.5% in the 1× arm
or above 36.4% in the 2× arm, sustained across a majority of the seven folds,
with positive total P&L at 2 ticks. Predictions 1 and 2 are narrow enough that
being wrong will be obvious immediately.

### Power check, run before any strategy code

Following entry 2's precedent. Signal distribution and bar coverage only; it
never looks at what price did after 03:00.

**Coverage is not a problem.** 1,895 London sessions have bars in the range
window; **1,892 are usable**, with a median of 476 one-minute bars in the range
window (complete is 475) and 120 in the entry window (complete is 120). Three
sessions are too thin to use.

**Overnight range height (points):**

| p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 8.2 | 12.0 | 17.5 | 28.5 | 43.0 | 59.2 | 97.4 | 188.0 |

**The 40-point cap skips 216 of 1,892 sessions — 11.4% — leaving 1,676
tradeable.** Recorded because entry 4's 10-point opening-range ceiling skipped
**70%** and made that strategy a low-volatility strategy by construction. This
cap does not do that, and the test has real power.

| Year | Sessions | Median range | Skipped | Tradeable |
|---|---|---|---|---|
| 2019 | 171 | 11.00 | 1 | 170 |
| 2020 | 258 | 23.25 | 51 | 207 |
| 2021 | 259 | 15.50 | 15 | 244 |
| 2022 | 258 | 24.38 | 38 | 220 |
| 2023 | 258 | 12.00 | 5 | 253 |
| 2024 | 259 | 13.50 | 13 | 246 |
| 2025 | 257 | 22.00 | 45 | 212 |
| 2026 | 172 | 31.88 | 48 | 124 |

**Sizing over the 1,676 tradeable sessions:** 1 contract 34.5%, 2 contracts
29.2%, 3 contracts 17.2%, 4 contracts 8.7%, 5 contracts 10.3%. Median risk $165,
mean $159, max $200; only 8 sessions risk under $100.

**Regime note, recorded in advance:** 2026's median overnight range is 31.88
points against 2023's 12.00, and 2026 skips 28% of its sessions against 2023's
2%. Any year-to-year difference in results must be read against that before
being attributed to the strategy.

### Pre-registered tests

1. **Per-fold P&L and Sharpe**, seven out-of-sample years 2020–2026, both arms,
   filter ON and OFF.
2. **The 2026 fold reported separately and alongside the seven.**
3. **Pooled hit rate against the break-even implied by the realised
   reward-to-risk** — computed from the trades' own realised stop and target
   distances, not from the nominal multiples, since the overshoot moves it.
4. **P&L and counts by exit reason:** stop / target / 09:25 flatten.
5. **Sizing distribution** — contracts per trade, and the realised risk
   distribution in dollars.
6. **Overshoot distribution** — entry price minus range edge, in points and as a
   fraction of range height.
7. **`eval_sim` pass probability**, per fold and pooled.
8. **Evaluations blown across the seven years.**
9. **Filter ON versus OFF**, including the **long-only-versus-long-only
   diagnostic** from entry 4: if the filter's advantage disappears when longs
   are compared with longs, it is a directional switch rather than a trend
   filter.
10. Everything repeated at **1 tick** as the optimistic sensitivity.

### Kill criteria — decided now

Any **one** kills an arm. **Each arm is judged independently** against these,
measured pooled over the seven out-of-sample years at the base 2 ticks per side.

1. **Pooled `eval_sim` pass probability below 25%.**
2. **More than 1 evaluation blown across the seven years.**
3. **Fewer than 4 of 7 folds profitable, *or* total P&L negative.** Survival
   requires **both**.

Non-fatal, with a reporting obligation:

4. **The verdict must state explicitly whether the breakout or the trend filter
   is carrying any result**, in those terms, whatever the outcome. Carried over
   from entry 4, including its obligation: a combined figure without attribution
   does not satisfy this entry.

No appeal, no re-grid, no third target multiple.

### Longer-term intent: copying trades across multiple funded accounts

Unchanged and repeated because it governs how any accepted result would be used.
Copying identical trades across N funded accounts **multiplies outcomes in both
directions and is not diversification**: the same losing day draws down every
account at once and a trailing-drawdown breach terminates all of them on the
same date. It is one bet at N times the size with N times the fees. Only the
evaluation *attempt* is diversified, and only while accounts start at different
times on different price paths.

### Implementation details the spec left open, fixed before any run

Following entry 2's precedent: these were not settled by the specification
above, they are settled now, **before any outcome has been observed**, and they
do not alter the kill criteria.

**Fill resolution: 1-minute bars.** The spec fixes 5-minute bars for the
*signal* and is silent on fills. Entry, stop and target all resolve on 1-minute
bars, and **if one bar touches both the stop and the target, the stop is assumed
filled** - the convention entries 1, 4 and 5 all use. Signals are therefore
generated on the 1-minute index with the 5-minute candles built internally, the
same shape ORB-2 uses.

**The entry bar is included in the exit search.** The fill happens inside it, so
the rest of that minute can reach the stop. Trades entering and exiting inside
one bar are counted and reported, since rule 6's 30-second floor is not
verifiable at this resolution.

**A trigger in the window is honoured even if its action bar is not.** A
5-minute candle labelled 04:55 closes at 04:59:59 and is acted on at the 05:00
open. The window governs the *trigger*, not the fill, so that trade is taken.

**The 09:25 flatten lands on the open of the 09:25 bar**, not the close of the
bar before it, matching "flatten at 09:25".

**Minimum range coverage: 60 five-minute bars.** A complete 19:00-02:55 window
holds 107. Sessions below the floor are skipped and counted; the power check
found only 3 of 1,895 that thin.

**Roll days are skipped on the trade date D**, not on the date the range began.

### Verdict: REJECTED — both arms, both filter states, both slippage levels

**Date:** 2026-09-05
**Spec frozen at:** `068927d`; implementation details at `6fe805f`
**Verdict commit:** `9b28b3c`
**Code:** `strategies/london.py`, `strategies/trend.py` (`intraday_ema`),
`backtests/run_london.py`
**Reports:** `backtests/results/london_report_slip2.txt` (base),
`..._slip1.txt` (optimistic)

Seven out-of-sample years, 2020–2026, nothing selected. **All eight
configurations fail all three kill criteria.**

#### Pooled out-of-sample, base case (2 ticks per side)

| | 1×/ON | 1×/OFF | 2×/ON | 2×/OFF |
|---|---|---|---|---|
| Trades | 686 | 1,025 | 686 | 1,025 |
| Net P&L | −$7,702 | **−$19,502** | −$9,765 | **−$20,781** |
| Sharpe | −1.37 | −2.27 | −1.52 | −2.08 |
| Profit factor | 0.816 | 0.716 | 0.785 | 0.721 |
| Max drawdown | −$9,150 | −$20,994 | −$10,828 | −$23,471 |
| Targets / stops | 197 / 144 | 286 / 249 | 47 / 155 | 82 / 269 |
| 09:25 flattens | 345 | 490 | 484 | 674 |
| **Target share** | **57.77%** | **53.46%** | **23.27%** | **23.36%** |
| Break-even (realised) | 58.19% | 58.35% | 39.34% | 39.54% |
| Random-walk value | 52.19% | 52.17% | 35.31% | 35.29% |
| Evaluations blown | 5 | 12 | 5 | 14 |
| Pass probability | 0.83% | 0.10% | 1.38% | 0.62% |
| Folds profitable | 1 of 7 | 1 of 7 | 0 of 7 | 2 of 7 |

At **1 tick** (optimistic): −$3,842 / −$13,580 / −$5,905 / −$14,859, pass
probabilities 3.94% / 0.56% / 4.45% / 1.81%, evaluations blown 4 / 8 / 4 / 12,
and at most 2 of 7 folds profitable. **Every arm fails every criterion at both
cost assumptions.** Halving the assumed slippage removes about a third of the
loss and changes no verdict.

#### Kill criteria

| # | Criterion | 1× arm | 2× arm | |
|---|---|---|---|---|
| 1 | Pass probability ≥ 25% | 0.10% / 0.83% | 0.62% / 1.38% | **FAIL** |
| 2 | Evaluations blown ≤ 1 | 12 / 5 | 14 / 5 | **FAIL** |
| 3 | ≥4 of 7 folds **and** P&L > 0 | 1 of 7, −$19,502 | 2 of 7, −$20,781 | **FAIL** |

#### Criterion 4: is the breakout or the filter carrying any result?

**Neither carries a positive result. The filter reduces the loss by about 40%
and is not statistically distinguishable from zero; the breakout bracket is
close to break-even and the 09:25 flatten is what loses the money.**

| | dE (ON − OFF) | SE | t | Long-only dE |
|---|---|---|---|---|
| 1× arm | **+$7.80** / trade | $6.46 | **+1.21** | +$5.24 (SE $8.22) |
| 2× arm | +$6.04 / trade | $7.46 | +0.81 | +$4.03 (SE $9.41) |

**This is the first entry in which the trend filter did not fail its own
diagnostic.** Entry 4's filter measured `dE = +$0.13` (t = +0.03) and *reversed*
to −$5.97 when longs were compared with longs. Here the filter is positive on
both measures and survives the long-only comparison, at 60.5% long against
OFF's 55.4% — so it is not merely a directional switch. But `t = +1.21` is not
a result, the ON arms still lose $7,702 and $9,765, and eight configurations
were examined. **Recorded as the first non-failure of a trend filter in this
log, not as evidence one works.**

#### My random-walk benchmark was wrong, and the way it was wrong is the finding

The entry pre-registered that the driftless random-walk value `a / (a + b)` is
the baseline the 03:00 volume step must beat. **That benchmark is invalid under
a time limit, and I should have said so before the run rather than after.**

`a / (a + b)` is the probability of touching one barrier before the other with
*unlimited* time. Entry 6 flattens at 09:25, so the farther barrier is
systematically under-reached. The distortion scales with how far the target
sits:

| | Target share | Random walk | Departure | z |
|---|---|---|---|---|
| 1×/OFF | 53.46% | 52.17% | **+1.29** | +0.60 |
| 1×/ON | 57.77% | 52.19% | **+5.58** | +2.06 |
| 2×/OFF | 23.36% | 35.29% | **−11.93** | −4.68 |
| 2×/ON | 23.27% | 35.31% | **−12.04** | −3.58 |

The 2× arm's twelve-point shortfall is **not evidence the signal is bad**; it is
the time limit truncating a distant target — 674 of its 1,025 trades never
resolve at all. In the 1× arm the two barriers are near-equidistant (stop =
range + overshoot ≈ 1.07× range against a 1.0× range target), so truncation
falls on both roughly equally and the comparison is close to fair.

**On that fair comparison the London breakout sits slightly above a driftless
random walk** — +1.29 points unfiltered (z = +0.60), +5.58 filtered (z = +2.06).
Entry 4's ORB sat within **0.2** points of its own random-walk value. So the
03:00 volume step does appear to add something entry 1's opening range did not.

**It is not enough, and it is not established.** The unfiltered departure is
inside noise. The filtered z of +2.06 is nominally significant but is one of
eight configurations examined, which is roughly where a single z ≈ 2 is expected
by chance, and no significance test was pre-registered. Above all it fails the
economic test: break-even at 2 ticks is 58.35% and the arm delivered 53.46%.

#### Where the money actually goes: the flatten, and its sign is arithmetic

Mean P&L per trade by exit reason, base case:

| | 1×/ON | 1×/OFF | 2×/ON | 2×/OFF |
|---|---|---|---|---|
| Target | +$141.78 | +$140.98 | +$306.60 | +$304.88 |
| Stop | −$197.31 | −$197.51 | −$198.85 | −$199.42 |
| **09:25 flatten** | **−$20.93** | **−$21.72** | **+$13.73** | **+$11.67** |

**The flatten loses in the 1× arm and earns in the 2× arm, and that sign is
determined by the bracket's geometry rather than by anything the market does.**
A trade surviving to 09:25 is conditioned into the band between its stop and its
target. In the 1× arm that band is (−1.07R, +1.0R) — the stop is *farther*
because of the overshoot — so its midpoint is negative and the flatten loses. In
the 2× arm the band is (−1.07R, +2.0R), midpoint strongly positive, and the
flatten earns.

**This is the third confirmation of the truncation account from entry 4's
2026-09-04 addendum, and the first in the negative direction.** Entry 4's
bracket (stop 10, target 18) had a band midpoint of +4 points and a flatten mean
of +$39.91; entry 5 cut the hold to 45 minutes and the premium collapsed to
+$2.64; here an inverted band produces a *negative* premium. The account has now
predicted the sign and the magnitude of an unresolved-trade population three
times across three different strategies.

**The consequence for entry 6 is the sharpest number in it.** In the 1×/ON arm
at 2 ticks, the bracket alone nets **−$481** across 341 resolved trades — within
a rounding error of break-even, and by far the best any breakout bracket has
managed in this log. The 345 flattens then lose **$7,221**, which is the entire
result. At 1 tick the bracket is actually **+$1,814** and the flattens still
turn it into a −$3,842 loss.

**So the strategy does not fail because the breakout is worthless. It fails
because half its trades never resolve, and the unresolved half is priced against
it by the geometry of its own bracket.**

#### Predictions scored — five of six wrong

This is the worst-performing prediction set in the log, and it is recorded as
such rather than softened.

**1. 1× target share inside 45–53% — WRONG.** 53.46% unfiltered (just outside)
and 57.77% filtered (well outside).

**2. 2× target share inside 29–36% — WRONG.** 23.3%, twelve points below the
band, for the time-limit reason above that the prediction did not account for.

**3. Departure from random walk no larger than entry 4's 0.2 points — WRONG.**
+1.29 unfiltered, +5.58 filtered, −12 in the 2× arm.

**4. Both arms negative, with the 1× arm losing more in dollars — HALF WRONG.**
Both negative, correct. But the **2× arm lost more** in every configuration
(−$20,781 against −$19,502 unfiltered; −$9,765 against −$7,702 filtered). The
reasoning behind the ordering was that a 1:1 payoff punishes a sub-50% hit rate
harder than a 2:1 payoff punishes a sub-33% one — which ignored that the 2× arm
converts most of its trades into flattens and takes its losses through a far
worse target-to-stop count (82 against 269).

**5. The trend filter changes little — WRONG.** It improves mean P&L per trade
by $7.80 and cuts the loss roughly in half. Not significant, but not "little"
either, and unlike entry 4's it survives the long-only diagnostic.

**6. Volatility-scaled brackets do not rescue it — CORRECT.** Both arms
rejected on every criterion at both cost levels.

**One prediction of six held, and it was the least specific.** The pattern in
the misses is consistent: every one came from carrying entry 4's opening-range
findings onto a different session and a different bracket geometry without
checking whether the machinery transferred. The random-walk benchmark in
particular was imported wholesale and is simply not valid where a time limit
binds.

#### Other pre-registered reporting

**Sizing.** Over the 1,025 unfiltered trades: 1 contract 34.1%, 2 contracts
29.8%, 3 contracts 17.1%, 4 contracts 8.9%, 5 contracts 10.1% — within a point
of the power check's forecast on every bucket.

**Overshoot.** Median **1.00 point**, mean 1.62, p90 3.25, **max 50.50**. As a
fraction of the range: median 7.1%, mean 10.0%.

**Realised risk, and the sizing rule's breach.** Median **$180**, mean $176,
**max $416** against a rule that sizes for $200. **240 of 1,025 trades (23.4%)
risk more than the stated $200**, exactly as the entry predicted when it kept
the rule as written. One trade's stop-loss exceeded the **$400 internal daily
loss limit** — the sizing rule permits a position whose full stop is larger than
the limit that is supposed to contain the day. No daily-loss halt fired in any
configuration, so it never bound in practice, but the rule allows it and that is
a defect worth recording.

**Skipped sessions**, 2020–2026: 1,025 traded, 470 no break inside the window,
344 with no London session at all (weekends — a Sunday group holds only the
18:00–18:59 reopen), 210 over the 40-point cap, 26 roll days, 2 too thin.
Range-cap skips had a median range of 54.25 points and a maximum of 188.

**The 2026 fold, separately.** 1×/ON +$661 (Sharpe 1.53, 58 trades), 1×/OFF
+$325 (81 trades), 2×/ON −$372, 2×/OFF +$32. 2026 is the only profitable year
for the 1× unfiltered arm. On 58–81 trades this is not a measurement, and the
entry's pre-registered regime note applies: 2026's median overnight range is
31.88 points against 2023's 12.00.

**Rule 6.** **Zero** trades entered and exited inside one 1-minute bar in any
configuration, so the 30-second minimum hold is fully verifiable here — unlike
entry 4, where 7 trades were not.

### What was learned

**The 03:00 volume step is the first breakout mechanism in this log to leave a
mark, and the mark is too small to pay for itself.** Entry 4's ORB was
indistinguishable from a coin flip to two-tenths of a point. The London 1× arm
sits 1.3 points above its random-walk value unfiltered and 5.6 filtered. That is
the difference between "no effect" and "an effect inside the noise, smaller than
the spread" — a real distinction, and not one that changes the decision.

**The bracket was nearly break-even and the exit rule destroyed it.** −$481
across 341 resolved trades at the base cost is the best breakout bracket this
project has measured. The strategy still lost $7,702 because 345 unresolved
trades were flattened out of a band skewed against them. **Where a bracket is
asymmetric, the unresolved population is not neutral — its expected value is the
band's midpoint, and that is a design choice, not a market outcome.**

**A benchmark imported without checking its assumptions is worse than no
benchmark.** The random-walk value was pre-registered as the thing to beat and
is invalid wherever a time limit binds. It produced a twelve-point "shortfall"
in the 2× arm that means nothing about the signal. Future entries using
`a / (a + b)` must state whether the holding period is long enough for it to
apply, and if it is not, must compare against a time-limited simulation instead.

**Five wrong predictions out of six.** Every miss came from assuming entry 4's
opening-range results would transfer to a different session and a different
bracket. They did not, in both directions: the signal was better than predicted
and the exit was far worse.

### Next

**Do not test a third target multiple, a different flatten time, or a different
range window.** Entry 6 fixed two arms in advance precisely so that a third
could not be justified by whatever the first two returned, and the flatten
finding above is exactly the kind of result that invites one. The observation
that a wider target makes the flatten profitable is arithmetic, not an edge; the
2× arm already has that property and lost more money.

**The mechanism remains unestablished.** No order-flow evidence was gathered
here either. The one thing that would justify returning to breakouts is what
entry 1 asked for and has never been produced: direct measurement of who takes
the other side of a range break and under what constraint.

**If anything in this entry is worth carrying forward it is the exit, not the
entry.** Three entries have now shown that the fate of unresolved trades is set
by bracket geometry rather than by the signal. An idea about *that* would be a
different hypothesis with a different mechanism, and it would need its own entry
written before any code — not an extension of this one.

### Addendum, 2026-09-05 — the sizing rule breached its own risk cap

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** This records a defect the run measured and fixes it for future entries.

Entry 6 sized positions as `contracts = floor($200 / (range_pts × $5))`,
computed from the **range height**, while the realised stop sat at the range
height **plus the overshoot** from the entry to the range edge. The entry kept
that as written, matching the source, and required the breach to be measured
rather than assumed small.

**Measured, on the 1,025 unfiltered trades:**

| | |
|---|---|
| Realised risk, median | **$180** |
| Realised risk, mean | $176 |
| Realised risk, **max** | **$416** |
| Trades risking more than the stated $200 | **240 of 1,025 (23.4%)** |
| Trades whose stop exceeded the **$400 daily loss limit** | **1** |

The overshoot was a median of 1.00 point and a mean of 1.62 (7.1% and 10.0% of
the range), with a maximum of 50.50 points. No daily-loss halt fired in any
configuration, so the limit never actually bound — but **the rule permitted a
position whose full stop-loss was larger than the daily loss limit that is
supposed to contain the day**, which is a guard defeating itself by arithmetic.

#### Standing rule for future entries

**Any future use of range-based sizing must size off the realised stop distance,
or cap at the daily loss limit, whichever binds first.** Concretely, the
position must satisfy both:

```
contracts × stop_distance × point_value  <=  risk_budget
contracts × stop_distance × point_value  <=  rules.DAILY_LOSS_LIMIT
```

where `stop_distance` is the distance from the actual fill to the actual stop,
not a proxy for it. A rule that sizes off a quantity the stop does not use is
not a risk rule; it is an estimate of one.

**This binds new strategies, not replications.** Entry 7 reproduces entry 6's
specification unchanged, sizing flaw included, because altering a constant would
make it something other than a replication. That exemption is deliberate,
applies only to a like-for-like reproduction of an already-frozen spec, and does
not extend to any entry proposing a strategy of its own.

### Addendum, 2026-09-11 — re-priced at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 this entry assumed (Lucid
support, 2026-09-11, article 11508978; method in entry 1's addendum of the
same date). All eight saved streams were re-priced exactly. This entry sizes
per session, so each trade's size was recovered from the commission it
carried — 1 to 5 contracts, $1.50 to $7.50 a trade. Nothing is selected in
this entry's walk-forward and no stream carries a daily-loss exit, so the
re-pricing is the full effect of the correction.

**Base case, 2 ticks, pooled out-of-sample 2020–2026:**

| | 1×/ON | 1×/OFF | 2×/ON | 2×/OFF |
|---|---|---|---|---|
| Net P&L as scored | −$7,703 | −$19,503 | −$9,765 | −$20,781 |
| **Net P&L re-priced** | **−$5,387** | **−$15,949** | **−$7,449** | **−$17,228** |
| Folds profitable | 1 of 7 → 1 of 7 | 1 of 7 → 1 of 7 | 0 of 7 → 0 of 7 | 2 of 7 → 2 of 7 |
| Evaluations blown | 5 → 4 | 12 → 10 | 5 → 5 | 14 → 12 |
| Pass probability | 0.83% → 2.28% | 0.10% → 0.34% | 1.38% → 2.83% | 0.62% → 1.20% |
| Payout probability ($52,100) | 3.90% → 7.83% | 0.81% → 1.70% | 5.22% → 8.40% | 2.54% → 4.20% |
| Max drawdown | $9,150 → $6,996 | $20,994 → $17,688 | $10,828 → $8,649 | $23,471 → $20,080 |

Optimistic case, 1 tick: 1×/ON −$3,843 → −$1,527 (2 of 7 folds, 8.52% pass,
2 blown); 1×/OFF −$13,580 → −$10,027; 2×/ON −$5,905 → −$3,589; 2×/OFF
−$14,859 → −$11,305.

**Would any kill criterion have resolved differently? No, on any of the eight
configurations.** Pass probability at best 2.83% against 25%; evaluations
blown at best 4 against 1; folds profitable at best 2 of 7 against 4, with
every total negative.

**Break-even, restated — and one sub-finding changes sign.** The verdict's
"break-even at 2 ticks is 58.35%" was the realised figure for 1×/OFF, mean
stop over mean stop plus mean target. Re-priced: 1×/ON **58.19% → 57.00%**,
1×/OFF **58.35% → 57.14%**, 2×/ON 39.34% → 38.48%, 2×/OFF 39.54% → 38.68%.
Target shares are unchanged (57.77%, 53.46%, 23.27%, 23.36%). The 1×/OFF arm
still falls short of its break-even by 3.7 points. **The 1×/ON arm's bracket
alone now clears its break-even**: 57.77% against 57.00%, where at $1.25 it
sat 0.4 points under. In dollars the resolved trades — 197 targets and 144
stops — go from **−$481 to +$896**, and the arm's 345 flattens at 09:25 go
from −$7,221 to −$6,283. The arm loses $5,387 in total. This is the finding
the verdict already recorded, sharpened: the bracket was a rounding error
from break-even and is now fractionally past it, and the exit rule is the
whole loss. It is not evidence of an edge — a $1,377 swing across 341 trades
is $4 a trade, inside the noise the verdict measured — and the London family
stays closed.


---

## 7. Replication of entry 6's one non-null finding, on MNQ — REJECTED

**Date:** 2026-09-05
**Spec frozen at:** the commit adding this entry
**Code:** not yet written
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** **MNQ**, 5-minute bars resampled from 1-minute, 19:00 ET
through 09:25 ET
**Replicates:** entry 6 (`068927d` spec, `9b28b3c` verdict), 1× arm, filter ON,
base case 2 ticks per side.

### This is a replication, not a strategy

**No new strategy is proposed and no new mechanism is claimed.** Entry 6 is
rejected and stays rejected. This entry exists to answer one question about one
number, on data the specification has never touched.

Entry 6 produced exactly one result that was not null: in the trend-filtered 1×
arm, the target share of bracket outcomes was **57.77%** against a random-walk
benchmark of **52.19%** — a departure of **+5.58 points, z = 2.06**. Every other
figure in that entry was a loss or inside noise.

That number is the only reason the London family is not already closed, and it
has three specific weaknesses:

1. **It was one of eight configurations examined** (two arms × filter ON/OFF ×
   two slippage levels), with no correction for that.
2. **No significance test was pre-registered.** The z was computed after the
   fact, which is exactly the practice this log exists to prevent.
3. **The benchmark it was measured against was wrong.** Entry 6's own verdict
   established that `a / (a + b)` assumes unlimited time and is invalid where a
   flatten truncates the horizon.

This entry fixes all three at once: a different instrument, one pre-registered
test, and a corrected benchmark.

### The benchmark entry 6 used was wrong, so the MES figure is recomputed too

**A replication that compared a corrected MNQ number against an uncorrected MES
number would measure nothing.** So the corrected benchmark below is applied to
**both** instruments, and entry 6's MES departure is recomputed under it.

**This is stated in advance because it can dissolve the finding without MNQ
saying anything at all.** If the corrected MES departure falls below +2 points
or z below 1.65, then entry 6's non-null result was an artefact of its own
benchmark and there was never anything to replicate. That outcome is recorded as
a failure to replicate exactly as an MNQ null would be, and it closes the family
the same way.

### The corrected benchmark: a de-meaned bootstrap, fixed now

The question is what fraction of trades would resolve at the target first if the
price process had **no drift**, given the **same barrier distances** and the
**same time limit**. That is not `a / (a + b)`, which assumes the horizon is
unbounded.

**Procedure, fixed here and not to be varied after results are seen:**

For each realised trade in the arm under test:

1. Take the **1-minute arithmetic price changes** of the series from the entry
   bar through the flatten bar — the trade's actual holding window, at its
   actual length.
2. **Subtract the window's own mean change**, so the resampled process has zero
   drift by construction while keeping the window's realised volatility and the
   fat tails of its return distribution.
3. **Resample those de-meaned changes with replacement** to the same length, and
   rebuild a price path forward from the **actual entry price**.
4. Apply the **same stop, the same target and the same flatten bar**, with the
   stop assumed filled when one step reaches both — identical to the live rule.
5. Repeat **1,000 times per trade**, seeded at **0**.

**The benchmark target share is the pooled count of target-first outcomes
divided by the pooled count of resolved outcomes** across every replication of
every trade — flattens excluded on both sides, exactly as the observed target
share excludes them.

Sampling with replacement rather than permuting is deliberate: a permutation
holds the terminal price fixed and would test only path order, not drift.

### Pre-registered test — one test, decided now

**Observed target share versus the bootstrap benchmark, pooled 2020–2026, on the
MNQ 1× arm with the filter ON at 2 ticks per side.**

```
departure = observed_target_share - benchmark_target_share      (percentage points)
z         = departure / sqrt( p0 (1 - p0) / n_resolved )        p0 = benchmark
```

**One-sided z-test, H1: observed > benchmark, α = 0.05, critical value
z = 1.65.** One test, pre-registered, on one configuration. No other cell of
entry 6's grid is tested for significance here and none may be added afterwards.

The same statistic is computed on MES for comparison, and reported, but **MNQ is
the replication and MES is the recomputation of the original**.

### Kill criterion — decided now

**A departure below +2 points, or z below 1.65, means entry 6's finding is
recorded as NOT REPLICATED and the London family closes.** No further London
entry, no further instrument, no further arm.

The +2-point floor sits alongside the significance test on purpose: a departure
could clear z = 1.65 on a large sample while being far too small to matter
economically, and entry 6 already showed that a 5.58-point departure loses
money. Both conditions must hold.

For completeness and comparability, **entry 6's three kill criteria are also
reported** on the MNQ run — pooled pass probability against 25%, evaluations
blown against 1, and folds profitable against 4 of 7 with positive total P&L.
**They are not the replication question.** Entry 6 already failed all three on
MES and this entry expects the same on MNQ; a strategy that fails them can still
carry a real statistical departure, and it is the departure being tested.

### Prediction on record, and which the honest prior favours

**If entry 6's departure was signal**, MNQ shows a departure of the **same
positive sign with z ≥ 1.65**.

**If it was noise**, MNQ sits **within ±1 point of the benchmark**, with z
indistinguishable from zero.

**The honest prior favours noise, and by some distance.** Three reasons, all
available before the run:

**The observed z is almost exactly what eight draws of noise produce.** The
expected maximum of `n` independent standard normals is approximately
`sqrt(2 ln n)`; for the eight configurations entry 6 examined that is
**2.04**. The reported z was **2.06**. The single most extreme result from eight
looks at the data landed within two-hundredths of where pure chance puts it.

**Entry 6's prediction record was five wrong out of six**, and its verdict
attributed every miss to carrying assumptions across contexts without checking
whether the machinery transferred. The same caution applies to carrying a z
across instruments.

**Four prior breakout entries produced nothing**, and the mechanism has never
been measured independently of backtest P&L in any of them.

**Recorded plainly: this entry expects to close the London family.** It is being
run because a single pre-registered test on untouched data is cheap, and because
recording a clean failure to replicate is worth more than leaving a
marginally-significant number in the log unchallenged.

### Instrument translation — what changes and what does not

**Every constant of entry 6's specification is unchanged.** The range window
(19:00–02:55 ET), the entry window (03:00–05:00), the first 5-minute close
outside the range acted on at the next candle's open, first break only, one
trade per day, the 200-period EMA of 5-minute closes on the continuous 23-hour
series with the filter ON, the stop at the far side of the range, the 1× target,
the 09:25 flatten, roll days skipped, $1.25 per contract per side, 2 ticks of
slippage per side, the $400 daily loss limit, the 5-contract cap, and no
trailing-drawdown halt.

**What follows from the contract rather than from the spec:** MNQ is **$2.00 per
point** against MES's $5.00. The risk rule is stated in dollars and is unchanged
— `contracts = floor($200 / (range_pts × point_value))`, clamped to [1, 5] — so
its expression in points moves with the contract. **The skip threshold becomes a
range wider than 100 points** (`$200 / $2`), where MES's was 40.

This is the same rule, not a different one. Restating the threshold in points
would have made it a different rule.

**Entry 6's sizing defect is reproduced deliberately.** The 2026-09-05 addendum
to entry 6 requires future range-based sizing to size off the realised stop
distance or cap at the daily limit. **That rule does not apply here**, because
changing a constant would make this something other than a replication. The
exemption is limited to this entry.

### Data

MNQ.v.0, `ohlcv-1m`, GLBX.MDP3, **2019-05 through 2026-08**, matching the MES
cache's span.

**Cost is estimated before anything is pulled and the pull stops above $15.**
Databento metadata calls are free and timeseries calls are not; the project has
spent about $12 to date. The estimate is reported whatever it is.

**`data/validate.py` is run on the result and its report recorded in the
verdict**, including bar count, span, session coverage and any gaps. A
replication on unvalidated data would be worthless, and the MNQ series has never
been checked in this project.

**If MNQ.v.0 does not resolve, or the estimate exceeds $15, or validation shows
material gaps in the 19:00–05:00 window, this entry records that and stops.**
A failed data step is not a failure to replicate and must not be recorded as
one.

### Pre-registered reporting

1. The **cost estimate** and the **`validate.py` report** for the MNQ pull.
2. **Observed target share, benchmark target share, departure and z** for MNQ
   1×/ON at 2 ticks — the test.
3. **The same four figures recomputed for MES**, so the comparison is
   like-for-like.
4. **Entry 6's three kill criteria** on MNQ, for completeness.
5. Trade count, net P&L, per-fold P&L for 2020–2026, and the exit-reason
   breakdown, so the MNQ run can be read against the MES one.
6. **The bootstrap's own diagnostics:** replications run, mean resolved fraction,
   and the benchmark's dispersion across trades.

### Verdict: NOT REPLICATED — the London family closes

**Date:** 2026-09-05
**Spec frozen at:** `bff2bdc`
**Verdict commit:** `b5d25ba`
**Code:** `backtests/bootstrap_benchmark.py`, `backtests/run_entry7.py`
**Report:** `backtests/results/entry7_report.txt`
**Data:** `data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet`

#### The data step

MNQ.v.0 estimated at **$9.4256** against the $15 cap and was pulled:
**2,581,801 bars, 42.58 MB, 2019-05-05 to 2026-08-31**, matching the MES span.
Project data spend goes from about $12 to about $21.43.

`validate.py` reported **no OHLC integrity violations and no zero-volume RTH
bars**. Its findings are the expected ones: 7 exchange holidays, 64 half-days,
14 sessions with overnight bars but no RTH, and 9 bar-to-bar moves above 2%
(March 2020, the 2022 CPI prints, April 2025).

That report is RTH-focused and this strategy trades 19:00–09:25, so the
overnight window was checked separately, as the entry requires: **1,888 usable
London sessions against MES's 1,892, and 1,720 against 1,721 across 2020–2026.**
No material gaps. The entry proceeded rather than stopping.

#### The test

| | MES (recomputed) | MNQ (replication) |
|---|---|---|
| Trades | 686 | 438 |
| Targets / stops | 197 / 144 | 134 / 100 |
| 09:25 flattens | 345 | 204 |
| **Observed target share** | **57.77%** | **57.26%** |
| **Bootstrap benchmark** | **53.81%** | **53.61%** |
| **Departure** | **+3.96 pts** | **+3.65 pts** |
| **z (one-sided)** | **+1.47** | **+1.12** |
| Clears z = 1.65? | no | **no** |
| Clears +2.0 points? | yes | yes |

Bootstrap: 1,000 replications per trade, seed 0, mean resolved fraction 52.67%
(MES) and 56.93% (MNQ), per-trade benchmark dispersion 0.069 on both.

**MNQ's departure clears the +2-point floor but its z is 1.12, below the
pre-registered 1.65. The kill criterion fires on either condition. NOT
REPLICATED, and the London family closes.**

#### Entry 6's benchmark was doing most of the work

| | Departure | z |
|---|---|---|
| MES under `a / (a + b)` (entry 6's figure) | +5.59 | **+2.06** |
| MES under the corrected bootstrap | **+3.96** | **+1.47** |

**Entry 6's headline result does not survive its own correction.** The
infinite-horizon formula put the benchmark at 52.18% where the finite-horizon
bootstrap puts it at 53.81% — and that 1.6-point difference is enough to drop
the z from 2.06 to 1.47, below the threshold this entry pre-registered.

The entry anticipated this exact outcome and recorded in advance that it would
count as a failure to replicate: *"that recomputation can dissolve the finding
without MNQ saying anything at all."* It largely did. **MNQ then agreed.**

#### What actually happened is more interesting than either prediction

The pre-registered prediction had two branches. **Neither is what occurred.**

- *If signal:* MNQ shows the same sign with **z ≥ 1.65**. It did not — z = 1.12.
- *If noise:* MNQ sits **within ±1 point** of the benchmark. It did not — the
  departure is +3.65 points.

**Both instruments show a small positive departure of nearly identical size —
+3.96 and +3.65 points — and neither reaches significance.** That is a third
outcome: not "the effect vanished on a new instrument", but "the same small
effect appears on both and is too small to distinguish from noise at these
sample sizes."

The distinction matters for what the log now says. This is **not** evidence that
London breakouts are a coin flip; it is evidence that if anything is there, it
is smaller than 686 and 438 trades can resolve, and far smaller than the spread.
Entry 6 already showed the economic version of the same statement: a 57.77%
target share against a 58.35% break-even loses money.

**The honest prior in the entry was right about the conclusion and wrong about
the mechanism of it.** The reasoning recorded in advance was that the reported
z = 2.06 sat within two-hundredths of `sqrt(2 ln 8) = 2.04`, the expected
maximum of eight noise draws. That argument predicted a null on MNQ. What the
data shows instead is a consistent sub-threshold departure on both — the
selection-effect argument was sound about the significance and wrong about the
point estimate.

#### A bug nearly produced a false positive, and it is worth recording

The first run of this test reported **MNQ departure +16.94 points, z = +4.69,
REPLICATED**. That was wrong, and the cause was mine.

`run_london.apply_daily_limit` hardcoded the **MES** contract spec. Running MNQ
through it marked positions at $5.00 a point instead of $2.00, inflating open
P&L two and a half times and firing **61 spurious daily-loss exits**. Those
exits removed would-be *stops* from the bracket — the daily-loss guard truncates
losing trades before they reach their stop — which pushed the observed target
share from 57.26% to 70.2% while the benchmark, which does not model the daily
limit, stayed put.

**The failure mode is the dangerous one: a silent unit mismatch that inflates
exactly the statistic under test, in the favourable direction.** It was caught
because the exit-reason table showed a `loss_limit_flatten` row on MNQ that MES
did not have, and stop/target means near ±$380 against a $200 risk budget.
`apply_daily_limit` now takes the spec as an argument.

Recorded because the corrected MNQ figures are the ones above, and because
anything that reads this log should know that a positive replication was on the
screen for several minutes before it turned out to be a hardcoded constant.

#### Entry 6's kill criteria on MNQ — for completeness, not the test

| | MES | MNQ |
|---|---|---|
| Net P&L | −$7,702 | **+$116** |
| Sharpe | −1.37 | 0.03 |
| Profit factor | 0.816 | 1.005 |
| Max drawdown | −$9,150 | −$2,686 |
| Folds profitable | 1 of 7 | **4 of 7** |
| Evaluations blown | 5 | **1** |
| Pass probability | 0.83% | **13.56%** |

**MNQ passes two of entry 6's three criteria and fails the first.** Net P&L is
+$116 across 438 trades — indistinguishable from zero, on a strategy that lost
$7,702 on MES — with 4 of 7 folds profitable and 1 evaluation blown.

**This does not reopen anything, and the entry pre-registered that it would
not.** Criterion 1 fails at 13.56% against a 25% floor, so the strategy is
rejected on MNQ as it was on MES. More to the point, entry 7's own kill
criterion has already fired: the replication question is the significance test,
and it failed. A near-breakeven P&L on the one instrument that was tested second
is exactly the shape of result that invites a third instrument, and the entry
forbids one.

**No pooled MES+MNQ test is performed.** Pooling was not pre-registered, it
would be a second test chosen after seeing that neither single test cleared, and
running it is precisely the practice this entry exists to avoid.

#### Other reporting

**Exit reasons.** MES: flatten −$20.93 mean, stop −$197.31, target +$141.78.
MNQ: flatten **−$7.53**, stop −$177.34, target +$144.67. The flatten is negative
on both, as entry 6's geometry argument requires for a 1× arm — the surviving
band runs from −(range + overshoot) to +range, so its midpoint is below zero.
**Fourth confirmation of the truncation account, now on a second instrument.**

**Instrument translation.** MNQ's median overnight range is **59.8 points**
against MES's 17.0, and the $200 risk rule at $2.00 a point caps the range at
100 points rather than 40. The cap therefore bites much harder: **668 sessions
skipped on MNQ against 210 on MES**, leaving 438 trades against 686. The rule is
identical in dollars; its effect on the sample is not. Realised risk stayed
inside the budget on MNQ — median $170, max $309, **zero** trades above the $400
daily limit, against MES's one.

### What was learned

**A benchmark can be the finding.** Entry 6's only non-null result lost its
significance to the correction of its own benchmark, before MNQ was consulted at
all. The infinite-horizon formula was not a rounding error: it moved the
comparison point by 1.6 points and the z by 0.6.

**Replication caught what a bigger sample would not have.** The point estimates
agree closely across instruments; what failed was significance, on both. Running
the same test on a second instrument turned "one marginal result" into "two
consistent sub-threshold results", which is a much clearer thing to record than
either alone.

**Pre-registering the test, not just the strategy, is what made this readable.**
Entry 6 computed a z after the fact on one of eight configurations. Entry 7 fixed
one configuration, one benchmark, one threshold and one direction before looking.
The difference is that entry 6's number could be argued about and entry 7's
cannot.

### Next

**The London family is closed.** No further London entry, no third instrument,
no further arm, no pooled test. Entries 1, 4, 5, 6 and 7 have now tested
breakout continuation across two sessions, two instruments, three signal
definitions, four holding periods and two benchmarks, and produced nothing that
survives a pre-registered test.

**Entry 1's condition has still never been met and remains the only route back.**
Any future breakout idea must first establish the counterparty claim
independently of backtest results — order-flow evidence about who takes the
other side of a range break and under what constraint. That is a data purchase
and should be priced before it is started.

### Addendum, 2026-09-11 — re-priced at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 assumed (Lucid support,
2026-09-11, article 11508978; method in entry 1's addendum of the same
date). Both saved streams were re-priced exactly, sizes recovered from the
commission column.

**The statistic under test does not depend on commission.** This entry's one
criterion is the departure of the observed target share from a de-meaned
bootstrap benchmark, and its z-score. Target and stop counts are fixed by the
price path: MES 197/144 for a 57.77% share, MNQ 134/100 for 57.26%. Neither
count, neither benchmark, and neither z (+1.47, +1.12, both under 1.65)
moves. **NOT REPLICATED stands, and no criterion would have resolved
differently.**

What does move is the P&L reported for context:

| 1×/ON, 2 ticks, pooled 2020–2026 | MES, $1.25 | **MES, $0.50** | MNQ, $1.25 | **MNQ, $0.50** |
|---|---|---|---|---|
| Trades | 686 | 686 | 438 | 438 |
| Net P&L | −$7,703 | **−$5,387** | +$116 | **+$1,085** |
| Folds profitable | 1 of 7 | 1 of 7 | 4 of 7 | 4 of 7 |
| Evaluations blown | 5 | 4 | 1 | 1 |
| Pass probability | 0.83% | 2.28% | 13.56% | 19.96% |
| Payout probability ($52,100) | 3.90% | 7.83% | 28.69% | 37.42% |
| Realised break-even | 58.19% | 57.00% | 55.07% | 54.33% |

**MNQ is re-priced at the same $0.50 on the assumption that Lucid's micro
rate is common to MES and MNQ. Only MES was confirmed.** If MNQ differs, the
right-hand column moves by $2 × 438 per $1 of difference a side; nothing in
this entry's verdict rests on it. The MES column is entry 6's 1×/ON arm and
matches its addendum of the same date, including the bracket-alone sign
change recorded there.


---

## 8. orborb flat 1030 — REJECTED
**Date:** 2026-09-08
**Submitted via:** `/submit` in Discord
**Code:** `strategies/generated/orborb_flat_1030.py` (generated; not yet written at the time this entry was committed)

### Mechanism claimed

Plumbing test — reproduces entry 5. Breakout continuation, not established.

**Who is on the other side:**

None claimed. This is a pipeline test

### Kill criteria, pre-registered

Pooled eval_sim pass probability below 25%; more than 1 eval blown; fewer than 4 of 7 folds profitable or total P&L negative.

### Description as submitted

15-minute opening range breakout on MES. Range = 9:30–9:45 candle high/low. At 9:45 place a buy stop 1 point above the high and a sell stop 1 point below the low; first fill wins, other cancels. Skip the day if range is over 10 points. Fixed 10-point stop, 18-point target, 4 contracts. Cancel unfilled orders and flatten any open position at 10:30 ET. One trade per day.

### Verdict

Not yet run. To be filled in by the walk-forward, with the commit
hash recorded in a follow-up commit.

---

### Attempt 2 — 2026-09-09

The previous attempt did not reach review: generation, screening or the test suite failed.

Re-submitted under the same name. The pre-registration above is
unchanged and is still the claim being tested - only the
implementation was regenerated. A retry that altered the mechanism,
the counterparty or the kill criteria would be a different
hypothesis and needs its own entry.

### Verdict: REJECTED

**Date:** 2026-09-09
**Code:** `strategies/generated/orborb_flat_1030.py`
**Reports:** `backtests/results/orborb_flat_1030_folds.csv`, `orborb_flat_1030_trades.csv`

Walk-forward over 7 yearly folds at 2 ticks of slippage per side and $1.25 commission per side.

| | |
|---|---|
| Trades | 588 |
| Net P&L | $-4,587.50 |
| Folds profitable | 0 of 7 |
| Sharpe | -3.40 |
| Profit factor | 0.580 |
| Max drawdown | $-4,716.25 |
| Worst day | $-57.50 |
| Avg duration | 30.3 min |
| Profit from <=5s holds | 0.00% |
| Pass probability | 0.00% |
| Evaluations blown | 2 |

**Rule 13 is not satisfied:**

- profitable in 0 of 7 folds, needs 4
- total walk-forward P&L $-4,587.50 is not positive

In-sample results are never evidence, and this is the out-of-sample answer.


**Verdict commit:** `9c638fe`

### Diagnostic, 2026-09-09 — where the 110-trade gap against entry 5 comes from

Added after the run. **This changes neither strategy** and is not a
pre-registered test; it exists because the reproduction reported 588 trades
against entry 5's 478 and an unexplained gap of that size would undermine the
claim that the pipeline reproduces anything.

**The signal logic is not the explanation.** Run over identical bars, entry 5's
`orb_flat_1030` (`ORB2` with `entry5_params()`) and the generated
`orborb_flat_1030` differ on **four sessions in seven years**, and the
generated one trades *fewer*, never more.

| Window | `orb_flat_1030` | `orborb_flat_1030` |
|---|---|---|
| Full parquet, 2019-05-05 .. 2026-08-31 | 592 | 588 |
| **Entry 5's window, 2020-01-01 .. 2026-08-31** | **478** | **476** |
| 2019 only, 2019-05-05 .. 2019-12-31 | 114 | 112 |

The reference reproduces entry 5's recorded 478 exactly on entry 5's window.

**110 of the gap is the evaluation window, not the strategy.**
`backtests/run_generated.py` scores the whole parquet, which begins
2019-05-05. Entry 5 was scored on 2020–2026, so the 112 sessions the generated
run took in 2019 were never in entry 5's count. 112 extra from 2019 minus the
2 below is the 110 observed.

**This is a defect in `run_generated.py`, not in the generated strategy.** A
walk-forward that silently uses a wider span than the entry it is compared
against produces numbers that cannot be read alongside the log. It is recorded
here rather than fixed silently; fixing it changes every future generated
verdict, so it belongs in its own change.

**Contract size does not affect the trade count at all.** The generated
strategy scored 476 trades on entry 5's window at both 1 and 4 contracts —
sizing scales P&L, not the number of signals. The reported "588 at 1 contract
vs 478 at 4" therefore compares two things that differ by window, not by size.
Size does matter downstream: entry 5's trailing-drawdown halt cut the run to
**45 trades at 4 contracts** where the same signals at 1 contract leave 169,
which is the effect entry 5 already records.

#### The two sessions that are a genuine difference of interpretation

Both are cases where the generated strategy **stands aside and entry 5 trades**.
Neither adds trades.

**1. A trigger on the last bar of the entry window (3 sessions:
2019-06-06, 2019-09-06, 2020-02-18).** All three trigger on the 10:29 bar.
Entry 5 fills intrabar at the stop level on that bar and exits at the 10:30
open — a 60-second trade. The generated strategy fills at the *next* bar's
open, so its action timestamp is 10:30, which is not before the cancel, and it
declines: `if action_ts.time() >= window_end: return  # nothing left to act on
before the cancel`.

The disagreement is over what a resting stop order does in the window's final
minute — filled where it rests, or not worth opening one minute before the
flatten. Entry 5 transcribes a Pine script that fills at the level; the
generated module reads "cancel unfilled orders and flatten at 10:30" as
closing the entry window too. Both are defensible readings of the same
sentence, and one of them produces trades whose entire life is one bar.

**2. A single bar spanning both triggers (1 session: 2023-05-23).** The 09:45
bar ran 4181.50 to 4194.75, through the buy stop at 4193.25 *and* the sell stop
at 4184.25. One-minute bars carry no intrabar path, so which filled first is
unknowable. Entry 5 resolves it and takes the long; the generated strategy
returns rather than guess: `if hit_up and hit_dn: return  # which stop filled
first is unresolvable; stand aside`.

Entry 4 recorded this same case as `ambiguous_both_stops` and counted it. The
generated module declines it instead — the more conservative reading, and the
one that does not manufacture a fill the data cannot support.

#### What this does not say

It does not say either strategy is correct. Entry 5 is REJECTED and its verdict
stands; nothing here revises it, and this entry's own verdict is unaffected.
What it establishes is that the pipeline reproduced entry 5's *signal* to
within two sessions in seven years, and that the headline discrepancy was an
artefact of comparing different spans.

### Addendum, 2026-09-09 — re-scored on the corrected span

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** The diagnostic above found that `run_generated.py` scored the whole
parquet from 2019-05-05 while every other entry was scored on 2020–2026.
`run_generated.SCORE_START` is now imported from `walkforward.build_folds()`
— 2020-01-01 — with earlier bars supplied to the strategy as indicator history
only. This is the same strategy, the same code, the same 1 contract, the same
2 ticks per side; the only change is the span.

| | Original run (2019-05 .. 2026-08) | Corrected span (2020-01 .. 2026-08) |
|---|---|---|
| Trades | 588 | **476** |
| Net P&L | −$4,587.50 | **−$3,531.25** |
| Folds profitable | 0 of 7 | **0 of 7** |
| Sharpe | — | **−3.08** |
| Profit factor | — | **0.615** |
| Max drawdown | — | **−$3,592.50** |
| Pass probability | 0.00% | **0.00%** |
| Evaluations blown | 2 | **1** |

The 476 is the figure the diagnostic predicted from entry 5's own window
(476 generated against entry 5's 478), which is the check that the correction
did what it claims. The 112 trades removed were all 2019 and were never part
of any comparable entry; one of the two blown evaluations was among them.

Per fold, corrected span:

| Year | Trades | Net P&L | Sharpe | PF | Max DD |
|---|---|---|---|---|---|
| 2020 | 91 | −$1,057.50 | −5.70 | 0.41 | −$1,087.50 |
| 2021 | 112 | −$192.50 | −0.87 | 0.86 | −$388.75 |
| 2022 | 14 | −$180.00 | −5.51 | 0.41 | −$255.00 |
| 2023 | 86 | −$356.25 | −1.52 | 0.79 | −$458.75 |
| 2024 | 117 | −$1,122.50 | −3.62 | 0.58 | −$1,173.75 |
| 2025 | 44 | −$586.25 | −5.38 | 0.43 | −$608.75 |
| 2026 | 12 | −$36.25 | −0.88 | 0.88 | −$167.50 |

Rule 13 fails on both counts on either span: 0 of 7 folds profitable and a
negative total. REJECTED stands. Recorded so the verdict's numbers can be
read against the log's other entries, which they could not before.

### Addendum, 2026-09-11 — re-scored under the trailing halt

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** Until today `backtests/run_generated.py` applied the $400 daily loss
limit but not the $1,500 end-of-day trailing halt, so a generated verdict was
scored on a more permissive basis than entry 5's runner, which applied both.
The halt now lives in `engine.apply_internal_guards` — the daily loss limit,
then the trailing halt on the loss-limited stream — and both runners call it;
`tests/test_guard_parity.py` holds them to it, and entry 5's reports came out
byte-identical before and after the move. A generated verdict is now decided
on the guarded stream, with the same signals without the halt reported
alongside as the comparable basis: the basis entries 1 and 4 were measured
on, and the one this entry's verdict and the addendum above were scored on.

Same strategy, same code, same 1 contract, same 2 ticks per side, same span.
**The halt-OFF column reproduces the addendum above figure for figure**, which
is the check that nothing else moved. The payout probability is new — see
entry 4's addendum of the same date for its definition.

| 2020-01-01 .. 2026-08-31, 1 contract, 2 ticks | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | **214** | 476 |
| Net P&L | **−$1,495.00** | −$3,531.25 |
| Folds profitable | 0 of 7 | 0 of 7 |
| Sharpe | −3.45 | −3.08 |
| Profit factor | 0.568 | 0.615 |
| Max drawdown | −$1,507.50 | −$3,592.50 |
| Worst day | −$57.50 | −$57.50 |
| Pass probability | 0.00% | 0.00% |
| Payout probability ($52,100) | 0.00% | 0.00% |
| Evaluations blown | 0 | 1 |
| Sessions blocked by the halt | **262** | — |

**The halt fires on 2022-11-17 and never releases.** At 1 contract the stream
loses $1,057.50 in 2020 and $192.50 in 2021, and eleven trades into 2022 the
end-of-day balance sits $1,507.50 under its peak. The remaining 262 sessions
with a signal — the rest of 2022 and all of 2023–2026 — are blocked, so
four of the seven folds contain no trades. Entry 5's run at 4 contracts
halted after 45 trades in 2020; at a quarter of the size the same losses
take nearly three years to reach the same line, and the outcome is the same.

The 214 trades taken are 180 flattens at 10:30 (mean −$3.67), 26 stops
(−$57.50) and 8 targets (+$82.50). The blow-up count is 0 on the guarded
stream because the internal halt stops trading before the firm's $2,000 line
is reachable — the degeneracy entry 5 recorded against its own criterion 2,
and the reason a blow-up count is only read on the comparable basis.

Rule 13 fails on both bases: 0 of 7 folds and a negative total. REJECTED
stands. `backtests/results/orborb_flat_1030_trades.csv` and `_folds.csv` now
hold the guarded stream; the stream the verdict was scored on is
`orborb_flat_1030_trades_nohalt.csv`, identical in content to what the
verdict's reports line pointed at.

### Addendum, 2026-09-11 — re-run at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 assumed (Lucid support,
2026-09-11, article 11508978). `run_generated.py` now reads the rate from
`rules.COMMISSION_PER_SIDE`; this is a **full re-run**, not a re-pricing, so
both guards were re-decided at the new cost. Same code, 1 contract, 2 ticks,
same span. The halt-OFF column is exactly the previous addendum's figure plus
$1.50 × 476 trades, which is the check that only the commission changed.

| 2020-01-01 .. 2026-08-31, 1 contract, 2 ticks | Halt ON, $1.25 | **Halt ON, $0.50** | Halt OFF, $1.25 | **Halt OFF, $0.50** |
|---|---|---|---|---|
| Trades | 214 | **313** | 476 | 476 |
| Net P&L | −$1,495.00 | **−$1,479.25** | −$3,531.25 | **−$2,817.25** |
| Folds profitable | 0 of 7 | 0 of 7 | 0 of 7 | 0 of 7 |
| Sharpe | −3.45 | −2.08 | −3.08 | −2.46 |
| Profit factor | 0.568 | 0.716 | 0.615 | 0.678 |
| Max drawdown | −$1,507.50 | −$1,510.00 | −$3,592.50 | −$2,898.25 |
| Pass probability | 0.00% | 0.00% | 0.00% | 0.00% |
| Payout probability ($52,100) | 0.00% | 0.00% | 0.00% | 0.00% |
| Evaluations blown | 0 | 0 | 1 | 1 |
| Sessions blocked by the halt | 262 | **163** | — | — |

**The halt now fires on 2024-01-26 instead of 2022-11-17.** At $1.50 less a
trade, the stream reaches the $1,500 trail 99 trades later: 2020 −$921, 2021
−$25, 2022 −$159, 2023 −$227, and ten trades into 2024 the balance is $1,510
under its peak. The 313 trades taken are 249 flattens at 10:30 (mean
+$0.58), 50 stops (−$56.00) and 14 targets (+$84.00). Per fold, halt OFF:
2020 −$921, 2021 −$25, 2022 −$159, 2023 −$227, 2024 −$947, 2025 −$520,
2026 −$18 — every year negative at either rate.

**Would any kill criterion have resolved differently? No.** Pre-registered:
pass probability below 25% (0.00% either way); more than one evaluation blown
(1, on the comparable basis, either way — passes as before); fewer than 4 of 7
folds profitable or a negative total (0 of 7 and negative on both bases at
both rates). Rule 13 fails on both bases. REJECTED stands. The results CSVs
now hold the $0.50 streams; `--commission 1.25` reproduces the $1.25 ones.

## 9. orb full day test — REJECTED
**Date:** 2026-09-09
**Submission name:** `orb_full_day_test`
**Submitted via:** `/submit` in Discord
**Code:** `strategies/generated/orb_full_day_test.py` (generated; not yet written at the time this entry was committed)

### Mechanism claimed

Plumbing test — reproduces entry 4, filter OFF arm. Breakout continuation not established; not claimed.

**Who is on the other side:**

None claimed. Pipeline test against a known verdict.

### Kill criteria, pre-registered

Pooled eval_sim pass probability below 25%; more than 1 eval blown across seven years; fewer than 4 of 7 folds profitable or total 7-year P&L negative.

### Description as submitted

15-minute opening range breakout on MES. Range = 9:30–9:45 candle high/low. At 9:45 place a buy stop 1 point above the high and a sell stop 1 point below the low; first fill wins, other cancels. Skip the day if range is over 10 points. Fixed 10-point stop, 18-point target, 4 contracts. Cancel unfilled entry orders at 11:30 ET. Hold any open position until stop, target, or a forced flatten at 15:55 ET. One trade per day. No trend filter.

### Verdict

Not yet run. To be filled in by the walk-forward, with the commit
hash recorded in a follow-up commit.

---

### Addendum, 2026-09-09 — scored span corrected before any run

No walk-forward has been run for this entry: the attempt never reached the
registry, because its generated test suite failed
(`test_opening_range_only_uses_pre_0945_bars`, the model's own no-lookahead
check against its own strategy). Between that attempt and the next,
`run_generated.py`'s scored span was corrected to 2020-01-01 onward, matching
every other entry — see entry 8's diagnostic and addendum. Whatever verdict
this entry eventually receives will be on the corrected span from the start;
there are no earlier numbers to reconcile.

### Attempt 2 — 2026-09-09

The previous attempt did not reach review: generation, screening or the test suite failed.

Re-submitted under the same name. The pre-registration above is
unchanged and is still the claim being tested - only the
implementation was regenerated. A retry that altered the mechanism,
the counterparty or the kill criteria would be a different
hypothesis and needs its own entry.

### Attempt 3 — 2026-09-09

The previous attempt did not reach review: generation, screening or the test suite failed.

Re-submitted under the same name. The pre-registration above is
unchanged and is still the claim being tested - only the
implementation was regenerated. A retry that altered the mechanism,
the counterparty or the kill criteria would be a different
hypothesis and needs its own entry.

### Verdict: REJECTED

**Date:** 2026-09-09
**Code:** `strategies/generated/orb_full_day_test.py`
**Reports:** `backtests/results/orb_full_day_test_folds.csv`, `orb_full_day_test_trades.csv`

Walk-forward over 7 yearly folds at 2 ticks of slippage per side and $1.25 commission per side, **4 contract(s)**.

| | |
|---|---|
| Scored span | 2020-01-01 .. 2026-08-31 (earlier bars used as indicator history only) |
| Contracts | 4 (declared by the strategy) |
| Trades | 503 |
| Net P&L | $-8,450.00 |
| Folds profitable | 2 of 7 |
| Sharpe | -1.08 |
| Profit factor | 0.865 |
| Max drawdown | $-9,595.00 |
| Worst day | $-230.00 |
| Avg duration | 149.9 min |
| Profit from <=5s holds | 0.00% |
| Pass probability | 9.43% |
| Evaluations blown | 8 |

**Rule 13 is not satisfied:**

- profitable in 2 of 7 folds, needs 4
- total walk-forward P&L $-8,450.00 is not positive

In-sample results are never evidence, and this is the out-of-sample answer.


**Verdict commit:** `29178fd`

### Addendum, 2026-09-11 — re-scored under the trailing halt

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** `run_generated.py` now applies both internal guards — the $400 daily
loss limit and the $1,500 end-of-day trailing halt, from
`engine.apply_internal_guards`, the same call entry 5's runner makes — and
reports the same signals without the halt alongside as the comparable basis.
See entry 8's addendum of the same date for the change and the parity test.
The verdict above was scored without the halt, on the basis entries 1 and 4
used; **the halt-OFF column reproduces it figure for figure**, and adds the
payout probability (entry 4's addendum of the same date defines it).

| 2020-01-01 .. 2026-08-31, 4 contracts, 2 ticks | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | **27** | 503 |
| Net P&L | **−$1,570.00** | −$8,450.00 |
| Folds profitable | 0 of 7 | 2 of 7 |
| Sharpe | −4.22 | −1.08 |
| Profit factor | 0.552 | 0.865 |
| Max drawdown | −$1,570.00 | −$9,595.00 |
| Worst day | −$230.00 | −$230.00 |
| Pass probability | 0.06% | 9.43% |
| Payout probability ($52,100) | 0.63% | 18.26% |
| Evaluations blown | 0 | 8 |
| Sessions blocked by the halt | **476** | — |

**The halt fires on 2020-05-29, after 27 trades, and never releases.** Fourteen
stops at −$230, four targets at +$330 and nine 15:55 flattens averaging
+$36.67 put the end-of-day balance $1,570 under its peak inside five months.
The 476 later sessions with a signal — June 2020 through August 2026 — are
blocked; six of the seven folds contain no trades. This is the guard doing
what it is for. Entry 4 measured the same signals with no halt and counted
blow-ups instead; run with its own guards at 4 contracts, the strategy stops
trading in year one, as entry 5 did.

Rule 13 fails on both bases: 0 of 7 folds and −$1,570 under the guards, 2 of 7
and −$8,450 without them. REJECTED stands.
`backtests/results/orb_full_day_test_trades.csv` and `_folds.csv` now hold
the guarded stream; the stream the verdict was scored on is
`orb_full_day_test_trades_nohalt.csv`, identical in content to what the
verdict's reports line pointed at.

### Addendum, 2026-09-11 — re-run at the confirmed $0.50 commission

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** MES commission is $0.50 a side, not the $1.25 assumed (Lucid support,
2026-09-11, article 11508978). A **full re-run** of `run_generated.py` at the
new default, both guards re-decided; same code, 4 contracts, 2 ticks, same
span. The halt-OFF column is exactly the verdict's figure plus $6.00 × 503
trades, which is the check that only the commission changed.

| 2020-01-01 .. 2026-08-31, 4 contracts, 2 ticks | Halt ON, $1.25 | **Halt ON, $0.50** | Halt OFF, $1.25 | **Halt OFF, $0.50** |
|---|---|---|---|---|
| Trades | 27 | **30** | 503 | 503 |
| Net P&L | −$1,570.00 | **−$1,520.00** | −$8,450.00 | **−$5,432.00** |
| Folds profitable | 0 of 7 | 0 of 7 | 2 of 7 | 2 of 7 |
| Sharpe | −4.22 | −3.60 | −1.08 | −0.69 |
| Profit factor | 0.552 | 0.605 | 0.865 | 0.911 |
| Max drawdown | −$1,570 | −$1,520 | −$9,595 | −$6,697 |
| Pass probability | 0.06% | 0.19% | 9.43% | **13.79%** |
| Payout probability ($52,100) | 0.63% | 1.29% | 18.26% | 23.96% |
| Evaluations blown | 0 | 0 | 8 | 8 |
| Sessions blocked by the halt | 476 | **473** | — | — |

**The halt fires on 2020-06-23 instead of 2020-05-29** — three trades later,
still inside the first half of year one: 16 stops at −$224, 5 targets at
+$336 and 9 flattens averaging +$42.67. Per fold, halt OFF: 2020 −$4,135;
2021 +$2,567; 2022 −$311; 2023 −$248; 2024 −$1,640; 2025 −$2,113; 2026
+$448. Against entry 4's filter-OFF arm re-priced at the same rate and
slippage — −$5,946 over 504 trades — this is −$5,432 over 503, the same 6%
apart as before.

**Would any kill criterion have resolved differently? No.** Pass probability
13.79% against a 25% line; 8 evaluations blown against a limit of 1; 2 of 7
folds against 4, with a negative total on both bases. Rule 13 fails on both
bases. REJECTED stands. The results CSVs now hold the $0.50 streams;
`--commission 1.25` reproduces the $1.25 ones.

## 10. London 1×/ON held through the US session — REJECTED
**Date:** 2026-09-11
**Spec frozen at:** the commit adding this entry
**Code:** not yet written
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** MES **and** MNQ, 5-minute bars resampled from 1-minute, 19:00 ET
through 15:55 ET. **Both instruments are required; each is tested in full.**
**Source:** entry 6's specification, which was the YouTube "London Breakout
Strategy the Right Way" translated from AUDUSD to MES. **The 09:25 flatten in
entry 6 was the operator's addition, not the source's** (operator's statement,
2026-09-11); this entry removes it.

**Operator decisions, confirmed 2026-09-11 before this entry was frozen:** the
sizing rule is the log's corrected realised-stop rule, not entry 6's range-based
one; the early-close flatten is 12:55 ET; rule 13 stands as criterion 1
alongside the two raised bars; entry 7's +2-point floor is kept as criterion 4;
and the bar sitting above the effect that motivated the entry is intended.

### This entry is forbidden by entry 7, and is being written anyway

Entry 7's Next section says, in terms: **"The London family is closed. No further
London entry, no third instrument, no further arm, no pooled test."** Entry 6's
Next section says: **"Do not test a third target multiple, a different flatten
time, or a different range window."**

**This is a further London entry with a different flatten time.** The operator
has directed it after seeing both verdicts. Recording the conflict here rather
than omitting it is the whole function of this log: an entry that quietly ignores
its predecessors' prohibitions is how a rejected idea gets re-run until it passes,
and the reader six months from now needs to see that this one did not sneak past.

**What entry 6 itself left open, and this entry takes:** its Next section also
says *"If anything in this entry is worth carrying forward it is the exit, not the
entry … An idea about that would be a different hypothesis with a different
mechanism, and it would need its own entry written before any code."* This is
that entry. It changes the exit and only the exit.

**What is not different:** the signal, the breakout-continuation mechanism, and
the absence of any independent evidence for it. Entry 1's condition — order-flow
evidence about who takes the other side of a range break — has still never been
met. This entry does not meet it either.

### This idea is informed by a post-hoc finding, and the prior is biased upward

Stated first because it governs everything below.

Entry 6's verdict found, **after the run**, that the 1×/ON arm's bracket alone
was nearly break-even while its 345 flattens at 09:25 carried the whole loss.
Re-priced at the confirmed $0.50 commission (entry 6's addendum, 2026-09-11):
the 197 targets and 144 stops net **+$896**, the flattens **−$6,283**, the arm
**−$5,387**. Entry 7 found the same shape on MNQ: 204 of 438 trades flattened,
net negative. **This entry exists because of that table.** Its idea was chosen by
looking at the exit-reason breakdown of the very data it will be tested on.

Three consequences, all recorded in advance:

1. **Neither instrument is untouched data for this question.** MES has been
   examined eight times in entry 6 and recomputed in entry 7; MNQ was entry 7's
   replication and its flatten population has been seen. There is no out-of-sample
   instrument left in this family. The only protection is the pre-registered bar
   below, set higher than the effect that motivated it.
2. **The finding that motivated this entry is arithmetic, not evidence.** Entry 6
   recorded that a trade surviving to its flatten is conditioned into the band
   between stop and target and is worth the band's midpoint — a design property of
   an asymmetric bracket, confirmed four times in this log. That the flattens lost
   money says the band was skewed against them. It says nothing about whether
   those trades would have resolved favourably given more time; under a driftless
   process they resolve at the benchmark rate, which is *below* the bracket's
   break-even, and would have lost anyway.
3. **The prior is therefore biased upward and is corrected by raising the bar,
   not by pretending the bias is absent.** Entry 6's measured departure was
   **+3.96 points, z = 1.47** on MES and **+3.65, z = 1.12** on MNQ. The bar below
   requires roughly **+4.9 points on MES and +6.1 on MNQ** (power arithmetic under
   *Kill criteria*). **An effect exactly as large as the one that motivated this
   entry fails this test.** That is intended: the motivating estimate is the
   biased one.

**Prior for survival: low**, and lower than entry 6's, because the thing being
tested has already been looked at.

### Mechanism claimed

Entry 6's mechanism is unchanged and is restated only where the extended hold
adds to it. The overnight range (19:00–02:55 ET) is built by a small population
in thin liquidity; the 03:00 ET European cash open brings a materially larger
population that reprices the index; the first break of the range carried by that
volume continues.

**What the extended hold adds:** the claim that the repricing begun at 03:00 is
*not complete by 09:25*, and that the US cash open at 09:30 — the largest volume
step of the day — resolves the position at stop or target rather than leaving it
to be flattened from the middle of the band. Concretely: at 09:25 half of entry
6's trades were unresolved; the claim is that most of those resolve during the
US session, and that they resolve at the target more often than a driftless
process would.

**Who is on the other side:** as entry 6 — overnight participants inside the
range, stopped out as it breaks — plus, for the extended hold, **US-session
participants who fade the overnight move at the cash open**: gap-fade and
mean-reversion flows that sell an overnight rally into 09:30 liquidity. The
claim is that these flows are absorbed and the overnight direction reasserts.

**This is a behavioural claim, not a forced flow, and it is the entry's weakest
point** — the same weakness entry 6 recorded, now doubled, because the second
counterparty is asserted rather than measured. Nobody is obliged to fade the open.
Entry 4's post-hoc addendum found no measurable late-session drift after its
target exits; nothing in this log has measured US-session continuation of an
overnight move.

### Signal definition

**Identical to entry 6's 1×/ON arm in every respect except the two marked
CHANGED.** Nothing may change after results are seen.

**Bars.** 5-minute candles resampled from 1-minute data, labelled by opening
minute, left-closed.

**Overnight range.** High and low of 19:00 ET through 02:55 ET inclusive. The
window spans midnight: the range for a trade on date D begins at 19:00 on D−1.
`london_date()` in `strategies/london.py` is the only place that mapping lives.

**Entry window.** 03:00–05:00 ET. The first 5-minute candle that closes outside
the range triggers; the position is taken at the next candle's open. First break
only. One trade per day. The opposite side is never traded that day.

**Trend filter.** 200-period EMA of 5-minute closes on the continuous 23-hour
series, seeded with a 200-bar simple mean, read at 03:00 from closes strictly
before 03:00. Long only above, short only below. **Filter ON only. No OFF arm.**

**Stop.** The opposite side of the overnight range.

**Target.** 1× range height from entry. **One arm. No 2× arm.**

**Flatten — CHANGED.** **15:55 ET** if still open, on the open of the 15:55 bar.
Entry 6 flattened at 09:25. **On an early-close session the flatten is 12:55
ET** — five minutes before the 13:00 close, the same offset 15:55 has from the
16:00 cash close — and `rules.flatten_deadline` (13:00 on those days) is the
backstop. 15:55 does not exist on those days and entry 6 never had to decide
this; the operator confirmed 12:55 before the freeze.

**Sizing — CHANGED, as the log requires.** Entry 6 sized off the range height
and its 2026-09-05 addendum recorded that this permitted a stop larger than the
daily loss limit; the standing rule for every new strategy since is to size off
the realised stop. This entry is a new strategy, not a replication, so it is
bound:

```
stop_distance = |fill − far side of range|            (range height + overshoot)
contracts     = floor( $200 / (stop_distance × point_value) ), clamped to [1, 5]
must satisfy  contracts × stop_distance × point_value <= rules.DAILY_LOSS_LIMIT
skip the session if stop_distance × point_value > $200
```

`point_value` is $5.00 for MES and $2.00 for MNQ, taken from `engine.ContractSpec`
— never hardcoded (entry 7 recorded the MNQ-at-$5 bug). Keeping entry 6's
range-based sizing would have been a second override of the log; the operator
chose the corrected rule before the freeze.

**No entries on roll days** (trade date D).

**Costs.** **$0.50 per contract per side commission** — Lucid's confirmed MES
rate (`rules.COMMISSION_PER_SIDE`, support article 11508978); **MNQ is assumed
to carry the same rate, and only MES was confirmed.** **2 ticks of slippage per
side as the base case**, 1 tick as the optimistic sensitivity — entry 6's
convention, kept for the same reason: the 03:00 window is thin.

**Internal guards, on the standard in force since 2026-09-11:** the $400 daily
loss limit marked to market, the 5-contract cap, the 30-second minimum hold,
**and the $1,500 end-of-day trailing halt**, via `engine.apply_internal_guards`.
The same signals without the halt are reported alongside as the comparable
basis; the evaluation blow-up count is read there, as entry 5 established it
must be.

**Evaluation structure.** Seven out-of-sample yearly folds, 2020–2026, **nothing
selected**: one fixed configuration, so every fold is a genuine out-of-sample
observation of one rule. 2019 is partial and reportable outside the criteria.
The 2026 fold is partial (ends 2026-08-31) and is reported separately and
alongside the seven. **Both instruments run the identical specification.**

### Rules compatibility: the first strategy to hold from the overnight session into RTH

Checked rather than assumed. A position entered at 03:00–05:05 ET and held to
15:55 ET: `rules.is_entry_allowed` is True across the entry window;
`rules.must_flatten` is False at 15:55 on a normal day (deadline 16:30) and
**True from 13:00 on an early-close day**, which is why the 12:55 flatten above
exists. The trade sits entirely inside calendar date D, so `rules.session_date`
groups it correctly for the daily loss limit; only the range construction crosses
midnight, as in entry 6.

**New exposure this entry creates and entry 6 did not:** the position is open
through the 09:30 cash open and the 10:00 economic releases with a stop one range
height plus overshoot away — a median of about 18 points on MES. The daily loss
limit is marked bar by bar through that window. With sizing capped at $200 of
realised risk, a single trade's stop cannot reach the $400 limit, so
`loss_limit_flatten` is not expected to fire; **its count is reported.**

The entry-guard caveat entry 6 recorded stands: 03:00 entries pass because 03:00
is numerically before an afternoon cutoff, not because `rules.py` models an
overnight session.

### The benchmark: entry 7's de-meaned bootstrap, at the new horizon

The random-walk closed form `a / (a + b)` is invalid under a time limit (entry 6).
The benchmark is entry 7's **de-meaned bootstrap**, procedure unchanged, **with
the horizon set to the 15:55 flatten**: for each realised trade, resample the
window's own 1-minute `(high, low, close)` changes with replacement over the
trade's actual entry-to-15:55 length, drift removed, rebuild from the actual
entry price, apply the same stop, target and flatten, stop-first when one step
reaches both, 1,000 replications, seed 0. `bootstrap_benchmark.benchmark` takes
the flatten time as a parameter; it is called with 15:55 (12:55 on early-close
days) and nothing else about it changes.

**The benchmark must be recomputed for this horizon. Entry 7's 53.81% / 53.61%
were for a 09:25 horizon and must not be reused.** A longer horizon resolves more
of each replication, and the benchmark share moves accordingly; whatever it is,
it is the number the observed share is measured against.

The observed target share is `targets / (targets + stops)`, flattens excluded,
exactly as the benchmark excludes them — and **the flatten count itself is the
first thing this entry reports**, because if the extended hold does not resolve
most of entry 6's unresolved half, the mechanism claimed above is absent before
any share is compared.

### Pre-registered tests

On each instrument, pooled 2020–2026 at 2 ticks, filter ON, one configuration:

1. **Exit-reason counts and P&L:** stop / target / 15:55 flatten. **Against entry
   6's 345 of 686 flattened on MES and 204 of 438 on MNQ.**
2. **Observed target share, benchmark share at the 15:55 horizon, departure in
   points, and one-sided z**, per entry 7's formula: `z = departure /
   sqrt(p0 (1 − p0) / n_resolved)`, `p0` = benchmark.
3. **Per-fold P&L and Sharpe**, seven years, plus 2026 separately.
4. **`eval_sim` pass probability and payout probability**, pooled and per fold,
   on the guarded stream; **evaluations blown** on the comparable (halt-OFF)
   stream; sessions blocked by the trailing halt.
5. **Realised break-even** from the trades' own mean stop and mean target
   (entry 6's test 3), and the observed share against it.
6. **Sizing, realised risk and overshoot distributions**, and confirmation that
   no trade's stop exceeded $200 or the daily loss limit.
7. **Hold-time distribution**, average duration, and rule 7's share of profit
   from trades held 5 seconds or less (must be 0%).
8. **Everything repeated at 1 tick** as the optimistic sensitivity.
9. **Reproduction check before any 15:55 run:** the new code, with the flatten
   set to 09:25 and costs at $1.25, must reproduce entry 6's 1×/ON stream on MES
   **trade for trade** (686 trades, −$7,702.50) and entry 7's on MNQ (438,
   +$116.00). *(With the corrected sizing this reproduction must be run with
   entry 6's range-based sizing switched on; the sizing change is then the only
   difference at the 15:55 run, and its effect is reported separately.)* A code
   change that cannot reproduce the frozen stream is not a test of the flatten.

### Kill criteria — decided now

Measured on the guarded stream, pooled over the seven out-of-sample years, at the
base 2 ticks per side, filter ON. **Every criterion must pass on MES *and* on
MNQ. A failure on either instrument kills the entry as a whole; there is no
per-instrument survival.**

1. **Rule 13, the standing standard:** profitable in at least 4 of 7 folds
   **and** positive total P&L after commission and slippage, at 2 ticks.
2. **Pooled `eval_sim` pass probability ≥ 35%.** Entry 6 asked 25%; the bar is
   raised because the idea was chosen after seeing this data.
3. **Departure of observed target share from the 15:55-horizon bootstrap
   benchmark with one-sided z ≥ 2.5.** Entry 7 asked 1.65. A one-sided z of 2.5
   is p ≈ 0.006 — approximately the Bonferroni line for the ten looks this
   family has now had at these two series (eight configurations in entry 6,
   one in entry 7, one here).
4. **Departure ≥ +2 points** (carried from entry 7 and kept by the operator), so
   a statistically clean but economically trivial departure cannot pass on
   sample size alone.

**Power arithmetic, so the bar is understood before the run.** If most of entry
6's flattens resolve, MES has roughly 650 resolved trades and MNQ roughly 420.
At a benchmark near 54%, the standard error of the share is about 1.95 points on
MES and 2.43 on MNQ, so z = 2.5 needs a departure of about **+4.9 points on MES
and +6.1 on MNQ**. Entry 6 measured +3.96 and +3.65 at the shorter horizon.
**The test is set above the effect that motivated it, on both instruments, by
design.**

Non-fatal, with a reporting obligation:

5. The verdict must state, in those terms, **how much of any result is the
   extended hold** — the P&L of the trades that would have been flattened at
   09:25 and were not — **and how much is the sizing change**. A combined figure
   does not satisfy this entry.

No appeal, no second flatten time, no OFF arm, no 2× arm, no third instrument.

### Pre-registered: slippage at the 03:00 open is the dominant unknown, and this test cannot resolve it

Since the commission correction, slippage is the larger part of every round
turn: at 2 ticks it is $5.00 of a $6.00 round turn on MES, $2.00 of $3.00 on
MNQ. **Every entry in this entry's stream is filled at 03:05–05:05 ET, the
thinnest window this project trades**, and the 2-tick assumption there is a
guess with no measurement behind it. A 1-tick figure is reported as the
optimistic sensitivity and a 3-tick figure as the pessimistic one, **but no
backtest can settle which is right.** Only live fills against the signal level
at 03:00 can — on a live account, fill by fill — and TradingView paper fills
cannot see it. **A verdict of ACCEPTED here would therefore be conditional on
an unmeasured cost, and must say so on its face.** The 15:55 exit, by contrast,
lands in RTH liquidity where 1 tick is realistic; the asymmetry is noted.

### Prediction on record

Nothing below has been computed. No 15:55 run has been made and no outcome
inspected.

1. **The flatten share falls sharply**: from 50.3% of MES trades (345/686) to
   **under 20%**, and similarly on MNQ. The US session resolves most of what
   09:25 left open. *(If this fails, the mechanism as claimed is absent.)*
2. **The newly resolved trades resolve at about the benchmark rate**, so the
   pooled departure **shrinks** from entry 6's +3.96 toward zero rather than
   growing. The bracket edge entry 6 measured was concentrated in trades that
   resolved *quickly*; trades that took all day to resolve are the ones a
   driftless process describes best.
3. **z is below 2.5 on at least one instrument**, and pass probability is below
   35% on both.
4. **Total P&L at 2 ticks is negative on MES and small on MNQ**, because the
   bracket's realised break-even sits near 55% and the pooled share lands within
   two points of the benchmark.

**What would falsify the pessimism:** z ≥ 2.5 and pass probability ≥ 35% and
rule 13, on both instruments, at 2 ticks. Predictions 1 and 2 are the diagnostic
pair: if 1 holds and 2 fails — flattens vanish *and* the departure grows — the
extended hold did what was claimed. If 1 fails, nothing else matters.

### Data and cost

Both parquets exist: `data/mes_v_0_ohlcv_1m_2019-05_2026-08.parquet` and
`data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet`, 2019-05-05 to 2026-08-31,
validated in entries 6 and 7. **No data purchase.** The bootstrap at a 15:55
horizon resamples roughly twice as many bars per trade as entry 7's 09:25 run
(about 15 minutes); expect 30–40 minutes per instrument, run in the background.

### Longer-term intent: copying trades across multiple funded accounts

Unchanged and repeated because it governs how any accepted result would be used.
Copying identical trades across N funded accounts multiplies outcomes in both
directions and is not diversification: the same losing day draws down every
account at once and a trailing-drawdown breach terminates all of them on the
same date. It is one bet at N times the size with N times the fees.

### Verdict

Not yet run. To be filled in after the walk-forward on both instruments, with
the commit hash recorded in a follow-up commit. **If ACCEPTED, the verdict must
carry, on its first line, that the idea was chosen after seeing the exit-reason
table of the data it was tested on, and that its cost basis rests on an
unmeasured 03:00 slippage assumption.**

### Verdict: REJECTED

**Rejected on the pre-registered criteria, on the data the idea was drawn from.** The upward bias of the prior recorded above did not carry the entry over a bar set above the effect that motivated it.

**Date:** 2026-09-11
**Code:** `strategies/london.py` (`flatten_time`, `early_close_flatten_time`, `size_on`), `backtests/run_entry10.py`
**Reports:** `backtests/results/entry10_report.txt`; per instrument `entry10_<mes|mnq>_slip2.csv` (standard) and `_slip2_nohalt.csv` (comparable), `_slip1`/`_slip3` sensitivities, `_folds.csv`, and the attribution streams `_0925_stop.csv` and `_0925_range.csv`

Entry 6's 1×/ON specification held to 15:55 ET (12:55 on early-close sessions), realised-stop sizing, $0.50 a side commission, **2 ticks of slippage per side as the base case**, 2020-01-01 .. 2026-08-31, nothing selected. Guards: the $400 daily loss limit per size group, then the $1,500 end-of-day trailing halt on the combined stream — the two engine functions `apply_internal_guards` composes, in its order, composed per size group because size varies per session (exact given one trade a day). The benchmark is entry 7's de-meaned bootstrap at the 15:55 horizon, early-close days truncated at 12:55, seed 0.

#### Kill criteria — every line must pass on both instruments

| # | Criterion | MES | MNQ | Line | |
|---|---|---|---|---|---|
| 1 | Rule 13: >= 4 of 7 folds profitable and P&L > 0 | 0 of 7, $-1,298 | 1 of 7, $1,239 | >= 4 of 7 and > 0 | **FAIL** |
| 2 | Pooled pass probability | 1.29% | 50.82% | >= 35% | **FAIL** |
| 3 | z against the 15:55-horizon benchmark, one-sided | +0.35 | +1.16 | >= 2.5 | **FAIL** |
| 4 | Departure from the benchmark, points | +1.89 | +5.47 | >= +2 | **FAIL** |

**REJECTED.** Failed on: MES: Rule 13: >= 4 of 7 folds profitable and P&L > 0; MES: Pooled pass probability; MES: z against the 15:55-horizon benchmark, one-sided; MES: Departure from the benchmark, points; MNQ: Rule 13: >= 4 of 7 folds profitable and P&L > 0; MNQ: z against the 15:55-horizon benchmark, one-sided. Any one failure on either instrument kills the entry as a whole, as pre-registered.

#### MES

| 2020–2026, 2 ticks, $0.50 | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | 117 | 669 |
| Net P&L | $-1,297.50 | $-2,583.25 |
| Mean per trade | $-11.09 | $-3.86 |
| Folds profitable | 0 of 7 | 2 of 7 |
| Sharpe | -1.29 | -0.41 |
| Profit factor | 0.836 | 0.947 |
| Max drawdown | $-1,538.50 | $-3,591.75 |
| Worst day | $-213.00 | $-230.00 |
| Pass probability | 1.29% | 10.00% |
| Payout probability ($52,100) | 5.28% | 21.29% |
| Evaluations blown (read on the comparable basis) | 2 | 2 |
| Sessions blocked by the trailing halt | 552 | — |
| Avg hold | 440.6 min | 328.5 min |
| Profit from <=5s holds | 0.00% | 0.00% |

**Exits, standard stream:** 47 targets, 40 stops, 30 flattens (25.6% of trades), 0 daily-loss flattens, 0 daily-loss halts. **Observed target share 54.02%** on 87 resolved trades against a **benchmark of 52.14%** (117 trades bootstrapped, 1,000 replications, resolved fraction 78.63%): **departure +1.89 points, z = +0.35**. Realised break-even from mean stop and mean target: 56.30%.

**Sizing and risk:** contracts {1: 62, 2: 31, 3: 15, 4: 4, 5: 5}; realised risk median $158.75, mean $155.10, max $198.75; overshoot median 1.00 pts, mean 1.50, max 6.50.

Per fold, standard: 2020 $-613.75; 2021 $-683.75; 2022 $0.00; 2023 $0.00; 2024 $0.00; 2025 $0.00; 2026 $0.00.

**1 tick sensitivity:** standard 272 trades, $-100.25, 1 of 7 folds, pass 15.51%, payout 29.81%; comparable $884.25, 1 blown.
**3 ticks sensitivity:** standard 96 trades, $-1,529.25, 0 of 7 folds, pass 0.29%, payout 1.57%; comparable $-6,050.75, 4 blown.

**Attribution (criterion 5), standard basis at the base case:** the 15:55/stop-sized stream nets $-1,297.50 over 117 trades; the same signals flattened at 09:25 with stop sizing net $-789.75 over 185; entry 6's rule (09:25, range sizing) at these costs nets $-871.00 over 108. **The extended hold is worth $-507.75; the sizing change is worth $81.25.**

#### MNQ

| 2020–2026, 2 ticks, $0.50 | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | 124 | 406 |
| Net P&L | $1,239.00 | $-505.00 |
| Mean per trade | $9.99 | $-1.24 |
| Folds profitable | 1 of 7 | 4 of 7 |
| Sharpe | 1.10 | -0.14 |
| Profit factor | 1.156 | 0.983 |
| Max drawdown | $-1,535.00 | $-4,005.00 |
| Worst day | $-206.00 | $-215.00 |
| Pass probability | 50.82% | 13.87% |
| Payout probability ($52,100) | 67.05% | 27.38% |
| Evaluations blown (read on the comparable basis) | 1 | 1 |
| Sessions blocked by the trailing halt | 282 | — |
| Avg hold | 328.4 min | 298.5 min |
| Profit from <=5s holds | 0.00% | 0.00% |

**Exits, standard stream:** 65 targets, 47 stops, 12 flattens (9.7% of trades), 0 daily-loss flattens, 0 daily-loss halts. **Observed target share 58.04%** on 112 resolved trades against a **benchmark of 52.56%** (124 trades bootstrapped, 1,000 replications, resolved fraction 91.40%): **departure +5.47 points, z = +1.16**. Realised break-even from mean stop and mean target: 54.53%.

**Sizing and risk:** contracts {1: 83, 2: 27, 3: 10, 4: 2, 5: 2}; realised risk median $160.25, mean $154.16, max $200.00; overshoot median 3.88 pts, mean 4.93, max 19.50.

Per fold, standard: 2020 $1,600.50; 2021 $-56.50; 2022 $-305.00; 2023 $0.00; 2024 $0.00; 2025 $0.00; 2026 $0.00.

**1 tick sensitivity:** standard 126 trades, $1,339.00, 2 of 7 folds, pass 52.80%, payout 68.77%; comparable $53.00, 1 blown.
**3 ticks sensitivity:** standard 124 trades, $1,054.00, 1 of 7 folds, pass 44.77%, payout 61.79%; comparable $-1,063.00, 2 blown.

**Attribution (criterion 5), standard basis at the base case:** the 15:55/stop-sized stream nets $1,239.00 over 124 trades; the same signals flattened at 09:25 with stop sizing net $835.00 over 157; entry 6's rule (09:25, range sizing) at these costs nets $1,390.50 over 339. **The extended hold is worth $404.00; the sizing change is worth $-555.50.**

In-sample results are never evidence, and this is the out-of-sample answer on both instruments.


**Verdict commit:** `82006bf`

### Addendum, 2026-09-11 — attribution like-for-like, and the sample the halt left the z test

Added after the verdict; **the verdict is unchanged and this does not reopen
it.** Two things the frozen block does not say properly.

**1. Criterion 5's attribution, like-for-like.** The block compared the three
streams on the standard basis, and each stream's trailing halt fired on its own
date — the 15:55 stream after 117 MES trades, the 09:25 stop-sized stream after
185 — so the differences mixed truncation with the effect being attributed.
Recomputed on the comparable (halt-OFF) basis at the base case, where the 15:55
and 09:25 stop-sized streams take **identical entries at identical sizes**
(checked), so A − B is purely the exit. C is entry 6's rule at these costs and
reproduces entry 6's and entry 7's re-priced figures exactly.

| Halt OFF, 2 ticks, $0.50 | MES | MNQ |
|---|---|---|
| A: 15:55, stop sizing | −$2,583.25 over 669 (342 targets / 276 stops / 51 flattens) | −$505.00 over 406 (211 / 176 / 19) |
| B: 09:25, stop sizing | −$4,712.00 over 669 (195 / 143 / 331) | −$669.50 over 406 (124 / 94 / 188) |
| C: 09:25, range sizing — entry 6 | −$5,386.50 over 686 | +$1,085.00 over 438 |
| **Extended hold, A − B** | **+$2,128.75** | **+$164.50** |
| **Sizing change, B − C** | **+$674.50** | **−$1,754.50** |
| Total, A − C | +$2,803.25 | −$1,590.00 |

The extended hold is worth $2,129 on MES and $165 on MNQ, and neither arm is
profitable with it. **What it did to the target share is the finding.** On MES
the 280 trades that 09:25 had flattened resolved at 147 targets to 133 stops —
**52.5%, the benchmark rate** — and pulled the pooled share from 57.69% to
55.34%, under the 56.30% realised break-even. On MNQ the 169 newly resolved
trades went 87 to 82, **51.5%**, pulling the share from 56.88% to 54.52%,
exactly at the 54.53% break-even. **Prediction 2 held on both instruments: the
trades the US session resolved resolved like a driftless process, and the
departure shrank.** Entry 6's bracket edge lived in the trades that resolved
quickly; giving the rest all day did not add to it. The exit was not hiding an
edge.

On MNQ the sizing change cost $1,755. Realised-stop sizing skips 32 sessions
whose one-contract stop is over budget and sizes smaller wherever the overshoot
is large — and MNQ's overshoot (median 3.88 points, mean 4.93) is four times
MES's. That, not the exit, is most of the distance between entry 7's +$1,085 and
this entry's −$505; it is the corrected rule pricing risk entry 6's rule ignored.

**2. The sample the halt left the z test.** The entry pre-registered criterion 3
on the guarded stream, and its power arithmetic assumed roughly 650 resolved
trades on MES and 420 on MNQ. The trailing halt ended both streams in their
second year — 552 MES sessions and 282 MNQ sessions blocked — leaving **87 and
112 resolved trades**. At those sizes the standard error of the share is about
5.4 and 4.7 points, so z = 2.5 needed a departure of roughly **+13 and +12
points**, three times the effect the entry was written to detect. Criterion 3
was therefore close to unpassable for any effect this log has ever measured.
**That is a property of the pre-registration, not a reason to revise the
verdict** — the entry chose the guarded stream and the halt did what it is for.
Recorded so that the next entry combining a trailing halt with a share test
states which stream the test runs on and what sample the halt will leave it.
On the unhalted streams the observed shares are 55.34% (MES) and 54.52% (MNQ);
no benchmark was bootstrapped for those streams and no z is claimed for them.

**What this closes.** Entry 6's Next section said the exit, not the entry, was
the one thing worth carrying forward. It has now been carried forward, on both
instruments, with the bar raised for the bias, and it did not pay: the
unresolved half of the bracket resolves at the driftless rate when given the
day. **The London family is closed again, and this time on the exit as well as
the entry.** No further flatten time, no further instrument. Entry 1's condition
— order-flow evidence about who takes the other side of a range break — remains
the only route back for any breakout idea.

## 11. Turn-of-month institutional flows, intraday — REJECTED

**Date:** 2026-09-11
**Spec frozen at:** `a6d9af1`
**Verdict commit:** `2bf4fc4`
**Code:** `strategies/tom.py`, `backtests/run_entry11.py`, `research/power_check_tom.py`
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** MES, 1-minute bars, RTH only (09:30 to 15:55 ET)
**Source:** the operator's specification, 2026-09-11, transcribed below. The
published effect is Etula, Rinne, Suominen & Vaittinen (2020), *Dash for Cash:
Monthly Market Impact of Institutional Liquidity Needs*.

**Operator decisions, stated before this entry was frozen:** long only; the
signal arm carries no stop because it measures drift; a second arm carries a
15-point stop as the Lucid-safe form; 4 contracts; $0.50 a side; 1 tick a side
is the base case because every fill is in RTH, with 2 ticks as the sensitivity;
roll days skipped; both internal guards on the standard basis with the halt-OFF
stream alongside; the control is the open-to-close return of every non-window
session; yearly folds 2020–2026 with nothing selected; a power check before any
strategy code; the per-day breakdown is reported and never selected on.

### Mechanism claimed

Pension funds, mutual funds and retirement plans receive contributions and pay
obligations on a monthly schedule. The cash arrives and leaves at month
boundaries, and the institutions that hold it deploy contributions into the
market and raise cash for redemptions and benefit payments in the last one to
two and the first one to three trading days of the month. That trading is
driven by the calendar, not by price: a plan that must fund a payment on the
first of the month sells whatever the market is offering, and a plan that
received contributions invests them on the schedule its mandate sets. Etula,
Rinne, Suominen and Vaittinen (2020) document the resulting turn-of-month
pattern in equity index returns and attribute it to exactly this institutional
liquidity cycle, with the selling pressure into month end reversing into buying
in the first days of the new month.

**Who is on the other side:** the institution meeting a dated obligation, which
is constrained by the date and not by the price. A pension paying benefits on
the first cannot wait for a better level; a fund investing month-start
contributions is measured against the benchmark from the day the cash arrives
and is penalised for holding it. Their counterparty in the market is the
liquidity provider who absorbs the flow and is paid for it in price impact. The
edge claimed here is to sit on the liquidity provider's side of that flow in
the days the schedule makes it predictable. The constraint is contractual and
calendar-bound, which puts this in the same class of counterparty claim as
entry 2 and a different class from the breakout family, whose counterparty was
never identified.

**Stated honestly, before any data:** most of the published effect is measured
close to close and accrues over several days, and a substantial part of any
multi-day equity premium is overnight. The intraday open-to-close slice that
this project's rules allow it to hold may carry none of it. This entry does not
test whether the turn-of-month effect exists; it tests whether the part of it
that can be captured between 09:30 and 15:55 ET, on MES, after costs, is
distinguishable from zero. That may well be nothing, and a null here says
nothing against the published finding.

### Signal definition

Fixed now, in this order, and not to be changed after any number is seen.

**The trading calendar.** A trading day is a session in the bar data with a
09:30 ET RTH bar, **excluding sessions on which the cash equity market is
closed**. The operator specified the CME trading calendar; CME lists holiday
Globex-only sessions (Martin Luther King Day, Presidents Day, Memorial Day,
Juneteenth, Independence Day, Labor Day, Thanksgiving Day) as trading days with
a 13:00 ET close, and on those days no fund flow executes because the cash
market is shut. Counting them would shift T+1 onto a day the mechanism does
not run; in September 2024 and 2025 Labor Day would have been T+1 or T+2.
Those sessions are removed from the calendar. The three true half-days on which
the cash market is open until 13:00 (the day after Thanksgiving; December 24
when December 25 falls Tuesday to Friday; July 3 when July 4 falls Tuesday to
Friday) **stay in the calendar** so that T+2 stays T+2, and are **skipped as
trade days** because they have no 15:55. `loader.detect_early_close_dates`
flags both kinds; the classification between them is by the three date rules
above and nothing else.

**Window days.** For every month boundary observed in the data: **T-1** is the
last trading day of the month, **T+1, T+2, T+3** are the first three trading
days of the following month. Four window days per month. A boundary is only
labelled when both sides of it are in the data, so the first and last partial
months of the file contribute no mislabelled days.

**Trade days.** Window days that are not a roll day (`loader.detect_roll_dates`)
and not an early-close session. Skipped window days are counted and reported.

**Entry.** Long, at the open of the bar labelled 09:30 ET, on every trade day.
Exactly one trade per session.

**Exit.** At the open of the bar labelled 15:55 ET, the price as of 15:55:00,
labelled `flatten_1555`. The exit is marked on the 15:55 bar per the
interface convention in `strategies/base.py`, so the engine fills it at that
bar's open, exactly as entry 10's flatten did.

**Two arms, identical entries.**

- **Signal arm:** no stop. Its gross points per trade *are* the session's
  open-to-15:55 return; the engine's guards then apply to it like any other
  stream, and the daily loss limit will flatten it on a session that falls 20
  points at 4 contracts. This arm is reported under both guard bases; it is
  what the trade stream of the mechanism looks like.
- **Stop arm:** a 15-point stop below the entry fill, checked bar by bar from
  the bar after the entry bar, filled at the stop level, stop-first. This is
  the Lucid-safe form: 15 points at 4 contracts is $300 plus costs, inside the
  $400 daily limit, so one trade cannot reach the daily loss limit and
  `loss_limit_flatten` is expected never to fire on it (its count is
  reported). Entry-bar stop breaches, where the 09:30 bar's low is already
  through the stop, are not acted on and are counted as a diagnostic; at 15
  points on a one-minute bar the expected count is zero.

**Size.** 4 contracts, fixed. Below the 5-contract cap. Rule 4's cap is the
engine's; nothing here restates it.

**Costs.** `rules.COMMISSION_PER_SIDE` ($0.50 a side, $1.00 a round turn) and
**1 tick a side as the base case**, because every fill in this entry is at
09:30 or 15:55 in RTH liquidity, or at a 15-point stop level during RTH. 2
ticks a side is the sensitivity and is where rule 13 is decided. One round
turn per contract at the base case is $1.00 + $2.50 = $3.50, which is **0.70
MES points**; at 2 ticks it is $6.00, 1.20 points. Both figures are computed
from `engine.CostModel` and the MES spec in the runner, never written down.

**The control.** The open-to-15:55 return, in points, of **every eligible
non-window session** in the same span: sessions with a 09:30 bar and a 15:55
bar, not a roll day, not an early close, not T-1 to T+3. Same instrument, same
bars, same horizon, same exclusions. The only thing that differs between the
window and control populations is the calendar label.

**Guards.** `engine.apply_internal_guards` at 4 contracts, in its order: the
$400 daily loss limit marked to market, then the $1,500 end-of-day trailing
halt on the loss-limited stream. That is the standard basis. The same signals
with the halt off are the comparable basis, reported alongside; **evaluations
blown are read on the comparable stream**, as entries 5 and 10 established,
because a halt that ends a stream in year one leaves nothing to blow.

**Scored span.** From the first walk-forward fold's test start
(`walkforward.build_folds()[0].test_start`, 2020-01-01) to the end of the data
(2026-08-31), imported, not restated. Signals are generated over the whole
file and sliced.

### The walk-forward here is seven out-of-sample years, nothing selected

There is no parameter grid. Every constant above is fixed by the operator's
specification. The yearly folds of `backtests/walkforward.py` are therefore
seven consecutive out-of-sample observations of one rule, as in entries 4, 5,
6, 7 and 10, and "walk-forward" here means that and nothing more.

### Rules compatibility

Entry at 09:30 is before the 16:20 cutoff; the 15:55 exit is before the 16:30
flatten on a normal day; early-close sessions are never traded so rule 2's
13:00 deadline is never reached with a position open. The hold is about six
and a half hours, so rule 6's 30-second floor and rule 7's five-second measure
are untouched: rule 7's share must read 0.00%, and any other value is a bug.
The trade sits entirely inside one calendar date, so `rules.session_date`
groups it correctly for the daily loss limit. Sessions are independent, so
generating signals over the whole history and slicing by date is exact, as
`tests/test_scan.py` asserts for the other strategies.

### Power check, run before any strategy code

**Sample size guard, fixed now:** the count of eligible window sessions per
fold year. **If any fold year 2020–2025 has fewer than 20 eligible window
sessions, or 2026 (eight months) has fewer than 20, the entry stops here** and
records why. The check reads calendar labels and the bar index only; it never
computes a return, so it is a power check and not a peek. It also reports how
many window days each exclusion removes (cash-closed holiday sessions, true
half-days, roll days) so the calendar decision above is visible in numbers
before any outcome is.

**Power arithmetic, so the bar is understood before the run.** Roughly 48
window days a year less exclusions gives about 300 pooled window sessions and
about 1,350 control sessions over 2020–2026. The standard deviation of the
MES open-to-15:55 return over this span is on the order of 25 to 40 points,
2020 at the top of that range. At 30 points and those counts the standard
error of the difference of means is about 1.9 points, so a Welch t of 2.0
needs a pooled excess of roughly **3.5 to 4 points a day**, about five times
one round turn. **The t criterion, not the cost criterion, is the binding
one**, and an effect the size of one round turn cannot pass it at this sample.
That is intended: an excess that clears costs but not significance is a
pattern, not a hypothesis, and the operator set both lines.

### Pre-registered tests

Pooled 2020–2026, at the base case, one configuration:

1. **The reproduction check, first, gating.** The signal arm's unguarded
   one-contract trade stream, priced at zero commission and zero slippage,
   must have exactly the eligible window sessions as its trade dates and its
   `gross_points` must equal the bar-derived open-to-15:55 return on every one
   of them, to floating-point tolerance. The 4-contract stream's gross P&L
   must be exactly four times the 1-contract stream's. The stop arm's entry
   times and entry prices must be identical to the signal arm's. **A strategy
   that does not reproduce the returns the signal test is run on is not a
   test of the mechanism, and the entry stops there.**
2. **Window versus control, pre-cost:** n, mean, standard deviation and
   median of the open-to-15:55 return in points for the window and the
   control; the difference of means; Welch's t and its one-sided p, window
   greater than control.
3. **Per year, the same**, 2020 to 2026, and whether the window mean exceeds
   the control mean in that year.
4. **Per window day, T-1, T+1, T+2, T+3:** n, mean, standard deviation,
   difference against the control and its t. **Reported, never selected on**:
   the criteria below use the pooled four-day window and nothing else.
5. **Both arms, both guard bases, 4 contracts, at 1 tick:** trades, net P&L,
   mean per trade, folds profitable, Sharpe, profit factor, max drawdown, exit
   reasons, `eval_sim` pass and payout probability, evaluations blown on the
   comparable stream, sessions blocked by the halt, daily-loss flattens, and
   every rule 11 figure: max daily loss, worst day as a percentage of total
   profit, best day as a percentage of total profit, average trade duration,
   and the share of profit from trades held five seconds or less.
6. **Everything in 5 repeated at 2 ticks.**
7. **Diagnostics:** window days skipped as early-close or roll sessions, by
   year; entry-bar stop breaches on the stop arm; `loss_limit_flatten` counts
   on both arms.

### Kill criteria — decided now

Any **one** failure kills the entry. Criteria 1 and 2 are the mechanism test
on pre-cost returns, in the manner entry 2 established for a calendar effect;
3 and 4 are the account test on the stop arm; 5 is CLAUDE.md's standing
standard.

1. **Pooled excess and significance.** The window mean open-to-15:55 return
   minus the control mean, pre-cost, in points, must exceed **one round turn
   per contract at the base case** (0.70 points, computed from the cost
   model), **and** Welch's t for the difference must be **at least 2.0**.
   Either shortfall fails.
2. **Window beats control in at least 4 of 7 years**, on the yearly means of
   test 3.
3. **Stop arm, standard (guarded) stream, 1 tick: `eval_sim` pass probability
   at least 25%.** Entry 10 recorded that a trailing halt truncates the sample
   a pooled test runs on; the stream this criterion runs on is therefore
   named here, and the number of trading days the halt left it is reported
   beside the figure. The comparable stream's pass probability is reported
   alongside and decides nothing.
4. **Stop arm, comparable (halt-OFF) stream, 1 tick: evaluations blown at
   most 1**, by `engine.count_evaluation_blowups` against the firm's $2,000
   line.
5. **Rule 13 on the stop arm, standard stream: profitable in at least 4 of 7
   folds and positive total P&L after commission and slippage, at 1 tick and
   at 2 ticks.** *This criterion is not in the operator's list of 2026-09-11.
   It is added because CLAUDE.md rule 13 is the acceptance standard for every
   strategy in this repository and an entry cannot opt out of it; the log
   says so here rather than applying it silently.*

No appeal, no second stop distance, no short arm, no different window, no
different exit time, no selection on the per-day breakdown. If the per-day
table shows one day carrying the whole effect, that is a finding for the
*What was learned* section and a different entry with a different mechanism,
not a re-run of this one.

### Prediction on record

Nothing below has been computed. No return has been inspected.

1. **The pooled intraday excess is small and below one round turn:** between
   −1 and +1 point a day, so criterion 1 fails on the cost line and on the t
   line together. The published effect is close to close and multi-day; the
   09:30-to-15:55 slice of it, on an index future in 2020–2026, is expected to
   be inside the noise.
2. **If any excess is present it concentrates on T-1 and T+1**, the days
   nearest the boundary, with T+2 and T+3 near zero. Reported, not acted on.
3. **Criterion 2 lands at 3 or 4 of 7**, the coin-flip range, because yearly
   means over roughly 44 sessions each have standard errors of 4 to 6 points
   and the true excess is smaller than that.
4. **The stop arm is near break-even before costs and negative after**, with
   pass probability under 25% and 1 to 2 evaluations blown on the halt-OFF
   stream at 4 contracts; the trailing halt is likely to fire at least once
   in 2020 or 2022 on a stream whose daily standard deviation at 4 contracts
   is around $600.
5. **The signal arm under the daily loss limit flattens on some sessions**,
   since a 20-point open-to-close drop is routine in 2020 and 2022; the
   count is reported and is not evidence of anything.

**What would falsify the pessimism:** criteria 1 through 5 all passing. The
diagnostic pair is 1 and 2: an excess above the cost line with t above 2.0 in
a majority of years would say the intraday slice is real; anything less says
the operator's own prior held.

**Prior for survival: low.** The counterparty is well named and well
constrained, which is why the entry is worth writing; the claimed effect is
being measured in the part of the day where it is least expected to live.

### Data and cost

`data/mes_v_0_ohlcv_1m_2019-05_2026-08.parquet`, 2019-05-05 to 2026-08-31,
already on disk and validated in entries 1 to 10. **No data purchase.** The
run is a single pass over the sessions plus two `eval_sim` calls per basis per
cost level; minutes, not hours. MNQ is not tested: the operator specified MES.

### Longer-term intent: copying trades across multiple funded accounts

Unchanged and repeated because it governs how any accepted result would be
used. Copying identical trades across N funded accounts multiplies outcomes in
both directions and is not diversification: the same losing day draws down
every account at once and a trailing-drawdown breach terminates all of them on
the same date. It is one bet at N times the size with N times the fees.

### Power check result and reproduction, 2026-09-11 — run before the verdict

`research/power_check_tom.py`, on the calendar only. 1,890 sessions carry a
09:30 bar; 64 are early closes, of which 15 are cash half-days kept in the
calendar and 49 are cash-closed holiday sessions removed from it; 29 roll
days, none of which falls on a window day.

| Year | Window | Eligible | Control | T-1 | T+1 | T+2 | T+3 | Skipped |
|---|---|---|---|---|---|---|---|---|
| 2020 | 48 | 48 | 202 | 12 | 12 | 12 | 12 | 0 |
| 2021 | 48 | 48 | 203 | 12 | 12 | 12 | 12 | 0 |
| 2022 | 48 | 48 | 200 | 12 | 12 | 12 | 12 | 0 |
| 2023 | 48 | 47 | 197 | 12 | 11 | 12 | 12 | 1 early close |
| 2024 | 48 | 46 | 200 | 11 | 12 | 12 | 11 | 2 early closes |
| 2025 | 48 | 46 | 197 | 11 | 12 | 12 | 11 | 2 early closes |
| 2026 | 31 | 31 | 133 | 7 | 8 | 8 | 8 | 0 |
| **Pooled** | **319** | **314** | **1,332** | 77 | 79 | 80 | 78 | 5 |

The five skipped window days are all cash half-days: 2023-07-03 (T+1),
2024-07-03 (T+3), 2024-11-29 (T-1), 2025-07-03 (T+3), 2025-11-28 (T-1).
**Every fold year clears the 20-session floor; the smallest is 2026 at 31.
The power check passed.** The pooled counts are within the power arithmetic
above (about 300 against about 1,350), so the t criterion remains the binding
one.

**On the ordering.** The calendar functions and the strategy class live in
one module, `strategies/tom.py`, and were written in one pass, test-first,
before the power check ran. The check reads `tom.session_calendar`, which
touches the bar index and never a price; no return had been computed when
the table above was produced. That is the letter of "before any strategy
code" bent and its purpose kept, and it is recorded rather than implied.

**Reproduction (pre-registered test 1): passed on the second run.** The first
run diverged by three sessions: the signal arm took 314 trades against 311
labelled window sessions, the extras being 2, 3 and 6 January 2020. The
strategy was right and the comparison population was wrong: it had been built
on bars sliced to the scored span, which cannot see the December 2019
boundary, while the signals are generated over the whole file and sliced. The
population is now built the same way (`run_entry11.scored_returns`, with a
test pinning the case), and the second run reproduced all 314 window sessions
session for session, the 4-contract stream at exactly four times the
1-contract stream, and the stop arm's entries identical to the signal arm's.
The strategy's trade set did not change between the two runs.

### Verdict

Not yet run. To be filled in after the run, with the commit hash recorded in a
follow-up commit. If ACCEPTED, the verdict must carry on its first line that
the effect was measured on the intraday slice only and that its cost basis
rests on a slippage assumption no backtest can measure.

### Verdict: REJECTED

**Rejected on the pre-registered criteria.** The intraday slice of the turn-of-month effect, on MES over 2020–2026, did not clear the lines the operator set before the run.

**Date:** 2026-09-11
**Code:** `strategies/tom.py`, `backtests/run_entry11.py`, `research/power_check_tom.py`
**Reports:** `backtests/results/entry11_report.txt`; `entry11_returns.csv` (every eligible session with its label and open-to-15:55 points); per arm `entry11_<signal|stop>_slip<1|2>.csv` (standard) and `_nohalt.csv` (comparable); `entry11_folds.csv` (stop arm, 1 tick, standard)

Long MES at the 09:30 open on T-1, T+1, T+2, T+3 by the cash trading calendar, flat at the 15:55 open, 4 contracts, $0.50 a side, **1 tick a side as the base case** (0.70 points a round turn), 2020-01-01 .. 2026-08-31, nothing selected. Guards: `engine.apply_internal_guards` at 4 contracts, the daily loss limit then the trailing halt; halt OFF alongside.

#### Kill criteria — any one failure kills the entry

| # | Criterion | Result | Line | |
|---|---|---|---|---|
| 1 | Pooled window excess over control, pre-cost, and its t | +4.95 pts, t = +1.94 | > 0.70 pts and t >= 2.0 | **FAIL** |
| 2 | Years in which the window mean beats the control mean | 5 of 7 | >= 4 of 7 | **PASS** |
| 3 | Stop arm, standard stream, 1 tick: eval_sim pass probability | 4.75% (11 trading days) | >= 25% | **FAIL** |
| 4 | Stop arm, comparable stream, 1 tick: evaluations blown | 10 | <= 1 | **FAIL** |
| 5 | Rule 13 on the stop arm, standard stream: >= 4 of 7 folds profitable and P&L > 0 at 1 and 2 ticks | 1 tick 0 of 7, $-474; 2 ticks 0 of 7, $-584 | >= 4 of 7 and > 0, both | **FAIL** |

**REJECTED.** Failed on: Pooled window excess over control, pre-cost, and its t; Stop arm, standard stream, 1 tick: eval_sim pass probability; Stop arm, comparable stream, 1 tick: evaluations blown; Rule 13 on the stop arm, standard stream: >= 4 of 7 folds profitable and P&L > 0 at 1 and 2 ticks.

#### The mechanism test, pre-cost, in MES points

**Pooled:** window n = 314, mean +5.20, sd 40.48, median +7.00; control n = 1,332, mean +0.26, sd 40.99, median +2.00. **Difference +4.95 points, Welch t = +1.94** (df 476), one-sided p = 0.026. One round turn at the base case is 0.70 points.

| Year | Window n | Window mean | Control n | Control mean | Difference | t | Window beats control |
|---|---|---|---|---|---|---|---|
| 2020 | 48 | +8.06 | 202 | -0.68 | +8.74 | +1.52 | yes |
| 2021 | 48 | -2.92 | 203 | +3.59 | -6.51 | -1.26 | no |
| 2022 | 48 | +3.24 | 200 | -2.40 | +5.64 | +0.68 | yes |
| 2023 | 47 | +5.65 | 197 | +1.35 | +4.30 | +0.98 | yes |
| 2024 | 46 | -3.23 | 200 | +0.61 | -3.84 | -0.68 | no |
| 2025 | 46 | +8.43 | 197 | +0.38 | +8.05 | +0.97 | yes |
| 2026 | 31 | +23.44 | 133 | -1.75 | +25.19 | +2.79 | yes |

Window beats control in **5 of 7** years.

**Per window day, reported and not selected on:**

| Day | n | Mean | sd | Difference vs control | t |
|---|---|---|---|---|---|
| T-1 | 77 | +3.59 | 43.17 | +3.33 | +0.66 |
| T+1 | 79 | +2.24 | 42.08 | +1.98 | +0.41 |
| T+2 | 80 | +12.02 | 39.34 | +11.76 | +2.59 |
| T+3 | 78 | +2.81 | 37.05 | +2.55 | +0.59 |

#### Stop arm (15-point stop), the Lucid-safe form — 1 tick (base case)

| 2020–2026, 4 contracts, 1 tick, $0.50 | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | 11 | 314 |
| Trading days | 11 | 314 |
| Net P&L | $-474.00 | $14,369.00 |
| Mean per trade | $-43.09 | $45.76 |
| Folds profitable | 0 of 7 | 5 of 7 |
| Sharpe | -2.04 | 1.38 |
| Profit factor | 0.748 | 1.247 |
| Max drawdown | $-1,577.00 | $-7,161.00 |
| Max daily loss (worst day) | $-314.00 | $-314.00 |
| Worst day as % of total profit | n/a | -2.2% |
| Best day as % of total profit (rule 8) | n/a | 15.5% |
| Pass probability | 4.75% | 43.73% |
| Payout probability ($52,100) | 11.39% | 54.05% |
| Evaluations blown (read on the comparable basis) | 10 | 10 |
| Sessions blocked by the trailing halt | 303 | — |
| Daily-loss flattens | 0 | 0 |
| Avg trade duration | 186.8 min | 199.3 min |
| Profit from <=5s holds | 0.00% | 0.00% |

Exits, standard stream: flatten_1555: 5 trades, mean $282.00, total $1,410.00; stop: 6 trades, mean $-314.00, total $-1,884.00. Daily-loss halts: 0.
Per fold, standard: 2020 $-474.00; 2021 $0.00; 2022 $0.00; 2023 $0.00; 2024 $0.00; 2025 $0.00; 2026 $0.00.

#### Stop arm (15-point stop), the Lucid-safe form — 2 ticks (sensitivity)

| 2020–2026, 4 contracts, 2 ticks, $0.50 | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | 11 | 314 |
| Trading days | 11 | 314 |
| Net P&L | $-584.00 | $11,229.00 |
| Mean per trade | $-53.09 | $35.76 |
| Folds profitable | 0 of 7 | 5 of 7 |
| Sharpe | -2.52 | 1.08 |
| Profit factor | 0.700 | 1.187 |
| Max drawdown | $-1,657.00 | $-8,601.00 |
| Max daily loss (worst day) | $-324.00 | $-324.00 |
| Worst day as % of total profit | n/a | -2.9% |
| Best day as % of total profit (rule 8) | n/a | 19.7% |
| Pass probability | 3.06% | 39.52% |
| Payout probability ($52,100) | 8.12% | 50.02% |
| Evaluations blown (read on the comparable basis) | 10 | 10 |
| Sessions blocked by the trailing halt | 303 | — |
| Daily-loss flattens | 0 | 0 |
| Avg trade duration | 186.8 min | 199.3 min |
| Profit from <=5s holds | 0.00% | 0.00% |

Exits, standard stream: flatten_1555: 5 trades, mean $272.00, total $1,360.00; stop: 6 trades, mean $-324.00, total $-1,944.00. Daily-loss halts: 0.
Per fold, standard: 2020 $-584.00; 2021 $0.00; 2022 $0.00; 2023 $0.00; 2024 $0.00; 2025 $0.00; 2026 $0.00.

#### Signal arm (no stop), the mechanism's trade stream — 1 tick (base case)

| 2020–2026, 4 contracts, 1 tick, $0.50 | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | 10 | 314 |
| Trading days | 10 | 314 |
| Net P&L | $-415.00 | $10,854.00 |
| Mean per trade | $-41.50 | $34.57 |
| Folds profitable | 0 of 7 | 4 of 7 |
| Sharpe | -1.67 | 0.93 |
| Profit factor | 0.776 | 1.154 |
| Max drawdown | $-1,518.00 | $-8,247.00 |
| Max daily loss (worst day) | $-534.00 | $-674.00 |
| Worst day as % of total profit | n/a | -6.2% |
| Best day as % of total profit (rule 8) | n/a | 20.6% |
| Pass probability | 8.53% | 36.84% |
| Payout probability ($52,100) | 16.87% | 47.38% |
| Evaluations blown (read on the comparable basis) | 11 | 11 |
| Sessions blocked by the trailing halt | 304 | — |
| Daily-loss flattens | 4 | 152 |
| Avg trade duration | 240.1 min | 238.9 min |
| Profit from <=5s holds | 0.00% | 0.00% |

Exits, standard stream: flatten_1555: 6 trades, mean $239.33, total $1,436.00; loss_limit_flatten: 4 trades, mean $-462.75, total $-1,851.00. Daily-loss halts: 152.
Per fold, standard: 2020 $-415.00; 2021 $0.00; 2022 $0.00; 2023 $0.00; 2024 $0.00; 2025 $0.00; 2026 $0.00.

#### Signal arm (no stop), the mechanism's trade stream — 2 ticks (sensitivity)

| 2020–2026, 4 contracts, 2 ticks, $0.50 | Halt ON (standard) | Halt OFF (comparable) |
|---|---|---|
| Trades | 10 | 314 |
| Trading days | 10 | 314 |
| Net P&L | $-515.00 | $3,939.00 |
| Mean per trade | $-51.50 | $12.54 |
| Folds profitable | 0 of 7 | 3 of 7 |
| Sharpe | -2.07 | 0.35 |
| Profit factor | 0.728 | 1.054 |
| Max drawdown | $-1,588.00 | $-13,372.00 |
| Max daily loss (worst day) | $-544.00 | $-684.00 |
| Worst day as % of total profit | n/a | -17.4% |
| Best day as % of total profit (rule 8) | n/a | 56.1% |
| Pass probability | 5.96% | 30.26% |
| Payout probability ($52,100) | 13.20% | 40.90% |
| Evaluations blown (read on the comparable basis) | 13 | 13 |
| Sessions blocked by the trailing halt | 304 | — |
| Daily-loss flattens | 4 | 155 |
| Avg trade duration | 240.1 min | 236.8 min |
| Profit from <=5s holds | 0.00% | 0.00% |

Exits, standard stream: flatten_1555: 6 trades, mean $229.33, total $1,376.00; loss_limit_flatten: 4 trades, mean $-472.75, total $-1,891.00. Daily-loss halts: 155.
Per fold, standard: 2020 $-515.00; 2021 $0.00; 2022 $0.00; 2023 $0.00; 2024 $0.00; 2025 $0.00; 2026 $0.00.

#### Diagnostics

Window sessions skipped in the scored span: 5 (2023-07-03 T+1 early_close, 2024-07-03 T+3 early_close, 2024-11-29 T-1 early_close, 2025-07-03 T+3 early_close, 2025-11-28 T-1 early_close). Entry-bar stop breaches on the stop arm: 1.

In-sample results are never evidence, and this is the out-of-sample answer on seven yearly folds with nothing selected.

### What was learned

**The intraday slice is not nothing, and it is not enough.** The pooled window
excess over the control is +4.95 points a day before costs, seven times one
round turn, with a one-sided p of 0.026 — and a Welch t of 1.94 against a
pre-registered line of 2.0. Criterion 1 failed on significance alone, by
six-hundredths of a t. That is the pre-registration working as intended: the
line was set before the run, the entry said in advance that the t criterion
would be the binding one, and a result that lands just under it is rejected
exactly as one that lands far under it. Moving the line now is what this log
exists to prevent.

**The prior was wrong in the direction it least expected.** Prediction 1 said
the excess would be under one round turn; it was seven. Prediction 3 said
criterion 2 would land at 3 or 4 of 7; it landed at 5. Prediction 2 said any
effect would sit on T-1 and T+1; it sits on **T+2** (+11.76 points, t 2.59),
with T-1, T+1 and T+3 all inside two to three points of the control. That
is reported and not selected on, as pre-registered, and it is precisely the
table the kill criteria were written to be immune to: dropping T+2 leaves the
window excess at +2.62 points with t 0.91, so a post-hoc entry on T+2 alone
would be built on the one label out of four that happened to carry the pooled
number.

**A partial year carries much of the pooled excess.** 2026 contributes 31
window sessions at a mean of +23.44 points (t 2.79 on its own). Without 2026
the pooled excess is +2.73 points with t 1.04 and p 0.15. Both figures are
post-hoc diagnostics computed after the verdict and decide nothing; they are
recorded because the next reader should know that the near-miss on criterion
1 rests substantially on eight months of one year.

**The Lucid-safe form is not Lucid-safe at 4 contracts, and that failure is
sizing, not signal.** On the comparable stream the stop arm nets +$14,369
over 314 trades, 5 of 7 folds profitable, profit factor 1.25, Sharpe 1.38,
pass probability 43.7% — the best comparable-basis figures in this log by a
wide margin. It also blows ten evaluations, because a day at 4 contracts has a
standard deviation of about $526 and the firm's $2,000 trail is under four of
those; the same stream passes seven evaluations, which is what a 43.7% pass
probability looks like as a sequence. Under the internal guards the $1,500
halt fires on 2020-04-01 after eleven trades and never releases, leaving the
standard stream with 11 trading days, 0 of 7 folds, and a pass probability of
4.75% measured on nothing. Criteria 3, 4 and 5 all failed on that geometry.
The entry fixed 4 contracts by operator specification and did not pre-register
a size sensitivity, so whether one contract survives the trail is a question
this entry cannot answer and does not.

**The trailing halt lesson from entry 10 applied on the day it was written
down.** Entry 10's addendum said a pre-registration combining a trailing halt
with a pooled test must state what sample the halt will leave. This entry
named the stream and reported the day count beside the figure (11 days), which
made the truncation visible rather than misleading — the figure reads as
meaningless on its face, which is the correct reading. It did not change the
outcome and was not meant to.

**The calendar decision mattered on two days a year and was the right call.**
Removing the cash-closed holiday sessions from the count kept T+1 on the first
day the cash market was open. The five half-days lost to the no-15:55 rule
were all T-1 or T+3 in July and November; none was a T+2.

**Rule 7 read 0.00% on every stream and rule 6 was never near**, as an
entry holding six hours must. The consistency check (rule 8) on the
comparable stop arm at 1 tick reads 15.5%, inside the 30% line; at 2 ticks on
the comparable signal arm it reads 56.1% because that stream's total profit is
small, which is the metric behaving correctly on a near-zero denominator.

### Next

**Do not re-run this entry at a different size, a different stop, a T+2-only
window, or without 2026.** Each of those is a change chosen after seeing the
table above, and each is the route by which a rejected entry gets re-tested
until it passes. Sizing was fixed by the operator before the run; the T+2
concentration is a post-hoc reading of a four-label table; and 2026 is data,
not an outlier to be argued away.

**What would justify a new entry, written before any code:** the mechanism
test here came within 0.06 of its t line on 314 sessions, and the honest way
to resolve a near-miss is more out-of-sample data, not a reworked test. Two
routes exist. The first is time: each further month adds four window sessions,
and a pre-registered replication on data after 2026-08-31 — same calendar,
same horizon, same control, t line fixed in advance, with a sample size stated
before the run — would be entry 7's replication logic applied here. The
second is a second instrument: MNQ has been in the file since entry 7, and a
replication of the pooled test on it, pre-registered with its own line and
its own prediction, would ask the same question of a series this entry never
looked at. Either must name in advance which days are in the window, and
neither may be T+2 alone.

**Any tradeable form must be sized to the trail before it is written, not
after.** A strategy whose daily standard deviation is a quarter of the firm's
trailing drawdown blows evaluations at a rate no signal can outrun. A future
entry on this mechanism should state its contract count from the daily
standard deviation of the control population, which is now known, and
pre-register the evaluation-blown criterion on that size.

**The counterparty claim held up better than any breakout entry's and the
entry still died.** That is worth recording: a well-named, well-constrained
counterparty is necessary for an entry to be worth writing and is not
sufficient for it to pass. The mechanism is real in the literature, visible
here at the noise floor, and not capturable at this size under these rules.

---

## 12. Turn-of-month intraday — replication on MNQ, and a forward test on MES — PROPOSED

**Date:** 2026-09-12
**Spec frozen at:** the commit adding this entry
**Code:** not yet written. `strategies/tom.py` is reused unchanged; the runner
and the power check's instrument flag are new.
**Note:** an entry cannot contain its own commit hash. The verdict commit is
recorded in a one-line follow-up commit, never by amending.
**Instrument:** Part A — MNQ, 1-minute bars, 09:30 to 15:55 ET, 2020-01-01 to
2026-08-31. Part B — MES, the same, on sessions after 2026-08-31 only.
**Predecessor:** entry 11, REJECTED at `2bf4fc4`, whose Next section names
exactly these two routes and forbids every other.

**Operator decisions, 2026-09-12, before this entry was frozen:** both parts,
and **each must pass** for the entry to be ACCEPTED; the significance line
stays at Welch t ≥ 2.0 as in entry 11; the stop arm is **one contract**; the
stop is entry 11's 15 MES points expressed as a fraction of MES's control
standard deviation and applied to each instrument's own.

### This is a replication, not a new strategy

**No new mechanism is claimed and entry 11 stays rejected.** Entry 11 measured
the intraday slice of the turn-of-month effect on MES and found a pooled
excess of +4.95 points a day over the control, seven times one round turn,
with Welch t = 1.94 against a line of 2.0 and a one-sided p of 0.026. It
failed criterion 1 by six-hundredths of a t. Its account criteria failed on
sizing: four contracts against a $2,000 trail blew ten evaluations on a
stream that netted +$14,369.

That leaves one number worth one more look, and entry 11 said how: on data
the question has never touched, with the line fixed in advance, never on T+2
alone, sized to the trail. This entry is that look, twice.

**Mechanism, restated and unchanged.** Pension, mutual-fund and
retirement-plan cash flows cluster at month boundaries and are deployed in
the last one to two and first one to three trading days regardless of price
(Etula, Rinne, Suominen & Vaittinen 2020). **The counterparty** is the
institution meeting a dated obligation: constrained by the calendar, not the
price, and paying for liquidity in impact. The liquidity provider absorbs that
flow and is compensated; the claimed edge is to stand where the provider
stands on the days the schedule makes the flow predictable. Entry 11 recorded,
and this entry repeats, that most of the published effect is close to close
and multi-day, so the intraday slice may carry none of it.

### Part A — MNQ, 2020–2026

**What is unchanged from entry 11, line for line.** The cash trading calendar
(`tom.trading_days`: sessions with a 09:30 bar less the cash-closed holiday
sessions, half-days kept in the count and skipped as trade days). The window
(T-1, T+1, T+2, T+3 at every observed month boundary). Long at the 09:30 open,
flat at the 15:55 open, one trade per session, roll days and early closes
skipped. The control: the open-to-15:55 return of every eligible non-window
session. The scored span from `walkforward.build_folds()`. The costs:
`rules.COMMISSION_PER_SIDE`, 1 tick a side as the base case, 2 ticks as the
sensitivity. The guards: `engine.apply_internal_guards`, standard basis with
the halt-OFF stream alongside, evaluations blown read on the comparable stream.
The reproduction check, the per-day table reported and never selected on, and
every rule 11 figure.

**What follows from the contract.** MNQ is $2.00 a point with a $0.50 tick.
One round turn at the base case is $1.00 commission plus $1.00 slippage, which
is **1.00 MNQ point** (computed from the cost model, not written down). The MNQ
commission is assumed equal to MES's confirmed $0.50 a side; only MES was
confirmed (handoff §3.3), and the assumption is stated here so a later
correction can re-price the stream exactly.

**Sizing: one contract, fixed.** Entry 11's lesson was that a tradeable form
must be sized to the trail before it is written. The control population's
daily standard deviation on MES is 40.99 points, $205 a day per contract, so
the firm's $2,000 trail is about ten daily standard deviations at one contract
and 2.4 at four. MNQ's control standard deviation in dollars is expected to be
larger (the Nasdaq-100 moves more than the S&P 500 in percentage terms and the
contract is $2 a point on a higher index), putting the trail nearer five to
seven daily standard deviations at one contract. One contract is the minimum
the instrument allows and the only size this entry tests.

**The stop, instrument-aware.** Entry 11's 15-point stop was **0.366 of MES's
control standard deviation** (15 / 40.99). The stop arm here uses
`stop_points = round_to_tick(0.366 × sd_control)` where `sd_control` is the
standard deviation of the **control** population's open-to-15:55 return on the
instrument under test, computed at run time and reported. The control
population is every eligible non-window session, so computing its spread does
not look at a window outcome. Entry 7's MNQ overshoot lesson is why the stop is
not carried over in points: the same number of points is a different rule on a
different contract.

**Power check, run before any runner code.** `research/power_check_tom.py`
with the MNQ parquet, calendar only: eligible window sessions per fold year.
**Floor: 20 in every fold year**, as in entry 11, or the entry stops. MNQ's
file spans the same dates as MES's and its early-close and roll sets are
detected from its own bars, so the counts are expected to match entry 11's 314
and 1,332 closely but need not be identical.

**Power arithmetic.** Welch's t is scale-free, so the line sits at the same
place in standard-deviation units as entry 11's: with about 314 window and
1,332 control sessions, t = 2.0 needs a pooled excess of roughly **0.125
control standard deviations**. Entry 11's MES excess was 4.95 / 40.99 =
**0.121**. **The test is set at almost exactly the effect MES showed.** If the
MNQ effect is the same size, criterion 1 is a coin flip; if it is smaller, it
fails; only a larger effect passes comfortably. That is what replicating a
near-miss at the same line means, and it is intended.

### Part B — MES, forward data only

**The same test on MES sessions after 2026-08-31**, which no entry has seen.
Same calendar, window, horizon, control, costs, guards, one contract, the stop
at 0.366 of the forward control population's standard deviation.

**Trigger, fixed now:** Part B runs once the forward MES file holds **at least
96 eligible window sessions** — about two calendar years — and not before. The
data is pulled with `data/extend.py`, cost estimated and shown before any pull.

**Power, stated honestly.** At 96 window and roughly 400 control sessions with
a standard deviation near 41 points, the standard error of the difference is
about 4.6 points, so t = 2.0 needs an excess of roughly **9 points a day**,
almost twice entry 11's +4.95. Against an effect of entry 11's size Part B has
on the order of **20–25% power**. A larger trigger would fix that only by
waiting: 50% power against +4.95 points needs about 330 window sessions, seven
years. The operator chose both parts with each required to pass, so **this
entry cannot be ACCEPTED before Part B runs, and Part B as specified is more
likely than not to fail even if the effect is real.** Recorded so the verdict,
whatever it is, is read with that in view. Part B's terms are fixed here and
are not revised after Part A's result is known; any change to them after that
point would be a new entry.

### Rules compatibility

As entry 11: entry at 09:30 before the 16:20 cutoff, exit at 15:55 before the
16:30 flatten, early closes never traded, the trade inside one calendar date,
a six-and-a-half-hour hold so rules 6 and 7 are untouched and rule 7 must read
0.00%. One contract is inside the 5-contract cap; at one contract a stop of
about 0.37 daily standard deviations cannot reach the $400 daily loss limit on
either instrument, so `loss_limit_flatten` is expected never to fire on the
stop arm and its count is reported.

### Pre-registered tests

Part A, pooled 2020–2026 at the base case, one configuration per arm:

1. **Reproduction, first, gating:** the signal arm's unguarded one-contract
   zero-cost stream must reproduce the eligible window sessions and their
   open-to-15:55 returns session for session; the stop arm's entries must be
   identical to the signal arm's. A strategy that does not reproduce the
   population it is tested on stops the entry.
2. **Window versus control, pre-cost:** n, mean, standard deviation, median,
   difference, Welch t, one-sided p.
3. **Per year**, and whether the window mean exceeds the control mean.
4. **Per window day**, T-1, T+1, T+2, T+3: n, mean, difference, t. **Reported,
   never selected on.** Whether MNQ shows entry 11's T+2 concentration is a
   finding for *What was learned* and decides nothing here.
5. **Both arms, both guard bases, one contract, 1 tick:** trades, net P&L,
   mean per trade, folds profitable, Sharpe, profit factor, max drawdown, exit
   reasons, `eval_sim` pass and payout probability with the day count beside
   them, evaluations blown on the comparable stream, sessions blocked by the
   halt, daily-loss flattens, and every rule 11 figure.
6. **Everything in 5 at 2 ticks.**
7. **The stop in points** as derived, with the control standard deviation it
   came from; skipped window days by year; entry-bar stop breaches.

Part B, when triggered: tests 1–7 on the forward MES sessions, with "per
year" meaning each forward calendar year that has at least one window session.

### Kill criteria — decided now

**Part A, any one failure kills the entry:**

1. Pooled window excess over control, pre-cost, in MNQ points, **greater than
   one round turn per contract at the base case** (1.00 point, computed) **and
   Welch t ≥ 2.0.**
2. Window mean beats control mean in **at least 4 of 7 years.**
3. Stop arm, standard (guarded) stream, 1 tick: `eval_sim` pass probability
   **≥ 25%**, with the stream's trading-day count reported beside it.
4. Stop arm, comparable (halt-OFF) stream, 1 tick: **evaluations blown ≤ 1.**
5. Rule 13 on the stop arm, standard stream: **≥ 4 of 7 folds profitable and
   P&L > 0 at 1 tick and at 2 ticks.**

**Part B, any one failure kills the entry:** criteria 1, 3, 4 and 5 as above
on the forward MES data (criterion 1 at MES's 0.70-point round turn), and
criterion 2 as the window mean beating the control in a **majority of the
forward years.**

**Verdict logic.** If Part A fails, the entry is REJECTED at Part A and Part B
is never run. If Part A passes, the entry **stays PROPOSED** with a dated note
recording Part A's result, and the verdict is written only when Part B has
run; ACCEPTED requires both. No appeal, no second size, no second stop rule,
no T+2-only window, no third instrument, no shortening of Part B's trigger.

### Prediction on record

Nothing below has been computed on MNQ or on any forward data.

1. **Part A, criterion 1: more likely to fail than pass, about one in four.**
   If entry 11's effect is real and the same size in standard-deviation units
   on MNQ, the line is a coin flip; weighting that by a prior near 40% that the
   intraday effect is real at all gives roughly 20–25%. The pooled excess is
   predicted **positive** in any case, between 0 and 0.15 control standard
   deviations.
2. **The per-day table does not reproduce entry 11's T+2 concentration.** An
   effect concentrated on one of four labels with t 2.59 is what one label out
   of four looks like when noise picks a winner; on MNQ the four days are
   predicted to sit within noise of each other.
3. **Criterion 2 lands at 4 or 5 of 7**, as entry 11's did.
4. **The one-contract stop arm fails criteria 3 and 5 and passes 4.** At one
   contract the daily standard deviation is a few hundred dollars, the $3,000
   target is far away relative to the day-to-day noise so pass probability is
   low, and the $1,500 internal halt is predicted to fire at least once in the
   2020 or 2022 drawdowns, leaving the guarded stream with a truncated fold
   table. Evaluations blown on the comparable stream: zero or one.
5. **Part B fails on power if it is ever reached**, for the reason stated.

**Prior for survival: very low**, and for the Part A mechanism test alone,
about one in four.

### Data and cost

Part A: `data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet`, on disk since entry
7 and validated there. **No purchase.** Part B: a forward MES pull when the
trigger is near; estimate shown before any spend.

### Longer-term intent: copying trades across multiple funded accounts

Unchanged from every prior entry: N accounts running one strategy is one bet
at N times the size with N times the fees, and a single trailing-drawdown
breach ends all of them on the same day.

### Power check and reproduction, Part A, 2026-09-12 — run before the result

`research/power_check_tom.py --parquet data/mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet`,
calendar only. MNQ's file has the same 1,890 sessions with a 09:30 bar, the
same 64 early closes (15 cash half-days kept, 49 cash-closed holiday sessions
removed) and the same 29 roll days as MES's. Eligible window sessions per fold
year: 48, 48, 48, 47, 46, 46, 31 for 2020 to 2026, **314 pooled against 1,327
control** (MES had 1,332; five MNQ sessions lack a 15:55 bar). The same five
half-days are skipped. **Every fold year clears the floor; the power check
passed.** The power arithmetic above stands: t = 2.0 needs about 0.125 control
standard deviations.

**Reproduction (pre-registered test 1): passed on the first run.** The signal
arm reproduced all 314 MNQ window sessions session for session at one contract
and zero cost, and the stop arm's entries were identical to the signal arm's.
The population is built over the whole file and sliced, as entry 11's
correction requires. No strategy code changed for this entry;
`strategies/tom.py` is entry 11's module unchanged, and the runner
(`backtests/run_entry12.py`) derives the stop from the control population at
run time and recomputes entry 11's MES control standard deviation from the
bars before every run, refusing if it has moved from the recorded 40.99.

### Verdict

Not yet run. Part A's result is recorded here when it is in, with the commit
hash in a follow-up commit; the entry's verdict is written only when both
parts have run, or when either has failed.

---

## Template for new entries

```
## N. <Name> — PROPOSED | ACCEPTED | REJECTED

**Date:** YYYY-MM-DD
**Commit:** <hash>
**Code:** <path, or "not yet written">
**Note:** an entry cannot contain its own commit hash. Record the verdict
commit in a one-line follow-up commit immediately after, never by amending.
**Instrument:** MES | MNQ, timeframe

### Mechanism claimed
Why this should work. What structural feature of the market creates it.

**Who is on the other side:** name the counterparty and why they trade the
losing side knowingly. "Retail" is not an answer; state what constrains them.

### Parameters tested
Grid, and the defaults.

### Walk-forward verdict
Per-fold table, totals, rank correlation. Filled in after the run, never before.

### What was learned
Including what would have to be true for the idea to be revisited.
```
