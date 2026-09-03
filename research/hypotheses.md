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
