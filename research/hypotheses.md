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
**Verdict commit:** recorded in the one-line follow-up commit to this one
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
