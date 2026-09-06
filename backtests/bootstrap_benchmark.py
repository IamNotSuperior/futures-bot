"""Entry 7's corrected random-walk benchmark: a de-meaned bootstrap.

Entry 6 measured its target share against ``a / (a + b)``, the driftless
random-walk probability of touching one barrier before the other. Entry 6's own
verdict then established that this is **invalid under a time limit**: the
formula assumes an unbounded horizon, and the 09:25 flatten truncates the
farther barrier. The distortion is large - it produced an apparent twelve-point
"shortfall" in the 2x arm that said nothing about the signal.

This module answers the question the formula was standing in for: **given the
same barrier distances and the same time limit, what fraction of trades would
resolve at the target first if the price process had no drift?**

Procedure, fixed by entry 7 at ``bff2bdc``
------------------------------------------
For each realised trade:

1. Take the 1-minute price changes over the trade's holding window - entry bar
   through the flatten bar, at its actual length.
2. Subtract the window's own mean change, so the resampled process is driftless
   by construction while keeping the window's realised volatility and the fat
   tails of its distribution.
3. Resample with replacement to the same length and rebuild forward from the
   actual entry price.
4. Apply the same stop, target and flatten, stop-first when one step reaches
   both.
5. 1,000 replications per trade, seed 0.

The benchmark is the pooled target-first count over the pooled resolved count -
flattens excluded on both sides, exactly as the observed target share excludes
them.

Whole bars, not just closes
---------------------------
Step 1 resamples the ``(high, low, close)`` triplet of each minute jointly,
expressed as offsets from the previous close, rather than closes alone. The live
rule checks the stop and target against bar highs and lows, so a close-only
bootstrap would systematically under-count barrier touches and would not be
"the same stop and the same target" in any meaningful sense. De-meaning
subtracts the window's mean close-change from all three, which shifts each bar
bodily and leaves its intrabar geometry intact.

Sampling with replacement rather than permuting is deliberate: a permutation
holds the terminal price fixed and would test path order rather than drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REPLICATIONS = 1_000
SEED = 0


@dataclass(frozen=True)
class BenchmarkResult:
    """Pooled benchmark plus the diagnostics entry 7 requires."""

    target_share: float          # the benchmark, targets / (targets + stops)
    n_trades: int
    replications: int
    resolved_fraction: float     # mean share of paths that hit a barrier
    per_trade_dispersion: float  # sd of the per-trade benchmark
    pooled_targets: int
    pooled_stops: int

    def z_against(self, observed: float, n_resolved: int) -> float:
        """One-sided z for an observed share against this benchmark."""
        p0 = self.target_share
        if not (0.0 < p0 < 1.0) or n_resolved <= 0:
            return float("nan")
        se = np.sqrt(p0 * (1.0 - p0) / n_resolved)
        return float((observed - p0) / se) if se > 0 else float("nan")


def _by_day(bars: pd.DataFrame, flatten_time) -> dict:
    """Pre-slice the bars once, by calendar date, up to the flatten.

    Built once per instrument rather than per trade. Scanning the whole frame
    inside the per-trade loop is O(trades x bars) - about 1.2 billion row
    comparisons on this data - and turns a minute of work into an hour.
    """
    times = np.array([t.time() for t in bars.index])
    early = bars[times <= flatten_time]
    keys = np.array([t.date() for t in early.index])
    return {day: frame for day, frame in early.groupby(keys)}


def _window(day_frame: pd.DataFrame, entry_ts: pd.Timestamp) -> pd.DataFrame:
    """Bars from the entry bar through the flatten, within one session."""
    if day_frame is None or day_frame.empty:
        return pd.DataFrame()
    return day_frame[day_frame.index >= entry_ts]


def _simulate(dc: np.ndarray, dh: np.ndarray, dl: np.ndarray,
              entry_price: float, stop: float, target: float,
              direction: str, replications: int,
              rng: np.random.Generator) -> tuple[int, int, int]:
    """Returns (targets, stops, unresolved) over ``replications`` paths."""
    n = dc.size
    if n < 2:
        return 0, 0, replications

    idx = rng.integers(0, n, size=(replications, n))
    s_dc, s_dh, s_dl = dc[idx], dh[idx], dl[idx]

    # prev_close for step t is the entry price plus the changes before t.
    cum = np.cumsum(s_dc, axis=1)
    prev = np.empty_like(cum)
    prev[:, 0] = entry_price
    prev[:, 1:] = entry_price + cum[:, :-1]

    highs = prev + s_dh
    lows = prev + s_dl

    if direction == "long":
        hit_stop, hit_target = lows <= stop, highs >= target
    else:
        hit_stop, hit_target = highs >= stop, lows <= target

    big = n + 1
    first_stop = np.where(hit_stop.any(axis=1), hit_stop.argmax(axis=1), big)
    first_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), big)

    # Stop-first when a single step reaches both, matching the live rule.
    stops = int(np.sum((first_stop <= first_target) & (first_stop < big)))
    targets = int(np.sum((first_target < first_stop) & (first_target < big)))
    return targets, stops, replications - targets - stops


def benchmark(trades: pd.DataFrame, bars: pd.DataFrame, flatten_time,
              replications: int = REPLICATIONS,
              seed: int = SEED) -> BenchmarkResult:
    """Pooled de-meaned-bootstrap target share for a set of trades.

    ``trades`` must carry ``entry_time``, ``direction``, ``entry_price``,
    ``stop_price`` and ``target_price``. The horizon is always entry to the
    flatten, regardless of when the real trade actually exited: the benchmark
    asks what a driftless path would do with the same time available.
    """
    rng = np.random.default_rng(seed)
    pooled_t = pooled_s = pooled_u = 0
    per_trade: list[float] = []
    by_day = _by_day(bars, flatten_time)

    for _, row in trades.iterrows():
        entry_ts = pd.Timestamp(row["entry_time"])
        window = _window(by_day.get(entry_ts.date()), entry_ts)
        if len(window) < 3:
            continue
        close = window["close"].to_numpy(float)
        high = window["high"].to_numpy(float)
        low = window["low"].to_numpy(float)

        prev_close = close[:-1]
        dc = close[1:] - prev_close
        dh = high[1:] - prev_close
        dl = low[1:] - prev_close

        drift = dc.mean()
        dc, dh, dl = dc - drift, dh - drift, dl - drift

        t, s, u = _simulate(
            dc, dh, dl, float(row["entry_price"]), float(row["stop_price"]),
            float(row["target_price"]), str(row["direction"]), replications, rng,
        )
        pooled_t += t
        pooled_s += s
        pooled_u += u
        if t + s:
            per_trade.append(t / (t + s))

    resolved = pooled_t + pooled_s
    total = resolved + pooled_u
    return BenchmarkResult(
        target_share=(pooled_t / resolved) if resolved else float("nan"),
        n_trades=len(per_trade),
        replications=replications,
        resolved_fraction=(resolved / total) if total else float("nan"),
        per_trade_dispersion=float(np.std(per_trade)) if per_trade else float("nan"),
        pooled_targets=pooled_t,
        pooled_stops=pooled_s,
    )


def infinite_horizon(trades: pd.DataFrame) -> float:
    """``a / (a + b)`` - the benchmark entry 6 used, for comparison only.

    Kept so the verdict can show how far the invalid formula sat from the
    corrected one, not because it is a defensible benchmark.
    """
    entry = trades["entry_price"].astype(float)
    a = (entry - trades["stop_price"].astype(float)).abs()
    b = (trades["target_price"].astype(float) - entry).abs()
    return float(a.mean() / (a.mean() + b.mean()))
