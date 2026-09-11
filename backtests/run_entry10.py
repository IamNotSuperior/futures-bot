"""Entry 10: London 1x/ON held through the US session. Frozen at ``fd0c5c8``.

The entry's specification is entry 6's 1x/ON arm with two changes: the flatten
moves from 09:25 to 15:55 ET (12:55 on early-close sessions) and sizing follows
the log's corrected realised-stop rule. Both instruments, MES and MNQ, run the
identical configuration and both must pass.

Build order, fixed by the entry and by the operator
---------------------------------------------------
1. ``--reproduce`` - the entry's ninth pre-registered test, run **first**. The
   new code, with the flatten at 09:25, costs at $1.25 and entry 6's range-based
   sizing switched on for this check only, must reproduce entry 6's 1x/ON stream
   on MES (686 trades, -$7,702.50) and entry 7's on MNQ (438, +$116.00) trade
   for trade. A code change that cannot reproduce the frozen streams is not a
   test of the flatten.
2. The 15:55 run, **only after the operator confirms the reproduction**. It is
   not in this module yet, on purpose: nothing here can produce a 15:55 number
   before that confirmation.

    python backtests/run_entry10.py --reproduce
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from engine import MES, MNQ, CostModel, build_trades, price_trades  # noqa: E402
from london import LondonBreakout, LondonParams  # noqa: E402
from run_london import apply_daily_limit  # noqa: E402
from scan import slice_by_date  # noqa: E402
from trend import intraday_ema  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"
START, END = date(2020, 1, 1), date(2026, 8, 31)

INSTRUMENTS = {
    "MES": dict(parquet="mes_v_0_ohlcv_1m_2019-05_2026-08.parquet", spec=MES,
                saved="entry7_mes_1x_on.csv"),
    "MNQ": dict(parquet="mnq_v_0_ohlcv_1m_2019-05_2026-08.parquet", spec=MNQ,
                saved="entry7_mnq_1x_on.csv"),
}

#: Entry 6's base case and rule 13's survival bar.
BASE_SLIPPAGE_TICKS = 2.0

#: What entries 6 and 7 were scored at. The reproduction must run at exactly
#: this, or the saved streams cannot match to the cent.
REPRODUCTION_COSTS = CostModel(commission_per_side=rules.ASSUMED_COMMISSION_PER_SIDE,
                               slippage_ticks=BASE_SLIPPAGE_TICKS)


def ENTRY6_PARAMS(point_value: float) -> LondonParams:  # noqa: N802
    """Entry 6's 1x/ON arm: the module defaults, on the given contract."""
    return LondonParams(target_multiple=1.0, use_trend_filter=True,
                        point_value=point_value)


def ENTRY10_PARAMS(point_value: float) -> LondonParams:  # noqa: N802
    """Entry 10 as frozen: entry 6 with the flatten at 15:55 (12:55 on an
    early close) and realised-stop sizing. Nothing else differs."""
    return LondonParams(target_multiple=1.0, use_trend_filter=True,
                        point_value=point_value,
                        flatten_time=time(15, 55),
                        early_close_flatten_time=time(12, 55),
                        size_on="stop")


def build(name: str, params: LondonParams, costs: CostModel):
    """One instrument, one parameter set, through entry 6's pricing pipeline.

    Size varies per session, so trades are priced at one contract and each row
    scaled by its own size; the daily loss limit is applied per size group,
    which is exact given one trade a day (``run_london``). The trailing halt is
    **not** applied here - this is the comparable basis, and the standard basis
    is built on top of it by the caller.
    """
    cfg = INSTRUMENTS[name]
    spec = cfg["spec"]
    if params.point_value != spec.point_value:
        raise ValueError(f"{name}: params carry point_value {params.point_value}, "
                         f"the contract is {spec.point_value}")
    bars = loader.load_bars(str(PROJECT_ROOT / "data" / cfg["parquet"]))
    rolls = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)
    ema = intraday_ema(bars, 5, 200)

    strat = LondonBreakout(params, trend_ema=ema, roll_dates=rolls,
                           early_close_dates=early)
    signals = strat.generate_signals(bars)

    win_signals = slice_by_date(signals, START, END)
    win_bars = slice_by_date(bars, START, END)
    priced = price_trades(build_trades(win_signals, win_bars), spec, costs, 1)

    sizes = win_signals["contracts"].dropna()
    n = priced["entry_time"].map(sizes)
    if n.isna().any():
        raise RuntimeError("a trade has no size; signals and trades disagree")
    n = n.astype(int)
    out = priced.copy()
    out["contracts"] = n
    for col in ("gross_pnl", "commission", "slippage_cost", "net_pnl"):
        out[col] = out[col] * n

    out, halts = apply_daily_limit(out, win_bars, costs, spec)

    for col in ("stop_price", "target_price"):
        out[col] = out["entry_time"].map(win_signals[col].dropna()).astype(float)
    if out[["stop_price", "target_price"]].isna().any().any():
        raise RuntimeError("a trade is missing its barrier levels")

    return bars, win_bars, out, strat.diagnostics, halts


# ---------------------------------------------------------------------------
# The reproduction check
# ---------------------------------------------------------------------------

COMPARED = ["entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "exit_reason", "contracts", "net_pnl"]


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Timestamps in ET whatever the source; a saved CSV carries strings that
    span EST and EDT, which pandas will not infer as one offset."""
    out = frame.copy().reset_index(drop=True)
    for col in ("entry_time", "exit_time"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.tz_convert(rules.ET)
    out["contracts"] = out["contracts"].astype(int)
    return out[COMPARED]


def compare_streams(ours: pd.DataFrame, saved: pd.DataFrame) -> list[str]:
    """Trade-for-trade differences between a fresh stream and a saved one.

    Empty means identical on every compared column. A count mismatch is
    reported alone, because row-by-row comparison is meaningless after one.
    """
    a, b = _normalise(ours), _normalise(saved)
    if len(a) != len(b):
        return [f"trade count: ours {len(a)} trades, saved {len(b)} trades"]
    diffs: list[str] = []
    for col in COMPARED:
        if col in ("entry_price", "exit_price", "net_pnl"):
            mismatch = ~np.isclose(a[col].astype(float).to_numpy(),
                                   b[col].astype(float).to_numpy(), atol=1e-6)
        elif col in ("entry_time", "exit_time"):
            mismatch = (a[col] != b[col]).to_numpy()
        else:
            mismatch = (a[col].astype(str) != b[col].astype(str)).to_numpy()
        for i in np.flatnonzero(mismatch):
            diffs.append(f"row {i} ({a['entry_time'].iloc[i].date()}): {col} "
                         f"ours={a[col].iloc[i]!r} saved={b[col].iloc[i]!r}")
    return diffs


def reproduce() -> bool:
    """Run the new code at entry 6's settings and diff against the saved streams."""
    line = "=" * 92
    print(line)
    print("ENTRY 10 - reproduction check (pre-registered test 9)")
    print(f"  entry 6 params: flatten 09:25, range-based sizing, filter ON, 1x")
    print(f"  costs: ${REPRODUCTION_COSTS.commission_per_side:.2f}/side, "
          f"{REPRODUCTION_COSTS.slippage_ticks:g} ticks/side")
    print(line, flush=True)

    all_ok = True
    for name, cfg in INSTRUMENTS.items():
        saved_path = RESULTS / cfg["saved"]
        if not saved_path.exists():
            print(f"\n{name}: saved stream {saved_path.name} is not on disk; "
                  f"cannot compare.")
            all_ok = False
            continue
        print(f"\n{name}: building ...", flush=True)
        _, _, ours, _, halts = build(name, ENTRY6_PARAMS(cfg["spec"].point_value),
                                      REPRODUCTION_COSTS)
        saved = pd.read_csv(saved_path)
        out = RESULTS / f"entry10_reproduction_{name.lower()}.csv"
        ours.to_csv(out, index=False)
        diffs = compare_streams(ours, saved)
        print(f"  ours:  {len(ours):,} trades, net ${ours['net_pnl'].sum():,.2f}, "
              f"{halts} daily-loss halts")
        print(f"  saved: {len(saved):,} trades, net ${saved['net_pnl'].sum():,.2f}  "
              f"({saved_path.name})")
        if diffs:
            all_ok = False
            print(f"  DIVERGES - {len(diffs)} difference(s); first ten:")
            for d in diffs[:10]:
                print(f"    {d}")
        else:
            print(f"  IDENTICAL trade for trade on {', '.join(COMPARED)}")
        print(f"  written: {out.name}")

    print()
    print(line)
    print("REPRODUCTION " + ("PASSED on both instruments" if all_ok else "FAILED"))
    print(line)
    return all_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", action="store_true",
                    help="run the pre-registered reproduction check")
    args = ap.parse_args(argv)
    if not args.reproduce:
        print("Entry 10's 15:55 run is not available until the operator has "
              "confirmed the reproduction check. Run with --reproduce.")
        return 2
    return 0 if reproduce() else 1


if __name__ == "__main__":
    raise SystemExit(main())
