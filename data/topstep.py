"""The TopstepX/ProjectX bar repository as a second, read-only bar source.

    venv\\Scripts\\python.exe data\\topstep.py --compare --symbol MES
    venv\\Scripts\\python.exe data\\topstep.py --compare --symbol MNQ

``load_topstep_bars`` reads one CSV from a clone of
https://github.com/axb0306/cme-futures-ohlc (TopstepX bars via the ProjectX
gateway, one contract per symbol hardcoded and rolled by hand) into the same
shape ``loader.load_bars`` gives the Databento cache, so the two can be laid
side by side. ``compare`` does exactly that and ``write_report`` saves the
result as ``backtests/results/data_check_<symbol>.json`` for the viewer's
data-sources panel.

**Databento remains the only backtest source.** The repository's one-minute
span is about three months against the cache's seven years, so nothing here
can feed a walk-forward; its value is as an independent check that the
cache matches what a prop-firm platform's own feed shows. This module never
writes the cache or the forward file. Symbols are validated against
``rules.ALLOWED_INSTRUMENTS``.

The clone is expected at ``TOPSTEP_DATA_DIR`` or, by default, in
``cme-futures-ohlc`` beside the project. It carries no license and stays
outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("data", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader  # noqa: E402
import rules  # noqa: E402

RESULTS = PROJECT_ROOT / "backtests" / "results"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
TIMEFRAME_RE = re.compile(r"^(tick|1min|5min|15min|30min|1h|4h|daily)$")
COLUMNS = ["open", "high", "low", "close", "volume"]


def topstep_dir() -> Path:
    """Where the clone lives: the environment first, the sibling folder else."""
    env = os.environ.get("TOPSTEP_DATA_DIR")
    return Path(env) if env else PROJECT_ROOT.parent / "cme-futures-ohlc"


def _symbol(symbol: str) -> str:
    if symbol not in rules.ALLOWED_INSTRUMENTS:
        raise ValueError(
            f"symbol {symbol!r} is not allowed; this project trades "
            f"{' and '.join(sorted(rules.ALLOWED_INSTRUMENTS))} only (rule 1)"
        )
    return symbol


def _timeframe(timeframe: str) -> str:
    if not TIMEFRAME_RE.match(timeframe):
        raise ValueError(f"timeframe {timeframe!r} is not one the repository publishes")
    return timeframe


def topstep_file(symbol: str, timeframe: str = "1min", directory: Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else topstep_dir()
    symbol, timeframe = _symbol(symbol), _timeframe(timeframe)
    matches = sorted((directory / symbol).glob(f"{symbol}_{timeframe}_*_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"no {symbol}_{timeframe}_<start>_<end>.csv under {directory / symbol}; "
            f"clone axb0306/cme-futures-ohlc there or set TOPSTEP_DATA_DIR"
        )
    return matches[-1]


def load_topstep_bars(symbol: str, timeframe: str = "1min",
                      directory: Path | None = None) -> pd.DataFrame:
    """Bars indexed by ET like ``loader.load_bars``; timestamps in the file are UTC."""
    path = topstep_file(symbol, timeframe, directory)
    frame = pd.read_csv(path)
    ts = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True))
    out = frame[COLUMNS].copy()
    out.index = ts.tz_convert(rules.ET)
    out.index.name = "ts_event_et"
    return out.sort_index()


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = out.index.tz_convert("UTC")
    return out


def compare(symbol: str, cache_path: Path | None = None, directory: Path | None = None,
            tick: float = 0.25) -> dict:
    """How the repository's one-minute bars agree with the cache over their overlap."""
    symbol = _symbol(symbol)
    cache_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    t_path = topstep_file(symbol, "1min", directory)
    t = _utc(load_topstep_bars(symbol, "1min", directory))
    db = _utc(loader.load_bars(cache_path))
    lo, hi = t.index.min(), t.index.max()
    db_span = db.loc[lo:hi]
    common = t.index.intersection(db_span.index)
    a, b = t.loc[common], db_span.loc[common]

    columns: dict[str, dict] = {}
    for col in ("open", "high", "low", "close"):
        d = (a[col].astype(float) - b[col].astype(float)).abs()
        columns[col] = {
            "identical_pct": round(float((d == 0).mean() * 100), 2) if len(d) else 0.0,
            "within_tick_pct": round(float((d <= tick).mean() * 100), 2) if len(d) else 0.0,
            "max_abs_diff": float(d.max()) if len(d) else 0.0,
            "mean_abs_diff": round(float(d.mean()), 4) if len(d) else 0.0,
        }
    vol_same = float((a["volume"] == b["volume"]).mean() * 100) if len(common) else 0.0

    # Days on which the feeds disagree by more than a tick on average: with a
    # hand-rolled contract on one side and a volume roll on the other, these
    # are the roll days, and the size of the gap is the calendar spread.
    diff = a["close"].astype(float) - b["close"].astype(float)
    by_day = diff.groupby(common.date).agg(["mean", "max", "min", "size"])
    differing = [
        {"date": str(day), "mean_diff": round(float(r["mean"]), 2),
         "max_diff": round(float(r["max"]), 2), "min_diff": round(float(r["min"]), 2),
         "bars": int(r["size"])}
        for day, r in by_day.iterrows() if abs(float(r["mean"])) > tick
    ]

    # Bar labelling: shifting the repository a minute earlier should make
    # things worse, not better, if both label by opening minute.
    shifted = t.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=1)
    c2 = shifted.index.intersection(db_span.index)
    shifted_same = float((shifted.loc[c2, "close"].astype(float)
                          == db_span.loc[c2, "close"].astype(float)).mean()) if len(c2) else 0.0
    labelled_by_open = columns["close"]["identical_pct"] / 100 >= shifted_same

    def rth_counts(frame: pd.DataFrame) -> pd.Series:
        et = frame.copy()
        et.index = et.index.tz_convert(rules.ET)
        r = et.between_time("09:30", "15:59")
        return pd.Series(1, index=r.index.date).groupby(level=0).size()

    counts = pd.concat([rth_counts(t).rename("topstep"), rth_counts(db_span).rename("databento")],
                       axis=1).fillna(0).astype(int)
    mismatches = [{"date": str(day), "topstep": int(r["topstep"]), "databento": int(r["databento"])}
                  for day, r in counts.iterrows() if r["topstep"] != r["databento"]]

    return {
        "symbol": symbol,
        "source": "TopstepX via ProjectX gateway (axb0306/cme-futures-ohlc)",
        "topstep_file": t_path.name,
        "cache_file": cache_path.name,
        "computed": datetime.now(timezone.utc).astimezone().isoformat(timespec="minutes"),
        "topstep_rows": int(len(t)),
        "topstep_span": [str(lo), str(hi)],
        "cache_rows_in_span": int(len(db_span)),
        "common_rows": int(len(common)),
        "columns": columns,
        "volume_identical_pct": round(vol_same, 2),
        "labelled_by_open_minute": bool(labelled_by_open),
        "differing_days": differing,
        "rth_sessions_compared": int(len(counts)),
        "rth_count_mismatches": mismatches,
    }


def write_report(symbol: str, cache_path: Path | None = None, directory: Path | None = None,
                 results: Path | None = None) -> Path:
    results = Path(results) if results is not None else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    report = compare(symbol, cache_path, directory)
    out = results / f"data_check_{symbol.lower()}.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true", help="write the agreement report")
    ap.add_argument("--symbol", default="MES", choices=sorted(rules.ALLOWED_INSTRUMENTS))
    ap.add_argument("--cache", type=Path, default=None,
                    help="Databento parquet to compare against (default: the MES cache)")
    ap.add_argument("--dir", type=Path, default=None, help="the clone (default: TOPSTEP_DATA_DIR)")
    args = ap.parse_args(argv)
    if not args.compare:
        ap.print_help()
        return 2
    out = write_report(args.symbol, args.cache, args.dir, RESULTS)
    report = json.loads(out.read_text(encoding="utf-8"))
    c = report["columns"]["close"]
    print(f"{args.symbol}: {report['common_rows']:,} bars in common; close identical "
          f"{c['identical_pct']:.1f}%, within a tick {c['within_tick_pct']:.1f}%; "
          f"{len(report['differing_days'])} day(s) differ by more than a tick; "
          f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
