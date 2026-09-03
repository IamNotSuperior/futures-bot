"""Render one ORB trade as an annotated candlestick chart.

    python backtests/plot_trade.py --date 2026-08-25

Everything drawn comes from the strategy itself: the session's bars are
resampled and passed through :class:`OpeningRangeBreakout`, and the opening
range, entry bar, stop, target, exit bar and P&L are read off its output and
the engine's pricing. Nothing is typed in by hand.

That is the point of the chart. A picture drawn from numbers a human retyped
verifies the human's arithmetic; a picture drawn from the strategy's own output
verifies the strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in ("data", "strategies", "backtests"):
    sys.path.insert(0, str(PROJECT_ROOT / folder))

import loader  # noqa: E402
import rules  # noqa: E402
from engine import MES, CostModel, build_trades, price_trades  # noqa: E402
from orb import ORBParams, OpeningRangeBreakout, resample_bars  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "mes_v_0_ohlcv_1m_2019-05_2026-08.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"

UP = "#1b7f5a"
DOWN = "#c0392b"
OR_LINE = "#1f5f8b"
STOP_C = "#c0392b"
TARGET_C = "#1b7f5a"
ENTRY_C = "#d97706"
EXIT_C = "#5b21b6"


def session_context(date_str: str, params: ORBParams):
    """Bars, signals and the priced trade for one session, from the strategy."""
    bars = loader.load_bars(PARQUET)
    roll_dates = loader.detect_roll_dates(bars)
    early = loader.detect_early_close_dates(bars)

    day = pd.Timestamp(date_str).date()
    lo = pd.Timestamp(day, tz=rules.ET)
    hi = lo + pd.Timedelta(days=1)
    day_bars = bars[(bars.index >= lo) & (bars.index < hi)]
    if day_bars.empty:
        raise SystemExit(f"No bars for {date_str}")

    bars5 = resample_bars(day_bars, params.bar_minutes)
    strat = OpeningRangeBreakout(params, roll_dates=roll_dates,
                                 early_close_dates=early)
    # Sessions are independent, so generating on this session alone matches
    # generating over the whole history and slicing (asserted in test_scan.py).
    signals = strat.generate_signals_resampled(bars5)
    trades = price_trades(build_trades(signals, bars5), MES, CostModel(), 1)
    return bars5, signals, trades


def draw_candles(ax, bars: pd.DataFrame) -> None:
    """Plain OHLC candles on an integer x-axis, so gaps do not stretch time."""
    from matplotlib.patches import Rectangle

    for i in range(len(bars)):
        o = float(bars["open"].iloc[i])
        h = float(bars["high"].iloc[i])
        low = float(bars["low"].iloc[i])
        c = float(bars["close"].iloc[i])
        colour = UP if c >= o else DOWN
        ax.vlines(i, low, h, color=colour, linewidth=0.9, zorder=2)
        height = abs(c - o)
        bottom = min(o, c)
        if height < 1e-9:  # a doji still needs to be visible
            ax.hlines(o, i - 0.32, i + 0.32, color=colour, linewidth=1.4, zorder=3)
        else:
            ax.add_patch(Rectangle(
                (i - 0.32, bottom), 0.64, height,
                facecolor=colour, edgecolor=colour, linewidth=0.6, zorder=3,
            ))


def plot_trade(date_str: str, params: ORBParams, out_path: Path,
               pad_bars: int = 2) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    # Every label here contains dollar amounts, and matplotlib treats paired $
    # as mathtext - it silently italicised "commission -2.50 slippage" on the
    # first render. Turning parsing off is safer than escaping each label.
    plt.rcParams["text.parse_math"] = False

    bars5, signals, trades = session_context(date_str, params)
    if trades.empty:
        raise SystemExit(f"The strategy took no trade on {date_str}")
    trade = trades.iloc[0]

    entry_ts = trade["entry_time"]
    exit_ts = trade["exit_time"]
    direction = trade["direction"]
    long = direction == "long"

    entry_i = int(bars5.index.get_indexer([entry_ts])[0])
    exit_i = int(bars5.index.get_indexer([exit_ts])[0])

    # 09:30 through the exit bar, plus a little context on the right.
    last_i = min(len(bars5) - 1, exit_i + pad_bars)
    view = bars5.iloc[: last_i + 1]

    sig_row = signals.loc[entry_ts]
    stop = float(sig_row["stop_price"])
    target = float(sig_row["target_price"])
    or_height = float(sig_row["opening_range_height"])

    # The opening range as the strategy computed it.
    opening_end = (
        pd.Timestamp(f"{bars5.index[0].date()} 09:30")
        + pd.Timedelta(minutes=params.opening_range_minutes)
    ).time()
    opening = view[view.index.time < opening_end]
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    or_last_i = len(opening) - 1

    fig = plt.figure(figsize=(max(13.0, len(view) * 0.22), 8.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.28)
    ax = fig.add_subplot(grid[0])
    note = fig.add_subplot(grid[1])
    note.axis("off")

    draw_candles(ax, view)

    # --- opening range -----------------------------------------------------
    ax.add_patch(Rectangle(
        (-0.5, or_low), or_last_i + 1.0, or_high - or_low,
        facecolor=OR_LINE, alpha=0.13, edgecolor="none", zorder=1,
    ))
    ax.hlines([or_high, or_low], -0.5, or_last_i + 0.5,
              color=OR_LINE, linewidth=2.0, zorder=4)
    # Extended faintly, because the breakout is measured against these all day.
    ax.hlines([or_high, or_low], or_last_i + 0.5, len(view) - 0.5,
              color=OR_LINE, linewidth=1.0, linestyle=":", alpha=0.65, zorder=4)
    # Labels live in the right margin with the stop and target labels; inside
    # the plot they land on candles.
    ax.text(len(view) - 0.6, or_high, f" OR high {or_high:,.2f}",
            va="center", fontsize=9, color=OR_LINE, fontweight="bold")
    ax.text(len(view) - 0.6, or_low, f" OR low {or_low:,.2f}",
            va="center", fontsize=9, color=OR_LINE, fontweight="bold")
    ax.text(-0.4, or_high + (or_high - or_low) * 0.16,
            f"opening range 09:30-{opening_end.strftime('%H:%M')}",
            ha="left", fontsize=9, color=OR_LINE)

    # --- stop and target ---------------------------------------------------
    ax.hlines(stop, entry_i - 0.5, len(view) - 0.5, color=STOP_C,
              linewidth=1.6, linestyle="--", zorder=4)
    ax.hlines(target, entry_i - 0.5, len(view) - 0.5, color=TARGET_C,
              linewidth=1.6, linestyle="--", zorder=4)
    ax.text(len(view) - 0.6, stop, f" stop {stop:,.2f}", va="center",
            fontsize=9, color=STOP_C, fontweight="bold")
    ax.text(len(view) - 0.6, target, f" target {target:,.2f}", va="center",
            fontsize=9, color=TARGET_C, fontweight="bold")

    # --- entry -------------------------------------------------------------
    entry_price = float(trade["entry_price"])
    span = float(view["high"].max() - view["low"].min())
    offset = span * 0.13
    arrow_from = entry_price - offset if long else entry_price + offset
    ax.annotate(
        f"ENTRY {direction.upper()}\n{entry_price:,.2f}",
        xy=(entry_i, entry_price), xytext=(entry_i, arrow_from),
        ha="center", va="top" if long else "bottom", fontsize=9.5,
        fontweight="bold", color=ENTRY_C,
        arrowprops=dict(arrowstyle="-|>", color=ENTRY_C, linewidth=2.0),
        zorder=6,
    )
    ax.axvline(entry_i, color=ENTRY_C, alpha=0.28, linewidth=1.1, zorder=1)

    # --- exit --------------------------------------------------------------
    exit_price = float(trade["exit_price"])
    ax.plot([exit_i], [exit_price], marker="X", markersize=13, color=EXIT_C,
            zorder=7, markeredgecolor="white", markeredgewidth=1.0)
    exit_off = offset if long else -offset
    ax.annotate(
        f"EXIT {trade['exit_reason']}\n{exit_price:,.2f}",
        xy=(exit_i, exit_price), xytext=(exit_i, exit_price + exit_off),
        ha="center", va="bottom" if long else "top", fontsize=9.5,
        fontweight="bold", color=EXIT_C,
        arrowprops=dict(arrowstyle="-|>", color=EXIT_C, linewidth=2.0),
        zorder=6,
    )
    ax.axvline(exit_i, color=EXIT_C, alpha=0.28, linewidth=1.1, zorder=1)
    ax.axvspan(entry_i, exit_i, color="#94a3b8", alpha=0.10, zorder=0)

    # --- axes --------------------------------------------------------------
    step = 3 if len(view) <= 40 else 6
    ticks = list(range(0, len(view), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([view.index[i].strftime("%H:%M") for i in ticks],
                       rotation=0, fontsize=8.5)
    ax.set_xlim(-1.0, len(view) + 3.5)
    # Stop and target must be inside the view even when price never approached
    # them - an off-axes target line is exactly the thing being verified.
    lo_y = min(float(view["low"].min()), stop, target)
    hi_y = max(float(view["high"].max()), stop, target)
    pad = (hi_y - lo_y) * 0.13
    ax.set_ylim(lo_y - pad * 1.5, hi_y + pad)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("MES price")
    ax.set_title(
        (f"ORB  {direction.upper()}  MES  {bars5.index[0].date()}   "
         f"net ${trade['net_pnl']:,.2f}"),
        fontsize=13, fontweight="bold", loc="left", pad=34,
    )
    # Placed in a strip above the axes: inside the plot it covers the opening
    # range on a short and the target on a long.
    ax.legend(
        handles=[
            Line2D([], [], color=OR_LINE, lw=2, label="opening range high/low"),
            Line2D([], [], color=STOP_C, lw=1.6, ls="--", label="stop"),
            Line2D([], [], color=TARGET_C, lw=1.6, ls="--", label="target"),
            Line2D([], [], color=ENTRY_C, lw=2, label="entry bar"),
            Line2D([], [], color=EXIT_C, lw=0, marker="X", ms=9, label="exit"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.005, 1.0, 0.06), mode="expand",
        ncol=5, frameon=False, fontsize=8.8, borderaxespad=0.0,
    )

    # --- derivation panel --------------------------------------------------
    sign = "-" if long else "+"
    inv = "+" if long else "-"
    stop_dist = or_height * params.stop_multiple
    target_dist = stop_dist * params.target_multiple
    held = int(trade["duration_seconds"] // 60)

    lines = [
        f"Opening range   high {or_high:,.2f}  -  low {or_low:,.2f}"
        f"   =  height {or_height:,.2f} pts",
        f"Stop            entry {entry_price:,.2f} {sign} "
        f"({or_height:,.2f} x {params.stop_multiple:g} OR)  =  {stop:,.2f}"
        f"      [{stop_dist:,.2f} pts risk]",
        f"Target          entry {entry_price:,.2f} {inv} "
        f"({stop_dist:,.2f} x {params.target_multiple:g} stop)  =  {target:,.2f}"
        f"   [{target_dist:,.2f} pts reward]",
        f"Trigger         first {params.bar_minutes}m close beyond the range; "
        f"filled at the NEXT bar's open, so no lookahead",
        f"Outcome         {trade['exit_reason']} at {exit_price:,.2f} after "
        f"{held} min   |   gross ${trade['gross_pnl']:,.2f}  "
        f"commission -${trade['commission']:,.2f}  "
        f"slippage -${trade['slippage_cost']:,.2f}  "
        f"=  net ${trade['net_pnl']:,.2f}",
    ]
    note.text(0.005, 0.94, "\n".join(lines), va="top", ha="left",
              fontsize=9.6, family="DejaVu Sans Mono", linespacing=1.75)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--opening-range", type=int, default=15)
    ap.add_argument("--stop-multiple", type=float, default=1.0)
    ap.add_argument("--target-multiple", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    params = ORBParams(
        opening_range_minutes=args.opening_range,
        stop_multiple=args.stop_multiple,
        target_multiple=args.target_multiple,
    )
    out = Path(args.out) if args.out else RESULTS_DIR / f"orb_trade_{args.date}.png"
    path = plot_trade(args.date, params, out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
