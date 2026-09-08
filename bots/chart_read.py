"""`/read` - a description of the chart. Not a signal, and not an opinion.

Everything here is arithmetic on bars plus the guards in ``rules.py``. There is
deliberately **no bias label, no confidence score and no model opinion**: the
command answers "where is price relative to the levels a discretionary trader
would already be watching, and what would the rules permit right now", and
stops there. Anything beyond that would be a strategy, and a strategy needs a
pre-registered entry in ``research/hypotheses.md`` before any code - which is
exactly what this module is not.

Why the disclaimer is a constant rather than a docstring line: it is asserted
by a test and rendered at the top of every embed, so it cannot be dropped by
an edit that forgets it.

Data
----
The desk and the research bot are separate processes, so there is no shared
in-memory buffer to read. The desk appends each completed bar to
``journal/live_bars/<session date>.csv`` (see ``Desk.on_bar``) and this module
merges those over the parquet history:

  * parquet supplies the history the 200-period EMA and the 20-day average
    range need - one session of live bars cannot seed either;
  * the live file supplies today, which parquet never has.

If the live file is missing or stale the read still works from parquet alone
and **says so, with the as-of date**. That matters more than it looks: the
cached parquet ends 2026-08-31, so a read taken without the desk running would
otherwise present a week-old "prior day close" as though it were yesterday's.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date as date_type, time as time_type
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("data", "journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import pretrade  # noqa: E402
import rules  # noqa: E402
import store  # noqa: E402

DISCLAIMER = ("This is a description of the chart, not a signal. "
              "No tested strategy is behind this read.")

LIVE_BARS_DIR = PROJECT_ROOT / "journal" / "live_bars"

#: A live file whose last bar is older than this is treated as stale and the
#: read falls back to parquet. A 1-minute feed should produce a bar a minute;
#: five gives room for a slow alert without accepting yesterday's file.
LIVE_STALE_AFTER = pd.Timedelta(minutes=5)

RTH_START = time_type(9, 30)
RTH_LAST = time_type(15, 59)
OPENING_RANGE_END = time_type(9, 45)
#: The overnight session runs from the Globex reopen to the RTH open.
OVERNIGHT_START = time_type(18, 0)

EMA_PERIODS = (20, 50, 200)
ATR_PERIOD = 14
DAILY_EMA_PERIOD = 50
AVG_RANGE_SESSIONS = 20

#: Minimum clearance between entry and the level a stop is placed behind,
#: as a multiple of ATR. A stop inside half an ATR of entry is inside the
#: instrument's ordinary noise and will be taken out by it.
ATR_BUFFER_MULT = 0.5
#: The stop sits this many ticks beyond the level, not exactly on it.
STOP_PAD_TICKS = 2
#: Below this, the bracket is flagged. Not a veto - the read does not have
#: opinions - but an unflagged 1:1 would be a real omission.
MIN_REWARD_RISK = 1.5

#: Round numbers traders watch, per instrument.
ROUND_INCREMENT = {"MES": 25.0, "MNQ": 100.0}

TIMEFRAMES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}


class ReadError(RuntimeError):
    """Something the user should see as a message, not a traceback."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def parse_timeframe(text: str) -> int:
    key = (text or "5m").strip().lower()
    if key not in TIMEFRAMES:
        raise ReadError(
            f"unknown timeframe `{text}`; use one of "
            f"{', '.join(sorted(TIMEFRAMES, key=lambda k: TIMEFRAMES[k]))}"
        )
    return TIMEFRAMES[key]


def parquet_for(symbol: str) -> Path:
    return (PROJECT_ROOT / "data" /
            f"{symbol.lower()}_v_0_ohlcv_1m_2019-05_2026-08.parquet")


def live_path(symbol: str, day: date_type, directory: Path = LIVE_BARS_DIR) -> Path:
    return Path(directory) / f"{symbol.upper()}_{day}.csv"


def append_live_bar(symbol: str, timestamp, o: float, h: float, l: float,
                    c: float, v: float, directory: Path = LIVE_BARS_DIR) -> Path:
    """Append one completed bar to today's live file. Called by the desk.

    Append-only CSV, one file per session date, so a crash truncates at most
    the final line and a new session never has to rewrite an old file.
    """
    ts = rules.to_et(timestamp)
    path = live_path(symbol, rules.session_date(ts), directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new:
            handle.write("timestamp,open,high,low,close,volume\n")
        handle.write(f"{ts.isoformat()},{o},{h},{l},{c},{v}\n")
    return path


def load_live(symbol: str, day: date_type,
              directory: Path = LIVE_BARS_DIR) -> pd.DataFrame:
    path = live_path(symbol, day, directory)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    # utc=True then convert: a session spanning a DST change would otherwise
    # carry mixed offsets and refuse to parse into one column.
    frame["timestamp"] = (pd.to_datetime(frame["timestamp"], utc=True)
                          .dt.tz_convert(rules.ET))
    return frame.set_index("timestamp").sort_index()


@dataclass(frozen=True)
class BarSource:
    """Where the bars came from, and how current they are."""

    label: str
    live_bars: int
    last_bar: pd.Timestamp | None
    stale: bool
    as_of: date_type | None

    def warning(self, now: pd.Timestamp) -> str | None:
        """The caveat to print, or None when the data is current."""
        if self.as_of is None:
            return None
        today = rules.session_date(now)
        if self.live_bars and not self.stale:
            return None
        gap = (today - self.as_of).days
        if gap <= 0:
            return None
        return (f"desk feed not running - reading cached bars, most recent "
                f"session {self.as_of} ({gap} day(s) old). Levels below "
                f"describe that session, not today.")


def load_bars(symbol: str, now: pd.Timestamp,
              directory: Path = LIVE_BARS_DIR,
              parquet: Path | None = None) -> tuple[pd.DataFrame, BarSource]:
    """Parquet history with today's live bars merged over the top."""
    symbol = rules.require_allowed_instrument(symbol)
    now = rules.to_et(now)
    path = Path(parquet) if parquet else parquet_for(symbol)

    history = pd.DataFrame()
    if path.exists():
        import loader  # noqa: PLC0415

        history = loader.load_bars(path)

    live = load_live(symbol, rules.session_date(now), directory)
    stale = True
    last_live = None
    if not live.empty:
        last_live = live.index[-1]
        stale = (now - last_live) > LIVE_STALE_AFTER

    frames = [f for f in (history, live) if not f.empty]
    if not frames:
        raise ReadError(
            f"no bars for {symbol}: neither {path.name} nor a live file for "
            f"{rules.session_date(now)} exists."
        )
    bars = pd.concat(frames)
    # Live wins on collision - it is the more recent observation of the bar.
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in bars.columns:
            raise ReadError(f"bar data is missing the `{column}` column")

    label = ("desk live feed + parquet history" if (not live.empty and not stale)
             else "parquet cache" if live.empty
             else "parquet cache (live file stale)")
    return bars, BarSource(
        label=label,
        live_bars=0 if live.empty else len(live),
        last_bar=last_live,
        stale=stale,
        as_of=rules.session_date(bars.index[-1]),
    )


def resample(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Left-closed, labelled by opening minute - the repo-wide convention."""
    if minutes <= 1:
        return bars
    out = bars.resample(f"{minutes}min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return out.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(values: pd.Series, period: int) -> float | None:
    """SMA-seeded EMA, matching ``strategies/trend.ema``'s convention.

    Returns None rather than a number when there are fewer than ``period``
    observations. There is no defensible EMA value before the seed, and
    printing one would be inventing a level.
    """
    values = pd.Series(values).dropna()
    if len(values) < period:
        return None
    seed = float(values.iloc[:period].mean())
    alpha = 2.0 / (period + 1)
    out = seed
    for v in values.iloc[period:]:
        out = alpha * float(v) + (1 - alpha) * out
    return float(out)


def atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> float | None:
    """Wilder's ATR. None when there are not enough bars to seed it."""
    if len(bars) < period + 1:
        return None
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    prev_close = bars["close"].astype(float).shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1).dropna()
    if len(true_range) < period:
        return None
    out = float(true_range.iloc[:period].mean())
    for v in true_range.iloc[period:]:
        out = (out * (period - 1) + float(v)) / period
    return out


def session_vwap(bars: pd.DataFrame, day: date_type) -> float | None:
    """Volume-weighted average price over today's RTH bars so far."""
    rth = bars[(bars.index.map(rules.session_date) == day)]
    rth = rth.between_time(RTH_START, RTH_LAST)
    if rth.empty:
        return None
    volume = rth["volume"].astype(float)
    typical = (rth["high"].astype(float) + rth["low"].astype(float)
               + rth["close"].astype(float)) / 3.0
    total = volume.sum()
    if total <= 0:
        return float(typical.mean())
    return float((typical * volume).sum() / total)


def rth_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-session RTH high/low/close, indexed by session date."""
    rth = bars.between_time(RTH_START, RTH_LAST)
    if rth.empty:
        return pd.DataFrame(columns=["high", "low", "close"])
    grouped = rth.groupby(rth.index.map(rules.session_date))
    out = pd.DataFrame({
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
    })
    out.index.name = "session_date"
    return out


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    name: str
    price: float

    def distance(self, spot: float) -> float:
        return self.price - spot

    def side(self, spot: float) -> str:
        return "above" if self.price > spot else "below" if self.price < spot else "at"


def round_numbers(spot: float, symbol: str) -> list[Level]:
    """The nearest round number strictly either side of ``spot``.

    Strictly, because a level at the current price is not a level a bracket
    can use - zero distance is neither a stop nor a target. When price sits
    exactly on a round number (6800.00 with a 25-point step) the naive
    ``floor`` lands on the price itself, so the pair steps out to 6775 / 6825
    rather than silently dropping the level below.
    """
    step = ROUND_INCREMENT.get(symbol.upper(), 25.0)
    below = float(np.floor(spot / step) * step)
    if below >= spot:
        below -= step
    above = below + step
    if above <= spot:
        above += step
    return [Level(f"round {step:g} below", below),
            Level(f"round {step:g} above", above)]


def compute_levels(bars: pd.DataFrame, now: pd.Timestamp,
                   symbol: str = "MES") -> list[Level]:
    """Every structural level the read reports, unsorted.

    Levels are only included when the data actually supports them - an
    opening range before 09:45 is incomplete, and reporting a partial one as
    "the" opening range would be wrong rather than merely early.
    """
    now = rules.to_et(now)
    today = rules.session_date(now)
    spot = float(bars["close"].iloc[-1])
    levels: list[Level] = []

    sessions = rth_sessions(bars)
    prior = sessions[sessions.index < today]
    if not prior.empty:
        last = prior.iloc[-1]
        label = prior.index[-1]
        levels += [
            Level(f"prior day high ({label})", float(last["high"])),
            Level(f"prior day low ({label})", float(last["low"])),
            Level(f"prior day close ({label})", float(last["close"])),
        ]

    # Overnight: the Globex reopen on the previous calendar evening through to
    # the RTH open. Anchored on the previous *session* in the data, so a
    # Monday read spans the Sunday reopen rather than Friday evening.
    overnight = bars[(bars.index < pd.Timestamp.combine(today, RTH_START)
                      .tz_localize(rules.ET))]
    if not prior.empty:
        start = (pd.Timestamp.combine(prior.index[-1], OVERNIGHT_START)
                 .tz_localize(rules.ET))
        overnight = overnight[overnight.index >= start]
    if not overnight.empty:
        levels += [
            Level("overnight high", float(overnight["high"].max())),
            Level("overnight low", float(overnight["low"].min())),
        ]

    today_bars = bars[bars.index.map(rules.session_date) == today]
    opening = today_bars.between_time(RTH_START, "09:44")
    if not opening.empty and now.time() >= OPENING_RANGE_END:
        levels += [
            Level("opening range high", float(opening["high"].max())),
            Level("opening range low", float(opening["low"].min())),
        ]

    vwap = session_vwap(bars, today)
    if vwap is not None:
        levels.append(Level("session VWAP", vwap))

    levels += round_numbers(spot, symbol)
    return levels


def split_levels(levels: list[Level], spot: float) -> tuple[list[Level], list[Level]]:
    """(above, below), each sorted by distance from ``spot``."""
    above = sorted([l for l in levels if l.price > spot], key=lambda l: l.price)
    below = sorted([l for l in levels if l.price < spot],
                   key=lambda l: l.price, reverse=True)
    return above, below


# ---------------------------------------------------------------------------
# The bracket
# ---------------------------------------------------------------------------

@dataclass
class Bracket:
    """What the rules would permit for one direction, or why they would not."""

    direction: str
    allowed: bool = False
    blocks: list[str] = field(default_factory=list)
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    stop_level: str = ""
    target_level: str = ""
    contracts: int = 0
    risk_dollars: float = 0.0
    reward_risk: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def thin(self) -> bool:
        return self.reward_risk is not None and self.reward_risk < MIN_REWARD_RISK


def size_for(instrument: str, entry: float, stop: float, budget: float) -> int:
    """Largest contract count whose all-in risk fits ``budget`` and the cap.

    ``store.risk_dollars`` includes two ticks of slippage and the round-turn
    commission, so this is sized against what the stop actually costs rather
    than the bare stop distance. Returns 0 when even one contract does not
    fit - which is a real answer, not an error.
    """
    for n in range(rules.POSITION_CAP, 0, -1):
        if store.risk_dollars(instrument, entry, stop, n) <= budget:
            return n
    return 0


def build_bracket(direction: str, spot: float, levels: list[Level],
                  atr_value: float | None, symbol: str,
                  state: pretrade.AccountState, now: pd.Timestamp,
                  early_close_dates=(), roll_dates=(), closed_dates=()) -> Bracket:
    """Entry at spot, stop behind structure, target at the opposing level.

    The guards are not re-implemented here: the bracket is handed to
    ``pretrade.evaluate`` - the same function the manual journal and the desk
    use - and its blocks are reported verbatim. That is what makes "the rules
    would block this right now" mean the same thing everywhere.
    """
    bracket = Bracket(direction=direction, entry=float(spot))
    spec = store.spec_for(symbol)
    above, below = split_levels(levels, spot)
    long = direction == "long"

    if atr_value is None:
        bracket.blocks.append(
            f"not enough bars for an {ATR_PERIOD}-period ATR, so no stop "
            f"distance can be justified"
        )
        return bracket

    buffer = ATR_BUFFER_MULT * atr_value
    pad = STOP_PAD_TICKS * spec.tick_size
    stop_side, target_side = (below, above) if long else (above, below)

    candidates = [l for l in stop_side if abs(spot - l.price) >= buffer]
    if candidates:
        level = candidates[0]
        bracket.stop = level.price - pad if long else level.price + pad
        bracket.stop_level = level.name
    else:
        # Every structural level is inside the noise band. Falling back to a
        # pure ATR stop is stated rather than silently substituted.
        bracket.stop = spot - buffer if long else spot + buffer
        bracket.stop_level = f"no level beyond {ATR_BUFFER_MULT:g} ATR - ATR stop"
        bracket.notes.append(
            f"nearest {'support' if long else 'resistance'} is inside "
            f"{ATR_BUFFER_MULT:g} ATR ({buffer:,.2f} pts); stop is the ATR "
            f"buffer, not structure"
        )

    if target_side:
        bracket.target = target_side[0].price
        bracket.target_level = target_side[0].name
    else:
        bracket.blocks.append("no opposing level above/below price to target")
        return bracket

    risk_points = abs(spot - bracket.stop)
    reward_points = abs(bracket.target - spot)
    if risk_points <= 0:
        bracket.blocks.append("stop resolves to entry; no risk distance")
        return bracket
    # Reward:risk on raw points, the conventional reading. The dollar risk
    # below is all-in (slippage + commission), so the two are not the same
    # ratio and are deliberately reported separately.
    bracket.reward_risk = reward_points / risk_points

    budget = min(
        rules.DAILY_LOSS_LIMIT + state.today_realised_pnl,
        rules.INTERNAL.trailing_drawdown_stop
        - rules.drawdown_from_peak(state.balance, state.peak_eod_balance),
    )
    bracket.contracts = size_for(symbol, spot, bracket.stop, budget)
    if bracket.contracts == 0:
        bracket.blocks.append(
            f"one contract risks "
            f"${store.risk_dollars(symbol, spot, bracket.stop, 1):,.2f}, more "
            f"than the ${budget:,.2f} left in today's budget"
        )
        return bracket
    bracket.risk_dollars = store.risk_dollars(
        symbol, spot, bracket.stop, bracket.contracts)

    decision = pretrade.evaluate(
        pretrade.TicketRequest(
            instrument=symbol, direction=direction, entry_price=spot,
            stop_price=bracket.stop, contracts=bracket.contracts,
            thesis="(read preview)",
        ),
        state, now, early_close_dates, roll_dates, closed_dates,
    )
    bracket.blocks = list(decision.blocks)
    bracket.allowed = decision.allowed
    if bracket.thin:
        bracket.notes.append(
            f"reward:risk {bracket.reward_risk:.2f} is below "
            f"{MIN_REWARD_RISK:g}"
        )
    return bracket


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------

@dataclass
class ChartRead:
    """Everything `/read` reports. Rendering is separate; this holds no text."""

    symbol: str
    timeframe: str
    now: pd.Timestamp
    spot: float
    source: BarSource
    source_warning: str | None
    emas: dict
    daily_ema: float | None
    prior_close: float | None
    atr: float | None
    today_range: float | None
    avg_range: float | None
    levels: list
    state: pretrade.AccountState
    session: dict
    long: Bracket
    short: Bracket
    #: The calendar this read was built against. Carried so that logging a
    #: bracket re-evaluates against the same roll/early-close/holiday sets the
    #: read used, rather than a freshly-derived and possibly different one.
    calendar: dict = field(default_factory=dict)

    @property
    def ema_stack(self) -> str:
        """The 20/50/200 ordering, stated as an ordering and nothing more.

        Deliberately not translated into "bullish" or "bearish". The stack is
        a fact about three numbers; what it implies is the operator's call,
        and a label here would be the opinion this command does not offer.
        """
        have = [(p, self.emas.get(p)) for p in EMA_PERIODS]
        if any(v is None for _, v in have):
            got = [str(p) for p, v in have if v is not None]
            return f"incomplete - only {', '.join(got) or 'none'} seeded"
        values = [v for _, v in have]
        if values[0] > values[1] > values[2]:
            return "20 > 50 > 200"
        if values[0] < values[1] < values[2]:
            return "20 < 50 < 200"
        return "mixed (not stacked)"


def session_snapshot(now: pd.Timestamp, state: pretrade.AccountState,
                     early_close_dates=(), roll_dates=(),
                     closed_dates=()) -> dict:
    """Clock, calendar and budget. Every threshold read from rules.py."""
    now = rules.to_et(now)
    day = rules.session_date(now)
    entry_cut = rules.entry_deadline(now, early_close_dates)
    flatten = rules.flatten_deadline(now, early_close_dates)

    def _until(target: time_type) -> str:
        deadline = pd.Timestamp.combine(day, target).tz_localize(rules.ET)
        delta = deadline - now
        if delta.total_seconds() <= 0:
            return "passed"
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        return f"{hours}h {rem // 60:02d}m" if hours else f"{rem // 60}m"

    budget = rules.DAILY_LOSS_LIMIT + state.today_realised_pnl
    return {
        "day": day,
        "entry_cutoff": entry_cut,
        "to_entry_cutoff": _until(entry_cut),
        "flatten": flatten,
        "to_flatten": _until(flatten),
        "roll_day": rules.is_roll_day(day, roll_dates),
        "early_close": day in set(early_close_dates),
        "holiday": day in set(closed_dates),
        "realised_pnl": state.today_realised_pnl,
        "budget_left": max(0.0, budget),
        "open_position": state.open_position,
    }


def read(symbol: str = "MES", timeframe: str = "5m",
         now: pd.Timestamp | None = None,
         bars: pd.DataFrame | None = None,
         source: BarSource | None = None,
         journal_path: Path | None = None,
         directory: Path = LIVE_BARS_DIR,
         early_close_dates=(), roll_dates=(), closed_dates=()) -> ChartRead:
    """Build the whole read. Pure apart from reading bars and the journal."""
    symbol = rules.require_allowed_instrument(symbol)
    minutes = parse_timeframe(timeframe)
    now = rules.to_et(now or pd.Timestamp.now(tz=rules.ET))

    if bars is None:
        bars, source = load_bars(symbol, now, directory)
    if source is None:
        source = BarSource("supplied", 0, None, True,
                           rules.session_date(bars.index[-1]))

    # Never let a bar from the future into the read - a replay, or a clock
    # skew, would otherwise price the bracket off a bar that has not happened.
    bars = bars[bars.index <= now]
    if bars.empty:
        raise ReadError(f"no {symbol} bars at or before {now:%Y-%m-%d %H:%M}")

    frame = resample(bars, minutes)
    if frame.empty:
        raise ReadError(f"no {timeframe} bars for {symbol}")
    spot = float(frame["close"].iloc[-1])
    day = rules.session_date(now)
    atr_value = atr(frame)

    emas = {p: ema(frame["close"], p) for p in EMA_PERIODS}
    sessions = rth_sessions(bars)
    prior = sessions[sessions.index < day]
    prior_close = float(prior["close"].iloc[-1]) if not prior.empty else None
    daily_ema = (ema(prior["close"], DAILY_EMA_PERIOD)
                 if len(prior) >= DAILY_EMA_PERIOD else None)

    today_rows = sessions[sessions.index == day]
    today_range = (float(today_rows["high"].iloc[0] - today_rows["low"].iloc[0])
                   if not today_rows.empty else None)
    recent = prior.tail(AVG_RANGE_SESSIONS)
    avg_range = (float((recent["high"] - recent["low"]).mean())
                 if not recent.empty else None)

    journal_path = Path(journal_path) if journal_path else store.TRADES_PATH
    state = pretrade.AccountState.from_journal(day, journal_path)

    levels = compute_levels(bars, now, symbol)
    common = dict(early_close_dates=early_close_dates, roll_dates=roll_dates,
                  closed_dates=closed_dates)
    return ChartRead(
        symbol=symbol, timeframe=timeframe, now=now, spot=spot, source=source,
        source_warning=source.warning(now), emas=emas, daily_ema=daily_ema,
        prior_close=prior_close, atr=atr_value, today_range=today_range,
        avg_range=avg_range, levels=levels, state=state,
        session=session_snapshot(now, state, **common),
        long=build_bracket("long", spot, levels, atr_value, symbol, state,
                           now, **common),
        short=build_bracket("short", spot, levels, atr_value, symbol, state,
                            now, **common),
        calendar=common,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pts(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def format_levels(r: ChartRead) -> str:
    above, below = split_levels(r.levels, r.spot)
    lines = []
    for level in reversed(above):
        lines.append(f"  {level.price:>10,.2f}  +{level.price - r.spot:>7,.2f}  "
                     f"{level.name}")
    lines.append(f"> {r.spot:>10,.2f}  {'spot':>8}  <- current price")
    for level in below:
        lines.append(f"  {level.price:>10,.2f}  {level.price - r.spot:>8,.2f}  "
                     f"{level.name}")
    return "```\n" + "\n".join(lines) + "\n```"


def format_bracket(b: Bracket, symbol: str) -> str:
    if b.blocks:
        return ("**Rules would block this entry**\n"
                + "\n".join(f"- {reason}" for reason in b.blocks))
    lines = [
        f"entry `{b.entry:,.2f}`  stop `{b.stop:,.2f}`  target `{b.target:,.2f}`",
        f"stop behind: {b.stop_level}",
        f"target at: {b.target_level}",
        f"size **{b.contracts}** {symbol}   risk **${b.risk_dollars:,.2f}**   "
        f"R:R **{b.reward_risk:.2f}**",
    ]
    if b.thin:
        lines.append(f"**R:R is below {MIN_REWARD_RISK:g}**")
    lines += [f"_{n}_" for n in b.notes if "reward:risk" not in n]
    return "\n".join(lines)


def format_read(r: ChartRead) -> dict:
    """Embed fields. The disclaimer is added by the caller, always first."""
    s = r.session
    trend = [
        f"EMA stack ({r.timeframe}): **{r.ema_stack}**",
        "  " + "   ".join(f"{p}: {_pts(r.emas.get(p))}" for p in EMA_PERIODS),
    ]
    if r.prior_close is not None:
        trend.append(f"vs prior close {r.prior_close:,.2f}: "
                     f"**{r.spot - r.prior_close:+,.2f}**")
    if r.daily_ema is not None:
        side = "above" if r.spot > r.daily_ema else "below"
        trend.append(f"daily {DAILY_EMA_PERIOD}-EMA {r.daily_ema:,.2f}: "
                     f"price **{side}**")

    vol = [f"ATR({ATR_PERIOD}) on {r.timeframe}: **{_pts(r.atr)}** pts"]
    if r.today_range is not None and r.avg_range:
        vol.append(f"today's range {r.today_range:,.2f} vs "
                   f"{AVG_RANGE_SESSIONS}-day avg {r.avg_range:,.2f} "
                   f"(**{100.0 * r.today_range / r.avg_range:.0f}%**)")

    flags = []
    if s["holiday"]:
        flags.append("**exchange holiday**")
    if s["roll_day"]:
        flags.append("**ROLL DAY - no entries**")
    if s["early_close"]:
        flags.append("**early close**")
    session_lines = [
        f"entry cutoff {s['entry_cutoff']:%H:%M} (in **{s['to_entry_cutoff']}**)",
        f"flatten {s['flatten']:%H:%M} (in **{s['to_flatten']}**)",
        f"today realised **${s['realised_pnl']:,.2f}**   "
        f"budget left **${s['budget_left']:,.2f}**",
        f"open position **{s['open_position']:+d}**",
    ]
    if flags:
        session_lines.insert(0, "  ".join(flags))

    return {
        f"{r.symbol} {r.timeframe}  @  {r.spot:,.2f}":
            f"as of {r.now:%Y-%m-%d %H:%M:%S %Z}  -  {r.source.label}"
            + (f"\n**{r.source_warning}**" if r.source_warning else ""),
        "Trend context": "\n".join(trend),
        "Levels": format_levels(r),
        "Volatility": "\n".join(vol),
        "Session": "\n".join(session_lines),
        "If you trade this - LONG": format_bracket(r.long, r.symbol),
        "If you trade this - SHORT": format_bracket(r.short, r.symbol),
    }
