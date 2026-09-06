# Futures Bot — Intraday CME MES/MNQ Trading System

An intraday systematic trading project for CME Micro E-mini futures (MES, MNQ).
This repo currently contains only scaffolding — no strategy or execution logic
has been written yet. See [CLAUDE.md](CLAUDE.md) for the hard risk/behavior
rules that all future code in this project must follow.

## Project structure

```
futures-bot/
├── CLAUDE.md          # Hard rules: instruments, risk limits, backtest requirements
├── README.md          # This file
├── requirements.txt   # Python dependencies
├── venv/              # Local virtual environment (not committed)
├── data/              # Data pipeline + cached bars
│   ├── loader.py      #   credentials, bar loading, roll/early-close detection
│   ├── fetch.py       #   Databento pull, cost-gated
│   └── validate.py    #   data-quality report
├── strategies/        # Signal logic and the shared rule module
│   ├── rules.py       #   CLAUDE.md hard rules as enforceable functions
│   ├── base.py        #   Strategy interface
│   └── orb.py         #   opening-range breakout
│   ├── london.py      #   London breakout of the overnight range (entry 6)
│   ├── trend.py        #   daily and intraday EMAs, no-lookahead by construction
│   ├── registry.py    #   strategy registry + gated promotion
│   └── registry.yaml  #   where every strategy stands
├── backtests/         # Backtest runners, parameter scans, performance reports
├── bots/              # Discord bots
│   ├── research.py    #   read-only research bot (slash commands)
│   ├── runners.py     #   the work behind them, with no Discord in it
│   ├── desk.py        #   SHADOW desk bot: feed -> guards -> ticket -> Discord
│   ├── feed.py        #   webhook receiver + parquet replay, one Bar type
│   ├── tickets.py     #   the guard pass-through; calls pretrade.evaluate
│   ├── broker.py      #   BrokerAdapter + PaperAdapter (no live adapter)
│   └── desk_state.py  #   crash-safe state, atomically written
├── journal/           # Manual paper-trading discipline layer
├── research/          # hypotheses.md - the pre-registration log
├── execution/         # Live/paper execution: order routing, risk enforcement,
│                      #   position/time guards, broker integration (not built)
├── pine/              # TradingView scripts
│   └── bar_feed.pine  #   posts every closed 1m bar to the desk webhook
├── start_bot.bat      # Launch the research bot in its own console window
├── start_desk.bat     # Launch the desk bot (shadow mode) in its own window
└── tests/             # pytest suite + smoke scripts
```

`strategies/rules.py` is imported by both the backtest and the execution layer,
so a risk rule cannot be enforced in one and skipped in the other. Strategies
themselves return signals only — they have no way to express a position size or
a risk decision.

## Running things

```powershell
venv\Scripts\python.exe -m pytest tests\ -q       # test suite
venv\Scripts\python.exe data\fetch.py --estimate  # cost estimate, no spend
venv\Scripts\python.exe data\validate.py data\<file>.parquet
venv\Scripts\python.exe tests\smoke_orb.py        # eyeball ORB signals
venv\Scripts\python.exe strategies\registry.py   # where every strategy stands
```

### The research bot

`start_bot.bat` in the project root launches the Discord research bot in its own
console window. Double-click it, or run it from a terminal:

```powershell
.\start_bot.bat
```

It changes to the project directory itself, so it works from anywhere, and it
checks that `venv\Scripts\python.exe` and `.env` exist before starting. The
window stays open after the bot exits so a startup error is readable. Ctrl+C in
that window stops it.

The bot reads `DISCORD_TOKEN` from `.env` (gitignored) and syncs its slash
commands to every guild it is in on startup, which is immediate. Commands:

| Command | What it does |
|---|---|
| `/backtest <strategy> <start> <end>` | Metrics embed plus an equity-curve PNG |
| `/walkforward <strategy>` | Per-fold table from the saved run |
| `/evalsim <strategy>` | Pass probability and expected attempts |
| `/hypotheses [n]` | The log: all entries, or one in detail |
| `/status` | The strategy registry |

**The bot is read-only.** Nothing it exposes promotes a strategy, writes to the
journal, or places an order. Promotion goes through `Registry.promote`, which is
deliberately not reachable from chat.

`/walkforward` and `/evalsim` read saved run outputs and report when they were
produced rather than recomputing — the ORB walk-forward takes about 55 minutes,
and a chat command that silently starts an hour of work is a worse answer than
one that says where its numbers came from.

- **`data/`** — scripts and storage for pulling and caching historical MES/MNQ
  intraday data, plus any cleaned/resampled datasets used by backtests.
- **`strategies/`** — pure signal-generation logic: given price/feature data,
  produce entry/exit signals. Strategies do not place orders or manage risk
  directly — that's the execution layer's job.
- **`backtests/`** — vectorbt-based backtest runners and parameter scans that
  apply a strategy's signals to historical data with realistic commission and
  slippage, and produce the required performance report (see CLAUDE.md rule 9).
- **`execution/`** — everything involved in actually trading: risk guards
  (position cap, daily loss limit, min hold time, 4:20 PM ET entry cutoff,
  4:30 PM ET force-flatten), order routing, and eventually the Tradovate API
  integration for live/paper trading.

### The desk bot — SHADOW MODE

`start_desk.bat` launches `bots/desk.py` in its own console window. It consumes
1-minute bars, runs the strategies the registry names, puts every signal
through the same guards a manual trade goes through, and posts the outcome to
Discord.

**It places no orders, and it is not evidence of anything.** The only strategy
it runs is `orb2`, which `research/hypotheses.md` entry 4 **REJECTED**, and
every ticket it posts is labelled `SHADOW - strategy rejected, no orders`. It
exists to test the plumbing before a strategy earns its way to `paper` status —
`docs/HANDOFF.md` §6 gates that, and the gate is not met. `PaperAdapter` is the
only broker adapter, and `broker.require_live_eligible` refuses any strategy
below `live` status whatever the per-account flag says.

```powershell
.\start_desk.bat                                     # live webhook + Discord

venv\Scripts\python.exe bots\desk.py --replay --start 2026-08-24 ^
    --end 2026-08-28 --speed 60                      # replay, 60x real time
venv\Scripts\python.exe bots\desk.py --replay --start 2026-08-10 ^
    --end 2026-08-14 --speed 0                       # as fast as possible
venv\Scripts\python.exe bots\desk.py --no-discord    # stdout only
```

**Live data** comes from a TradingView alert running `pine/bar_feed.pine`,
POSTing JSON to `http://127.0.0.1:8787/bar`. The alert **must** be set to *Once
Per Bar Close* — on *Once Per Bar* it fires on every tick of the forming bar and
the desk would compute signals from a bar whose high and low were still moving.
`GET /health` reports accepted/rejected counts and the last bar seen.

**Replay** feeds cached parquet bars down the identical code path, so it tests
the live path rather than a parallel one. Guards read the **bar's** timestamp,
not the wall clock — otherwise every bar of a historical session would be
blocked as after-hours.

**Journals are separate.** Shadow tickets go to `journal/shadow_trades.jsonl`;
`journal/trades.jsonl` is the manual journal that hypothesis entry 3's 60-trade
gate counts, and nothing the desk does touches it. Both the shadow journal and
`journal/desk_state.json` are gitignored — they are test output, not evidence.

**Ops posts**: a 09:25 pre-market check (data flowing, calendar, roll day), a
heartbeat every 15 minutes *during the session only*, a 16:35 daily summary
read from the journal, and a startup reconciliation that shouts about any open
position the state file cannot explain.

Channel IDs come from `.env` — see `.env.example`. With none set the desk still
runs and prints every post to stdout.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Phased plan

1. **Data** — Build the pipeline to fetch, clean, and store intraday MES/MNQ
   historical data (`data/`). Establish a consistent bar format (e.g. 1-minute
   OHLCV) that later phases build on.

2. **Single strategy** — Implement one well-defined intraday strategy
   (`strategies/`) with clear, deterministic entry/exit rules. Keep it simple
   and testable before considering variations.

3. **Backtest & parameter scan** — Run the strategy through a vectorbt backtest
   (`backtests/`) with commission and slippage modeled, and scan its parameters
   to understand sensitivity. Every run reports max daily loss, worst day as %
   of total profit, average trade duration, and the share of total profit from
   trades held 5 seconds or less, per CLAUDE.md.

4. **Out-of-sample validation** — Hold out a validation period not used during
   parameter selection, and confirm the strategy's edge and risk profile hold
   up on unseen data before moving further.

5. **TradingView paper forward test** — Forward-test the strategy in real time
   on TradingView paper trading to validate behavior against live market
   conditions (latency, real fills, real data feed) without financial risk.

6. **Execution via Tradovate API** — Only after the above phases hold up,
   implement live/paper order execution against the Tradovate API
   (`execution/`), with all CLAUDE.md risk rules enforced in code: 2-contract
   cap, $300 daily loss limit, 30-second minimum hold, 4:20 PM ET entry cutoff,
   and 4:30 PM ET force-flatten.

Each phase should be validated before moving to the next — this is a
sequential gate, not a checklist to rush through.
