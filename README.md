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
├── data/              # Raw and processed market data (historical bars/ticks)
├── strategies/         # Strategy signal logic (entries/exits), no execution/risk code
├── backtests/          # Backtest runners, parameter scans, performance reports
└── execution/           # Live/paper execution: order routing, risk enforcement,
                          # position/time guards, broker (Tradovate) integration
```

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
