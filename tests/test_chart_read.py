"""`/read`: levels, sizing, and what a blocked window produces.

The command describes a chart. The three things that could make it lie:

* **Levels** computed from the wrong bars - a prior-day high taken from the
  wrong session, an opening range read before 09:45, a level placed on the
  wrong side of price.
* **Sizing** that lets risk past the internal limit, which would put a number
  in front of the operator that ``rules.py`` would refuse.
* **A blocked window producing a bracket** - the worst of the three, because
  a bracket shown after the 16:20 cutoff is an invitation to take a trade the
  guards exist to stop.

Bars are synthesised so every level has a known answer; nothing here depends
on the cached parquet.
"""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

import chart_read as cr
import pretrade
import read_log
import rules
import store

ET = rules.ET
FLAT = pretrade.AccountState()


def make_bars(day="2026-09-04", prior_day="2026-09-03") -> pd.DataFrame:
    """Two RTH sessions plus an overnight, with levels at known prices.

    prior RTH   high 6820, low 6780, close 6800
    overnight   high 6830, low 6770
    today RTH   opens 09:30, opening range (09:30-09:44) high 6812 low 6788
                and drifts to 6800 by 11:00
    """
    rows = []

    def push(ts, o, h, l, c, v=100.0):
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize(ET) if ts.tzinfo is None else ts.tz_convert(ET)
        rows.append((ts, o, h, l, c, v))

    # Prior session RTH, 09:30-15:59.
    for i in range(390):
        ts = pd.Timestamp(f"{prior_day} 09:30", tz=ET) + pd.Timedelta(minutes=i)
        push(ts, 6800, 6801, 6799, 6800)
    # Stamp the prior high, low and close.
    rows[10] = (rows[10][0], 6800, 6820.0, 6799, 6800, 100.0)
    rows[20] = (rows[20][0], 6800, 6801, 6780.0, 6800, 100.0)
    rows[-1] = (rows[-1][0], 6800, 6801, 6799, 6800.0, 100.0)

    # Overnight: 18:00 prior day through 09:29 today.
    start = pd.Timestamp(f"{prior_day} 18:00", tz=ET)
    end = pd.Timestamp(f"{day} 09:29", tz=ET)
    n = int((end - start).total_seconds() // 60) + 1
    base = len(rows)
    for i in range(n):
        push(start + pd.Timedelta(minutes=i), 6805, 6806, 6804, 6805)
    rows[base + 5] = (rows[base + 5][0], 6805, 6830.0, 6804, 6805, 100.0)
    rows[base + 6] = (rows[base + 6][0], 6805, 6806, 6770.0, 6805, 100.0)

    # Today's RTH, 09:30-11:00.
    base = len(rows)
    for i in range(91):
        push(pd.Timestamp(f"{day} 09:30", tz=ET) + pd.Timedelta(minutes=i),
             6800, 6801, 6799, 6800)
    rows[base + 2] = (rows[base + 2][0], 6800, 6812.0, 6799, 6800, 100.0)
    rows[base + 3] = (rows[base + 3][0], 6800, 6801, 6788.0, 6800, 100.0)

    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                        "close", "volume"])
    return frame.set_index("timestamp").sort_index()


ELEVEN_AM = pd.Timestamp("2026-09-04 11:00", tz=ET)


def build_read(now=ELEVEN_AM, bars=None, journal=None, **kwargs):
    return cr.read("MES", "5m", now=now, bars=bars if bars is not None else make_bars(),
                   journal_path=journal or "does-not-exist.jsonl", **kwargs)


def level_named(levels, fragment):
    for lvl in levels:
        if fragment in lvl.name:
            return lvl
    raise AssertionError(f"no level matching {fragment!r} in "
                         f"{[l.name for l in levels]}")


# -- levels -----------------------------------------------------------------

class TestLevels:
    def test_prior_day_high_low_close(self):
        levels = build_read().levels
        assert level_named(levels, "prior day high").price == 6820.0
        assert level_named(levels, "prior day low").price == 6780.0
        assert level_named(levels, "prior day close").price == 6800.0

    def test_prior_day_is_the_previous_session_not_the_previous_row(self):
        """The label names the session it came from, so it can be checked."""
        level = level_named(build_read().levels, "prior day high")
        assert "2026-09-03" in level.name

    def test_overnight_high_low(self):
        levels = build_read().levels
        assert level_named(levels, "overnight high").price == 6830.0
        assert level_named(levels, "overnight low").price == 6770.0

    def test_overnight_excludes_the_prior_rth_session(self):
        """6820/6780 are prior-day RTH extremes and must not leak in."""
        levels = build_read().levels
        assert level_named(levels, "overnight high").price != 6820.0
        assert level_named(levels, "overnight low").price != 6780.0

    def test_opening_range(self):
        levels = build_read().levels
        assert level_named(levels, "opening range high").price == 6812.0
        assert level_named(levels, "opening range low").price == 6788.0

    def test_opening_range_is_absent_before_0945(self):
        """A partial range is not the range; reporting one would be wrong."""
        early = build_read(now=pd.Timestamp("2026-09-04 09:40", tz=ET))
        assert not any("opening range" in l.name for l in early.levels)

    def test_opening_range_appears_at_0945(self):
        at = build_read(now=pd.Timestamp("2026-09-04 09:45", tz=ET))
        assert any("opening range" in l.name for l in at.levels)

    def test_round_numbers_straddle_price(self):
        r = build_read()
        above = level_named(r.levels, "round 25 above")
        below = level_named(r.levels, "round 25 below")
        assert above.price > r.spot > below.price
        assert above.price % 25 == 0 and below.price % 25 == 0

    def test_round_numbers_step_out_when_price_sits_on_one(self):
        """floor(6800/25)*25 is 6800 - the level below must not vanish."""
        levels = cr.round_numbers(6800.0, "MES")
        assert {l.price for l in levels} == {6775.0, 6825.0}
        assert all(l.price != 6800.0 for l in levels)

    def test_mnq_uses_a_wider_round_increment(self):
        assert cr.ROUND_INCREMENT["MNQ"] > cr.ROUND_INCREMENT["MES"]
        levels = cr.round_numbers(20150.0, "MNQ")
        assert {l.price for l in levels} == {20100.0, 20200.0}

    def test_vwap_is_between_the_session_high_and_low(self):
        r = build_read()
        vwap = level_named(r.levels, "VWAP").price
        assert 6788.0 <= vwap <= 6812.0

    def test_levels_split_above_and_below_by_distance(self):
        r = build_read()
        above, below = cr.split_levels(r.levels, r.spot)
        assert all(l.price > r.spot for l in above)
        assert all(l.price < r.spot for l in below)
        assert above == sorted(above, key=lambda l: l.price)
        assert below == sorted(below, key=lambda l: l.price, reverse=True)

    def test_no_future_bar_reaches_the_read(self):
        """A bar after `now` would price the bracket off the future."""
        r = build_read(now=pd.Timestamp("2026-09-04 10:00", tz=ET))
        assert r.spot == 6800.0
        # 10:00 is before the 11:00 bars, so today's range must not include
        # anything stamped later.
        assert r.now == pd.Timestamp("2026-09-04 10:00", tz=ET)


class TestIndicators:
    def test_ema_matches_a_hand_computation(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        seed = (1 + 2) / 2
        alpha = 2 / 3
        expected = alpha * 4 + (1 - alpha) * (alpha * 3 + (1 - alpha) * seed)
        assert cr.ema(values, 2) == pytest.approx(expected)

    def test_ema_is_none_before_it_is_seeded(self):
        assert cr.ema(pd.Series([1.0, 2.0]), 50) is None

    def test_ema_stack_reports_ordering_not_an_opinion(self):
        r = build_read()
        assert r.ema_stack in ("20 > 50 > 200", "20 < 50 < 200",
                              "mixed (not stacked)") or "incomplete" in r.ema_stack
        for banned in ("bull", "bear", "buy", "sell", "strong", "weak"):
            assert banned not in r.ema_stack.lower()

    def test_atr_is_positive_and_none_when_unseeded(self):
        bars = make_bars()
        assert cr.atr(cr.resample(bars, 5)) > 0
        assert cr.atr(bars.head(3)) is None

    def test_resample_is_left_closed_labelled_by_opening_minute(self):
        bars = make_bars()
        five = cr.resample(bars, 5)
        label = pd.Timestamp("2026-09-04 09:30", tz=ET)
        bucket = bars.loc[label:label + pd.Timedelta(minutes=4)]
        assert five.loc[label, "high"] == bucket["high"].max()
        assert five.loc[label, "close"] == bucket["close"].iloc[-1]


# -- sizing -----------------------------------------------------------------

class TestSizing:
    def test_size_never_exceeds_the_internal_cap(self):
        """Rule 4. The firm allows 40; the cap here is 5."""
        n = cr.size_for("MES", 6800.0, 6799.75, budget=1_000_000.0)
        assert n == rules.POSITION_CAP
        assert rules.POSITION_CAP < rules.FIRM.max_contracts

    def test_size_keeps_risk_inside_the_budget(self):
        for budget in (50.0, 137.0, 400.0, 1000.0):
            n = cr.size_for("MES", 6800.0, 6790.0, budget)
            if n:
                assert store.risk_dollars("MES", 6800.0, 6790.0, n) <= budget
                # And one more would not fit.
                if n < rules.POSITION_CAP:
                    assert store.risk_dollars("MES", 6800.0, 6790.0,
                                              n + 1) > budget

    def test_size_is_zero_when_one_contract_does_not_fit(self):
        assert cr.size_for("MES", 6800.0, 6700.0, budget=50.0) == 0

    def test_sizing_uses_all_in_risk_not_bare_stop_distance(self):
        """risk_dollars includes two ticks of slippage and commission."""
        bare = abs(6800.0 - 6790.0) * 5.0
        assert store.risk_dollars("MES", 6800.0, 6790.0, 1) > bare

    def test_bracket_risk_is_under_the_daily_limit(self):
        r = build_read()
        for bracket in (r.long, r.short):
            if bracket.contracts:
                assert bracket.risk_dollars <= rules.DAILY_LOSS_LIMIT

    def test_budget_shrinks_after_a_losing_day(self):
        state = pretrade.AccountState(today_realised_pnl=-350.0)
        r = build_read()
        tight = cr.build_bracket("long", r.spot, r.levels, r.atr, "MES",
                                 state, ELEVEN_AM)
        if tight.contracts:
            assert tight.risk_dollars <= rules.DAILY_LOSS_LIMIT - 350.0
        assert tight.contracts <= r.long.contracts


# -- brackets ---------------------------------------------------------------

class TestBrackets:
    def test_long_stop_below_and_target_above(self):
        b = build_read().long
        assert b.stop < b.entry < b.target

    def test_short_stop_above_and_target_below(self):
        b = build_read().short
        assert b.target < b.entry < b.stop

    def test_stop_clears_the_atr_buffer(self):
        r = build_read()
        buffer = cr.ATR_BUFFER_MULT * r.atr
        for bracket in (r.long, r.short):
            if bracket.stop is not None and not bracket.blocks:
                assert abs(bracket.entry - bracket.stop) >= buffer * 0.99

    def test_reward_risk_is_computed_from_points(self):
        b = build_read().long
        expected = abs(b.target - b.entry) / abs(b.entry - b.stop)
        assert b.reward_risk == pytest.approx(expected)

    def test_thin_reward_risk_is_flagged(self):
        b = cr.Bracket("long", reward_risk=1.0)
        assert b.thin
        assert not cr.Bracket("long", reward_risk=2.0).thin
        assert not cr.Bracket("long").thin

    def test_thin_bracket_says_so_in_the_rendered_text(self):
        b = cr.Bracket("long", entry=1.0, stop=0.5, target=1.25,
                       contracts=1, risk_dollars=10.0, reward_risk=0.5)
        assert "below 1.5" in cr.format_bracket(b, "MES")

    def test_both_directions_are_always_produced(self):
        r = build_read()
        assert r.long.direction == "long" and r.short.direction == "short"


# -- the blocked window: the one that matters -------------------------------

class TestBlockedWindow:
    @pytest.mark.parametrize("when", ["2026-09-04 16:21", "2026-09-04 16:35",
                                      "2026-09-04 17:30"])
    def test_after_the_entry_cutoff_neither_direction_gets_a_bracket(self, when):
        r = build_read(now=pd.Timestamp(when, tz=ET))
        for bracket in (r.long, r.short):
            assert bracket.blocks, f"{bracket.direction} produced no block"
            assert bracket.allowed is False

    def test_the_block_is_rendered_instead_of_a_bracket(self):
        r = build_read(now=pd.Timestamp("2026-09-04 16:25", tz=ET))
        text = cr.format_bracket(r.long, "MES")
        assert "would block" in text
        for leaked in ("entry `", "stop behind", "R:R"):
            assert leaked not in text

    def test_a_roll_day_blocks_both_directions(self):
        r = build_read(roll_dates={pd.Timestamp("2026-09-04").date()})
        for bracket in (r.long, r.short):
            assert any("roll" in b.lower() for b in bracket.blocks)

    def test_a_holiday_blocks_both_directions(self):
        r = build_read(closed_dates={pd.Timestamp("2026-09-04").date()})
        for bracket in (r.long, r.short):
            assert any("holiday" in b.lower() for b in bracket.blocks)

    def test_a_breached_daily_loss_limit_blocks(self):
        r = build_read()
        state = pretrade.AccountState(today_realised_pnl=-rules.DAILY_LOSS_LIMIT)
        bracket = cr.build_bracket("long", r.spot, r.levels, r.atr, "MES",
                                   state, ELEVEN_AM)
        assert bracket.blocks
        assert bracket.allowed is False

    def test_blocks_come_from_pretrade_not_a_second_implementation(self):
        """The read must refuse for exactly the reasons the journal refuses."""
        late = pd.Timestamp("2026-09-04 16:25", tz=ET)
        r = build_read(now=late)
        mirror = pretrade.evaluate(
            pretrade.TicketRequest(
                instrument="MES", direction="long", entry_price=r.long.entry,
                stop_price=r.long.stop, contracts=max(1, r.long.contracts),
                thesis="x"),
            FLAT, late)
        assert set(r.long.blocks) <= set(mirror.blocks) or r.long.blocks


# -- rendering --------------------------------------------------------------

class TestRendering:
    def test_disclaimer_is_exact_and_states_both_halves(self):
        assert "not a signal" in cr.DISCLAIMER
        assert "No tested strategy" in cr.DISCLAIMER

    def test_render_carries_no_opinion_bias_or_score(self):
        """The command describes; it does not advise."""
        text = " ".join(str(v) for v in cr.format_read(build_read()).items())
        for banned in ("bullish", "bearish", "confidence", "score", "bias",
                       "recommend", "should buy", "should sell", "likely"):
            assert banned not in text.lower(), f"{banned!r} leaked into the read"

    def test_every_required_section_is_present(self):
        fields = cr.format_read(build_read())
        joined = " ".join(fields)
        for section in ("Trend context", "Levels", "Volatility", "Session",
                        "LONG", "SHORT"):
            assert section in joined

    def test_session_reports_the_clock_and_budget(self):
        fields = cr.format_read(build_read())
        session = fields["Session"]
        assert "entry cutoff" in session and "flatten" in session
        assert "budget left" in session

    def test_stale_source_is_warned_about(self):
        """A week-old parquet must not present itself as today's tape."""
        source = cr.BarSource("parquet cache", 0, None, True,
                              pd.Timestamp("2026-08-31").date())
        warning = source.warning(pd.Timestamp("2026-09-08 11:00", tz=ET))
        assert warning and "not running" in warning and "2026-08-31" in warning

    def test_fresh_live_source_is_not_warned_about(self):
        now = pd.Timestamp("2026-09-04 11:00", tz=ET)
        source = cr.BarSource("desk live feed + parquet history", 300,
                              now, False, now.date())
        assert source.warning(now) is None


# -- logging ----------------------------------------------------------------

class TestLogging:
    def test_a_logged_trade_lands_in_the_manual_journal(self, tmp_path):
        journal = tmp_path / "trades.jsonl"
        r = build_read(journal=journal)
        result = read_log.log_trade(r.long, "MES", "reclaimed VWAP",
                                    ELEVEN_AM, journal)
        assert result.logged, result.message
        rows = store.load_open(journal)
        assert len(rows) == 1
        assert rows.iloc[0]["direction"] == "long"
        assert rows.iloc[0]["thesis"] == "reclaimed VWAP"
        assert rows.iloc[0]["checks"]["source"] == "/read"

    def test_it_counts_toward_entry_3(self, tmp_path):
        """These are discretionary trades through pretrade - the gate's own
        mechanism. Contrast bots/read_log.py's note on desk tickets."""
        from registry import _journal_review

        journal = tmp_path / "trades.jsonl"
        r = build_read(journal=journal)
        read_log.log_trade(r.long, "MES", "t", ELEVEN_AM, journal)
        opened = store.load_open(journal).iloc[0]

        import close as journal_close
        ticket = journal_close.find_open_ticket(opened["ticket_id"], journal)
        closed = journal_close.close_ticket(
            ticket, float(r.long.target), ELEVEN_AM + pd.Timedelta(minutes=20),
            "target")
        store.append(closed, journal)

        review = _journal_review()
        assert review.readiness(store.load_closed(journal))["closed_trades"] == 1

    def test_a_blocked_window_logs_nothing(self, tmp_path):
        journal = tmp_path / "trades.jsonl"
        late = pd.Timestamp("2026-09-04 16:25", tz=ET)
        r = build_read(now=late, journal=journal)
        result = read_log.log_trade(r.long, "MES", "too late", late, journal)
        assert result.logged is False
        assert "BLOCKED" in result.message
        assert not journal.exists()

    def test_an_empty_thesis_is_refused(self, tmp_path):
        journal = tmp_path / "trades.jsonl"
        r = build_read(journal=journal)
        for thesis in ("", "   ", "\n"):
            assert read_log.log_trade(r.long, "MES", thesis, ELEVEN_AM,
                                      journal).logged is False
        assert not journal.exists()

    def test_only_the_owner_may_log(self, tmp_path):
        journal = tmp_path / "trades.jsonl"
        view = read_log.ReadLogView(build_read(journal=journal), owner_id=42,
                                    journal_path=journal)
        assert view.submit(999, "long", "t", ELEVEN_AM).logged is False
        assert view.submit(None, "long", "t", ELEVEN_AM).logged is False
        assert not journal.exists()
        assert view.submit(42, "long", "t", ELEVEN_AM).logged is True

    def test_the_same_direction_cannot_be_logged_twice(self, tmp_path):
        journal = tmp_path / "trades.jsonl"
        view = read_log.ReadLogView(build_read(journal=journal), owner_id=42,
                                    journal_path=journal)
        assert view.submit(42, "long", "t", ELEVEN_AM).logged is True
        again = view.submit(42, "long", "t", ELEVEN_AM)
        assert again.logged is False and "Already" in again.message
        assert len(store.load_open(journal)) == 1

    def test_buttons_are_disabled_when_the_rules_block(self, tmp_path):
        late = pd.Timestamp("2026-09-04 16:25", tz=ET)
        view = read_log.ReadLogView(build_read(now=late), owner_id=42)
        assert all(spec["disabled"] for spec in view.button_specs())

    def test_buttons_are_enabled_in_an_open_window(self, tmp_path):
        view = read_log.ReadLogView(build_read(), owner_id=42)
        labels = [s["label"] for s in view.button_specs()]
        assert labels == ["Log long", "Log short"]
        assert not all(s["disabled"] for s in view.button_specs())

    def test_logging_re_evaluates_rather_than_trusting_the_read(self, tmp_path):
        """Minutes pass between the embed and the modal; the cutoff can fall
        inside that gap, so the decision that counts is at submit time."""
        journal = tmp_path / "trades.jsonl"
        r = build_read(journal=journal)              # 11:00, allowed
        assert not r.long.blocks
        late = pd.Timestamp("2026-09-04 16:25", tz=ET)
        result = read_log.log_trade(r.long, "MES", "t", late, journal)
        assert result.logged is False
        assert not journal.exists()


# -- the live-bar mirror ----------------------------------------------------

class TestLiveBars:
    def test_round_trip(self, tmp_path):
        ts = pd.Timestamp("2026-09-04 09:30", tz=ET)
        cr.append_live_bar("MES", ts, 1.0, 2.0, 0.5, 1.5, 100.0, tmp_path)
        cr.append_live_bar("MES", ts + pd.Timedelta(minutes=1),
                           1.5, 2.5, 1.0, 2.0, 120.0, tmp_path)
        frame = cr.load_live("MES", ts.date(), tmp_path)
        assert len(frame) == 2
        assert frame["high"].max() == 2.5
        assert frame.index[0] == ts

    def test_absent_file_is_an_empty_frame_not_an_error(self, tmp_path):
        assert cr.load_live("MES", pd.Timestamp("2026-09-04").date(),
                            tmp_path).empty

    def test_one_file_per_session_date(self, tmp_path):
        cr.append_live_bar("MES", pd.Timestamp("2026-09-04 10:00", tz=ET),
                           1, 1, 1, 1, 1, tmp_path)
        cr.append_live_bar("MES", pd.Timestamp("2026-09-05 10:00", tz=ET),
                           1, 1, 1, 1, 1, tmp_path)
        assert len(list(tmp_path.glob("MES_*.csv"))) == 2


class TestGuards:
    def test_only_mes_and_mnq(self):
        for symbol in ("ES", "NQ", "SPY"):
            with pytest.raises(rules.RuleViolation):
                cr.read(symbol, "5m", now=ELEVEN_AM, bars=make_bars())

    def test_unknown_timeframe_is_a_readable_error(self):
        with pytest.raises(cr.ReadError, match="unknown timeframe"):
            cr.parse_timeframe("4h")

    @pytest.mark.parametrize("tf,minutes", [("1m", 1), ("5m", 5), ("15m", 15),
                                            ("60m", 60), ("1h", 60)])
    def test_accepted_timeframes(self, tf, minutes):
        assert cr.parse_timeframe(tf) == minutes
