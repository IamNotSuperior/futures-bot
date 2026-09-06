"""Tests for the strategy registry.

The tests that matter are the refusals. A gate that has only been tested on the
paths it allows has not been tested at all, so every transition that must fail
has a test here, including the ones nobody would write by accident: promoting
out of rejected, skipping a stage, and promoting to paper on an entry the log
has not accepted.
"""

import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

from registry import (
    ALLOWED_TRANSITIONS, STATUSES, HypothesisEntry, PromotionError, Registry,
    RegistryError, StrategyRecord, parse_hypotheses,
)

LOG = textwrap.dedent("""\
    # Strategy hypothesis log

    ## 1. Rejected Thing — REJECTED

    **Verdict commit:** `aaa1111`

    A paragraph that serves as the summary.

    ### Walk-forward verdict: REJECTED

    ## 2. Accepted Thing — ACCEPTED

    **Verdict commit:** `bbb2222`

    This one passed its walk-forward.

    ## 3. Open Thing — PROPOSED

    Still open, no verdict.

    ## Template for new entries
    """)


@pytest.fixture
def log_path(tmp_path) -> Path:
    p = tmp_path / "hypotheses.md"
    p.write_text(LOG, encoding="utf-8")
    return p


def make_registry(tmp_path, log_path, rows) -> Registry:
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump({"strategies": rows}), encoding="utf-8")
    return Registry.load(p, log_path)


def row(name, entry, status, class_path="orb:OpeningRangeBreakout",
        commit=None, verdict="v"):
    return {"name": name, "class_path": class_path, "hypothesis_entry": entry,
            "status": status, "verdict_commit": commit,
            "walkforward_verdict": verdict}


class TestParseHypotheses:
    def test_reads_number_title_and_status(self, log_path):
        entries = parse_hypotheses(log_path)
        assert set(entries) == {1, 2, 3}
        assert entries[1].status == "REJECTED"
        assert entries[2].status == "ACCEPTED"
        assert entries[3].status == "PROPOSED"
        assert entries[2].title == "Accepted Thing"

    def test_reads_the_verdict_commit(self, log_path):
        entries = parse_hypotheses(log_path)
        assert entries[1].verdict_commit == "aaa1111"
        assert entries[3].verdict_commit is None

    def test_accepted_flag(self, log_path):
        entries = parse_hypotheses(log_path)
        assert entries[2].accepted
        assert not entries[1].accepted
        assert not entries[3].accepted

    def test_summary_skips_metadata_lines(self, log_path):
        assert parse_hypotheses(log_path)[1].summary().startswith("A paragraph")

    def test_the_template_heading_is_not_an_entry(self, log_path):
        assert all(isinstance(n, int) for n in parse_hypotheses(log_path))

    def test_the_real_log_parses(self):
        entries = parse_hypotheses()
        assert set(entries) >= {1, 2, 3, 4, 5}
        assert entries[1].status == "REJECTED"
        assert entries[4].status == "REJECTED"
        assert entries[5].status == "REJECTED"
        assert entries[3].status == "PROPOSED"


class TestLoading:
    def test_rejects_an_unknown_status(self, tmp_path, log_path):
        with pytest.raises(RegistryError, match="unknown status"):
            make_registry(tmp_path, log_path, [row("a", 1, "banana")])

    def test_rejects_a_row_missing_fields(self, tmp_path, log_path):
        p = tmp_path / "registry.yaml"
        p.write_text(yaml.safe_dump({"strategies": [{"name": "a"}]}),
                     encoding="utf-8")
        with pytest.raises(RegistryError, match="missing"):
            Registry.load(p, log_path)

    def test_unknown_name_lists_what_it_knows(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("a", 1, "rejected")])
        with pytest.raises(KeyError, match="known"):
            reg.get("nope")

    def test_real_registry_loads_and_agrees_with_the_log(self):
        reg = Registry.load()
        assert reg.verify() == []
        assert set(reg.names()) >= {"orb", "orb2", "orb_flat_1030"}

    def test_entries_one_to_five_are_present(self):
        reg = Registry.load()
        assert sorted(r.hypothesis_entry for r in reg.all()) == [1, 2, 3, 4, 5]

    def test_the_four_rejected_entries_are_rejected(self):
        reg = Registry.load()
        by_entry = {r.hypothesis_entry: r for r in reg.all()}
        for n in (1, 2, 4, 5):
            assert by_entry[n].status == "rejected", f"entry {n}"

    def test_class_paths_import(self):
        reg = Registry.load()
        for r in reg.all():
            if r.class_path:
                assert r.load_class() is not None

    def test_an_entry_without_code_refuses_to_load_a_class(self):
        reg = Registry.load()
        with pytest.raises(RegistryError, match="not code"):
            reg.get("manual_discretionary").load_class()


class TestVerify:
    def test_flags_a_registry_that_disagrees_with_a_rejection(
            self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("a", 1, "testing")])
        problems = reg.verify()
        assert len(problems) == 1
        assert "REJECTED in the log" in problems[0]

    def test_flags_paper_status_on_a_merely_proposed_entry(
            self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("c", 3, "paper")])
        assert any("only PROPOSED" in p for p in reg.verify())

    def test_flags_a_missing_entry(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("z", 99, "proposed")])
        assert any("no entry 99" in p for p in reg.verify())

    def test_flags_a_mismatched_verdict_commit(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path,
                            [row("a", 1, "rejected", commit="deadbee")])
        assert any("does not match" in p for p in reg.verify())

    def test_agreement_is_silent(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path,
                            [row("a", 1, "rejected", commit="aaa1111")])
        assert reg.verify() == []


class TestTransitionsThatMustFail:
    """Every refusal. These are the tests the registry exists for."""

    def test_rejected_is_terminal_for_every_target(self, tmp_path, log_path):
        for target in STATUSES:
            reg = make_registry(tmp_path, log_path, [row("a", 1, "rejected")])
            if target == "rejected":
                with pytest.raises(PromotionError, match="already"):
                    reg.promote("a", target)
            else:
                with pytest.raises(PromotionError, match="terminal"):
                    reg.promote("a", target)

    def test_proposed_cannot_skip_to_paper(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "proposed")])
        with pytest.raises(PromotionError, match="not a transition"):
            reg.promote("b", "paper")

    def test_proposed_cannot_skip_to_live(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "proposed")])
        with pytest.raises(PromotionError, match="not a transition"):
            reg.promote("b", "live")

    def test_testing_cannot_skip_to_live(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "testing")])
        with pytest.raises(PromotionError, match="not a transition"):
            reg.promote("b", "live")

    def test_live_cannot_go_back_to_paper(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "live")])
        with pytest.raises(PromotionError, match="not a transition"):
            reg.promote("b", "paper")

    def test_testing_to_paper_refused_when_the_log_rejected_it(
            self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("a", 1, "testing")])
        with pytest.raises(PromotionError, match="REJECTED"):
            reg.promote("a", "paper")

    def test_testing_to_paper_refused_when_the_log_is_only_proposed(
            self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("c", 3, "testing")])
        with pytest.raises(PromotionError, match="PROPOSED"):
            reg.promote("c", "paper")

    def test_testing_to_paper_refused_when_there_is_no_entry(
            self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("z", 99, "testing")])
        with pytest.raises(PromotionError, match="no entry"):
            reg.promote("z", "paper")

    def test_promoting_to_the_same_status_is_refused(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "testing")])
        with pytest.raises(PromotionError, match="already"):
            reg.promote("b", "testing")

    def test_unknown_status_is_a_value_error(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "testing")])
        with pytest.raises(ValueError, match="unknown status"):
            reg.promote("b", "sideways")

    def test_promote_takes_no_override_argument(self):
        """The absence of a bypass is a feature and is asserted as one."""
        import inspect
        params = list(inspect.signature(Registry.promote).parameters)
        assert params == ["self", "name", "to_status"]
        assert not any(
            "force" in p or "override" in p or "skip" in p for p in params
        )

    def test_a_refused_promotion_does_not_touch_the_file(
            self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("a", 1, "rejected")])
        before = (tmp_path / "registry.yaml").read_text(encoding="utf-8")
        with pytest.raises(PromotionError):
            reg.promote("a", "testing")
        assert (tmp_path / "registry.yaml").read_text(encoding="utf-8") == before
        assert reg.get("a").status == "rejected"


class TestTransitionsThatMustSucceed:
    def test_proposed_to_testing(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "proposed")])
        assert reg.promote("b", "testing").status == "testing"

    def test_testing_to_paper_when_the_log_accepted_it(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "testing")])
        assert reg.promote("b", "paper").status == "paper"

    def test_anything_can_be_rejected(self, tmp_path, log_path):
        for start in ("proposed", "testing", "paper", "live"):
            reg = make_registry(tmp_path, log_path, [row("b", 2, start)])
            assert reg.promote("b", "rejected").status == "rejected"

    def test_a_successful_promotion_persists(self, tmp_path, log_path):
        reg = make_registry(tmp_path, log_path, [row("b", 2, "proposed")])
        reg.promote("b", "testing")
        reloaded = Registry.load(tmp_path / "registry.yaml", log_path)
        assert reloaded.get("b").status == "testing"

    def test_the_transition_table_covers_every_status(self):
        assert set(ALLOWED_TRANSITIONS) == set(STATUSES)
        for targets in ALLOWED_TRANSITIONS.values():
            assert targets <= set(STATUSES)


class TestPaperToLiveGate:
    """The gate reads journal/review.readiness, so it is exercised through it."""

    def _reg(self, tmp_path, log_path):
        return make_registry(tmp_path, log_path, [row("b", 2, "paper")])

    def _closed(self, n, pnl, contracts=1, entry_hour=10):
        idx = pd.date_range("2025-07-14 10:00", periods=n, freq="1D",
                            tz="America/New_York")
        return pd.DataFrame({
            "entry_time": idx,
            "exit_time": idx + pd.Timedelta(minutes=30),
            "net_pnl": [float(pnl)] * n,
            "contracts": [contracts] * n,
            "duration_seconds": [1800.0] * n,
            "session_date": [t.date() for t in idx],
        })

    def test_refused_with_no_journal(self, tmp_path, log_path, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
        with pytest.raises(PromotionError, match="no journal"):
            self._reg(tmp_path, log_path).promote("b", "live")

    def test_refused_below_sixty_clean_trades(self, tmp_path, log_path, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: self._closed(10, 50.0))
        with pytest.raises(PromotionError, match="rule-clean trades 10 of 60"):
            self._reg(tmp_path, log_path).promote("b", "live")

    def test_refused_on_negative_expectancy(self, tmp_path, log_path, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: self._closed(80, -50.0))
        with pytest.raises(PromotionError, match="must be positive"):
            self._reg(tmp_path, log_path).promote("b", "live")

    def test_refused_when_a_rule_violation_resets_the_count(
            self, tmp_path, log_path, monkeypatch):
        import store
        closed = self._closed(80, 50.0)
        closed.loc[0, "contracts"] = 99          # over the position cap
        monkeypatch.setattr(store, "load_closed", lambda *a, **k: closed)
        with pytest.raises(PromotionError, match="reset by"):
            self._reg(tmp_path, log_path).promote("b", "live")

    def test_refused_on_pass_probability(self, tmp_path, log_path, monkeypatch):
        """Clean and profitable, but nowhere near a 50% pass probability."""
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: self._closed(80, 1.0))
        with pytest.raises(PromotionError, match="pass probability"):
            self._reg(tmp_path, log_path).promote("b", "live")

    def test_the_refusal_names_every_failing_gate(
            self, tmp_path, log_path, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: self._closed(5, -10.0))
        with pytest.raises(PromotionError) as excinfo:
            self._reg(tmp_path, log_path).promote("b", "live")
        message = str(excinfo.value)
        assert "rule-clean trades" in message
        assert "expectancy" in message

    def test_allowed_when_every_gate_passes(self, tmp_path, log_path, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: self._closed(80, 400.0))
        reg = self._reg(tmp_path, log_path)
        assert reg.promote("b", "live").status == "live"


class TestPaperTradeCount:
    def test_counts_closed_tickets(self, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: pd.DataFrame({"net_pnl": [1.0, 2.0]}))
        assert Registry.load().paper_trade_count("orb") == 2

    def test_missing_journal_counts_zero(self, monkeypatch):
        import store
        monkeypatch.setattr(store, "load_closed",
                            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
        assert Registry.load().paper_trade_count("orb") == 0

    def test_status_table_renders(self):
        table = Registry.load().status_table()
        assert "orb" in table and "rejected" in table
