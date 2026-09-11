"""The /submit pipeline: the sandbox, the approval gate, and the ordering.

Three properties, each of which is the reason a step exists:

* **The file-write sandbox.** Generated code may land in two directories and
  nowhere else. A path that escapes by ``..``, an absolute path, a symlink, or
  a name that shadows ``rules`` must all be refused - and refused *before* the
  file is written, not cleaned up afterwards.
* **The approval gate.** Only the owner approves; only a submission whose
  tests passed can be approved; ``Registry.add`` cannot create anything above
  ``proposed``; rejection is terminal.
* **Verdict before embed.** The walk-forward verdict is written into
  ``hypotheses.md`` and committed before any result is posted. If that order
  inverts there is a window in which a human has acted on a number the log has
  not frozen, and the log's entire claim is that it has.

No test here calls the Claude API. ``generate.parse_response`` is exercised on
fixed strings; the network path is not the thing under test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import generate as generate_mod
import rules
import sandbox
import submissions
import submit_view
from registry import Registry, RegistryError

#: A fake snowflake, as in test_desk_discord.py. The real DESK_OWNER_ID is
#: configuration in .env and has no business in a test - the first version of
#: this file hard-coded the operator's actual id, which is what a pre-push
#: scan for .env values caught.
OWNER = 123456789012345678
STRANGER = 12345

CLEAN_STRATEGY = '''\
"""A generated strategy."""
from __future__ import annotations

import pandas as pd
import numpy as np
from base import Strategy, empty_signals, validate_signals


class MyStrategy(Strategy):
    name = "my_strategy"

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = empty_signals(bars.index)
        return validate_signals(out)
'''

CLEAN_TEST = '''\
import pandas as pd
import pytest


def test_placeholder():
    assert pd.Timestamp("2026-01-01").year == 2026
'''


def submission(name="my_strategy", **kw) -> generate_mod.Submission:
    base = dict(
        name=name,
        mechanism="Overnight inventory is unwound at the open.",
        counterparty="Dealers flat by 10:00, forced to lift offers.",
        kill_criteria="Fewer than 4 of 7 folds profitable, or negative total.",
    )
    base.update(kw)
    return generate_mod.Submission(**base)


# ---------------------------------------------------------------------------
# 1. the file-write sandbox
# ---------------------------------------------------------------------------

class TestWriteSandbox:
    @pytest.mark.parametrize("relative", [
        "strategies/generated/ok.py",
        "tests/generated/test_ok.py",
    ])
    def test_allowed_paths_resolve(self, relative):
        target = sandbox.resolve_target(relative)
        assert target.suffix == ".py"
        assert any(d.resolve() in target.parents for d in sandbox.ALLOWED_DIRS)

    @pytest.mark.parametrize("relative", [
        "strategies/rules.py",
        "strategies/engine.py",
        "backtests/engine.py",
        "bots/desk.py",
        "journal/store.py",
        "CLAUDE.md",
        ".env",
        "requirements.txt",
        "tests/test_rules.py",
        "strategies/generated/../rules.py",
        "strategies/generated/../../.env",
        "tests/generated/../../strategies/rules.py",
        "strategies/generated/sub/../../orb.py",
    ])
    def test_everything_else_is_refused(self, relative):
        with pytest.raises(sandbox.SandboxError):
            sandbox.resolve_target(relative)

    def test_absolute_paths_are_refused(self, tmp_path):
        for absolute in (str(tmp_path / "x.py"), "/etc/passwd",
                         "C:\\Windows\\System32\\drivers\\etc\\hosts"):
            with pytest.raises(sandbox.SandboxError, match="absolute|outside|'\\.\\.'"):
                sandbox.resolve_target(absolute)

    def test_the_named_protected_files_cannot_be_written(self):
        """rules.py and engine.py by name - the two the brief calls out."""
        for relative in ("strategies/rules.py", "backtests/engine.py"):
            before = (sandbox.PROJECT_ROOT / relative).read_text(encoding="utf-8")
            with pytest.raises(sandbox.SandboxError):
                sandbox.safe_write(relative, "raise SystemExit(1)")
            after = (sandbox.PROJECT_ROOT / relative).read_text(encoding="utf-8")
            assert before == after, f"{relative} was modified"

    def test_non_python_is_refused(self):
        with pytest.raises(sandbox.SandboxError, match="must be .py"):
            sandbox.resolve_target("strategies/generated/notes.txt")

    def test_a_symlink_out_of_the_sandbox_is_refused(self, tmp_path):
        link = sandbox.STRATEGY_DIR / "_escape_link"
        outside = tmp_path
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted on this machine")
        try:
            with pytest.raises(sandbox.SandboxError, match="outside the sandbox"):
                sandbox.resolve_target("strategies/generated/_escape_link/evil.py")
        finally:
            link.unlink()

    def test_safe_write_lands_in_the_sandbox(self, tmp_path):
        target = sandbox.safe_write("strategies/generated/_probe.py",
                                    CLEAN_STRATEGY)
        try:
            assert target.exists()
            assert target.read_text(encoding="utf-8") == CLEAN_STRATEGY
        finally:
            target.unlink(missing_ok=True)


class TestContentScreen:
    def test_clean_source_passes(self):
        assert sandbox.screen_source(CLEAN_STRATEGY).ok
        assert sandbox.screen_source(CLEAN_TEST).ok

    @pytest.mark.parametrize("source,fragment", [
        ("import os\n", "import os"),
        ("import subprocess\n", "subprocess"),
        ("import sys\n", "import sys"),
        ("import socket\n", "socket"),
        ("import shutil\n", "shutil"),
        ("import importlib\n", "importlib"),
        ("from pathlib import Path\n", "pathlib"),
        ("import requests\n", "requests"),
    ])
    def test_dangerous_imports_are_refused(self, source, fragment):
        result = sandbox.screen_source(source)
        assert not result.ok
        assert fragment in result.report()

    @pytest.mark.parametrize("source", [
        "x = eval('1+1')\n",
        "exec('x=1')\n",
        "f = open('/etc/passwd')\n",
        "m = __import__('os')\n",
        "c = compile('1', '<s>', 'eval')\n",
    ])
    def test_dynamic_execution_is_refused(self, source):
        assert not sandbox.screen_source(source).ok

    @pytest.mark.parametrize("source", [
        "import pandas as pd\nx = pd.DataFrame.__subclasses__()\n",
        "import pandas as pd\ny = pd.DataFrame.__globals__\n",
        "import pandas as pd\nz = (1).__class__.__bases__\n",
    ])
    def test_reflection_escapes_are_refused(self, source):
        assert not sandbox.screen_source(source).ok

    def test_secret_names_are_refused_even_as_strings(self):
        """A string-built lookup would not appear as an import in the AST."""
        result = sandbox.screen_source(
            "import pandas as pd\nKEY = 'ANTHROPIC_API_KEY'\n")
        assert not result.ok

    def test_relative_imports_are_refused(self):
        assert not sandbox.screen_source("from . import rules\n").ok

    def test_syntax_errors_are_reported_not_raised(self):
        result = sandbox.screen_source("def broken(\n")
        assert not result.ok and "does not parse" in result.report()

    def test_every_problem_is_reported_at_once(self):
        result = sandbox.screen_source("import os\nimport socket\nx = eval('1')\n")
        assert len(result.problems) >= 3

    def test_safe_write_refuses_screened_source_without_writing(self):
        relative = "strategies/generated/_never.py"
        with pytest.raises(sandbox.SandboxError, match="refused"):
            sandbox.safe_write(relative, "import os\n")
        assert not (sandbox.PROJECT_ROOT / relative).exists()


class TestGeneratedTestImports:
    """A generated test must be able to import the strategy it tests.

    The first live /submit died here: the test file was refused for importing
    its own strategy module, so a submission could never reach the suite.
    """

    def test_a_generated_test_may_import_its_own_strategy(self):
        source = ("import pandas as pd\n"
                  "import pytest\n"
                  "import orborb_flat_1030\n\n"
                  "def test_signals():\n"
                  "    assert orborb_flat_1030 is not None\n")
        result = sandbox.screen_source(source, allow_module="orborb_flat_1030")
        assert result.ok, result.report()

    def test_the_fully_qualified_spelling_is_also_accepted(self):
        for form in ("import strategies.generated.my_thing\n",
                     "from strategies.generated.my_thing import Thing\n",
                     "import my_thing\n",
                     "from my_thing import Thing\n"):
            result = sandbox.screen_source(form, allow_module="my_thing")
            assert result.ok, f"{form!r}: {result.report()}"

    def test_it_may_not_import_a_different_generated_strategy(self):
        """One submission must not reach into another's code."""
        for form in ("import other_strategy\n",
                     "from other_strategy import Thing\n",
                     "import strategies.generated.other_strategy\n"):
            result = sandbox.screen_source(form, allow_module="my_thing")
            assert not result.ok, f"{form!r} was allowed"

    def test_the_allowance_does_not_open_the_strategies_package(self):
        """`strategies.generated.x` must not admit `strategies.rules`."""
        for form in ("import strategies.rules\n",
                     "from strategies.engine import price_trades\n",
                     "import strategies\n"):
            result = sandbox.screen_source(form, allow_module="my_thing")
            assert not result.ok, f"{form!r} was allowed"

    def test_a_strategy_gets_no_allowance_at_all(self):
        """Only the test is given one; a strategy importing another is coupling."""
        assert not sandbox.screen_source("import other_strategy\n").ok
        assert not sandbox.screen_source("import my_thing\n").ok

    def test_the_allowance_does_not_relax_anything_else(self):
        source = "import my_thing\nimport os\n"
        result = sandbox.screen_source(source, allow_module="my_thing")
        assert not result.ok
        assert "import os" in result.report()

    def test_write_generated_passes_the_allowance_only_to_the_test(self):
        import inspect

        source = inspect.getsource(submissions.write_generated)
        strategy_call = source.index("strategies/generated/")
        test_call = source.index("tests/generated/")
        assert "allow_module=name" in source[test_call:]
        assert "allow_module" not in source[strategy_call:test_call]

    def test_the_live_failure_case_now_passes_end_to_end(self, tmp_path):
        """The exact shape that failed: a test importing its own strategy."""
        name = "probe_strategy"
        test_source = (
            "import pandas as pd\n"
            "import pytest\n"
            f"import {name}\n\n"
            f"def test_interface():\n"
            f"    assert {name}.__name__ == '{name}'\n"
        )
        written = sandbox.safe_write(f"tests/generated/test_{name}.py",
                                     test_source, allow_module=name)
        try:
            assert written.exists()
        finally:
            written.unlink(missing_ok=True)


class TestResubmission:
    """A failed attempt must not burn the name.

    The pre-registration is committed before any code exists, so a submission
    that dies at the screen or in the suite leaves an entry behind. Opening a
    second entry for the same claim would make the log say an idea was
    proposed twice when it was proposed once and built badly.
    """

    def write_log(self, tmp_path, extra=""):
        log = tmp_path / "hypotheses.md"
        log.write_text("# Log\n\n## 1. Something — REJECTED\n\nbody\n" + extra,
                       encoding="utf-8")
        return log

    def test_a_retry_reuses_the_entry(self, tmp_path):
        log = self.write_log(tmp_path)
        first = submissions.pre_register(submission(name="fresh_idea"), log,
                                         do_commit=False)
        second = submissions.pre_register(submission(name="fresh_idea"), log,
                                          do_commit=False)
        assert second.entry_number == first.entry_number
        assert second.attempt == 2
        body = log.read_text(encoding="utf-8")
        assert body.count("## 2. fresh idea") == 1, "a second entry was opened"
        assert "### Attempt 2 —" in body

    def test_the_note_is_dated_and_says_why(self, tmp_path):
        log = self.write_log(tmp_path)
        submissions.pre_register(submission(name="fresh_idea"), log,
                                 do_commit=False)
        submissions.pre_register(submission(name="fresh_idea"), log,
                                 do_commit=False, reason="the screen refused it")
        body = log.read_text(encoding="utf-8")
        assert "the screen refused it" in body
        assert re.search(r"### Attempt 2 — \d{4}-\d{2}-\d{2}", body)

    def test_attempts_keep_counting(self, tmp_path):
        log = self.write_log(tmp_path)
        numbers = [submissions.pre_register(submission(name="fresh_idea"), log,
                                            do_commit=False).attempt
                   for _ in range(4)]
        assert numbers == [1, 2, 3, 4]

    def test_attempts_are_counted_per_entry(self, tmp_path):
        """A retry elsewhere must not renumber this entry's attempts."""
        log = self.write_log(tmp_path)
        submissions.pre_register(submission(name="idea_one"), log, do_commit=False)
        submissions.pre_register(submission(name="idea_one"), log, do_commit=False)
        submissions.pre_register(submission(name="idea_one"), log, do_commit=False)
        second = submissions.pre_register(submission(name="idea_two"), log,
                                          do_commit=False)
        assert second.attempt == 1
        retry = submissions.pre_register(submission(name="idea_two"), log,
                                         do_commit=False)
        assert retry.attempt == 2

    def test_the_pre_registration_is_not_rewritten_by_a_retry(self, tmp_path):
        """The claim is unchanged; only the implementation is regenerated."""
        log = self.write_log(tmp_path)
        original = submission(name="fresh_idea")
        submissions.pre_register(original, log, do_commit=False)
        changed = submission(name="fresh_idea",
                             mechanism="A completely different claim.")
        submissions.pre_register(changed, log, do_commit=False)
        body = log.read_text(encoding="utf-8")
        assert original.mechanism in body
        assert "A completely different claim." not in body
        assert "would be a different" in body

    def test_a_name_that_reached_the_registry_stays_taken(self):
        for name in ("orb2", "london_1x", "manual_discretionary"):
            with pytest.raises(Exception) as exc:
                submissions.pre_register(submission(name=name), do_commit=False)
            assert "taken" in str(exc.value) or "shadow" in str(exc.value)

    def test_the_refusal_names_the_status(self):
        with pytest.raises(submissions.SubmissionError, match="rejected"):
            submissions.pre_register(submission(name="london_1x"),
                                     do_commit=False)

    def test_find_entry_matches_the_marker(self, tmp_path):
        log = self.write_log(tmp_path)
        run = submissions.pre_register(submission(name="fresh_idea"), log,
                                       do_commit=False)
        assert submissions.find_entry_for("fresh_idea", log) == run.entry_number
        assert submissions.find_entry_for("never_submitted", log) is None

    def test_find_entry_falls_back_to_the_title(self, tmp_path):
        """Entries written before the marker existed have only the title."""
        log = self.write_log(
            tmp_path, "\n## 2. legacy idea — PROPOSED\n\nno marker here\n")
        assert submissions.find_entry_for("legacy_idea", log) == 2

    def test_the_live_entry_8_is_findable(self):
        """entry 8 predates the marker; a retry must still reuse it."""
        found = submissions.find_entry_for("orborb_flat_1030")
        assert found == 8, f"expected entry 8, got {found}"

    def test_register_proposed_reuses_a_proposed_record(self, monkeypatch):
        """A retry that got this far once must not fail on a duplicate name."""
        import inspect

        source = inspect.getsource(submissions.register_proposed)
        assert "reusing the existing proposed record" in source
        assert 'record.status != "proposed"' in source

    def test_an_orphaned_module_gives_an_answer_not_a_traceback(self):
        """is_generated only asks whether the file exists.

        The failed live run left strategies/generated/orborb_flat_1030.py with
        no registry record, so /walkforward on it reached registry().get() and
        raised KeyError.
        """
        import runners

        orphans = [p.stem for p in sandbox.STRATEGY_DIR.glob("*.py")
                   if p.stem not in Registry.load().names()]
        if not orphans:
            pytest.skip("no orphaned generated module on disk")
        with pytest.raises(runners.WorkError, match="never reached review"):
            runners.run_walkforward_generated(orphans[0])

    def test_entry_body_is_scoped(self, tmp_path):
        log = self.write_log(
            tmp_path, "\n## 2. A — PROPOSED\n\nalpha\n\n## 3. B — PROPOSED\n\nbeta\n")
        assert "alpha" in submissions.entry_body(2, log)
        assert "beta" not in submissions.entry_body(2, log)


class TestScoredSpan:
    """A generated verdict is scored on the span every other entry used.

    The first run scored the whole parquet from 2019-05-05 and reported 588
    trades against entry 5's 478 on the same signals - the gap was entirely
    the span. Entry 8 carries the diagnostic. The fix imports the span from
    walkforward.py rather than writing 2020 down a second time.
    """

    def test_span_is_walkforwards_first_test_year(self):
        import run_generated
        import walkforward

        assert run_generated.SCORE_START == walkforward.build_folds()[0].test_start

    def test_span_is_not_a_restated_date(self):
        import inspect

        import run_generated

        source = inspect.getsource(run_generated)
        assert "SCORE_START = walkforward.build_folds()" in source
        assert "SCORE_START = date(" not in source

    def test_2019_is_history_not_sample(self):
        """Signals generated over 2019 are dropped; bars before it stay."""
        import pandas as pd
        import run_generated

        index = pd.date_range("2019-11-01 09:30", "2020-02-01 09:30",
                              freq="1D", tz=rules.ET)
        frame = pd.DataFrame({"entry_long": True}, index=index)
        sliced = run_generated.score_slice(frame, pd.Timestamp("2026-08-31").date())
        assert sliced.index.min().date() >= run_generated.SCORE_START
        assert len(sliced) < len(frame)
        assert (frame.index.date < run_generated.SCORE_START).any(), \
            "the fixture must actually contain pre-span rows"

    def test_the_verdict_block_states_the_span(self):
        import pandas as pd
        from datetime import date

        import run_generated

        result = run_generated.WalkforwardResult(
            name="x", folds=pd.DataFrame({"test_net_pnl": [1.0] * 7}),
            trades=pd.DataFrame({"net_pnl": [10.0]}),
            metrics={"sharpe": 1.0, "profit_factor": 1.5, "max_drawdown": -10.0,
                     "max_daily_loss": -5.0, "avg_duration_seconds": 600.0,
                     "microscalp_profit_pct": 0.0},
            pass_probability=0.3, blowups=0, profitable_folds=5,
            accepted=True, reasons=[], contracts=1, size_note="1",
            span=(date(2020, 1, 1), date(2026, 8, 31)))
        block = run_generated.verdict_block(result, 9)
        assert "| Scored span | 2020-01-01 .. 2026-08-31" in block
        assert "indicator history only" in block

    def test_the_verdict_block_reports_the_payout_milestone(self):
        """The $52,100 payout probability sits directly under the pass figure."""
        import pandas as pd
        from datetime import date

        import run_generated

        result = run_generated.WalkforwardResult(
            name="x", folds=pd.DataFrame({"test_net_pnl": [1.0] * 7}),
            trades=pd.DataFrame({"net_pnl": [10.0]}),
            metrics={"sharpe": 1.0, "profit_factor": 1.5, "max_drawdown": -10.0,
                     "max_daily_loss": -5.0, "avg_duration_seconds": 600.0,
                     "microscalp_profit_pct": 0.0},
            pass_probability=0.3, payout_probability=0.45, blowups=0,
            profitable_folds=5, accepted=True, reasons=[], contracts=1,
            size_note="1", span=(date(2020, 1, 1), date(2026, 8, 31)))
        block = run_generated.verdict_block(result, 9)
        pass_row = "| Pass probability | 30.00% |"
        payout_row = "| Payout probability | 45.00% |"
        assert pass_row in block and payout_row in block
        assert block.index(payout_row) > block.index(pass_row)

    def test_the_embed_reports_the_payout_milestone(self):
        import inspect

        import research

        source = inspect.getsource(research._run_generated_walkforward)
        assert '"Payout probability": f"{result.payout_probability:.2%}"' in source


class TestVerdictUnderTheTrailingHalt:
    """A generated verdict is scored under both internal guards.

    The standard stream (daily loss limit and the $1,500 trailing halt)
    decides rule 13. The same signals without the halt are reported alongside
    as the comparable basis, because a halt that fires in year one makes the
    fold test degenerate and entries 1 and 4 were measured without one.
    """

    def _result(self):
        import pandas as pd
        from datetime import date

        import run_generated

        comparable = run_generated.BasisSummary(
            trades=478, net_pnl=-11_780.0, profitable_folds=1, sharpe=-2.0,
            profit_factor=0.673, max_drawdown=-12_065.0, max_daily_loss=-500.0,
            pass_probability=0.0022, payout_probability=0.0129, blowups=6)
        return run_generated.WalkforwardResult(
            name="x", folds=pd.DataFrame({"test_net_pnl": [-1.0] * 7}),
            trades=pd.DataFrame({"net_pnl": [-1415.0]}),
            metrics={"sharpe": -3.0, "profit_factor": 0.527, "max_drawdown": -1500.0,
                     "max_daily_loss": -300.0, "avg_duration_seconds": 600.0,
                     "microscalp_profit_pct": 0.0},
            pass_probability=0.0, payout_probability=0.0001, blowups=0,
            profitable_folds=0, accepted=False,
            reasons=["profitable in 0 of 7 folds, needs 4"],
            contracts=4, size_note="4 (declared by the strategy)",
            span=(date(2020, 1, 1), date(2026, 8, 31)),
            dd_halts=433, comparable=comparable)

    def test_the_block_names_both_guards_and_the_sessions_blocked(self):
        import run_generated

        block = run_generated.verdict_block(self._result(), 8)
        assert "$400" in block and "$1,500" in block
        assert "| Sessions blocked by the trailing halt | 433 |" in block

    def test_the_comparable_basis_follows_the_verdict(self):
        import run_generated

        block = run_generated.verdict_block(self._result(), 8)
        verdict = block.index("**Rule 13 is not satisfied:**")
        comparable = block.index("#### Comparable basis, trailing halt OFF")
        assert comparable > verdict, "rule 13 is decided before the comparison"
        tail = block[comparable:]
        assert "| Trades | 478 |" in tail
        assert "| Net P&L | $-11,780.00 |" in tail
        assert "| Folds profitable | 1 of 7 |" in tail
        assert "| Pass probability | 0.22% |" in tail
        assert "| Payout probability | 1.29% |" in tail
        assert "| Evaluations blown | 6 |" in tail

    def test_the_reports_line_lists_both_bases(self):
        import run_generated

        block = run_generated.verdict_block(self._result(), 8)
        assert "`backtests/results/x_folds.csv`" in block
        assert "x_trades_nohalt.csv" in block

    def test_the_embed_reports_the_halt_and_the_comparable_basis(self):
        import inspect

        import research

        source = inspect.getsource(research._run_generated_walkforward)
        assert '"Sessions blocked by the trailing halt"' in source
        assert '"Halt OFF (comparable)"' in source


class TestCleanLogGuard:
    """The bot only ever commits its own append.

    Every bot commit of hypotheses.md is `git add` of the whole file, so an
    uncommitted edit already in the tree would ride along under a message
    that describes something else. That happened once - a diagnostic on entry
    8 was committed under "Pre-register entry 9". Not data loss; misattribution
    in a file whose value is that its history means what it says.
    """

    def _porcelain(self, monkeypatch, text):
        class Result:
            stdout = text

        monkeypatch.setattr(submissions, "git",
                            lambda *a, **k: Result())

    def test_a_dirty_log_is_refused_with_the_reason(self, monkeypatch):
        self._porcelain(monkeypatch, " M research/hypotheses.md\n")
        with pytest.raises(submissions.SubmissionError) as exc:
            submissions.require_clean_log()
        message = str(exc.value)
        assert "uncommitted changes" in message
        assert "research/hypotheses.md" in message
        assert "only commits its own append" in message

    def test_a_staged_edit_counts_as_dirty(self, monkeypatch):
        self._porcelain(monkeypatch, "M  research/hypotheses.md\n")
        with pytest.raises(submissions.SubmissionError):
            submissions.require_clean_log()

    def test_a_clean_log_passes(self, monkeypatch):
        self._porcelain(monkeypatch, "")
        submissions.require_clean_log()

    def test_submit_refuses_before_writing_anything(self, monkeypatch, tmp_path):
        """The check runs before the append, or the refusal would itself
        dirty the file it is refusing over."""
        order = []
        monkeypatch.setattr(submissions, "require_clean_log",
                            lambda path=None: order.append("check") or
                            (_ for _ in ()).throw(
                                submissions.SubmissionError("dirty")))
        monkeypatch.setattr(submissions, "append_entry",
                            lambda text, path=None: order.append("write"))
        log = tmp_path / "h.md"
        log.write_text("# Log\n", encoding="utf-8")
        with pytest.raises(submissions.SubmissionError, match="dirty"):
            submissions.pre_register(submission(name="fresh_idea"), log,
                                     do_commit=True)
        assert order == ["check"]

    @pytest.mark.parametrize("func", ["pre_register", "approve", "reject",
                                      "record_verdict"])
    def test_every_log_writer_guards(self, func):
        """All four bot paths that commit the log check first."""
        import inspect

        source = inspect.getsource(getattr(submissions, func))
        assert "require_clean_log" in source, f"{func} does not guard the log"

    def test_the_refusal_renders_as_a_message_in_discord(self):
        import inspect

        import research

        source = inspect.getsource(research.report_error)
        assert "SubmissionError" in source


class TestSuiteScoping:
    """Each submission is judged on the base suite plus its own test.

    Entry 9's attempt left a failing model-written test in tests/generated/.
    Because run_tests ran `tests/` wholesale, that file would have failed
    every later submission - and did fail the operator's plain `pytest tests/`.
    """

    def test_other_generated_tests_are_ignored(self):
        argv = submissions.test_command(
            own_test=sandbox.TEST_DIR / "test_mine.py")
        assert "tests/" in argv
        assert "--ignore=tests/generated" in argv
        assert "tests/generated/test_mine.py" in argv

    def test_the_own_test_comes_after_the_ignore(self):
        """pytest honours an explicit path over --ignore of its parent."""
        argv = submissions.test_command(own_test=sandbox.TEST_DIR / "test_mine.py")
        assert argv.index("--ignore=tests/generated") < \
            argv.index("tests/generated/test_mine.py")

    def test_without_an_own_test_only_the_base_suite_runs(self):
        argv = submissions.test_command()
        assert "--ignore=tests/generated" in argv
        assert not any("tests/generated/" in a for a in argv)

    def test_explicit_paths_bypass_the_scoping(self):
        argv = submissions.test_command(paths=["tests/test_rules.py"])
        assert "tests/test_rules.py" in argv
        assert "--ignore=tests/generated" not in argv

    def test_the_pipeline_passes_its_own_test_path(self):
        import inspect

        import research

        source = inspect.getsource(research._run_submission)
        assert "run.test_path" in source


class TestDeclaredContracts:
    """A generated strategy transcribes the size the submission stated.

    The first live run priced entry 5's reproduction at 1 contract against a
    description that says 4, which made the P&L incomparable to the entry it
    was reproducing. The size now travels on the class - and is still capped.
    """

    def strategy(self, contracts=None):
        import run_generated

        class Stub:
            name = "stub"

        if contracts is not None:
            Stub.contracts = contracts
        return Stub()

    def test_a_declared_size_is_used(self):
        import run_generated

        size, note = run_generated.resolve_contracts(self.strategy(4))
        assert size == 4
        assert "declared" in note

    def test_the_default_is_one(self):
        import run_generated

        size, note = run_generated.resolve_contracts(self.strategy())
        assert size == 1

    def test_a_size_above_the_internal_cap_is_clamped_and_reported(self):
        """Rule 4. The firm allows 40; a description saying so gets 5."""
        import run_generated

        size, note = run_generated.resolve_contracts(self.strategy(40))
        assert size == rules.POSITION_CAP == 5
        assert "CLAMPED" in note
        assert "40" in note, "the report must say what was asked for"

    @pytest.mark.parametrize("declared", [0, -3])
    def test_a_nonsense_size_falls_back_to_one(self, declared):
        import run_generated

        size, note = run_generated.resolve_contracts(self.strategy(declared))
        assert size == 1
        assert str(declared) in note

    def test_a_non_integer_size_falls_back_to_one(self):
        import run_generated

        size, note = run_generated.resolve_contracts(self.strategy("four"))
        assert size == 1
        assert "four" in note

    def test_an_explicit_override_wins(self):
        import run_generated

        size, note = run_generated.resolve_contracts(self.strategy(4), override=2)
        assert size == 2
        assert "command line" in note

    def test_the_size_reaches_the_verdict_block(self):
        import pandas as pd
        import run_generated

        result = run_generated.WalkforwardResult(
            name="x", folds=pd.DataFrame({"test_net_pnl": [1.0] * 7}),
            trades=pd.DataFrame({"net_pnl": [10.0]}),
            metrics={"sharpe": 1.0, "profit_factor": 1.5, "max_drawdown": -10.0,
                     "max_daily_loss": -5.0, "avg_duration_seconds": 600.0,
                     "microscalp_profit_pct": 0.0},
            pass_probability=0.3, blowups=0, profitable_folds=5,
            accepted=True, reasons=[], contracts=4,
            size_note="4 (declared by the strategy)")
        block = run_generated.verdict_block(result, 8)
        assert "4 contract(s)" in block
        assert "| Contracts | 4 (declared by the strategy) |" in block

    def test_the_embed_reports_the_size(self):
        """The number the operator reads must say which size produced it."""
        import inspect

        import research

        source = inspect.getsource(research._run_generated_walkforward)
        assert '"Contracts": result.size_note' in source

    def test_the_prompt_asks_for_transcription_not_inference(self):
        system = generate_mod.SYSTEM
        assert "`contracts`" in system
        assert "transcription of what the operator wrote" in system
        assert "do not compute it" in system.lower()

    def test_the_prompt_states_the_cap(self):
        assert "5-contract internal cap" in generate_mod.SYSTEM


class TestNameValidation:
    @pytest.mark.parametrize("name", ["rules", "engine", "base", "orb", "orb2",
                                      "store", "pretrade", "registry", "pytest"])
    def test_shadowing_names_are_refused(self, name):
        with pytest.raises(sandbox.SandboxError, match="shadow"):
            sandbox.validate_name(name)

    @pytest.mark.parametrize("name", ["", "a", "ab", "with-dash", "9lives",
                                      "with space", "x" * 41, "dots.here"])
    def test_malformed_names_are_refused(self, name):
        with pytest.raises(sandbox.SandboxError):
            sandbox.validate_name(name)

    def test_case_is_normalised_rather_than_refused(self):
        """`MyStrategy` becomes the module `mystrategy`, and the caller is
        told so by the return value rather than by a rejection."""
        assert sandbox.validate_name("VWAP_Fade") == "vwap_fade"
        assert sandbox.validate_name("  Gap_Close  ") == "gap_close"

    @pytest.mark.parametrize("name", ["vwap_fade", "gap_close", "my_strategy2"])
    def test_reasonable_names_pass(self, name):
        assert sandbox.validate_name(name) == name


class TestScrubbedEnvironment:
    def test_secrets_are_removed(self):
        env = sandbox.scrubbed_env({
            "PATH": "/bin", "ANTHROPIC_API_KEY": "sk-ant-real",
            "DISCORD_TOKEN": "tok", "DATABENTO_API_KEY": "db",
            "DESK_WEBHOOK_TOKEN": "w",
        })
        assert env == {"PATH": "/bin"}

    def test_the_test_subprocess_uses_it(self):
        """Not decoration: the screen catches mistakes, this removes the prize."""
        import inspect

        source = inspect.getsource(submissions.run_tests)
        assert "scrubbed_env" in source


# ---------------------------------------------------------------------------
# 2. the approval gate
# ---------------------------------------------------------------------------

def make_run(status=submissions.STATUS_AWAITING) -> submissions.SubmissionRun:
    run = submissions.SubmissionRun(submission=submission(), entry_number=99)
    run.status = status
    run.class_name = "MyStrategy"
    return run


class TestApprovalGate:
    def test_only_the_owner_may_approve(self, monkeypatch):
        called = []
        monkeypatch.setattr(submissions, "approve",
                            lambda run: called.append(run) or "abc1234")
        view = submit_view.ApprovalView(make_run(), OWNER)

        for user in (STRANGER, None):
            result = view.decide(user, "approve")
            assert result.acted is False
            assert result.message == submit_view.NOT_OWNER
        assert called == []

        assert view.decide(OWNER, "approve").acted is True
        assert len(called) == 1

    def test_unset_owner_means_nobody_can_approve(self):
        view = submit_view.ApprovalView(make_run(), None)
        assert view.decide(OWNER, "approve").acted is False

    def test_only_a_passing_submission_can_be_approved(self, monkeypatch):
        monkeypatch.setattr(submissions, "approve", lambda run: "abc1234")
        for status in (submissions.STATUS_PENDING, submissions.STATUS_FAILED):
            view = submit_view.ApprovalView(make_run(status), OWNER)
            result = view.decide(OWNER, "approve")
            assert result.acted is False
            assert "tests passed" in result.message

    def test_approval_happens_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            submissions, "approve",
            lambda run: (calls.append(1), setattr(run, "status",
                                                  submissions.STATUS_APPROVED),
                         "abc1234")[-1])
        view = submit_view.ApprovalView(make_run(), OWNER)
        assert view.decide(OWNER, "approve").acted is True
        again = view.decide(OWNER, "approve")
        assert again.acted is False and "Already" in again.message
        assert len(calls) == 1

    def test_buttons_disable_after_a_decision(self, monkeypatch):
        monkeypatch.setattr(submissions, "approve", lambda run: "abc1234")
        run = make_run()
        view = submit_view.ApprovalView(run, OWNER)
        assert not any(s["disabled"] for s in view.button_specs())
        run.status = submissions.STATUS_APPROVED
        assert all(s["disabled"] for s in view.button_specs())

    def test_approve_failure_is_surfaced_not_swallowed(self, monkeypatch):
        def boom(run):
            raise submissions.SubmissionError("registry refused")

        monkeypatch.setattr(submissions, "approve", boom)
        result = submit_view.ApprovalView(make_run(), OWNER).decide(OWNER, "approve")
        assert result.acted is False and "registry refused" in result.message


class TestRegistryAdd:
    def test_add_takes_no_status_argument(self):
        """A status argument would be a complete bypass of promote()."""
        import inspect

        params = list(inspect.signature(Registry.add).parameters)
        assert params == ["self", "name", "class_path", "hypothesis_entry"]
        assert "status" not in params
        assert "force" not in params

    def test_add_refuses_an_entry_that_does_not_exist(self):
        reg = Registry.load()
        with pytest.raises(RegistryError, match="no entry"):
            reg.add("brand_new_thing", "brand_new_thing:X", 9999)

    def test_add_refuses_a_duplicate_name(self):
        reg = Registry.load()
        with pytest.raises(RegistryError, match="already in the registry"):
            reg.add("orb", "orb:OpeningRangeBreakout", 1)

    def test_set_verdict_does_not_change_status(self, tmp_path):
        """A verdict is evidence; acting on it is promote()'s job."""
        import inspect

        source = inspect.getsource(Registry.set_verdict)
        assert "status=" not in source

    def test_promote_still_refuses_a_rejected_strategy(self):
        reg = Registry.load()
        with pytest.raises(Exception, match="terminal|rejected"):
            reg.promote("orb2", "paper")


# ---------------------------------------------------------------------------
# 3. verdict before embed
# ---------------------------------------------------------------------------

class TestVerdictOrdering:
    def test_record_verdict_commits_before_returning(self, monkeypatch):
        """The hash the embed prints cannot exist until the commit does."""
        # The clean-log guard is tested on its own; here it would read the
        # real working tree and refuse whenever the developer has an
        # uncommitted edit to the log - an environment-dependent failure.
        monkeypatch.setattr(submissions, "require_clean_log", lambda path=None: None)
        order = []
        monkeypatch.setattr(submissions, "set_entry_status",
                            lambda n, s, path=None: order.append(f"status:{s}"))
        monkeypatch.setattr(submissions, "append_to_entry",
                            lambda n, t, path=None: order.append("write"))
        monkeypatch.setattr(submissions, "commit",
                            lambda paths, msg: order.append("commit") or "dead15")

        class FakeRecord:
            status = "testing"

        class FakeRegistry:
            @staticmethod
            def load():
                return FakeRegistry()

            def set_verdict(self, name, verdict, verdict_commit=None):
                order.append(f"registry:{verdict_commit}")

            def get(self, name):
                return FakeRecord()

            def promote(self, name, to_status):
                order.append(f"promote:{to_status}")

        monkeypatch.setattr(submissions, "Registry", FakeRegistry)

        commit = submissions.record_verdict("x", 5, "### Verdict: REJECTED", "s")
        assert commit == "dead15"
        assert order.index("write") < order.index("commit")
        assert "registry:dead15" in order
        assert order.count("commit") == 2, "verdict commit + hash follow-up"
        # A REJECTED verdict must move the registry too, or the log and the
        # registry disagree - which is how this was found live.
        assert "promote:rejected" in order

    def test_an_accepted_verdict_does_not_auto_promote_to_paper(self, monkeypatch):
        """`paper` unblocks the desk bot; reaching it is an act, not a side
        effect of a background job finishing."""
        monkeypatch.setattr(submissions, "require_clean_log", lambda path=None: None)
        promotions = []
        monkeypatch.setattr(submissions, "set_entry_status",
                            lambda n, s, path=None: None)
        monkeypatch.setattr(submissions, "append_to_entry",
                            lambda n, t, path=None: None)
        monkeypatch.setattr(submissions, "commit", lambda paths, msg: "cafe01")

        class FakeRecord:
            status = "testing"

        class FakeRegistry:
            @staticmethod
            def load():
                return FakeRegistry()

            def set_verdict(self, *a, **k):
                pass

            def get(self, name):
                return FakeRecord()

            def promote(self, name, to_status):
                promotions.append(to_status)

        monkeypatch.setattr(submissions, "Registry", FakeRegistry)
        submissions.record_verdict("x", 5, "### Verdict: ACCEPTED", "s")
        assert promotions == [], "an accepted walk-forward promoted itself"

    def test_the_posted_hash_is_the_one_written_to_the_log(self, monkeypatch):
        monkeypatch.setattr(submissions, "require_clean_log", lambda path=None: None)
        written = []
        monkeypatch.setattr(submissions, "set_entry_status",
                            lambda n, s, path=None: None)
        monkeypatch.setattr(submissions, "append_to_entry",
                            lambda n, t, path=None: written.append(t))
        monkeypatch.setattr(submissions, "commit", lambda paths, msg: "beef99")

        class FakeRegistry:
            @staticmethod
            def load():
                return FakeRegistry()

            def set_verdict(self, *a, **k):
                pass

        monkeypatch.setattr(submissions, "Registry", FakeRegistry)
        commit = submissions.record_verdict("x", 5, "### Verdict: ACCEPTED", "s")
        assert any(commit in w for w in written), \
            "the commit hash must be recorded in the entry itself"

    def test_the_command_freezes_before_it_posts(self):
        """Static check on the ordering in research.py.

        A runtime test would need a gateway. What matters is that
        record_verdict is called before edit_original_response posts the
        embed, and that is visible in the source.
        """
        import inspect

        import research

        source = inspect.getsource(research._run_generated_walkforward)
        freeze = source.index("record_verdict")
        # The embed is built after the freeze, and posted after that.
        assert freeze < source.index("embed = _embed"), \
            "the embed is built before the verdict is frozen"
        assert "Frozen first" in source or "frozen" in source.lower()

    def test_verdict_block_states_the_cost_model(self):
        """Rule 10: no report on a frictionless fill model."""
        import run_generated

        assert run_generated.BASE_SLIPPAGE_TICKS == 2.0
        assert run_generated.COMMISSION_PER_SIDE == rules.COMMISSION_PER_SIDE

    def test_commission_is_read_from_rules_not_restated(self):
        """The $1.25 that every pre-2026-09-11 verdict carried was a literal
        here. The verified figure is imported, so it cannot drift."""
        import inspect

        import run_generated

        source = inspect.getsource(run_generated)
        assert "COMMISSION_PER_SIDE = rules.COMMISSION_PER_SIDE" in source
        assert "COMMISSION_PER_SIDE = 1.25" not in source
        assert "COMMISSION_PER_SIDE = 0.5" not in source

    def test_a_historical_verdict_can_be_reproduced_by_naming_its_commission(self):
        """``run(..., commission=rules.ASSUMED_COMMISSION_PER_SIDE)`` and the
        CLI's ``--commission`` exist so a pre-correction verdict is one flag
        away. The bot never passes either: from chat the base case is fixed."""
        import inspect

        import research
        import run_generated

        assert "commission" in inspect.signature(run_generated.run).parameters
        assert "--commission" in inspect.getsource(run_generated.main)
        bot_call = inspect.getsource(research._run_generated_walkforward)
        assert "commission=" not in bot_call

    def test_the_embed_states_the_commission_it_ran_at(self):
        import inspect

        import research

        source = inspect.getsource(research._run_generated_walkforward)
        assert "$1.25/side commission" not in source
        assert "result.commission_per_side" in source

    def test_acceptance_is_rule_13_and_lives_in_one_place(self):
        import run_generated

        assert run_generated.MIN_PROFITABLE_FOLDS == 4
        assert run_generated.TOTAL_FOLDS == 7


# ---------------------------------------------------------------------------
# the log: rule 12 ordering
# ---------------------------------------------------------------------------

class TestPreRegistration:
    def test_the_entry_is_the_operators_words(self):
        sub = submission()
        text = submissions.render_entry(7, sub)
        assert sub.mechanism in text
        assert sub.counterparty in text
        assert sub.kill_criteria in text
        assert "PROPOSED" in text

    def test_a_generated_draft_never_replaces_the_pre_registration(self):
        sub = submission()
        text = submissions.render_entry(7, sub, draft="A tidier paraphrase.")
        assert sub.mechanism in text
        assert "not the pre-registration" in text

    def test_missing_fields_are_refused(self):
        for field in ("mechanism", "counterparty", "kill_criteria"):
            sub = submission(**{field: "   "})
            assert field in sub.missing()
            with pytest.raises(submissions.SubmissionError, match="required"):
                submissions.pre_register(sub, do_commit=False)

    def test_a_duplicate_registry_name_is_refused(self):
        """`london_1x` is a registry name but not a reserved module name, so
        this reaches the registry check rather than the shadow check."""
        with pytest.raises(submissions.SubmissionError, match="already"):
            submissions.pre_register(submission(name="london_1x"),
                                     do_commit=False)

    def test_a_shadowing_name_is_refused_before_the_registry_is_consulted(self):
        with pytest.raises(sandbox.SandboxError, match="shadow"):
            submissions.pre_register(submission(name="orb2"), do_commit=False)

    def test_pre_register_writes_the_entry_before_any_code(self, tmp_path):
        log = tmp_path / "hypotheses.md"
        log.write_text("# Log\n\n## 1. Something — REJECTED\n\nbody\n",
                       encoding="utf-8")
        run = submissions.pre_register(submission(name="fresh_idea"), log,
                                       do_commit=False)
        assert run.entry_number == 2
        body = log.read_text(encoding="utf-8")
        assert "## 2. fresh idea — PROPOSED" in body
        # No code exists at this point, by construction.
        assert not (sandbox.STRATEGY_DIR / "fresh_idea.py").exists()

    def test_entry_is_inserted_before_the_template(self, tmp_path):
        log = tmp_path / "h.md"
        log.write_text("# Log\n\n## 1. A — REJECTED\n\nb\n\n"
                       "## Template for new entries\n\nstuff\n", encoding="utf-8")
        submissions.append_entry(submissions.render_entry(2, submission()), log)
        body = log.read_text(encoding="utf-8")
        assert body.index("## 2.") < body.index("## Template for new entries")

    def test_set_entry_status_rewrites_only_the_heading(self, tmp_path):
        log = tmp_path / "h.md"
        log.write_text("## 1. A — PROPOSED\n\nPROPOSED appears in the body too.\n",
                       encoding="utf-8")
        submissions.set_entry_status(1, "REJECTED", log)
        body = log.read_text(encoding="utf-8")
        assert "## 1. A — REJECTED" in body
        assert "PROPOSED appears in the body too." in body


class TestResponseParsing:
    def test_the_three_artefacts_are_split_out(self):
        raw = (f"{generate_mod.STRATEGY_MARK}\n{CLEAN_STRATEGY}\n"
               f"{generate_mod.TEST_MARK}\n{CLEAN_TEST}\n"
               f"{generate_mod.ENTRY_MARK}\n## draft\n"
               f"{generate_mod.END_MARK}\n")
        result = generate_mod.parse_response(raw, "my_strategy")
        assert "class MyStrategy" in result.strategy_source
        assert "def test_placeholder" in result.test_source
        assert result.entry_markdown == "## draft"
        assert result.class_name == "MyStrategy"

    def test_code_fences_are_stripped(self):
        raw = (f"{generate_mod.STRATEGY_MARK}\n```python\n{CLEAN_STRATEGY}```\n"
               f"{generate_mod.TEST_MARK}\n```python\n{CLEAN_TEST}```\n"
               f"{generate_mod.ENTRY_MARK}\nx\n{generate_mod.END_MARK}")
        result = generate_mod.parse_response(raw, "my_strategy")
        assert not result.strategy_source.startswith("```")
        assert sandbox.screen_source(result.strategy_source).ok

    def test_a_truncated_response_fails_loudly(self):
        raw = f"{generate_mod.STRATEGY_MARK}\n{CLEAN_STRATEGY}\n"
        with pytest.raises(generate_mod.GenerationError, match="cut short"):
            generate_mod.parse_response(raw, "my_strategy")

    def test_a_strategy_with_no_class_is_refused(self):
        raw = (f"{generate_mod.STRATEGY_MARK}\nx = 1\n"
               f"{generate_mod.TEST_MARK}\ny\n"
               f"{generate_mod.ENTRY_MARK}\nz\n{generate_mod.END_MARK}")
        with pytest.raises(generate_mod.GenerationError, match="no class"):
            generate_mod.parse_response(raw, "my_strategy")

    def test_generate_refuses_without_an_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(generate_mod.GenerationError, match="ANTHROPIC_API_KEY"):
            generate_mod.generate(submission(), api_key="")

    def test_the_prompt_carries_the_interface_and_the_rules(self):
        text = generate_mod.build_prompt(submission())
        assert "class Strategy" in text
        assert "Intraday CME Futures Trading Bot" in text or "Hard rules" in text
        assert "generate_signals" in text

    def test_tls_verification_is_not_weakened(self):
        """The truststore workaround must never become `verify=False`.

        anthropic 1.x installs truststore, which on Python 3.14 / Windows
        recurses forever in ssl.verify_mode and makes every request fail as
        APIConnectionError. The fix supplies a certifi-backed context so the
        handshake never reaches truststore. The tempting "fix" for a TLS error
        is to turn verification off, so the properties are asserted here
        rather than trusted to the comment.
        """
        import ast
        import inspect
        import ssl
        import textwrap

        # AST, not a substring scan: the docstring above deliberately contains
        # the phrase it warns against, and a text check would trip on prose
        # while missing `verify = False` written with spaces.
        tree = ast.parse(textwrap.dedent(inspect.getsource(generate_mod._http_client)))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in ("verify",
                                                              "check_hostname"):
                assert not (isinstance(node.value, ast.Constant)
                            and node.value.value is False), \
                    f"{node.arg}=False disables TLS verification"
            if isinstance(node, ast.Attribute):
                assert node.attr != "CERT_NONE", "CERT_NONE disables verification"

        client = generate_mod._http_client()
        if client is None:
            pytest.skip("certifi unavailable; the SDK builds its own client")
        context = ssl.create_default_context()
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_the_model_is_not_asked_for_an_opinion(self):
        system = generate_mod.SYSTEM.lower()
        for banned in ("confidence", "how likely", "rate the", "score the"):
            assert banned not in system
        assert "do not claim the strategy works" in system
