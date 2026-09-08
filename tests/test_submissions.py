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

import sys
from pathlib import Path

import pytest

import generate as generate_mod
import sandbox
import submissions
import submit_view
from registry import Registry, RegistryError

OWNER = 491708157341466624
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
        order = []
        monkeypatch.setattr(submissions, "set_entry_status",
                            lambda n, s, path=None: order.append(f"status:{s}"))
        monkeypatch.setattr(submissions, "append_to_entry",
                            lambda n, t, path=None: order.append("write"))
        monkeypatch.setattr(submissions, "commit",
                            lambda paths, msg: order.append("commit") or "dead15")

        class FakeRegistry:
            @staticmethod
            def load():
                return FakeRegistry()

            def set_verdict(self, name, verdict, verdict_commit=None):
                order.append(f"registry:{verdict_commit}")

        monkeypatch.setattr(submissions, "Registry", FakeRegistry)

        commit = submissions.record_verdict("x", 5, "### Verdict: REJECTED", "s")
        assert commit == "dead15"
        assert order.index("write") < order.index("commit")
        assert "registry:dead15" in order
        assert order.count("commit") == 2, "verdict commit + hash follow-up"

    def test_the_posted_hash_is_the_one_written_to_the_log(self, monkeypatch):
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
        assert run_generated.COMMISSION_PER_SIDE == 1.25

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

    def test_the_model_is_not_asked_for_an_opinion(self):
        system = generate_mod.SYSTEM.lower()
        for banned in ("confidence", "how likely", "rate the", "score the"):
            assert banned not in system
        assert "do not claim the strategy works" in system
