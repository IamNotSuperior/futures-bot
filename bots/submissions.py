"""The /submit pipeline: pre-register, generate, test, approve, walk forward.

The order of operations here is the product, so it is worth stating plainly.

1. **Pre-register.** The operator's mechanism, counterparty and kill criteria
   are appended to ``research/hypotheses.md`` as a PROPOSED entry and
   committed - **before** the API is called and before any code exists.
   CLAUDE.md rule 12: *"Every strategy gets a hypothesis entry before any code
   is written."* Writing the entry after generation would make the claim a
   description of whatever was produced rather than a prediction, which is the
   thing the log exists to prevent.
2. **Generate**, into the sandbox only (``bots/sandbox.py``).
3. **Test.** The full suite, in a subprocess with the secrets stripped out of
   the environment. A failure leaves the strategy unregistered.
4. **Register** at ``proposed``. Not higher - ``Registry.add`` cannot express
   anything else.
5. **Approve** moves it to ``testing`` and commits. Rejection is terminal, as
   it is everywhere else in this repo.
6. **Walk forward**, and write the verdict into the entry, commit it, and
   record the commit hash *before* the result is posted anywhere.

Step 6's ordering is not cosmetic. If the embed went out first, there would be
a window in which a number had been seen by a human but not yet frozen in the
log - and the log's whole claim is that a verdict is recorded before it is
acted on and never revised afterwards.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

sys.path.insert(0, str(PROJECT_ROOT / "bots"))

import generate as generate_mod  # noqa: E402
import sandbox  # noqa: E402
from registry import Registry, parse_hypotheses  # noqa: E402

log = logging.getLogger("bot.submissions")

HYPOTHESES = PROJECT_ROOT / "research" / "hypotheses.md"
TEMPLATE_MARK = "## Template for new entries"

STATUS_PENDING = "pending"
STATUS_FAILED = "failed"
STATUS_AWAITING = "awaiting_approval"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"


class SubmissionError(RuntimeError):
    """Shown to the user as a message rather than a traceback."""


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True, check=check)


def commit(paths: list[Path], message: str) -> str:
    """Stage exactly ``paths`` and commit. Returns the short hash.

    Named paths only - never ``git add -A``. A pipeline that staged everything
    would sweep whatever the operator happened to be editing into a commit
    whose message says it froze a hypothesis.
    """
    relative = [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                for p in paths if p.exists()]
    if not relative:
        raise SubmissionError("nothing to commit")
    git("add", "--", *relative)
    staged = git("diff", "--cached", "--name-only").stdout.strip()
    if not staged:
        return git("rev-parse", "--short", "HEAD").stdout.strip()
    git("commit", "-m", message)
    return git("rev-parse", "--short", "HEAD").stdout.strip()


# ---------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------

def next_entry_number(path: Path = HYPOTHESES) -> int:
    entries = parse_hypotheses(path)
    return (max(entries) + 1) if entries else 1


def render_entry(number: int, submission: generate_mod.Submission,
                 draft: str = "", today: date_type | None = None) -> str:
    """The PROPOSED entry, from the operator's own words.

    The model's draft is appended under its own heading and clearly labelled.
    It never replaces the operator's text: the pre-registration is what the
    operator claimed, and a paraphrase - however good - is a different claim.
    """
    import pandas as pd  # noqa: PLC0415
    import rules  # noqa: PLC0415

    day = today or pd.Timestamp.now(tz=rules.ET).date()
    title = submission.name.replace("_", " ")
    out = [
        "",
        f"## {number}. {title} — PROPOSED",
        "",
        f"**Date:** {day}",
        f"**Submitted via:** `/submit` in Discord",
        f"**Code:** `strategies/generated/{submission.name}.py` "
        f"(generated; not yet written at the time this entry was committed)",
        "",
        "### Mechanism claimed",
        "",
        submission.mechanism.strip(),
        "",
        "**Who is on the other side:**",
        "",
        submission.counterparty.strip(),
        "",
        "### Kill criteria, pre-registered",
        "",
        submission.kill_criteria.strip(),
        "",
    ]
    if submission.description.strip():
        out += ["### Description as submitted", "",
                submission.description.strip(), ""]
    if draft.strip():
        out += ["### Drafted expansion (generated, not the pre-registration)",
                "",
                "The operator's three statements above are the pre-registration.",
                "This expansion was drafted by the model afterwards and is",
                "recorded for readability only.",
                "", draft.strip(), ""]
    out += ["### Verdict", "",
            "Not yet run. To be filled in by the walk-forward, with the commit",
            "hash recorded in a follow-up commit.", "", "---"]
    return "\n".join(out)


def append_entry(text: str, path: Path = HYPOTHESES) -> None:
    """Insert before the template block, or append when there is none."""
    path = Path(path)
    body = path.read_text(encoding="utf-8")
    if TEMPLATE_MARK in body:
        head, _, tail = body.partition(TEMPLATE_MARK)
        body = head.rstrip() + "\n" + text + "\n\n" + TEMPLATE_MARK + tail
    else:
        body = body.rstrip() + "\n" + text + "\n"
    path.write_text(body, encoding="utf-8")


def set_entry_status(number: int, status: str, path: Path = HYPOTHESES) -> None:
    """Rewrite one entry's heading status. Headings are the log's authority."""
    path = Path(path)
    body = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(## {number}\. .*?) — (\w+)\s*$", re.MULTILINE)
    if not pattern.search(body):
        raise SubmissionError(f"no entry {number} heading to update")
    path.write_text(pattern.sub(rf"\1 — {status}", body, count=1),
                    encoding="utf-8")


def append_to_entry(number: int, text: str, path: Path = HYPOTHESES) -> None:
    """Append ``text`` to the end of entry ``number``'s body."""
    path = Path(path)
    body = path.read_text(encoding="utf-8")
    lines = body.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^## {number}\. ", line):
            start = i
        elif start is not None and line.startswith("## "):
            lines.insert(i, text + "\n")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    if start is None:
        raise SubmissionError(f"no entry {number} to append to")
    path.write_text(body.rstrip() + "\n\n" + text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

@dataclass
class SubmissionRun:
    """One trip through the pipeline. Held in memory by the bot."""

    submission: generate_mod.Submission
    entry_number: int
    status: str = STATUS_PENDING
    strategy_path: Path | None = None
    test_path: Path | None = None
    class_name: str = ""
    draft_entry: str = ""
    pretrade_commit: str = ""
    approve_commit: str = ""
    test_output: str = ""
    problems: list = field(default_factory=list)

    @property
    def registry_name(self) -> str:
        return self.submission.name


def pre_register(submission: generate_mod.Submission,
                 path: Path = HYPOTHESES, do_commit: bool = True) -> SubmissionRun:
    """Rule 12: the entry, committed, before any code exists."""
    sandbox.validate_name(submission.name)
    missing = submission.missing()
    if missing:
        raise SubmissionError(
            f"these fields are required and were empty: {', '.join(missing)}. "
            f"An idea that cannot name its counterparty is a pattern, not a "
            f"hypothesis."
        )
    reg = Registry.load()
    if submission.name in reg.names():
        raise SubmissionError(
            f"`{submission.name}` is already in the registry. A new idea needs "
            f"a new name; a verdict is never revised."
        )

    number = next_entry_number(path)
    append_entry(render_entry(number, submission), path)
    run = SubmissionRun(submission=submission, entry_number=number)
    if do_commit:
        run.pretrade_commit = commit(
            [path],
            f"Pre-register entry {number}: {submission.name}\n\n"
            f"Submitted through /submit. The mechanism, counterparty and kill\n"
            f"criteria above are the operator's own words, committed before\n"
            f"any code was generated - CLAUDE.md rule 12.\n\n"
            f"No implementation exists at this commit.",
        )
    return run


def write_generated(run: SubmissionRun, result: generate_mod.Generated) -> None:
    """Screen and write both files. Raises SandboxError on refusal."""
    name = run.submission.name
    run.strategy_path = sandbox.safe_write(
        f"strategies/generated/{name}.py", result.strategy_source)
    run.test_path = sandbox.safe_write(
        f"tests/generated/test_{name}.py", result.test_source)
    run.class_name = result.class_name
    run.draft_entry = result.entry_markdown


def run_tests(paths: list[str] | None = None, timeout: int = 900) -> tuple[bool, str]:
    """The full suite, in a subprocess with the secrets removed.

    Secrets are stripped rather than trusted to the AST screen: the screen
    catches mistakes, and this makes the consequence of one that slips through
    an empty variable instead of a live API key.
    """
    env = sandbox.scrubbed_env(dict(os.environ))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe"),
               "-m", "pytest", *(paths or ["tests/"]), "-q", "--no-header"]
    try:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"the test suite did not finish within {timeout}s"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def tail(text: str, limit: int = 1400) -> str:
    return text if len(text) <= limit else "...\n" + text[-limit:]


def register_proposed(run: SubmissionRun) -> None:
    """Only reached when the suite passed. Registers at ``proposed``."""
    reg = Registry.load()
    reg.add(run.registry_name,
            class_path=f"{run.submission.name}:{run.class_name}",
            hypothesis_entry=run.entry_number)
    run.status = STATUS_AWAITING


def approve(run: SubmissionRun) -> str:
    """Freeze the entry and move to ``testing``. Returns the commit hash."""
    if run.status != STATUS_AWAITING:
        raise SubmissionError(
            f"`{run.registry_name}` is {run.status}, not awaiting approval"
        )
    set_entry_status(run.entry_number, "PROPOSED")
    reg = Registry.load()
    reg.promote(run.registry_name, "testing")
    paths = [HYPOTHESES, PROJECT_ROOT / "strategies" / "registry.yaml"]
    if run.strategy_path:
        paths.append(run.strategy_path)
    if run.test_path:
        paths.append(run.test_path)
    run.approve_commit = commit(
        paths,
        f"Approve entry {run.entry_number}: {run.registry_name} to testing\n\n"
        f"The implementation and its test are frozen at this commit. The\n"
        f"pre-registration was committed before any code existed; this commit\n"
        f"is the code that will be walked forward against it.\n\n"
        f"No verdict yet - status is `testing`, not `paper`.",
    )
    run.status = STATUS_APPROVED
    return run.approve_commit


def reject(run: SubmissionRun) -> None:
    """Terminal, as everywhere else. The entry stays in the log."""
    set_entry_status(run.entry_number, "REJECTED")
    append_to_entry(
        run.entry_number,
        "### Verdict: REJECTED at review\n\n"
        "Rejected by the operator before any walk-forward was run. The entry\n"
        "stays in the log: the point of the log is to make it expensive to\n"
        "quietly re-test the same idea until it passes.",
    )
    commit([HYPOTHESES], f"Reject entry {run.entry_number}: "
                         f"{run.registry_name}\n\n"
                         f"Rejected at review, before any walk-forward. A\n"
                         f"rejected entry stays in the log with its verdict\n"
                         f"intact.")
    run.status = STATUS_REJECTED


def is_generated(name: str) -> bool:
    """True when ``name`` has a module under ``strategies/generated/``."""
    return (sandbox.STRATEGY_DIR / f"{name}.py").exists()


def record_verdict(name: str, entry_number: int, verdict_block: str,
                   summary: str) -> str:
    """Write the verdict into the log and the registry, then commit.

    **Called before the result embed is posted, never after.** Returns the
    commit hash, which the caller puts in the embed - so a posted result can
    always be pointed at the commit that froze it.

    Two commits, matching the log's existing convention: the verdict, then a
    one-line follow-up recording its own hash (which cannot be known until the
    first commit exists).
    """
    set_entry_status(entry_number, "REJECTED" if "REJECTED" in verdict_block
                     else "ACCEPTED")
    append_to_entry(entry_number, verdict_block)
    first = commit([HYPOTHESES], f"Entry {entry_number} verdict: {name}")

    append_to_entry(entry_number, f"**Verdict commit:** `{first}`")
    reg = Registry.load()
    reg.set_verdict(name, summary, verdict_commit=first)
    commit([HYPOTHESES, PROJECT_ROOT / "strategies" / "registry.yaml"],
           f"Record entry {entry_number}'s verdict commit hash")
    return first
