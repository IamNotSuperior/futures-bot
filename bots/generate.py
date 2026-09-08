"""Ask Claude for a Strategy implementation, a test, and a draft log entry.

One API call per submission. The prompt carries the operator's submission plus
three files that define what a correct answer looks like: ``strategies/base.py``
(the interface), ``CLAUDE.md`` (the hard rules), and one existing strategy as a
worked example.

Why delimited blocks rather than structured outputs
---------------------------------------------------
The three artefacts are *source files*. Returning them inside JSON means every
newline, quote and backslash in the generated code has to survive an escaping
round trip, and a single mis-escape corrupts a file that is then executed.
Sentinel-delimited blocks have no escaping layer to get wrong, and a truncated
response fails loudly at the parse rather than producing a file that is
syntactically valid but silently cut short.

What the model is *not* asked to do
-----------------------------------
It is not asked whether the strategy is good, whether it will be profitable,
or how confident it is. It writes an implementation of the operator's stated
mechanism and a test for it. The verdict comes from the walk-forward, and
CLAUDE.md rule 13 is explicit that in-sample results are never evidence.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("bot.generate")

MODEL = "claude-opus-5"
MAX_TOKENS = 32_000

#: Files that show the model what a correct answer looks like.
CONTEXT_FILES = (
    ("strategies/base.py", "The Strategy interface every strategy implements."),
    ("CLAUDE.md", "The project's hard rules. Non-negotiable."),
    ("strategies/orb.py", "An existing strategy, as a worked example of house style."),
)

STRATEGY_MARK = "===STRATEGY==="
TEST_MARK = "===TEST==="
ENTRY_MARK = "===ENTRY==="
END_MARK = "===END==="


class GenerationError(RuntimeError):
    """The API call or the parse failed. The message is shown to the user."""


@dataclass(frozen=True)
class Submission:
    """What the operator typed into the modal, plus any attachment."""

    name: str
    mechanism: str
    counterparty: str
    kill_criteria: str
    description: str = ""
    pine_source: str = ""

    def missing(self) -> list[str]:
        return [field for field in ("mechanism", "counterparty", "kill_criteria")
                if not getattr(self, field, "").strip()]

    def as_prompt_block(self) -> str:
        parts = [
            f"Strategy name: {self.name}",
            "",
            "## Mechanism claimed (the operator's words)",
            self.mechanism.strip(),
            "",
            "## Who is on the other side of the trade (the operator's words)",
            self.counterparty.strip(),
            "",
            "## Kill criteria, pre-registered (the operator's words)",
            self.kill_criteria.strip(),
        ]
        if self.description.strip():
            parts += ["", "## Description supplied with the submission",
                      self.description.strip()]
        if self.pine_source.strip():
            parts += ["", "## Attached Pine script", "```",
                      self.pine_source.strip()[:20_000], "```"]
        return "\n".join(parts)


@dataclass(frozen=True)
class Generated:
    strategy_source: str
    test_source: str
    entry_markdown: str
    class_name: str
    usage: dict


SYSTEM = """You write trading strategy code for a disciplined research
repository. You will be given a strategy submission, the Strategy interface it
must implement, the project's hard rules, and one existing strategy as an
example of house style.

Produce exactly three artefacts, each between its markers, in this order:

===STRATEGY===
<the complete contents of a Python module implementing the strategy>
===TEST===
<the complete contents of a pytest module testing it>
===ENTRY===
<a draft hypotheses.md entry in the project's format>
===END===

Hard requirements for the STRATEGY module:
- It subclasses Strategy from `base` and implements generate_signals.
- `import base`, `import rules`, `import pandas as pd`, `import numpy as np`
  are the ONLY imports permitted. Import by bare name - this repository has no
  packages. Do not import os, sys, pathlib, subprocess, importlib or anything
  that touches the filesystem, the network or the process.
- It reads no files and writes no files.
- Signals only. It must not size positions, apply a stop loss in dollars,
  decide a contract count, or enforce a time cutoff - the execution layer owns
  all of that and reads rules.py. There is nowhere in the interface for a
  strategy to express a risk decision, and that is deliberate.
- No lookahead. True at timestamp T in an entry column means "act at the open
  of bar T, decided using only bars strictly before T". A rolling calculation
  must be shifted so the value used on bar T was knowable at T-1.
- Set a class attribute `name` matching the strategy name given to you.

Hard requirements for the TEST module:
- Plain pytest, no fixtures from outside the file, `import pytest`,
  `import pandas as pd`, `import numpy as np` and the strategy module by bare
  name only.
- It must include a test that asserts no lookahead: construct bars where a
  future move would change an earlier signal if the strategy peeked, and
  assert the signal does not change.
- It must include a test that the signals frame satisfies the interface
  (the four required boolean columns, no bar carrying both a long and a short
  entry).

Hard requirements for the ENTRY:
- Use the operator's own mechanism, counterparty and kill criteria verbatim
  where given. Do not soften them, do not invent a counterparty, and do not
  add a verdict - the verdict comes from the walk-forward later.
- Do not claim the strategy works, is promising, or is likely to be
  profitable. State what is claimed and how it will be falsified.

Write nothing outside the markers. No preamble, no explanation, no commentary
after ===END===."""


def read_context() -> str:
    blocks = []
    for relative, why in CONTEXT_FILES:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        blocks.append(f"### {relative} - {why}\n```python\n"
                      f"{path.read_text(encoding='utf-8')}\n```")
    return "\n\n".join(blocks)


def build_prompt(submission: Submission) -> str:
    return "\n\n".join([
        "# The submission", submission.as_prompt_block(),
        "# Reference material", read_context(),
        f"Now produce the three artefacts for `{submission.name}`. The strategy "
        f"module will be written to strategies/generated/{submission.name}.py "
        f"and the test to tests/generated/test_{submission.name}.py, so the "
        f"test must import the strategy as `import {submission.name}`.",
    ])


def parse_response(text: str, name: str) -> Generated:
    """Split the three artefacts out, or say precisely what was missing."""
    def _between(start: str, end: str) -> str:
        pattern = re.escape(start) + r"\s*(.*?)\s*" + re.escape(end)
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else ""

    strategy = _between(STRATEGY_MARK, TEST_MARK)
    test = _between(TEST_MARK, ENTRY_MARK)
    entry = _between(ENTRY_MARK, END_MARK)

    missing = [label for label, value in
               (("strategy", strategy), ("test", test), ("entry", entry))
               if not value.strip()]
    if missing:
        raise GenerationError(
            f"the response did not contain {', '.join(missing)}. This usually "
            f"means the output was cut short. Nothing was written."
        )

    strategy = _strip_fence(strategy)
    test = _strip_fence(test)

    classes = re.findall(r"^class\s+(\w+)\s*\(", strategy, re.MULTILINE)
    if not classes:
        raise GenerationError("the generated strategy defines no class")
    # The Strategy subclass is the last class defined - a params dataclass, if
    # the model wrote one, comes first, following the house style in orb.py.
    return Generated(strategy, test, entry.strip(), classes[-1], {})


def _strip_fence(block: str) -> str:
    block = block.strip()
    if block.startswith("```"):
        block = re.sub(r"^```[a-zA-Z]*\n", "", block)
        block = re.sub(r"\n```$", "", block)
    return block.strip()


def _http_client():
    """An HTTP client that does not route TLS through ``truststore``.

    ``anthropic`` 1.x installs ``truststore``, which reads the Windows system
    certificate store. On this machine - Python 3.14 on Windows - truststore's
    ``_set_ssl_context_verify_mode`` recurses into ``ssl.py``'s ``verify_mode``
    property without terminating, so *every* request dies with a
    ``RecursionError`` that the SDK reports as ``APIConnectionError``. The same
    request through ``urllib`` succeeds, which is how it was isolated.

    The fix supplies an explicit certifi-backed context, so the handshake never
    reaches truststore. **Verification is not weakened**: the context is
    ``ssl.create_default_context``, which is ``CERT_REQUIRED`` with hostname
    checking on - a test asserts both, because "fix the TLS error" is exactly
    the change that tends to arrive as ``verify=False``.

    Returns None when certifi is unavailable, which lets the SDK build its own
    client - correct on a machine where truststore works.
    """
    try:
        import certifi  # noqa: PLC0415
        from anthropic import DefaultHttpxClient  # noqa: PLC0415
    except ImportError:  # pragma: no cover - certifi ships with httpx2
        return None
    import ssl  # noqa: PLC0415

    return DefaultHttpxClient(verify=ssl.create_default_context(
        cafile=certifi.where()))


def generate(submission: Submission, api_key: str | None = None,
             model: str = MODEL) -> Generated:
    """One streaming call to the Messages API. Raises GenerationError."""
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise GenerationError(
            f"the anthropic package is not installed ({exc}); "
            f"`pip install -r requirements.txt`"
        ) from None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not (key or "").strip():
        raise GenerationError(
            "ANTHROPIC_API_KEY is not set in .env. /submit needs it to reach "
            "the Claude API; add it and restart the bot."
        )

    client = anthropic.Anthropic(api_key=key, http_client=_http_client())
    try:
        # Streaming because the three artefacts together are long, and a
        # non-streaming request at this max_tokens risks an HTTP timeout.
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": build_prompt(submission)}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise GenerationError("ANTHROPIC_API_KEY was rejected.") from None
    except anthropic.RateLimitError as exc:
        retry = exc.response.headers.get("retry-after", "60")
        raise GenerationError(f"rate limited; retry in {retry}s") from None
    except anthropic.APIStatusError as exc:
        raise GenerationError(f"API error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError as exc:
        raise GenerationError(f"could not reach the API: {exc}") from None

    if message.stop_reason == "refusal":
        detail = getattr(message.stop_details, "explanation", "") or ""
        raise GenerationError(f"the model declined this request. {detail}".strip())

    text = "".join(b.text for b in message.content if b.type == "text")
    if message.stop_reason == "max_tokens":
        raise GenerationError(
            "the response hit the output limit and is incomplete. Nothing was "
            "written. Try a shorter description or a simpler strategy."
        )
    result = parse_response(text, submission.name)
    return Generated(
        result.strategy_source, result.test_source, result.entry_markdown,
        result.class_name,
        {"input_tokens": message.usage.input_tokens,
         "output_tokens": message.usage.output_tokens},
    )
