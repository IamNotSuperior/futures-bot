"""Where generated code may be written, and what it may contain.

Two layers, and it is worth being precise about what each one is for.

**The write sandbox** (:func:`safe_write`) confines files produced by the
Claude API to ``strategies/generated/`` and ``tests/generated/``. It resolves
the path first and compares real, resolved parents, so ``..`` segments,
absolute paths, and symlinks pointing out of the tree are all refused rather
than normalised into something that looks safe.

**The content screen** (:func:`screen_source`) rejects source that imports the
filesystem, the network, or the process, or that reaches for ``eval``,
``exec``, ``open`` or dunder escape hatches.

Why the second layer exists
---------------------------
The write sandbox constrains where files land. It does **not** constrain what
those files do once pytest imports and runs them - and running them is the
whole point of the pipeline. A generated module could read ``.env``, post it
somewhere, or delete the parquet cache at import time, and no path check would
see it. The screen is what stands in front of that.

**This is not a security boundary against an adversary.** A determined
attacker with code-execution can defeat an AST screen (``getattr`` chains,
string-built imports, C-extension paths). It is a boundary against *mistakes*
and *drift* - a model that reaches for ``os.environ`` because the strategy
needed a config value, or writes a test that shells out. That is the realistic
failure here, and the screen catches it loudly. Alongside it,
``submissions.run_tests`` executes the suite in a subprocess with the real
secrets stripped from the environment, so even code that gets past the screen
has nothing worth exfiltrating in reach.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The only two directories generated code may be written to.
STRATEGY_DIR = PROJECT_ROOT / "strategies" / "generated"
TEST_DIR = PROJECT_ROOT / "tests" / "generated"
ALLOWED_DIRS = (STRATEGY_DIR, TEST_DIR)

#: Modules a strategy legitimately needs. Everything else is refused.
#: `pandas`/`numpy` for the maths, `base`/`rules` for the interface and the
#: limits, `pytest` for the generated test. Nothing here can touch the
#: filesystem, the network or the process table.
ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "math", "statistics", "datetime", "dataclasses",
    "typing", "collections", "itertools", "functools", "enum", "abc",
    "base", "rules", "pytest", "__future__",
})

#: Names that are refused outright wherever they appear as a call or an
#: attribute. `open` is here because file access is the write sandbox's job,
#: not the generated strategy's.
BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview",
})

#: Dunder attributes that walk out of the sandbox by reflection.
BANNED_ATTRS = frozenset({
    "__globals__", "__subclasses__", "__bases__", "__mro__", "__code__",
    "__builtins__", "__loader__", "__closure__", "__reduce__", "__class__",
    "__dict__", "__getattribute__",
})

#: A generated file must not name these, in source, at all. Catches a
#: string-built import that the AST walk would not see as an import.
BANNED_SUBSTRINGS = (
    "os.environ", "subprocess", "socket", "shutil", "importlib",
    "sys.modules", "ANTHROPIC_API_KEY", "DISCORD_TOKEN", "DATABENTO_API_KEY",
)

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")

#: Names a submission may not take: they would shadow a real module on
#: sys.path, and `import rules` inside a generated strategy would then resolve
#: to the generated file instead of the guard module.
RESERVED_NAMES = frozenset({
    "rules", "base", "engine", "store", "pretrade", "close", "review",
    "loader", "fetch", "validate", "registry", "metrics", "scan", "trend",
    "orb", "orb2", "london", "eod_rebalance", "walkforward", "eval_sim",
    "desk", "research", "runners", "broker", "tickets", "feed", "approvals",
    "chart_read", "read_log", "sandbox", "generate", "submissions",
    "test", "tests", "conftest", "generated", "pandas", "numpy", "pytest",
})


class SandboxError(RuntimeError):
    """A write or a piece of source was refused. The message says why."""


@dataclass(frozen=True)
class Screening:
    ok: bool
    problems: list

    def report(self) -> str:
        return "\n".join(f"- {p}" for p in self.problems)


def validate_name(name: str) -> str:
    """A submission name that is safe as a module name and not a shadow."""
    name = (name or "").strip().lower()
    if not NAME_RE.match(name):
        raise SandboxError(
            f"`{name}` is not a usable strategy name: 3-40 characters, "
            f"lowercase letters, digits and underscores, starting with a letter."
        )
    if name in RESERVED_NAMES:
        raise SandboxError(
            f"`{name}` would shadow an existing module on sys.path. A "
            f"generated strategy importing `rules` must get the real guard "
            f"module, so this name is refused. Pick another."
        )
    return name


def resolve_target(relative: str | Path) -> Path:
    """The absolute path a generated file may be written to, or raise.

    Resolution happens **before** the containment check, so ``..`` and
    symlinks are compared as the real location they point at rather than as
    the text they were written as.
    """
    candidate = Path(relative)
    if candidate.is_absolute():
        raise SandboxError(
            f"refusing an absolute path: {relative}. Generated files are named "
            f"relative to the project root."
        )
    if any(part == ".." for part in candidate.parts):
        raise SandboxError(f"refusing a path containing '..': {relative}")

    target = (PROJECT_ROOT / candidate).resolve()
    for allowed in ALLOWED_DIRS:
        root = allowed.resolve()
        if target == root:
            break
        if root in target.parents:
            break
    else:
        allowed_names = ", ".join(
            str(d.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for d in ALLOWED_DIRS
        )
        raise SandboxError(
            f"refusing to write outside the sandbox: {relative} resolves to "
            f"{target}. Generated code may only be written under "
            f"{allowed_names}."
        )
    if target.suffix != ".py":
        raise SandboxError(f"generated files must be .py, got {target.name}")
    return target


def import_allowed(module: str, allow_module: str | None) -> bool:
    """Is ``module`` importable from generated source?

    ``allow_module`` is the one generated strategy the file under test is
    permitted to import - its own. Both spellings of that module are accepted,
    the bare name (the repository's flat-module convention, and what the
    generation prompt asks for) and the fully-qualified
    ``strategies.generated.<name>``.

    The dotted form is matched **exactly**, never by its root package. Allowing
    the root would permit ``strategies.rules`` and every other module in the
    tree, which is the opposite of the point: a generated test may reach its
    own strategy and nothing else, so one submission cannot import, subclass or
    monkeypatch another's code.
    """
    root = module.split(".")[0]
    if root in ALLOWED_IMPORTS:
        return True
    if not allow_module:
        return False
    return module in (allow_module, f"strategies.generated.{allow_module}")


def screen_source(source: str, filename: str = "<generated>",
                  allow_module: str | None = None) -> Screening:
    """Refuse source that reaches for the filesystem, network or process.

    ``allow_module`` additionally permits one generated strategy module - see
    :func:`import_allowed`. It is passed for a generated *test* and never for a
    generated *strategy*: a strategy importing another generated strategy has
    no legitimate reason to, and would couple two submissions that are supposed
    to be independent.

    Returns a :class:`Screening` rather than raising so every problem is
    reported at once - a caller fixing one import at a time would need as many
    API round trips as there are violations.
    """
    problems: list[str] = []
    permitted = sorted(ALLOWED_IMPORTS) + (
        [allow_module] if allow_module else [])
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return Screening(False, [f"does not parse: {exc}"])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not import_allowed(alias.name, allow_module):
                    problems.append(
                        f"line {node.lineno}: `import {alias.name}` - only "
                        f"{permitted} are permitted"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(
                    f"line {node.lineno}: relative import - generated modules "
                    f"are flat and must import by bare name"
                )
            elif not import_allowed(node.module or "", allow_module):
                problems.append(
                    f"line {node.lineno}: `from {node.module} import ...` - "
                    f"only {permitted} are permitted"
                )
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: `{node.id}` is not permitted")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            problems.append(
                f"line {node.lineno}: `.{node.attr}` is a reflection escape "
                f"hatch and is not permitted"
            )

    lowered = source.lower()
    for banned in BANNED_SUBSTRINGS:
        if banned.lower() in lowered:
            problems.append(f"source mentions `{banned}`, which is not permitted")

    return Screening(not problems, problems)


def safe_write(relative: str | Path, content: str, screen: bool = True,
               allow_module: str | None = None) -> Path:
    """Screen ``content``, then write it inside the sandbox. Returns the path.

    ``encoding="utf-8"`` explicitly: on Windows ``write_text`` defaults to
    cp1252 and raises on the first non-ASCII character *after* truncating the
    file, which would leave a half-written module behind.
    """
    target = resolve_target(relative)
    if screen:
        result = screen_source(content, str(target), allow_module=allow_module)
        if not result.ok:
            raise SandboxError(
                f"generated source for {target.name} was refused:\n"
                f"{result.report()}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def scrubbed_env(base: dict) -> dict:
    """``base`` with every real secret removed.

    Used for the subprocess that runs generated tests. The screen should stop
    code from reading these, but the screen is a check on mistakes rather than
    a containment boundary - so the values are not in the process at all.
    """
    secrets = {"ANTHROPIC_API_KEY", "DISCORD_TOKEN", "DATABENTO_API_KEY",
               "DESK_WEBHOOK_TOKEN", "DESK_OWNER_ID", "AWS_SECRET_ACCESS_KEY",
               "GITHUB_TOKEN"}
    return {k: v for k, v in base.items() if k not in secrets}
