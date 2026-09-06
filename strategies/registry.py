"""Strategy registry: where every idea stands, and the only way that changes.

The registry exists to make promotion expensive. Nothing here decides whether a
strategy is good; it decides whether the evidence required to advance it has
actually been produced and recorded, and it refuses when it has not.

    from registry import Registry
    reg = Registry.load()
    reg.status_table()
    reg.promote("orb2", "paper")      # raises: entry 4 is REJECTED

Two rules do most of the work:

**Rejected is terminal.** ``research/hypotheses.md`` says a verdict is filled in
after walk-forward and is not revised afterwards. A registry that let a rejected
strategy be promoted back into testing would be a bypass around that, so no
transition leaves ``rejected``.

**There is no override argument.** Not on :meth:`Registry.promote`, not
anywhere. A gate with an override is a gate that gets overridden at exactly the
moment it matters, and CLAUDE.md rule 9 forbids a bypass path for the risk
limits this feeds. If a gate is wrong, the fix is to produce the evidence or to
change the pre-registered rule in the log - not to pass a flag.

Thresholds are imported, never restated. The paper->live gate reads
``journal.review.readiness``, which is the same function ``journal/review.py``
prints from, so the bot, the CLI and the registry cannot drift apart.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Inserted in reverse so the resulting order is strategies, backtests, journal.
# That order matters: there are two modules called `review`, and journal's one
# imports backtests' one by bare name. If journal came first, that import would
# resolve to itself.
for _folder in ("journal", "backtests", "strategies"):
    _p = str(PROJECT_ROOT / _folder)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def _journal_review():
    """``journal/review.py``, loaded by path rather than by name.

    ``import review`` is ambiguous in this repo - ``backtests/review.py`` and
    ``journal/review.py`` both exist, and which one wins depends on the order
    some other module happened to build ``sys.path`` in. Loading this one by
    file path removes the ambiguity instead of relying on that order holding.
    """
    import importlib.util

    name = "journal_review"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "journal" / "review.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.yaml"
HYPOTHESES_PATH = PROJECT_ROOT / "research" / "hypotheses.md"

#: The lifecycle. Order is meaningful only for reading; transitions are the
#: authority, not adjacency.
STATUSES = ("proposed", "testing", "paper", "live", "rejected")

#: Which moves exist at all. Anything absent is refused before a gate is even
#: consulted. ``rejected`` maps to nothing: a verdict is not revised.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"testing", "rejected"}),
    "testing": frozenset({"paper", "rejected"}),
    "paper": frozenset({"live", "rejected"}),
    "live": frozenset({"rejected"}),
    "rejected": frozenset(),
}

#: Statuses the hypothesis log itself can carry, mapped to registry statuses.
#: The log is the source of truth for a verdict; the registry only mirrors it.
LOG_STATUS_TO_REGISTRY = {
    "PROPOSED": "proposed",
    "ACCEPTED": "testing",
    "REJECTED": "rejected",
}


class PromotionError(RuntimeError):
    """A promotion was refused. The message says which gate refused it."""


class RegistryError(RuntimeError):
    """The registry file or its agreement with the log is broken."""


# ---------------------------------------------------------------------------
# The hypothesis log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisEntry:
    number: int
    title: str
    status: str          # PROPOSED | ACCEPTED | REJECTED | UNKNOWN
    verdict_commit: str | None
    body: str

    @property
    def accepted(self) -> bool:
        return self.status == "ACCEPTED"

    def summary(self, limit: int = 700) -> str:
        """First substantive paragraph of the entry, for a chat reply."""
        for block in self.body.split("\n\n"):
            text = block.strip()
            if not text or text.startswith(("#", "|", "**Date:", "**Spec",
                                            "**Code:", "**Note:", "**Commit",
                                            "**Verdict commit", "**Instrument",
                                            "**Source:", "**Predecessor")):
                continue
            return text[:limit] + ("..." if len(text) > limit else "")
        return "(no summary text found)"


def parse_hypotheses(path: Path = HYPOTHESES_PATH) -> dict[int, HypothesisEntry]:
    """Read the log's ``## N. Title - STATUS`` headings and their bodies.

    The heading is the authority: every entry carries its verdict there, and
    the log's own convention is to update it when a verdict lands. Parsing the
    heading rather than hunting for a verdict section means a half-written
    entry reads as UNKNOWN instead of silently as a pass.
    """
    text = path.read_text(encoding="utf-8")
    entries: dict[int, HypothesisEntry] = {}

    lines = text.splitlines()
    starts: list[tuple[int, int, str, str]] = []
    for i, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        head = line[3:].strip()
        if "." not in head:
            continue
        number_part, rest = head.split(".", 1)
        if not number_part.strip().isdigit():
            continue
        # Titles use an em dash before the status.
        status = "UNKNOWN"
        title = rest.strip()
        for sep in ("—", " - ", "–"):
            if sep in rest:
                title, _, tail = rest.rpartition(sep)
                candidate = tail.strip().upper()
                if candidate in LOG_STATUS_TO_REGISTRY:
                    status = candidate
                    title = title.strip()
                break
        starts.append((i, int(number_part.strip()), title, status))

    for idx, (line_no, number, title, status) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[line_no + 1:end])
        commit = None
        for key in ("**Verdict commit:**", "**Commit:**"):
            if key in body:
                after = body.split(key, 1)[1].splitlines()[0]
                token = after.strip().strip("`").split()[0].strip("`")
                if token and token != "recorded":
                    commit = token
                break
        entries[number] = HypothesisEntry(number, title, status, commit, body)
    return entries


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyRecord:
    name: str
    class_path: str | None
    hypothesis_entry: int
    status: str
    verdict_commit: str | None
    walkforward_verdict: str

    def load_class(self):
        """Import the strategy class. Flat modules, so ``module:Class``."""
        if not self.class_path:
            raise RegistryError(
                f"{self.name} has no class_path; it is a log entry, not code"
            )
        module_name, _, class_name = self.class_path.partition(":")
        if not class_name:
            raise RegistryError(f"class_path must be 'module:Class', got "
                                f"{self.class_path!r}")
        return getattr(importlib.import_module(module_name), class_name)


class Registry:
    """The strategy table, loaded from and written back to ``registry.yaml``."""

    def __init__(self, records: Iterable[StrategyRecord],
                 path: Path = REGISTRY_PATH,
                 hypotheses_path: Path = HYPOTHESES_PATH,
                 shadow: Iterable[str] = ()) -> None:
        self._records = {r.name: r for r in records}
        self.path = path
        self.hypotheses_path = hypotheses_path
        #: Names the desk bot runs as plumbing tests. Not a status; see
        #: :meth:`shadow_names` and the comment block in ``registry.yaml``.
        self._shadow = tuple(dict.fromkeys(shadow))

    # -- loading and saving ------------------------------------------------

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH,
             hypotheses_path: Path = HYPOTHESES_PATH) -> "Registry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rows = raw.get("strategies") or []
        records = []
        for row in rows:
            missing = {"name", "class_path", "hypothesis_entry", "status"} - set(row)
            if missing:
                raise RegistryError(f"registry row missing {sorted(missing)}: {row}")
            if row["status"] not in STATUSES:
                raise RegistryError(
                    f"{row['name']}: unknown status {row['status']!r}; "
                    f"expected one of {STATUSES}"
                )
            records.append(StrategyRecord(
                name=row["name"],
                class_path=row.get("class_path"),
                hypothesis_entry=int(row["hypothesis_entry"]),
                status=row["status"],
                verdict_commit=row.get("verdict_commit"),
                walkforward_verdict=(row.get("walkforward_verdict") or "").strip(),
            ))
        shadow = raw.get("shadow") or []
        if not isinstance(shadow, list) or any(not isinstance(n, str) for n in shadow):
            raise RegistryError(
                f"`shadow` must be a list of strategy names, got {shadow!r}"
            )
        # A name that collides with a status is almost certainly someone
        # reaching for `shadow: paper` as though it were one.
        bad = [n for n in shadow if n in STATUSES]
        if bad:
            raise RegistryError(
                f"`shadow` holds status name(s) {bad}; it is a list of strategy "
                f"names, not a status. Shadow is not a promotion."
            )
        return cls(records, Path(path), Path(hypotheses_path), shadow)

    def save(self) -> None:
        payload = {"strategies": [
            {
                "name": r.name,
                "class_path": r.class_path,
                "hypothesis_entry": r.hypothesis_entry,
                "status": r.status,
                "verdict_commit": r.verdict_commit,
                "walkforward_verdict": r.walkforward_verdict,
            }
            for r in self.all()
        ]}
        # Carried through explicitly. `save` rebuilds the file from records, so
        # anything not named here is dropped - and silently dropping the shadow
        # list on the next promote() would turn the desk bot's run set into an
        # empty one with no error anywhere.
        if self._shadow:
            payload["shadow"] = list(self._shadow)
        self.path.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=88,
                           default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    # -- reading -----------------------------------------------------------

    def all(self) -> list[StrategyRecord]:
        return sorted(self._records.values(), key=lambda r: r.hypothesis_entry)

    def names(self) -> list[str]:
        return [r.name for r in self.all()]

    def get(self, name: str) -> StrategyRecord:
        try:
            return self._records[name]
        except KeyError:
            raise KeyError(
                f"no strategy {name!r} in the registry; known: {self.names()}"
            ) from None

    def shadow_names(self) -> list[str]:
        """Strategies listed for shadow running. **Not a status.**

        Shadow is a plumbing test list: it exercises the signal -> guard ->
        ticket path without any of it counting as evidence. A name here keeps
        whatever status its record carries, and appearing here never advances
        anything. :meth:`promote` does not read this list and cannot be
        influenced by it.
        """
        return list(self._shadow)

    def is_shadow(self, name: str) -> bool:
        return name in self._shadow

    def desk_strategies(self) -> list[tuple[StrategyRecord, str]]:
        """What the desk bot runs, as ``(record, mode)`` pairs.

        ``mode`` is ``"shadow"`` or ``"live-eligible"``. The split is the whole
        point: a shadow record is run so the plumbing can be tested, and a
        ``paper``/``live`` record is run because it earned its way there. The
        desk bot labels and routes them differently and must never collapse the
        two, so the distinction is made here rather than at the call site.

        A record that is both listed in ``shadow`` and at ``paper``/``live``
        status is reported as ``live-eligible``; :meth:`verify` flags the
        ``live`` half of that overlap as a contradiction to be resolved.
        """
        out: list[tuple[StrategyRecord, str]] = []
        for record in self.all():
            if record.status in ("paper", "live"):
                out.append((record, "live-eligible"))
            elif self.is_shadow(record.name):
                out.append((record, "shadow"))
        return out

    def hypotheses(self) -> dict[int, HypothesisEntry]:
        return parse_hypotheses(self.hypotheses_path)

    def entry_for(self, name: str) -> HypothesisEntry | None:
        return self.hypotheses().get(self.get(name).hypothesis_entry)

    def paper_trade_count(self, name: str) -> int:
        """Closed journal tickets. The journal is not per-strategy yet, so
        every strategy reads the same file; when it gains a strategy column
        this is the one place that has to change."""
        import store  # local: the journal is optional for registry reads
        try:
            closed = store.load_closed()
        except FileNotFoundError:
            return 0
        return int(len(closed))

    # -- agreement with the log -------------------------------------------

    def verify(self) -> list[str]:
        """Discrepancies between the registry and the hypothesis log.

        Returns an empty list when they agree. This is the check that stops the
        registry becoming a second, softer source of truth.
        """
        entries = self.hypotheses()
        problems: list[str] = []
        for r in self.all():
            entry = entries.get(r.hypothesis_entry)
            if entry is None:
                problems.append(
                    f"{r.name}: hypotheses.md has no entry {r.hypothesis_entry}"
                )
                continue
            if entry.status == "REJECTED" and r.status != "rejected":
                problems.append(
                    f"{r.name}: entry {entry.number} is REJECTED in the log but "
                    f"the registry says {r.status!r}"
                )
            if entry.status == "PROPOSED" and r.status in ("paper", "live"):
                problems.append(
                    f"{r.name}: entry {entry.number} is only PROPOSED in the log "
                    f"but the registry says {r.status!r}"
                )
            if (entry.verdict_commit and r.verdict_commit
                    and entry.verdict_commit != r.verdict_commit):
                problems.append(
                    f"{r.name}: verdict commit {r.verdict_commit} does not match "
                    f"the log's {entry.verdict_commit}"
                )

        # -- the shadow list ------------------------------------------------
        # Shadow is not a status, so the checks above do not see it at all. It
        # still has to agree with the table it names.
        for name in self._shadow:
            if name not in self._records:
                problems.append(
                    f"shadow lists {name!r}, which is not in the registry"
                )
                continue
            if self._records[name].status == "live":
                problems.append(
                    f"shadow lists {name!r}, but it is at 'live' status - a "
                    f"strategy cannot be both routed to a broker and run as a "
                    f"no-order plumbing test"
                )
        return problems

    # -- the only way status changes --------------------------------------

    def promote(self, name: str, to_status: str) -> StrategyRecord:
        """Move a strategy to ``to_status``, or refuse and say why.

        There is deliberately no override argument. Every refusal below is a
        refusal to act on evidence that does not exist yet.
        """
        record = self.get(name)
        if to_status not in STATUSES:
            raise ValueError(
                f"unknown status {to_status!r}; expected one of {STATUSES}"
            )
        if to_status == record.status:
            raise PromotionError(f"{name} is already {to_status!r}")

        allowed = ALLOWED_TRANSITIONS[record.status]
        if to_status not in allowed:
            if record.status == "rejected":
                raise PromotionError(
                    f"{name} is rejected, and a rejected entry is terminal. "
                    f"research/hypotheses.md records verdicts that are never "
                    f"revised; a new idea needs a new entry, not a promotion."
                )
            raise PromotionError(
                f"{name}: {record.status!r} -> {to_status!r} is not a transition "
                f"that exists. From {record.status!r} you may go to "
                f"{sorted(allowed) or 'nowhere'}."
            )

        if to_status == "paper":
            self._require_walkforward_pass(record)
        elif to_status == "live":
            self._require_paper_gate(record)

        promoted = replace(record, status=to_status)
        self._records[name] = promoted
        self.save()
        return promoted

    def _require_walkforward_pass(self, record: StrategyRecord) -> None:
        """testing -> paper: the log must record an accepted walk-forward."""
        entry = self.hypotheses().get(record.hypothesis_entry)
        if entry is None:
            raise PromotionError(
                f"{record.name}: hypotheses.md has no entry "
                f"{record.hypothesis_entry}, so no walk-forward is recorded."
            )
        if not entry.accepted:
            raise PromotionError(
                f"{record.name}: entry {entry.number} reads {entry.status} in "
                f"research/hypotheses.md. Promotion to paper needs a passed "
                f"walk-forward recorded there - profitable in a majority of "
                f"folds, positive total P&L after costs, and surviving at 2 "
                f"ticks of slippage."
            )

    def _require_paper_gate(self, record: StrategyRecord) -> None:
        """paper -> live: entry 3's gate, computed by the journal itself."""
        import store
        readiness = _journal_review().readiness

        try:
            closed = store.load_closed()
        except FileNotFoundError:
            raise PromotionError(
                f"{record.name}: no journal exists yet, so there are no "
                f"rule-clean trades to count. The gate needs 60."
            ) from None

        r = readiness(closed)
        if r["ready"]:
            return

        gates = r["gates"]
        reasons = []
        if not gates["clean_trades"]:
            reasons.append(
                f"rule-clean trades {r['clean_trades']} of 60"
                + (f" (reset by {r['violations']} rule violation(s))"
                   if r["violations"] else "")
            )
        if not gates["positive_expectancy"]:
            reasons.append(f"expectancy ${r['expectancy']:,.2f} per trade, "
                           f"must be positive")
        if not gates["pass_probability"]:
            prob = r["pass_probability"]
            reasons.append(
                "pass probability not computable yet"
                if prob is None else
                f"pass probability {prob:.2%}, must exceed 50%"
            )
        raise PromotionError(
            f"{record.name}: the paper gate is not met - " + "; ".join(reasons)
        )

    # -- presentation ------------------------------------------------------

    def status_table(self) -> str:
        rows = [
            f"{'strategy':<22}{'entry':>6}{'status':>11}{'paper':>8}  verdict commit",
            "-" * 78,
        ]
        for r in self.all():
            marker = "  [shadow]" if self.is_shadow(r.name) else ""
            rows.append(
                f"{r.name:<22}{r.hypothesis_entry:>6}{r.status:>11}"
                f"{self.paper_trade_count(r.name):>8}  "
                f"{r.verdict_commit or '-'}{marker}"
            )
        problems = self.verify()
        if self._shadow:
            rows.append("")
            rows.append(
                f"shadow (desk bot plumbing test, NOT a status, no orders): "
                f"{', '.join(self._shadow)}"
            )
        rows.append("")
        rows.append("registry agrees with hypotheses.md" if not problems
                    else "DRIFT:\n  " + "\n  ".join(problems))
        return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover - convenience
    print(Registry.load().status_table())
