---
name: warn-restated-threshold
enabled: true
event: file
action: warn
conditions:
  - field: file_path
    operator: regex_match
    pattern: ^(?!.*[\\/]tests[\\/])(?!.*[\\/]rules\.py$).*\.py$
  - field: content
    operator: regex_match
    pattern: (?-i:\b(DAILY_LOSS_LIMIT|TRAILING_DD_STOP|TRAILING_DD_WARN|POSITION_CAP|MIN_HOLD_SECONDS|MICROSCALP_SECONDS|MICROSCALP_PROFIT_FLAG_PCT|COMMISSION_PER_SIDE|ENTRY_CUTOFF|FLATTEN_TIME)\s*=\s*[0-9])|commission_per_side\s*=\s*(?!0\.0\b)[0-9]|\blimit\s*=\s*[0-9]{3,}
---

**A rule threshold may be restated here.** This edit assigns a number to a
name that `strategies/rules.py` owns, or passes a literal commission or a
three-digit `limit=` where the rules module should be read instead. CLAUDE.md
rule 9 and the handoff's "Never restate a threshold": every limit lives in
`rules.py` and is imported, because a hardcoded copy diverges silently the
moment the rule changes — `enforce_daily_loss_limit` once carried
`limit=300.0` after the limit moved to $400.

Allowed, with a warning. If this is `rules.COMMISSION_PER_SIDE` or another
import, fine. If it is a literal, import the constant instead.
