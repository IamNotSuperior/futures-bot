---
name: warn-commit-needs-approval
enabled: true
event: bash
action: warn
pattern: \bgit\s+(commit|push)\b
---

**Commit or push ahead.** The operator's standing rule is to ask before any
`git commit` or `git push`, however small. Confirm this one was approved in
the current conversation — an approval earlier in the session for different
work does not carry over. If `hypotheses.md` is in this commit, confirm the
entry's heading status and `**Verdict commit:**` line are what the log
requires, and that nothing else rides along in the same file.
