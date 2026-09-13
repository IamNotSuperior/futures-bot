---
name: block-tls-weakening
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: ^(?!.*[\\/]tests[\\/]).*\.py$
  - field: content
    operator: regex_match
    pattern: verify\s*=\s*False|check_hostname\s*=\s*False|CERT_NONE
---

**TLS verification is being weakened in production code.** This edit adds
`verify=False`, `check_hostname=False` or `CERT_NONE` to a Python file
outside `tests/`. The project's rule (handoff §5, "truststore recurses
forever"): fix a TLS error by adding a CA bundle, never by removing a check.
`generate._http_client()` is the pattern to follow, and
`test_tls_verification_is_not_weakened` will fail on this anyway.

Blocked. If the text is prose in a docstring that names the forbidden thing,
reword it so the literal does not appear, or make the edit in a test file.
