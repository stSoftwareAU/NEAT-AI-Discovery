## Summary

Closes #2322.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The rewritten 2108 sweep tests pass on milestone/2083 with the 2303 fix merged in. — evidence: `tests/issue 2108 chunk 08b recommendation core sweep.rs::a finite record set no longer drives the fan in correlation to nan, ::fan in candidates no longer depend on the order the caller lists neurons in, ::every float comparator in the swept files has a table row (precondition message rewritten to n` — reviewer: met
- **met** — The FILED FINDINGS contract, the 2103 ledger-scaffold and 2210 finalisation tests still pass. — evidence: `tests/issue 2103 chunk 08b ledger scaffold.rs (11 passed), tests/issue 2210 chunk 08b finalisation.rs (5 passed), FILED FINDINGS contract in tests/issue 2108 chunk 08b recommendation core sweep.rs::the recommendation core outcome links its filed findings and tests/issue 2109 chunk 08b batch successf` — reviewer: met
- **partial** — ./quality.sh passes. — evidence: `./quality.sh run: bash-syntax, shellcheck, cargo install pinning, PR summary layout, cargo deny (advisories/bans/licences/sources ok), debug build, format, clippy and type checks all passed` — reviewer: partial — reason: the run was cut off while compiling the test stage, so the full gate was not seen to pass end to end

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Checked against CONTRIBUTING.md (the repo has no CODING-STANDARDS.md): Australian English in all new prose and the assertion message; code cited by symbol, never by line number (Issue 1942); the Issue 1799 precondition assertion kept non-vacuous, only its message updated; no new dependencies, env va
