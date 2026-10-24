## Summary

Closes #2270.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **missing** — runtime dir rejects other uid exists, asserts false for geteuid()+1, asserts the refusal log line, and passes without root. — reviewer: missing — reason: Closure evidence unavailable: tool execution failed before the diff or tests could be inspected.
- **missing** — A serial test asserts that XDG RUNTIME DIR is unset after apply xdg runtime dir() when expected runtime dir(TMPDIR) is a symlink. — reviewer: missing — reason: Closure evidence unavailable: the added tests could not be inspected.
- **missing** — tests/issue 2096 env readers outside config.rs exists, with at least one test per accessor listed in B. Each test is [serial test::serial] and restores the environment. — reviewer: missing — reason: Closure evidence unavailable: the test file and accessor list could not be inspected.
- **missing** — Every float probe asserts is finite() for NaN, inf and -inf. Every integer probe asserts the [floor, ceiling] bound for 0, oversize, padded and unparsable input. — reviewer: missing — reason: Closure evidence unavailable: probe assertions could not be inspected.
- **missing** — Any probe that exposes a real defect is [ignore = "finding: …"] with its assertion intact, and is listed in the PR body for 13d-2 to adjudicate. There are no production-code changes in this PR. — reviewer: missing — reason: Closure evidence unavailable: the diff, ignored probes and PR body could not be inspected.
- **missing** — ./quality.sh passes. — reviewer: missing — reason: No quality-gate result could be verified.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

