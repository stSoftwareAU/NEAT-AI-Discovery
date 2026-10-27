## Summary

Added hostile environment-value probes for the numeric accessors and Linux runtime-directory checks. Environment guards preserve non-Unicode prior values and restore them during unwinding. No production behaviour changed. Closes #2270.

## Spec

### Intent and Rationale

- Pin the current validation, clamp and fallback behaviour before the follow-up ledger records it.

### Essential Design Decisions

- Environment values are captured with `var_os` and restored by drop guards, so panic paths and non-Unicode values are preserved.
- A failing probe remains an ignored `finding:` test with its assertion intact; no probe exposed a defect in this run.

### Undiscoverable Facts

- The other-uid test compares an owned 0700 directory against `geteuid().wrapping_add(1)`, so it does not require root or ownership changes.

## Evidence

Backend-only change. `cargo test --test issue_2096_env_readers_outside_config` passed all 12 probes. `cargo test --lib platform::tests` passed all 16 platform tests, including the other-uid refusal, same-owner control, symlink refusal, and existing world-writable case.

## Test Plan

- `cargo fmt --all -- --check` — passed.
- `cargo test --test issue_2096_env_readers_outside_config` — passed: 12 passed, 0 failed. Covers all listed integer accessors with zero, oversize, whitespace-padded and invalid values; `u32` overflow fallback; all float accessors with NaN, inf and -inf; both noise-floor overrides.
- `cargo test --lib platform::tests` — passed: 16 passed, 0 failed. Covers other-uid rejection and log, same-owner control, symlink refusal through `apply_xdg_runtime_dir`, and the existing world-writable rejection and unset-variable assertion.
- No assertions were removed from tests that existed on the milestone base. The world-writable test's environment save/restore now uses drop guards; its refusal, log and unset-variable assertions remain.
- No ignored probes or production changes. The tests run on the current environment, including the Linux uid and symlink paths.
- `cargo clippy --all-targets --all-features -- -D warnings`, `cargo fmt --all -- --check`, bash syntax and PR-summary-location checks — passed on the final head.
- `./quality.sh` — stopped at `cargo deny check`: RUSTSEC-2026-0391/0398/0403/0410 (zstd/zstd-safe/zstd-sys via parquet 59.3.0; the fix needs zstd 0.14, which parquet 59.3.0 does not allow). This predates the PR (the diff touches no dependency) and is tracked in #2432. The stages after it (clippy, fmt, the touched tests) were run individually; the full `cargo test` suite was left to CI.
- Red checks: `xdg_runtime_dir_apply_rejects_symlink` — not re-run with the guard broken; its sibling world-writable test is unchanged in what it asserts.

**Docs sweep** — grep: `XDG_RUNTIME_DIR`, `TMPDIR`, and the 11 accessor names in `README.md`, `docs/` excluding `docs/archive/`, and source doc comments; section: `docs/GPU_GUIDE.md#xdg_runtime_dir-warnings-on-linux` and matching `docs/CONFIGURATION.md` variable rows read; siblings: `max_add_neuron_candidates_per_target` and `max_coordinated_per_target_output` confirm the existing accessor/configuration set; updated: none — the manual already describes symlink, ownership, permissions and unset behaviour, and this test-only change adds no setting or runtime behaviour.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — runtime dir rejects other uid exists, asserts false for geteuid()+1 , asserts the refusal log line, and passes without root. — evidence: `src/analysis/utils/platform.rs::runtime dir rejects other uid (asserts !trusted for geteuid().wrapping add(1) and the "owned by another user" log line, with a same-owner positive control); cargo test --lib platform::tests passed 16/16 as a non-root user` — reviewer: met
- **met** — A serial test asserts that XDG RUNTIME DIR is unset after apply xdg runtime dir() when expected runtime dir(TMPDIR) is a symlink. — evidence: `src/analysis/utils/platform.rs::xdg runtime dir apply rejects symlink ( [serial], plants a symlink at expected runtime dir(TMPDIR), asserts XDG RUNTIME DIR is None, the "not a directory" refusal log, and the symlink left in place); passed` — reviewer: met
- **met** — tests/issue 2096 env readers outside config.rs exists, with at least one test per accessor listed in B. Each test is [serial test::serial] and restores the environment. — evidence: `tests/issue 2096 env readers outside config.rs: 12 [serial] hostile values tests, one per accessor (max add neuron candidates per target through the two remove low impact noise floor overrides), restoring env through a Drop guard` — reviewer: met
- **met** — Every float probe asserts is finite() for NaN, inf and -inf. Every integer probe asserts the [floor, ceiling] bound for 0, oversize, padded and unparsable input. — evidence: `tests/issue 2096 env readers outside config.rs::assert non finite falls back (is finite for NaN/inf/-inf), assert probes (floor/ceiling for 0, oversize, padded, unparsable) and the noise-floor tests' is finite assertions` — reviewer: met
- **met** — Any probe that exposes a real defect is [ignore = "finding: …"] with its assertion intact, and is listed in the PR body for 13d-2 to adjudicate. There are no production-code changes in this PR. — evidence: `No probe is ignored, because none exposed a defect. The diff against the milestone base touches only src/analysis/utils/platform.rs under mod tests , tests/issue 2096 env readers outside config.rs and the PR summary files, so there are no production-code changes.` — reviewer: met
- **partial** — ./quality.sh passes. — evidence: `The PR summary Test Plan records fmt, clippy and the two touched test targets passing individually.` — reviewer: partial — reason: The full ./quality.sh gate stopped at cargo deny check on RUSTSEC-2026-0391/0398/0403/0410. That failure predates this diff and is tracked in #2432. The stages after it were run individually, not through the gate.
- **unrequested** — Change not traceable to the issue: xdg runtime dir rejects preexisting world writable was refactored to use the new EnvGuard . — evidence: `src/analysis/utils/platform.rs::xdg runtime dir rejects preexisting world writable` — reviewer: unrequested — reason: Panic-safe env restore replaces manual save/restore, and every assertion is unchanged.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Checked against CONTRIBUTING.md and AGENTS.md, since the repo has no CODING-STANDARDS.md. Australian English, [serial] on every env-mutating test, SAFETY: comments on env writes, and drop-guard env restore are all compliant. The symlink test now pins the refusal reason, which satisfies "An Assertion
