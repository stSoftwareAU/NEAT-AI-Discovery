## Summary

`scripts/install-rustup.sh` now gives its `rustup-init` download a per-attempt transfer cap (`--max-time 300`) and a total retry budget (`--retry-max-time 1200`), alongside the existing `--connect-timeout 30`, `--retry 3` and SHA-256 digest verification. A peer that stops sending the body can no longer hold the installer indefinitely; on exhaustion curl fails and the script still prints `ERROR: failed to download` and returns 1. The `scripts/runlib.sh:568–570` call site is left for NEAT-AI-core (see below), so this is `Part of #2556`.

Version bump: `Cargo.toml` updated for this change (with `Cargo.lock` and `fuzz/Cargo.lock` restamped); the merged commit records the final version. The Develop merge retains 0.74.362 in the fuzz lockfile to match Cargo.toml.

## Spec

### Intent and Rationale

- `--connect-timeout` only bounds the connection phase; once connected, a stalled body hangs forever, so the digest check and failure path are never reached. `--max-time` bounds each attempt and `--retry-max-time` bounds the whole retry sequence.

### Essential Design Decisions

- Limits are the issue's suggested values (300 s per attempt, 1200 s overall), assumed adequate for the ~15 MB `rustup-init` on supported unattended hosts; a slower link than ~50 kB/s would be rejected.
- `scripts/runlib.sh` is deliberately not edited: `CONTRIBUTING.md` (lines 35–45) says it is a byte-identical copy of NEAT-AI-core's helper and `family-sync` re-copies it; the issue itself says to change it in NEAT-AI-core.

### Undiscoverable Facts

- The same omission exists in `scripts/runlib.sh`, which must be fixed in NEAT-AI-core `Develop`, outside this repository.

## Evidence

Backend/script change; no UI. The new test runs the real script against a stub `curl` (the harness already used by the other digest-verification tests, which logs the arguments it receives) and asserts the received flag/value pairs.

## Acceptance Criteria

- **missing** — "The same omission occurs in `scripts/runlib.sh:568–570`" — reason: that file is a vendored byte-identical copy of NEAT-AI-core's canonical helper (`CONTRIBUTING.md`); the fix belongs in NEAT-AI-core and `family-sync` copies it here.

## Test Plan

- Added `bounds_each_attempt_and_the_whole_download_with_a_time_budget` and the `Sandbox::curl_args` helper in `tests/issue_1911_rustup_digest_verification.rs`. Red checks: removing `--max-time 300` from the script made it fail; removing `--retry-max-time 1200` made it fail; both restored.
- Download failure still exits non-zero: covered by the existing `fails_loud_when_the_download_fails` in the same file.
- `cargo test -q --test issue_1911_rustup_digest_verification`: 13 passed. `issue_2097_rustup_pin_parity` and `issue_2072_canonical_runlib`: 17 passed. `cargo fmt --check`, `shellcheck`, `bash -n` and `cargo clippy --all-targets --all-features -D warnings`: passed.
- No existing test edited, so no assertions removed.
- Full `./quality.sh`: not completed — the cold build in this fresh worktree exceeded the 590 s tool cap during the clippy/check stages; clippy was then run separately and passed. CI runs the full gate.
<!-- vibe-quality-gate-skipped reason="cold build exceeded the single-command timeout; targeted checks run instead" -->
