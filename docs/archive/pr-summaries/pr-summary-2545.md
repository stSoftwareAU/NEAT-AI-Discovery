## Summary

`./bump-deps.sh` now also refreshes and age-checks `fuzz/Cargo.lock`, then finishes with `quality/fuzz_lockfile.sh check`, so a bump that raises a root requirement can no longer leave the fuzz lock unresolvable under `--locked`. Closes #2545.

Version bump: `Cargo.toml` is bumped by CI's `version-increment` job; the merged commit records the final version.

## Spec

### Intent and Rationale

- `fuzz/` is its own workspace, so the root `cargo update` (Phase 3) never touched `fuzz/Cargo.lock`. After `cargo upgrade --compatible` raised e.g. `bytemuck = "1.25"` to `"1.26"`, the fuzz lock kept 1.25 and `cargo metadata --locked --manifest-path fuzz/Cargo.toml` failed.
- The issue suggested per-package `cargo update --precise`. AGENTS.md rules that out for lockstep releases (Issue #2202), so the fuzz lock goes through the same rollback-and-reapply machinery (`reapply_safe_lock_changes`) as the root lock.

### Essential Design Decisions

- The whole fuzz lock is refreshed and gated (not only packages shared with the root), mirroring the root behaviour; every change is age-checked with `plan_lock_quarantine` against `VIBE_BUMP_QUARANTINE_HOURS`.
- `quality/fuzz_lockfile.sh sync` runs first, so the crate's own path-package version stamp is current and is not reported as a changed package by the age check.
- The Phase 3a reporting and recheck moved into `bump_deps::gate_lock_refresh` (shared by the root and fuzz locks); root output is unchanged except that its ERROR lines now carry a `(Cargo.lock)` label.
- The new phase runs only in network, non-dry-run mode. `--no-network` still gets Phase 3c's `sync` restamp, since a re-resolve cannot be age-checked offline. A failure exits 8.

### Undiscoverable Facts

- `.github/workflows/ci.yml`'s `version-increment` job runs `cargo update --workspace` and never touches `fuzz/Cargo.lock`; that gap predates this change and is outside it (CI is not modified without approval).

## Evidence

Backend/CLI shell change, no UI. Verified with the stubbed-cargo harness in `tests/bump_deps_test.sh` (new Test 30, cases 30a–30f). The real `cargo update` against a live `fuzz/` workspace and real crates.io publish times were not exercised.

```mermaid
flowchart LR
    A[cargo upgrade --compatible] --> B[cargo update root]
    B --> C[gate_lock_refresh Cargo.lock]
    C --> D[cargo check --locked]
    D --> E[fuzz_lockfile.sh sync]
    E --> F[cargo update fuzz]
    F --> G[gate_lock_refresh fuzz/Cargo.lock]
    G --> H[fuzz_lockfile.sh check]
```

## Test Plan

- `tests/bump_deps_test.sh` Test 30 (new): 30a refresh moves a shared package and runs sync before check; 30b an in-quarantine version is held back in the fuzz lock and the held-back counter accumulates; 30c a failing fuzz `cargo update` fails and `check` is not run; 30d a failing `check` fails loudly naming `fuzz/Cargo.lock`; 30e a missing fuzz lock is a no-op; 30f a failing `sync` fails and cargo is not run. Result: `timeout 600 bash tests/bump_deps_test.sh` passed, 133 passed, 0 failed.
- Red checks (each mutation applied, suite run, script restored): skip the fuzz `cargo update` → 10 failures across 30a–30c; skip the gate call → 4 failures (30a, 30b); skip `check` → 5 failures (30a, 30b, 30d); skip `sync` → 8 failures (30a, 30b, 30c, 30f).
- Changed call site: Phase 3a now calls `bump_deps::gate_lock_refresh`; Tests 26–29 still pass through `reapply_safe_lock_changes`, and the other tests that drive `bump-deps.sh` (`issue_1909`, `issue_1994`, `issue_1908`, `issue_1910`, `issue_1234`, `issue_1974_*`) passed. The root-lock path of `gate_lock_refresh` is shared code with the fuzz call and is covered by 30a/30b; there is no end-to-end test of the main flow (it needs the network).
- No existing assertion was removed or edited.
- `shellcheck bump-deps.sh tests/bump_deps_test.sh`, `bash -n`, `quality/bash_syntax.sh` and `quality/shellcheck.sh`: clean.
- Docs updated: `AGENTS.md` (Dependency Bumps), `CONTRIBUTING.md` (gate never upgrades dependencies paragraph) and `README.md` (fuzz/Cargo.lock paragraph). Remaining hits for `fuzz/Cargo.lock` / `fuzz_lockfile` / `bump-deps` re-read and still true (the "re-resolves nothing" wording in `quality.sh` and CONTRIBUTING.md:161 is scoped to `sync`).
- Twinned rules: the header comment of `bump-deps.sh`, AGENTS.md, CONTRIBUTING.md and README.md each restate the bump scope; all four now cover `fuzz/Cargo.lock`.

<!-- vibe-quality-gate-skipped reason="`./quality.sh` ran up to the unit-test stage (bash syntax, shellcheck, cargo-install pinning, PR-summary location, fuzz lock sync and check, cargo deny, type checks all passed) and was stopped by the 590 s foreground cap while the `cargo test` stage was still running; CI runs the full gate" -->

## Not checked

- Full `./quality.sh` test stage (stopped at the time cap, no failure seen).
- The real `cargo update` on `fuzz/` and real crates.io publish dates.
- Standards/Spec reviewer sub-agents were not dispatched: the issue states no acceptance-criteria list.
