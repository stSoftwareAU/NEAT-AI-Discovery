# Pin `cargo install` in CI workflows to a reviewed release with `--locked`

## Summary

Three CI workflows previously ran `cargo install <plugin>` with no
`--locked` flag and no `--version` pin, so each run resolved a fresh
dependency graph (and the plugin itself) from `crates.io`. A poisoned
new release of the plugin or any transitive dependency would have
executed via `build.rs` on the runner, with access to whatever
`GITHUB_TOKEN` / `ACTIONS_PUSH` token that workflow holds.

This PR pins all three call sites to a specific plugin version **and**
passes `--locked` so the plugin's own `Cargo.lock` is honoured. Both
flags together close the supply-chain hole described in
`supply-chain:unpinned-cargo-install` (stable ID `11cc27a3`).

Changes:

| File | Plugin | New install command |
| --- | --- | --- |
| `.github/workflows/ci.yml` | `cargo-outdated` | `cargo install --locked --version 0.16.0 cargo-outdated` |
| `.github/workflows/security.yml` | `cargo-audit` | `cargo install --locked --version 0.21.2 cargo-audit` |
| `.github/workflows/upgrade-dependencies.yml` | `cargo-edit` | `cargo install --locked --version 0.13.6 cargo-edit` |

A brief comment above each install records *why* the pin is required so
a future contributor does not silently drop the flags.

Closes #1223.

## Evidence

Backend / CI-only change — no UI to screenshot.

**TDD evidence.** A new shell test
(`tests/test_cargo_install_pinning.sh`) scans every workflow YAML for
`cargo install` invocations and asserts each one carries both
`--locked` and `--version <X.Y.Z>`. Before the workflow edits:

```text
FAIL: .github/workflows/security.yml:28 — 'cargo install' missing --locked: ...
FAIL: .github/workflows/upgrade-dependencies.yml:30 — 'cargo install' missing --locked: ...
FAIL: .github/workflows/ci.yml:59 — 'cargo install' missing --locked: ...
Results: 0 passed, 3 failed
```

After the workflow edits:

```text
Results: 3 passed, 0 failed
✅ All cargo install invocations are pinned with --locked and --version
```

The test is generic — it scans every `*.yml` / `*.yaml` under
`.github/workflows/`, so any future workflow that adds a new
`cargo install` line without both flags will fail this check.

```mermaid
flowchart LR
    A[PR opens or schedule fires] --> B[Runner installs Rust]
    B --> C{cargo install plugin}
    C -- before: no --locked, no --version --> D[Resolves latest plugin + transitive deps from crates.io]
    D --> E[Runs build.rs with workflow token]
    C -- after: --locked --version X.Y.Z --> F[Installs pinned version using plugin's Cargo.lock]
    F --> G[Runs build.rs with reviewed dep graph]
```

## Test Plan

- `tests/test_cargo_install_pinning.sh` — new TDD test asserting every
  `cargo install` line in `.github/workflows/*.yml` carries both
  `--locked` and `--version <X.Y.Z>`. Passes after the fix.
- `bash -n` on every `*.sh` under the repo — passes.
- `shellcheck -s bash tests/test_cargo_install_pinning.sh` — passes.
- Existing `tests/test_security_workflow_validation.sh` — still passes
  (the security workflow's `cargo audit`, `cargo-audit`, and
  `rustsec/audit-check` references remain in place).
- No Rust source files were modified, so the full `cargo build` / lint
  / unit-test pipeline is unaffected by this change.
