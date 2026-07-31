# PR Summary — Issue #1868

## Summary

The `version-increment` and `auto-format` jobs handed the write-capable
`ACTIONS_PUSH` PAT to `actions/checkout` without `persist-credentials: false`, so
the PAT was written into `.git/config` as an auth extraheader and stayed readable
by every later step in the job — including the `build.rs` scripts of
`cargo-outdated` and its transitive tree, executed by
`cargo install --locked --version 0.16.0 cargo-outdated`. Version pinning limits
the likelihood of a poisoned release but does nothing to stop a build script
reading a credential that need not be on disk at all.

Both checkouts now set `persist-credentials: false` and no longer receive the
PAT. The PAT is bound to `ACTIONS_PUSH_TOKEN` only on the individual steps that
talk to the remote (pull, fetch, push), which authenticate with an explicit
`https://x-access-token:${ACTIONS_PUSH_TOKEN}@github.com/${REPOSITORY}.git` URL.
The credential therefore never reaches disk, and the two jobs now follow the
`persist-credentials: false` standard the repository's other ten checkouts
already set. Pushes still use the PAT, so they keep re-triggering workflows as
`AGENTS.md` requires.

Closes #1868.

## Evidence

This is a CI/workflow change with no web interface, so no screenshot applies.
Evidence is the new test suite plus a clean `actionlint` run (the only two
findings are pre-existing `SC2086` infos on the untouched `Check for Changes`
step, present before and after this change).

Credential lifetime, before and after:

```mermaid
flowchart TD
    subgraph before["Before — PAT persisted"]
        B1[checkout with token, persist-credentials default true] --> B2[".git/config holds PAT"]
        B2 --> B3["cargo install cargo-outdated<br/>third-party build.rs can read .git/config"]
        B3 --> B4[git push origin — reads PAT from .git/config]
    end
    subgraph after["After — PAT per-step only"]
        A1[checkout, persist-credentials false, no token] --> A2[".git/config holds no credential"]
        A2 --> A3["cargo install cargo-outdated<br/>nothing to read"]
        A3 --> A4["git push https://x-access-token:$ACTIONS_PUSH_TOKEN@…<br/>PAT only in this step's env"]
    end
```

Test run:

```
test actions_push_pat_is_never_handed_to_checkout ... ok
test actions_push_pat_is_only_exposed_as_a_step_env_var ... ok
test every_checkout_disables_credential_persistence ... ok
test remote_facing_steps_authenticate_with_an_explicit_url ... ok
test result: ok. 4 passed; 0 failed
```

All four tests were written first and failed against the unfixed `ci.yml`
(`ci.yml:66 passes the write-capable ACTIONS_PUSH PAT to a checkout step`), then
passed after the change.

## Test Plan

Added `tests/issue_1868_actions_push_credential_persistence.rs`:

- `every_checkout_disables_credential_persistence` — every `actions/checkout`
  step in `ci.yml` sets `persist-credentials: false` (regression guard for the
  whole file, not just the two jobs fixed here).
- `actions_push_pat_is_never_handed_to_checkout` — no `token:` key is bound to
  `secrets.ACTIONS_PUSH`, with the offending line number in the failure message.
- `actions_push_pat_is_only_exposed_as_a_step_env_var` — every reference to
  `secrets.ACTIONS_PUSH` binds it to the `ACTIONS_PUSH_TOKEN` env var, and at
  least one reference remains so pushes keep re-triggering workflows.
- `remote_facing_steps_authenticate_with_an_explicit_url` — every step binding
  `ACTIONS_PUSH_TOKEN` uses it in an authenticated remote URL and never writes it
  back via `git remote set-url`.

Existing workflow-invariant suites re-run green: `issue_1286_workflow_permissions`
(the two jobs keep `contents: write`), `issue_1290_workflow_set_euo_pipefail`
(the reworked multi-line `run:` blocks keep `set -euo pipefail`), and
`issue_1292_actionlint_workflow`. Full `./quality.sh` passes.

## Documentation

- `AGENTS.md` — records the invariant and warns against reintroducing a
  `token:` key on a checkout step.
- `CONTRIBUTING.md` — CI pipeline section documents how the PAT now reaches the
  push steps.
