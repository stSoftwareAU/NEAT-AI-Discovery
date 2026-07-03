# PR Summary — Issue #1486

## Summary

Added `CODEOWNERS` coverage for the repository's high-blast-radius paths so a
pull request cannot merge them without review from a trusted team. The audit
(finding `BP-8be698853743`) flagged that no `CODEOWNERS` file existed while the
CI workflows run with privileged secrets beyond the default `GITHUB_TOKEN`:

- a write-capable PAT `secrets.ACTIONS_PUSH` used to push commits back to
  branches (`.github/workflows/ci.yml`);
- `secrets.SEMGREP_APP_TOKEN` (`.github/workflows/semgrep.yml`);
- `secrets.CODECOV_TOKEN` (`.github/workflows/cargo-quality.yml`).

Without owner review on `.github/workflows/`, a careless or malicious workflow
edit is a direct path to secret exfiltration or unauthorised pushes to protected
branches.

Changes:

- **`.github/CODEOWNERS`** — assigns `@stSoftwareAU/developers` as owner of the
  `*` default plus the sensitive paths: `.github/` and `.github/workflows/`,
  `Cargo.toml` / `Cargo.lock`, and the security-sensitive automation/policy
  (`deny.toml`, `renovate.json`, `SECURITY.md`).
- **`CONTRIBUTING.md`** — documents the code-owner policy and the
  branch-protection settings an admin must enable on `Develop` for `CODEOWNERS`
  to take effect.
- **`tests/issue_1486_codeowners.rs`** — repo-hygiene test asserting the real
  committed artefact.

The issue's example used `@stSoftwareAU/maintainers`, but that team does not
exist in the org (verified via the GitHub API). A `CODEOWNERS` file that names a
non-existent owner is flagged as invalid by GitHub and is **not enforced**, so
the file assigns the real `@stSoftwareAU/developers` team instead — keeping the
protection functional.

Closes #1486.

## Branch protection (repository setting — requires a human admin)

`CODEOWNERS` only enforces review once branch protection references it. These
are repository-level settings that cannot be committed as files; a repository
admin must enable them on the default branch (`Develop`), as now documented in
`CONTRIBUTING.md`:

- **Require a pull request before merging** with **Require review from Code
  Owners**.
- Block direct pushes and force-pushes.
- Require linear history.
- Confirm the `quality.sh` required status checks.
- (Defence-in-depth) Require signed commits.

```mermaid
flowchart LR
    PR[PR edits .github/workflows/**] --> CO{CODEOWNERS<br/>match?}
    CO -- yes --> REV[Require review from<br/>@stSoftwareAU/developers]
    REV --> BP{Branch protection<br/>enforces owner review?}
    BP -- yes --> MERGE[Merge allowed]
    BP -- no --> GAP[Gap: file present<br/>but not enforced]
    CO -- no --> UNOWNED[Unowned path]
```

## Evidence

Backend/repository-hygiene change — no web interface to screenshot. Verified via
the Rust test suite: the new test fails without a `CODEOWNERS` file and passes
once it is added.

```
running 5 tests
test codeowners_exists_and_is_non_empty ... ok
test every_rule_assigns_at_least_one_owner ... ok
test declares_a_default_fallback_owner ... ok
test covers_dependency_manifests ... ok
test covers_github_workflows ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured
```

`cargo clippy --all-targets --all-features -- -D warnings` and
`cargo check --all-targets --all-features` both pass.

> Note: `./quality.sh` runs `cargo upgrade --incompatible`, which bumped a GPU
> dependency to a version that fails to compile against the existing
> `src/analysis/gpu/*` source (a pre-existing incompatibility unrelated to this
> issue). That out-of-scope dependency bump was reverted so the baseline builds;
> `Cargo.toml` / `Cargo.lock` are unchanged by this PR.

## Test Plan

- Added `tests/issue_1486_codeowners.rs`:
  - `codeowners_exists_and_is_non_empty` — a `CODEOWNERS` file exists at the
    root, `.github/`, or `docs/` and is non-empty.
  - `every_rule_assigns_at_least_one_owner` — every rule names at least one
    `@user`, `@org/team`, or email owner.
  - `covers_github_workflows` — the privileged `.github/` CI paths are covered.
  - `covers_dependency_manifests` — `Cargo.toml` and `Cargo.lock` are covered.
  - `declares_a_default_fallback_owner` — a `*` default owner exists so no path
    is left unowned.
- Confirmed the test fails before `.github/CODEOWNERS` is added and passes after.
