## Summary

Removed the dead Deno/Mermaid validation block from `.github/workflows/markdown-lint.yml`. This repository is a pure Rust library (`Cargo.toml`, `src/lib.rs`, no `deno.json`/`deno.lock`) with no `worker/` directory, so the three steps that referenced `worker/deno/mod.ts` could never execute here:

- **`Detect Deno worker module`** — `[ -f worker/deno/mod.ts ]` always resolved to `present=false`.
- **`denoland/setup-deno`** — gated on `present == 'true'`, never ran.
- **`Validate Mermaid blocks`** — same gate, never ran.

The inert steps carried an ongoing maintenance cost (the `denoland/setup-deno` SHA pin still had to be kept current) and added review noise for a code path that can never run in this repo. Deleting them leaves the `markdownlint-cli2` gate intact and also drops the stale `setup-deno` pin so Renovate/`bump-deps` no longer track it here.

The job-level comment was trimmed to match (`optional Mermaid validation` removed) so the workflow description stays accurate.

Closes #1504.

## Evidence

Backend/CI-only change — no web interface to screenshot. Verification performed:

- `python3 -c "yaml.safe_load(...)"` → **YAML OK** (workflow parses cleanly after the edit).
- `grep -rn "worker/deno|setup-deno|check-mermaid|detect-deno" .github/` → **no dead references remaining**.

Resulting `markdownlint` job flow:

```mermaid
flowchart LR
    A[checkout] --> B[setup-node 22]
    B --> C[install markdownlint-cli2]
    C --> D[run markdownlint-cli2]
```

## Test Plan

This is a GitHub Actions workflow change with no Rust code impact, so the repo's `quality.sh` (fmt/clippy/test/build) gate is not exercised by the change. Validation instead:

- Confirmed the workflow YAML remains valid after removal.
- Confirmed no `worker/deno`, `setup-deno`, `check-mermaid`, or `detect-deno` references remain under `.github/`.
- Confirmed the `markdownlint-cli2` install + run steps are unchanged and intact.
