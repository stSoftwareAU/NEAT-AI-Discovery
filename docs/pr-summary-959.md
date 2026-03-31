## Summary

Fix GitHub Actions not running automatically on dependency upgrade PRs, and add
library dependency upgrades to `quality.sh` so issues are caught locally.
Closes #959.

### Root Cause - CI Not Triggering

The `upgrade-dependencies.yml` workflow created PRs using `GITHUB_TOKEN`, which
is a known GitHub limitation: events created by `GITHUB_TOKEN` do not trigger
other workflows. Changed to use `ACTIONS_PUSH` PAT (matching the pattern already
used by `version-increment` and `auto-format` jobs in `ci.yml`).

### Changes

1. **`quality.sh`**: Added `cargo upgrade --incompatible` + `cargo update` step
   before the licence audit, so dependencies are upgraded to latest versions
   (including major version bumps) during local quality checks.

2. **`.github/workflows/upgrade-dependencies.yml`**:
   - Switched from `GITHUB_TOKEN` to `ACTIONS_PUSH` PAT for checkout and PR
     creation, so the created PR triggers CI workflows.
   - Added `--incompatible` flag to `cargo upgrade` to include major version
     bumps.

3. **Dependency API migrations** (required by incompatible upgrades):
   - **wgpu 28 -> 29**: `bind_group_layouts` now takes `Option<&BindGroupLayout>`;
     `Instance::new` takes owned `InstanceDescriptor`.
   - **rand 0.8 -> 0.10**: `thread_rng()` renamed to `rng()`;
     `gen()` moved to `RngExt` trait; `distributions` module renamed to `distr`.
   - **criterion 0.5 -> 0.8**: `criterion::black_box` deprecated in favour of
     `std::hint::black_box`.
   - **arrow/parquet 57 -> 58, dashmap 5 -> 6, lz4_flex 0.11 -> 0.13,
     signal-hook 0.3 -> 0.4, uuid 1.22 -> 1.23, proptest 1.10 -> 1.11**:
     Compatible API changes, no code modifications required.

4. **`AGENTS.md`**: Updated quality gate documentation to reflect the new
   dependency upgrade step.

## Evidence

- `./quality.sh` passes cleanly with all upgraded dependencies
- All tests pass: `cargo test --lib --tests --all-features -- --test-threads=2`
- Clippy clean: `cargo clippy --all-targets --all-features -- -D warnings`

## Test Plan

- Added `tests/test_quality_upgrade.sh` verifying:
  - `cargo-upgrade` command is available
  - `cargo upgrade --incompatible --dry-run` succeeds
  - `Cargo.toml` has required dependency sections
  - `Cargo.lock` exists for reproducible builds
