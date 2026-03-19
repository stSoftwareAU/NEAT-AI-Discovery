## Summary

Add cargo doc build and fuzzing support for CI pipeline integration. Closes #875.

### What was done

1. **Documentation build check** (`scripts/doc-check.sh`): Standalone script that runs
   `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` to catch broken doc
   links, malformed doc comments, and missing documentation. Verified the codebase has
   zero doc warnings.

2. **Fuzzing CI script** (`scripts/fuzz-ci.sh`): Runs both fuzz targets
   (`fuzz_ffi_deserialisation`, `fuzz_ffi_entry_points`) for a configurable duration
   (default 30s each). Installs nightly toolchain and cargo-fuzz if needed.

3. **Fuzzing workflow template** (`docs/ci-fuzzing-workflow.yml`): Ready-to-use GitHub
   Actions workflow for fuzzing. Includes disk cleanup, nightly toolchain setup, and
   dependency caching. Can be copied to `.github/workflows/fuzzing.yml` by a maintainer.

4. **CI doc build step documentation** (`docs/ci-doc-build-step.md`): Documents the exact
   YAML step to add to the `quality` job in `ci.yml`.

### Note on workflow files

Workflow files (`.github/workflows/*.yml`) require the `workflow` OAuth scope to push,
which the automated worker does not have. The workflow changes are provided as
documentation/templates for a maintainer to apply:
- `docs/ci-fuzzing-workflow.yml` → copy to `.github/workflows/fuzzing.yml`
- `docs/ci-doc-build-step.md` → add the documented step to `.github/workflows/ci.yml`

## Evidence

- `cargo doc` builds with zero warnings (verified via `./scripts/doc-check.sh`)
- `./quality.sh` passes cleanly with all changes
- Both shell scripts pass `bash -n` syntax checks
- Fuzz targets already exist and are well-structured in `fuzz/fuzz_targets/`

## Test Plan

- Verified `./scripts/doc-check.sh` runs successfully with no warnings
- Verified `./scripts/fuzz-ci.sh` syntax is correct (`bash -n`)
- Verified `./quality.sh` passes with all changes
- README updated with new script references
