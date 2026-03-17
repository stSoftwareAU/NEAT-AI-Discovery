## Summary

Add `cargo doc --no-deps` with `RUSTDOCFLAGS="-D warnings"` to `quality.sh` to catch broken documentation links and doc warnings at build time. Fixes all 7 existing documentation warnings (broken intra-doc links and private item links). Closes #670.

## Changes

- **quality.sh**: Added documentation build step (`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`) after tests and before release build
- **AGENTS.md**: Updated quality gate and CI pipeline documentation to reflect the new step
- **Fixed 7 doc warnings**:
  - Escaped `[0,1]` and `[-1,1]` bracket notation in 4 doc comments that rustdoc interpreted as intra-doc links
  - Changed `[`private_const`]` doc links to backtick code spans in 3 places where the linked item is private

### CI workflow change (requires workflow token scope)

The following step should be added to `.github/workflows/ci.yml` in the `quality` job, after the "Build library" step:

```yaml
    - name: Build documentation
      env:
        RUSTDOCFLAGS: "-D warnings"
      run: cargo doc --no-deps
```

This could not be pushed from this PR due to OAuth token scope limitations.

## Evidence

This is a CI/build configuration change with no visual output. Evidence is the clean `quality.sh` run including the new `cargo doc --no-deps` step passing with `-D warnings`.

## Test Plan

- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` passes cleanly with zero warnings
- `./quality.sh` passes all checks including the new documentation build step
- No existing tests were modified or removed
