## Summary

Cleaned up the `docs/` directory by archiving all PR summary files into
`docs/archive/pr-summaries/`. Moved 98 `pr-summary-*.md` files from `docs/`
and 63 from `docs/archive/` into the new consolidated subdirectory. Updated
`CONTRIBUTING.md` to reference the new PR summary location and added a
`README.md` explaining the archive contents. Closes #606.

## Evidence

This is a file organisation change with no UI or performance impact. Verified
by running `./quality.sh` which passes cleanly — all 509 unit tests and
integration tests pass, and the release build succeeds.

Before: `docs/` contained 98 `pr-summary-*.md` files alongside 8 core
documentation files, making navigation difficult.

After: `docs/` contains only core documentation files
(`ANALYSIS_DEEP_DIVE.md`, `BENCHMARKS.md`, `DISCOVERY_TYPES.md`, `FFI_API.md`,
`GPU_GUIDE.md`, `IMPACT_CALCULATION.md`) plus the `archive/` and `discoveries/`
subdirectories.

## Test Plan

- No new tests required — this is a documentation reorganisation with no code changes
- Verified `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
- Verified no broken internal links remain in active documentation files
