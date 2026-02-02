## Summary

Archived 63 historical PR summary files (`pr-summary-164.md` through
`pr-summary-373.md`) from `docs/` into a new `docs/archive/` subdirectory.
This declutters the docs directory so the core documentation files
(`DISCOVERY_TYPES.md`, `IMPACT_CALCULATION.md`, etc.) are easier to find.

A `docs/archive/README.md` explains the purpose of the archived files.
`CONTRIBUTING.md` was updated to note the archive location.

Closes #374 (parent: #366).

## Evidence

Unable to generate screenshot: this is a file-organisation change with no
visual interface.

## Test Plan

- Verified `./quality.sh` passes (no broken references, all tests green)
- Confirmed no source code references to the moved files require updating
- Only reference found in `CONTRIBUTING.md` was updated to note the archive
