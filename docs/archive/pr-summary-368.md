## Summary

Created `AGENTS.md` as the single source of truth for AI coding agent guidelines,
consolidating scattered development conventions from README.md. Updated README.md
to reference AGENTS.md for detailed coding guidelines, testing philosophy, and CI
details — keeping README.md focused on user-facing documentation.

### What changed

- **New file: `AGENTS.md`** — Comprehensive coding guidelines covering:
  1. Project overview and sole mission
  2. Architecture and source layout
  3. Coding conventions (Australian English, DRY, KISS, Rust best practices)
  4. Testing philosophy (TDD, test outcomes not implementation, unit tests vs benchmarks)
  5. Quality gate (`./quality.sh` steps and CI pipeline)
  6. Build and install instructions
  7. FFI contract (exported symbols, JSON interface, memory management)
  8. GPU requirement (Metal/Vulkan, minimum system requirements)
  9. Key invariants (forward-only activation, atomic record writes)
  10. Environment variables reference
  11. Candidate types reference
  12. Quick reference commands

- **Updated: `README.md`** — Replaced verbose development guidelines, testing
  philosophy, and CI sections with concise summaries that point to AGENTS.md.
  User-facing content (API docs, troubleshooting, build commands) remains intact.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- `./quality.sh` passes cleanly (no source code changes, only documentation)
- Verified all cross-references between AGENTS.md and README.md are valid
- Australian English spelling used throughout AGENTS.md
