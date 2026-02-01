## Summary

Created `CONTRIBUTING.md` with development guidelines extracted from README.md,
following Rust ecosystem best practices. The README.md "Development" section has
been simplified to a quick-reference block with a link to the new file.

### Changes

- **CONTRIBUTING.md** (new): Covers getting started (prerequisites, building,
  testing), development workflow (TDD, quality gate, CI), code style (Australian
  English, Clippy, formatting), testing guidelines (unit tests vs benchmarks,
  test organisation, testing philosophy), PR process, and project structure.
- **README.md** (modified): Replaced the verbose Development section with a
  concise quick-reference and link to CONTRIBUTING.md. Added CONTRIBUTING.md to
  the Additional Documentation table.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.
The change is documentation-only.

## Test Plan

- No code changes; `./quality.sh` passes cleanly
- Verified all links in CONTRIBUTING.md reference existing files
- Australian English spelling used throughout
