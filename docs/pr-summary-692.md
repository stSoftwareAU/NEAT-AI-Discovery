## Summary

Add `clippy::filter_next` and `clippy::collapsible_if` denial lints to the Clippy invocation in CI (`ci.yml`), the local quality gate (`quality.sh`), and documentation (`AGENTS.md`) for consistency with sibling Rust repositories. Closes #692.

- `clippy::filter_next` — suggests `.find()` instead of `.filter().next()`
- `clippy::collapsible_if` — suggests merging nested `if` blocks where appropriate

No existing code violated these lints, so no source changes were required.

## Evidence

This is a CI/tooling change with no visual output. Verified by running `./quality.sh` locally — all checks pass cleanly with the new lints enabled.

## Test Plan

- Ran `cargo clippy --all-targets --all-features -- -D warnings -D clippy::uninlined_format_args -D clippy::filter_next -D clippy::collapsible_if` — zero warnings
- Ran full `./quality.sh` — all steps pass (fmt, clippy, check, tests, docs, release build)
