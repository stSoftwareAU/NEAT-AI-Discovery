## Summary

Updated the Version Management section in AGENTS.md to add a prominent reminder
that the version in `Cargo.toml` must always be incremented on any code change.
Without a version increment, remote and unattended machines will continue using
the old cached compiled library and never pick up new changes.

The update:
- Adds a critical callout explaining why version increments are essential
- Documents both the CI auto-increment mechanism and when manual increments are needed
- Adds a cross-reference from the CI Pipeline section to Version Management

Closes #725.

## Evidence

This is a documentation-only change (AGENTS.md). No UI, performance, or code
behaviour changes — verified by running `./quality.sh` which passed all checks.

## Test Plan

- No code changes; quality gate (`./quality.sh`) passes cleanly
- Verified the updated AGENTS.md renders correctly with proper markdown formatting
