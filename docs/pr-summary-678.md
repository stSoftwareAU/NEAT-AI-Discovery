## Summary

Added `upgrade-dry-run.txt` and `upgrade.log` to `.gitignore` and removed them from version control. These are artefacts from the dependency upgrade workflow and should not be tracked. Closes #678.

## Evidence

This is a configuration-only change (`.gitignore`). No UI or performance changes. Verified with:
- `git ls-files upgrade-dry-run.txt upgrade.log` returns empty (files no longer tracked)
- `./quality.sh` passes cleanly

## Test Plan

- No Rust code changes; no new tests required
- Verified files are untracked after `git rm --cached`
- Verified `.gitignore` entries prevent future tracking
