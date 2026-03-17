## Summary

Reorder CI pipeline so `version-increment` runs first with no dependencies,
preventing wasted runner time from re-triggered workflow runs after version bump
commits. Closes #690.

**Before:** `version-increment` depended on `auto-format`, so it ran late.
When it pushed a version-bump commit, all other jobs (quality, validation, etc.)
would re-run unnecessarily.

**After:** `version-increment` has no dependencies (runs first). All other jobs
(`auto-format`, `quality`, `validation`, `shell-checks`, `spell-check`,
`security`) depend on `version-increment`. On the first run, version-increment
may push a commit; the re-triggered run then skips version-increment (already
done) and runs all other jobs without further re-triggers.

### Dependency graph change

```
Before:
  auto-format -> version-increment -> quality
  (validation, shell-checks, spell-check, security run independently)

After:
  version-increment -> auto-format -> quality
  version-increment -> validation
  version-increment -> shell-checks
  version-increment -> spell-check
  version-increment -> security
```

## Evidence

This is a CI configuration-only change (`.github/workflows/ci.yml`). No Rust
source code, visual output, or performance characteristics are affected.
`quality.sh` passes cleanly with all existing tests.

## Test Plan

- Verified YAML syntax is valid
- Ran `./quality.sh` — all checks pass (fmt, clippy, check, tests, doc, release build)
- CI will validate the workflow executes correctly on the PR itself
