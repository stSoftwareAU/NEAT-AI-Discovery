## Summary

Bumped `actions/cache` in `.github/workflows/ci.yml` from v4.3.0 (commit `0057852b`, Node.js 20) to v5.0.5 (commit `27d5ce7f`, Node.js 24) to clear the runner deprecation warning. v5.0.5 was published on 2026-04-13, well outside the 24h supply-chain quarantine. Closes #1293.

## Evidence

Backend / CI workflow change — no UI to screenshot. The fix is a one-line SHA pin update:

```diff
-      uses: actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830 # v4.3.0
+      uses: actions/cache@27d5ce7f107fe9357f9df03efb73ab90386fccae # v5.0.5
```

SHA confirmed via `gh api repos/actions/cache/git/refs/tags/v5.0.5` → `27d5ce7f107fe9357f9df03efb73ab90386fccae`.

`./quality.sh` passed cleanly (fmt, clippy, all 171+ tests, release build) — the workflow change does not affect compiled code.

## Test Plan

- [x] `./quality.sh < /dev/null` passes
- [ ] CI run on this PR uses `actions/cache@v5.0.5` and no longer emits the Node.js 20 deprecation warning
