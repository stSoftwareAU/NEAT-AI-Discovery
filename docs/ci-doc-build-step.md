# CI Documentation Build Step

**Issue:** #875
**Status:** ⏳ **Pending proposal — not yet implemented.** No `cargo doc` step
exists in any workflow under `.github/workflows/`. The only documentation check
in CI today is the "Check documentation" step in `ci.yml` (`ci.yml:362-379`),
which merely greps `src/lib.rs` for `///` doc comments — it does **not** build
the docs or fail on broken doc links.

The following step should be added to the `quality` job in `.github/workflows/ci.yml`,
after the "Run tests" step and before any release build step:

```yaml
    - name: Build documentation
      run: RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

This mirrors the documentation build check already present in `quality.sh`
(the "Building documentation" step, `quality.sh:72-73`) and ensures that broken
doc links, malformed doc comments, and missing documentation for public items
are caught in CI.

## Flag consistency

The three doc-build invocations currently disagree on `--all-features`:

| Location | Command | `--all-features`? |
|----------|---------|-------------------|
| `quality.sh:73` | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | ❌ no |
| `scripts/doc-check.sh:13` | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` | ✅ yes |
| Proposed CI step (above) | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` | ✅ yes |

When this step is implemented, align all three on the same flag set (preferably
`--all-features`, so feature-gated public items are also doc-checked) so the CI,
the local script, and the quality gate cannot drift.

## Standalone Script

The same check is available as `scripts/doc-check.sh` for local use:

```bash
./scripts/doc-check.sh
```

## Note

Workflow files (`.github/workflows/*.yml`) require the `workflow` OAuth scope to push.
This change must be applied by a maintainer with appropriate permissions.
