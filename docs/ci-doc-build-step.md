# CI Documentation Build Step

**Issue:** #875

The following step should be added to the `quality` job in `.github/workflows/ci.yml`,
after the "Run tests" step and before any release build step:

```yaml
    - name: Build documentation
      run: RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

This mirrors the documentation build check already present in `quality.sh` (line 41)
and ensures that broken doc links, malformed doc comments, and missing documentation
for public items are caught in CI.

## Standalone Script

The same check is available as `scripts/doc-check.sh` for local use:

```bash
./scripts/doc-check.sh
```

## Note

Workflow files (`.github/workflows/*.yml`) require the `workflow` OAuth scope to push.
This change must be applied by a maintainer with appropriate permissions.
