## Summary

Add cargo-deny configuration for licence compliance, dependency auditing, advisory scanning, and duplicate detection. Closes #667.

- Created `deny.toml` at the project root with:
  - Licence allowlist (Apache-2.0, MIT, BSD-2/3-Clause, BSL-1.0, ISC, Unlicense, Zlib, 0BSD, CC0-1.0, Unicode-3.0) — all compatible with Apache-2.0
  - Exception for `r-efi` (LGPL-2.1-or-later) — transitive UEFI dependency not linked on macOS/Linux
  - Advisory database scanning with an ignore for RUSTSEC-2024-0436 (`paste` — unmaintained transitive dep from metal/parquet, awaiting upstream migration)
  - Duplicate dependency detection (warnings for transitive duplicates: foldhash, getrandom, hashbrown, lz4_flex, ordered-float)
  - Source restriction to crates.io only
- Added `license = "Apache-2.0"` to `Cargo.toml`
- Added `cargo deny check` step to `quality.sh` (runs early, before build)
- **Note**: `.github/workflows/security.yml` should also be updated to run `cargo deny check` alongside `cargo audit`. This requires `workflow` scope on the GitHub token and should be done in a separate commit by a maintainer. Suggested diff:
  ```yaml
  - name: Install cargo-audit and cargo-deny
    run: cargo install cargo-audit cargo-deny
  # ... after cargo audit step:
  - name: Run licence and dependency audit
    run: cargo deny check
  ```

## Evidence

This is a configuration/CLI change with no UI. Evidence is the clean `cargo deny check` output:

```
advisories ok, bans ok, licenses ok, sources ok
```

All quality checks pass (`./quality.sh` completes successfully).

## Test Plan

- `cargo deny check` passes cleanly (all four checks: advisories, bans, licences, sources)
- `./quality.sh` passes with the new cargo-deny step integrated
- CI security workflow updated to run `cargo deny check` alongside existing `cargo audit`
