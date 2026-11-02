## Summary

Registered the existing shared-cache contention Criterion benchmark with
`harness = false`, updated the suite count to 47, and bumped the crate version.
Closes #2414.

## Spec

### Intent and Rationale

- Cargo's auto-discovered test harness reported zero benchmarks for this Criterion target, so the suite was added to the existing explicit benchmark registry.

### Essential Design Decisions

- The benchmark implementation is unchanged; the manifest entry lets Cargo and both suite-discovery scripts invoke it as a Criterion benchmark.

### Undiscoverable Facts

- Before registration, `cargo bench --bench impact_cache_contention -- --list` reported `0 tests, 0 benchmarks`. After registration, it listed all three cases and Criterion measured each one.

## Evidence

This is a manifest and documentation change with no UI surface. The benchmark timings below verify execution only and make no speed-up claim.

## Test Plan

- Before registration, `./benchmark_compare.sh --list` and `./scripts/benchmark-ci.sh --list` each reported 46 suites without `impact_cache_contention`; after registration each reported 47 and included it.
- Before registration, `cargo bench --bench impact_cache_contention -- --list` exited successfully with `0 tests, 0 benchmarks`.
- After registration, `cargo bench --bench impact_cache_contention -- --list` listed `wide_200_neurons`, `deep_100_layers` and `mesh_10x20`.
- `timeout 300 cargo bench --bench impact_cache_contention -- --warm-up-time 0.1 --measurement-time 0.2 --sample-size 10 < /dev/null` passed: wide median 575.83 µs (range 369.76–853.91 µs), deep median 335.08 µs (range 272.52–423.47 µs), mesh median 2.3650 ms (range 1.9193–2.8148 ms).
- `timeout 120 cargo fmt --all -- --check < /dev/null` passed.
- `timeout 900 ./quality.sh < /dev/null` timed out with exit 124 on all three attempts. The second attempt reached the release-library build after its tests and documentation build; the third timed out while the unit-test suite was running. The full gate remains unverified.
