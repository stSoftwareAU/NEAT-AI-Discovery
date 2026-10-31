## Summary

`detect_topology_issues` no longer runs the uncancellable per-neuron O(H·O) nearest-output scan or the O(H·I) input `find`, and every loop now observes the analysis deadline and global cancellation. Closes #2369.

- New `detect_topology_issues_with_deadline` returns `BoundedScan<TopologyCandidate>`: valid candidates found before an interruption are kept and `truncation = Some(DeadlinePassed)` is set. `detect_topology_issues` is kept as a wrapper with no deadline, which still observes global cancellation.
- Long paths: all outputs sit at reverse-BFS distance 0, so the old `min_by_key` always picked the first unconnected output. A hidden neuron more than three hops from an output cannot already connect to one, so one output is selected once and reused. The existing `synapse_exists` filter stays as a guard against a stale cache.
- Starved neurons: the cached input UUID set is copied into a compact vector once; each starved neuron stops at its first input without an edge (`first_unconnected_input`, deadline checked per input).
- `compute_shortest_paths_to_output` takes the deadline and returns `None` when interrupted.
- The topology spec in `structural_specs.rs` passes the analysis deadline and logs a `tracing::warn!` when the scan stops early.
- Version bump: `Cargo.toml` version updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The issue asks for a bounded, cancellable scan. Every loop that scales with H, I, O or the record count checks cancellation, and the two per-neuron scans are removed rather than only guarded.

### Essential Design Decisions

- One output is reused for all long-path candidates. This relies on every output having BFS distance 0 and on a >3-hop neuron having no direct output edge; `synapse_exists` is still checked per neuron, so a stale cache yields `skip_target_uuid = None` rather than a duplicate edge.
- Partial results are returned and sorted, not discarded, matching the earlier cancellable detectors (#2236 / #2169).
- The issue's suggested wall-clock-style ratio test is not used: `CONTRIBUTING.md` forbids timing thresholds in unit tests. Cancellation is tested deterministically with `DeadlineOverrideGuard`; timings live in `benches/topology_scan.rs`.
- The deadline test module is `src/analysis/topology_deadline_tests.rs`, declared from `topology.rs` through `#[path]`. It sits outside `src/analysis/detection/` so the chunk 8a audit file set (swept files) is unchanged.

### Undiscoverable Facts

- The chunk 8a audit meta-tests (`tests/issue_2154_*`, `2280`, `2281`, `2282`, `2284`, `2294`, `2298`, `2300`) parse `docs/audits/security-sweep-chunk-08a-detection-neuron.md`: the six capacity sites now in `detect_topology_issues_with_deadline` each need a capacity row.

## Evidence

Backend-only change; no UI files touched.

Benchmark (`benches/topology_scan.rs`, criterion, `--warm-up-time 1 --measurement-time 2`; base is `0fdc998` with only the bench file added; the comparison uses the pre-existing `detect_topology_issues` API):

| Case | Base | Head |
| --- | --- | --- |
| `long_path_sparse` hidden=64 | 78.1 µs | 49.5 µs |
| `long_path_sparse` hidden=256 (outputs=64) | 676.6 µs | 217.1 µs |
| `long_path_dense` hidden=256 | 3.139 ms | 2.914 ms |
| `starved_sparse` hidden=256 | 334.4 µs | 332.5 µs |
| `starved_dense` hidden=256 | 6.695 ms | 6.590 ms |

A separate local run of `long_path_sparse_deadline/hidden=64` with a far-future deadline measured 105.31 µs (Criterion estimate, 87.620–123.78 µs interval, 10 samples). This run used the current working tree and a different machine from the comparison above, so the figures are not directly comparable.

The long-path O(H·O) scan is removed (about 3.1x faster at hidden=256 with many outputs). The starved-neuron fixtures find an unconnected input at the front of the scan, so their time is dominated by other work and is unchanged; the change there is the cancellation checks and the removal of the per-neuron set iteration, not a measured speed-up.

Security-fix evidence:

- Regression test: `tests/issue_2369_topology_scan.rs::existing_detector_stops_when_global_cancellation_is_requested`. Run against base `0fdc998` production code with the new test file copied in, it fails (the base returns candidates after cancellation is requested). `local_and_cached_topology_preserve_long_path_and_imbalance_candidates` preserves behaviour and passes on both base and head; the current test was rerun against a clean archive of base `0fdc998`.
- Original trigger closed: the per-neuron `output_uuids.iter().filter().min_by_key()` and `input_uuids.iter().find()` scans over all outputs and inputs are gone, and the detector, its BFS, record lookup, long-path, fan-in and input loops all call `deadline_passed`. Error aggregation checks the deadline after at most 1,024 record or error elements, as well as before each qualified neuron.

Not run: a break-the-guard check of each individual deadline check by deleting it; the deterministic tests below count the checks passed before the stop, which would shift if one were removed.

## Test Plan

- `src/analysis/topology_deadline_tests.rs`: `elapsed_deadline_is_reported_before_preparation`, `preparation_deadline_is_reported_during_record_lookup`, `long_path_loop_deadline_keeps_candidates_found_before_interruption`, `imbalance_loop_deadline_keeps_candidates_found_before_interruption`, `input_scan_checks_cancellation_between_existing_edges`. Passed.
- `src/analysis/module_dispatch_specs/structural_specs_issue_2369_tests.rs`: `topology_spec_forwards_elapsed_deadline_to_detector`. Passed.
- `tests/issue_2369_topology_scan.rs`: four tests, passed at head; the cancellation regression fails on base as stated above.
- Audit record meta-tests (`issue_2154`, `2280`, `2281`, `2282`, `2284`, `2294`, `2298`, `2300`): passed (8 and 42 tests) after the six capacity rows were added.
- Docs: `docs/DISCOVERY_TYPES.md` (topology scan bounds paragraph), `docs/audits/security-sweep-chunk-08a-detection-neuron.md` and `docs/BENCHMARKS.md` (topology suite row and both suite totals, now 45) updated.
- CI fix for PR #2481: `cargo test --test issue_1685_doc_link_integrity benchmarks_doc_matches_cargo_bench_targets -- --exact` reproduced the stale-count assertion before correcting the totals. The same command passes after the fix, and `cargo test --test issue_1685_doc_link_integrity --all-features -- --test-threads=2` passes all nine tests. The existing test derives the count from `Cargo.toml`; no test changes were needed. A docs sweep found both stale totals in `docs/BENCHMARKS.md`; the count in `pr-summary-2350.md` describes its historical change and remains true.
- Merged `origin/Develop` (including #2374) into the branch; the targeted tests above and `cargo test --lib topology` (14 passed) were re-run after the merge.
- Original implementation validation, before the documentation-only CI fix: `./quality.sh` stages (deny, build, fmt, clippy, check) completed clean up to its test stage; because the test stage exceeds a single foreground call, the remaining stages were run separately: `cargo test --lib --all-features` (1696 passed, 0 failed, 3 ignored), `cargo test --test '*' --all-features` (273 test binaries, all `ok`), `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` (clean) and `cargo build --release --lib` (clean).

- CI-fix validation: `timeout 900 ./quality.sh < /dev/null` passed Bash syntax, ShellCheck, install pinning, summary location, dependency audit, debug build, formatting, Clippy and type checks, then timed out (exit 124) compiling the full test stage. Not completed in this run: full test suite, rustdoc, release build or instrumented coverage. The targeted documentation suite passed as recorded above.

## Merge verification

- CI and code review: pending until observed.
