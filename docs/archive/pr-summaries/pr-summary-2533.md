## Summary

Non-serial tests in the `tests/infrastructure` binary that read or clear the process-global cancellation flag are now `#[serial]`, so the serial cancellation tests can no longer flip it under them. This covers the `RecordCache` tests named in the issue, the cache tests that build an LRU or compressed cache, and the analysis-pipeline tests. Test-only change, so no `Cargo.toml` version bump (the #2495 and #2531 test-only commits did not bump either). Closes #2533.

## Spec

### Intent and Rationale

- `RecordCache::get` calls `deadline_passed`, which is true whenever the global flag is set, even with no deadline. `#[serial]` only excludes other `#[serial]` tests, so parallel cache tests failed at random. I took the issue's first suggested fix (`#[serial]`) rather than a per-cache cancellation seam, which would change production code for a test-isolation problem.

### Essential Design Decisions

- Pipeline tests are serialised for a second reason: `analyze_all` calls `reset_cancellation()` at its head (`src/analysis/orchestration.rs`), so a parallel pipeline test can also clear a flag a serial test has just set and make that serial test fail.
- Left parallel on purpose: `stress_impact_computation_concurrent_determinism` and `stress_deadlock_detection_reports_clean_state_after_concurrent_work` (issue_832, `compute_impacts_public` only), and the detector tests in issue_487, issue_754 and issue_776 that #2495 already left parallel because their detectors do not read the flag.

### Undiscoverable Facts

- Observed with the unfixed binary: `--test-threads=8` with the filters `cache_ record_cache cancel deadline_passed reset_ memory_pressure` failed 2 of 60 runs, both in `cache_returns_consistent_data_across_concurrent_reads` (`.expect("Get should succeed")`, `issue_186_rwlock_cache.rs:141`).

## Evidence

Tests marked `#[serial]` (14):

- `tests/infrastructure/issue_186_rwlock_cache.rs`: all five tests.
- `tests/infrastructure/issue_1048_parquet_file_guard.rs`: `record_cache_file_guard_survives_path_deletion`, `lru_cache_holds_file_guard`, `compressed_cache_holds_file_guard` (the last two are included defensively; the reader checks the flag at batch boundaries).
- `tests/infrastructure/issue_196_cache_locality_benchmark.rs`: `cache_locality_optimization_preserves_correctness`.
- `tests/infrastructure/issue_228_zero_copy_buffer.rs`: `metadata_includes_zero_copy_status`.
- `tests/infrastructure/issue_832_deadlock_stress.rs`: the three tests that call `run_analysis_pipeline`.
- `tests/infrastructure/issue_988_graceful_gpu_error.rs`: `analyze_parallel_returns_structured_error_when_no_gpu`.

Binary re-check for other callers of `deadline_passed` / `cancellation::is_cancelled` reachable from non-serial tests: the remaining non-serial tests are signature, config, error-type, lock, logging or topology-cache-construction tests that do not reach them (`issue_1002` only checks signatures; `issue_677` fails JSON parsing before `analyze_all`).

## Test Plan

- Flake loop against the same filters, 8 test threads: unfixed 2 failures in 60 runs; fixed 0 failures in 300 runs. The failing assertion is the one quoted above.
- `cargo test --test infrastructure -- issue_186 issue_1048 issue_196 issue_228 issue_832 issue_988`: 29 passed. GPU-gated tests skip on this host.
- `cargo fmt --check` and `cargo clippy --tests -- -D warnings`: clean.
- No test assertion was added, removed or edited; only the `#[serial]` attribute, one `use serial_test::serial;` per file where missing, and a one-line comment per file.
- Not run on a GPU host: the GPU-gated pipeline tests (`issue_196`, `issue_228`, `issue_832`) therefore only compile here.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeds the 590 s foreground cap on this host (as in #2496/#2497/#2508/#2509/#2515); targeted tests, fmt and clippy were run; CI runs the full gate" -->

Not checked: the full `./quality.sh` locally; the GPU-gated tests executing on a GPU host.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
