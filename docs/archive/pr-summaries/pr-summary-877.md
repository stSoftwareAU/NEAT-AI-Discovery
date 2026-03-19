## Summary

Add benchmark regression detection infrastructure for CI. Closes #877.

Creates `scripts/benchmark-ci.sh`, a CI-friendly script with two modes:
- **Compile-only** (default): Verifies all benchmark targets compile via `cargo bench --no-run` — fast, no GPU required, suitable for every PR on standard GitHub Actions runners.
- **Compare**: Runs benchmarks against a saved baseline using the existing `benchmark_compare.sh` and fails if any benchmark regresses beyond a configurable threshold (default: 10%).

The script discovers benchmark suites from `Cargo.toml` automatically, validates inputs, and handles GPU-less environments gracefully.

## Evidence

- All 25 tests in `tests/benchmark_ci_test.sh` pass, covering: CLI flag parsing (--help, -h, --list, --compile-only, --compare, --threshold), error handling (unknown flags, missing threshold value, non-numeric threshold, invalid mode), benchmark discovery from Cargo.toml (finds all 22 suites), and environment variable override.
- `./quality.sh` passes cleanly with the new scripts.

## Test Plan

- Added `tests/benchmark_ci_test.sh` with 25 test assertions covering:
  - Script exists and is executable
  - `--help` and `-h` display usage information
  - Unknown options exit with error
  - `--threshold` requires a value and validates it is numeric
  - `--list` discovers all benchmark suites from Cargo.toml
  - `BENCHMARK_CI_MODE` environment variable is respected
  - `--compile-only` flag is accepted and shows correct mode
  - Benchmark suite count is >= 10

## Changes

- `scripts/benchmark-ci.sh` — New CI benchmark verification script
- `tests/benchmark_ci_test.sh` — Tests for the CI benchmark script
- `docs/BENCHMARKS.md` — Updated CI integration section with regression threshold documentation, disk space guidance, and GPU benchmark notes
