## Summary

Add benchmark regression tracking using Criterion's built-in baseline comparison. The new `benchmark_compare.sh` script saves baseline benchmark results and compares subsequent runs against them, reporting regressions that exceed a configurable threshold (default: 5%). Closes #576.

### What was added

- **`benchmark_compare.sh`** — Shell script that:
  - Discovers all 12 benchmark suites from `Cargo.toml` automatically
  - Saves baseline results via `--save-baseline` (stored in `target/criterion/`)
  - Compares current results against the baseline with configurable threshold
  - Reports regressions, improvements, and unchanged benchmarks
  - Exits with code 1 if any regressions exceed the threshold
  - Handles GPU-less environments by skipping unavailable benchmarks
  - Compatible with macOS (bash 3.2), Ubuntu, and AWS Linux

- **`docs/BENCHMARKS.md`** — Documentation covering:
  - Quick start workflow (save baseline, make changes, compare)
  - All commands and environment variables
  - How to interpret Criterion results
  - CI integration guidance
  - Updating baselines after intentional changes

## Evidence

This is a CLI/script change with no visual output. Evidence is provided by the integration tests.

Baselines are machine-specific (different CPUs/GPUs produce different timings), so they are stored locally in `target/criterion/` rather than committed to the repository. The script manages baselines per-machine.

## Test Plan

Added 6 integration tests in `tests/issue_576_benchmark_regression_tracking.rs`:

- `benchmark_suites_have_source_files` — verifies all 12 benchmark suites have corresponding `.rs` files in `benches/`
- `benchmark_compare_script_exists_and_is_executable` — verifies the script exists and passes bash syntax check
- `benchmark_compare_list_discovers_all_suites` — verifies `--list` discovers all 12 benchmark suites from `Cargo.toml`
- `benchmark_compare_help_shows_usage` — verifies `--help` documents all options
- `benchmark_compare_rejects_unknown_bench` — verifies `--bench nonexistent` fails with a clear error
- `benchmark_compare_without_baseline_reports_error` — verifies graceful handling when no baseline exists
