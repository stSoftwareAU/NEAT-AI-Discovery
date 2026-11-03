## Summary

Six boolean toggles (`NEAT_AI_DISCOVERY_VERBOSE`, `_GPU_TIMING`, `_NEURON_TARGETS_OUTPUT_ONLY`, `_QUIET_GPU`, `_TIMING`, `_GPU_METRICS`) were read with `std::env::var(..).is_ok()`, so `=0`, `=false` and `=no` turned them on. They now go through `parse_bool_env`, which enables only on `1`/`true`/`yes` (trimmed, case-insensitive), matching `docs/CONFIGURATION.md`. Closes #2467.

Version bump: `Cargo.toml` (and the `Cargo.lock` entry) bumped from 0.74.328 to 0.74.329.

## Spec

### Intent and Rationale

- `NEAT_AI_DISCOVERY_QUIET_GPU=0` made `apply_mesa_suppression` write `EGL_LOG_LEVEL`, `MESA_GLSL_CACHE_DISABLE` and `MESA_DEBUG`, hiding the diagnostics the operator wanted back. Routing all six sites through the one existing helper fixes that and removes the only boolean toggles that ignored the documented contract.

### Essential Design Decisions

- Added `parse_bool_value(raw: Option<&str>) -> bool` in `src/config/helpers.rs` as the pure seam; `parse_bool_env` delegates to it, so no second copy of the truthy set exists.
- The four `OnceLock` accessors stay cached. Their wiring is tested through child processes, because a cached value cannot be re-read in-process.
- Behaviour change for operators: a variable set to anything other than `1`/`true`/`yes` (for example `=on`) no longer enables these six toggles. `docs/CONFIGURATION.md` states this.
- Review clarification: the configuration reference places this default-off rule before the separate default-on exception and explicitly lists rejected examples, including `on` and empty.

### Undiscoverable Facts

- None.

## Evidence

Backend-only change; no UI files touched. Regression tests in `tests/issue_2467_bool_toggle_env_parsing.rs`. Each re-executes the test binary with all six variables set to one value, so the real cached accessors are read:

- `tests/issue_2467_bool_toggle_env_parsing.rs::off_values_disable_all_six_toggles` — `0`, `false`, `no`, `FALSE` and empty leave all six off.
- `tests/issue_2467_bool_toggle_env_parsing.rs::on_values_enable_all_six_toggles` — `1`, `true`, `yes`, ` YES ` enable all six (positive control).
- `tests/issue_2467_bool_toggle_env_parsing.rs::unset_toggles_are_off`.

Each child prints a marker the parent asserts on, so a skipped probe cannot pass. Unit tests `parse_bool_value_accepts_one_true_yes` and `parse_bool_value_rejects_everything_else` are in `src/config/helpers.rs`.

## Test Plan

- Red check: with `src/config` restored to the base version and the new test kept, `off_values_disable_all_six_toggles` failed with ``accessor `verbose` returned true but expected false``. With the fix it passes.
- Existing tests edited: none changed an assertion. Only comments changed in `tests/issue_2272_config_small_modules.rs` and `tests/infrastructure/observability.rs`, so no assertion was removed.
- Callers checked: `quiet_gpu()` is read by `apply_mesa_suppression` in `src/analysis/utils/platform.rs`; its docs already say `=1`, so no change was needed. `timing_enabled` and `gpu_metrics_enabled` in `src/observability/mod.rs` delegate to the config accessors.
- Docs: `docs/CONFIGURATION.md` (presence-only paragraph and three row sentences removed), doc comments in `src/config/*.rs`, `src/observability/mod.rs`, `src/observability/gpu_metrics.rs` and `src/observability/phase_timer.rs`. Grepped `README.md`, `docs/` (excluding archive and audits) and `src` for `presence-only`, `0` included, `any set value` and `set (any value)`; none remain except the intentional "before Issue #2467" history sentence.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --lib config::` and the `issue_1611`, `issue_2096`, `issue_2272` tests passed.

Original trigger closed: `NEAT_AI_DISCOVERY_QUIET_GPU=0` (and `false`/`no`) now yields `quiet_gpu() == false`, so no Mesa variable is written; the same holds for the other five toggles, and every site uses the one helper, so no equivalent bypass remains.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh takes about 15 minutes, beyond the 600 s foreground cap; targeted fmt, clippy and tests listed in the Test Plan were run; CI runs the full gate" -->
