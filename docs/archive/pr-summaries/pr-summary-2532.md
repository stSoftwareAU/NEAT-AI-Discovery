## Summary

Replaced the unrun shell test scripts with Rust integration tests that run in the standard Cargo test suite. The tests exercise the real `bump-deps.sh` and `scripts/benchmark-ci.sh` with isolated projects and deterministic command stubs. Closes #2532.

Version bump: `Cargo.toml` and both lockfiles record version `0.74.354`.

## Spec

### Intent and Rationale

- The shell scripts were not run by CI, leaving their useful checks outside the normal test gate. Rust integration tests put that coverage under `cargo test` and `./quality.sh`.

### Essential Design Decisions

- Tests use temporary manifests and lockfiles, fixed publish-time fixtures, and a stubbed Cargo executable so quarantine and compile outcomes are repeatable.
- The lockstep regression test observes the real Cargo no-op for a single exact-pinned package, then verifies the production script reapplies the safe family in one multi-package update.

### Undiscoverable Facts

- `cargo update --manifest-path <temp>/Cargo.toml -p js-sys@0.3.107 --offline` printed `Locking 0 packages to highest Rust 1.99.1 compatible versions`; the lockfile SHA-256 stayed unchanged (`5c240…`). The test stub mirrors that no-op behaviour.

## Evidence

- Backend and CLI-only change; no workflow files changed. Rust integration tests exercise both real scripts.
- `tests/issue_2532_shell_script_coverage.rs::safe_lockstep_family_is_reapplied_in_one_multi_package_update` verifies js-sys, wasm-bindgen and web-sys move together and are reported kept while a package with a young partner is rejected. Only the grouped Cargo call moves the family.
- `tests/issue_2532_shell_script_coverage.rs::reapply_holds_young_partners_and_transitives_while_keeping_safe_updates` checks young lockstep partners and transitive/new packages are held, independent safe updates remain, and failures name their causes. `fully_aged_refresh_is_unchanged_and_does_not_invoke_cargo` and `manifest_that_requires_a_young_package_fails_with_the_package_cause` cover the no-op and impossible-safe-resolution paths.
- The 29 `bump_deps_test.sh` cases map to the new Rust tests as follows: CLI/config and dry-run cases to `bump_cli_help_config_and_invalid_options_are_reported` and `bump_no_network_dry_run_is_read_only_and_repeatable`; helper parsing, manifest rollback, publish-time fixtures, epoch and lock helpers to `bump_source_helpers_parse_diff_restore_and_read_lock_versions` and `bump_publish_fixture_and_lock_helpers_fail_loud_and_cover_transitives`; quarantine planning to `bump_planner_quarantines_young_updates_new_packages_and_unknown_publish_times`; shell cases 26–29 to the three lock reapplication tests named above, plus two grouped-update failure-path tests (next bullet). The second-precision boundary cases remain covered by `tests/issue_1909_quarantine_second_precision.rs`; manifest validation remains covered by `tests/issue_1908_manifest_coverage.rs` and `tests/issue_1910_bump_deps_no_fixed_tmp.rs`; offline notice behaviour remains covered by `tests/issue_1994_no_network_notice.rs`.
- `lockstep_update_that_drags_in_a_young_package_is_rolled_back_and_rejected` covers the `lockstep update drags in-quarantine` branch of `bump_deps::reapply_stuck_lock_changes`: the grouped `cargo update` also bumps a young `pair-b`, so the lockfile must be restored to its pre-grouped state and every stuck member (js-sys, wasm-bindgen, web-sys, pair-a) reported `rejected` with that reason naming `pair-b`. `failed_lockstep_update_leaves_the_lockfile_unchanged_and_is_rejected` covers the `lockstep cargo update failed` branch: the grouped call exits non-zero, the lockfile is unchanged and every stuck member is rejected with the cargo error.
- The 10 `benchmark_ci_test.sh` cases map to `benchmark_cli_validation_and_discovery_work_in_an_isolated_manifest`, `benchmark_mode_selection_and_compile_results_use_cargo_stub`, and `benchmark_compare_mode_delegates_with_validated_threshold`. Compile success and failure use Cargo stubs, replacing the timeout-based probe. Threshold validation also remains covered by `tests/issue_1918_benchmark_threshold_validation.rs`.
- The Cargo stub models the observed exact-pin no-op and records its invocation arguments; benchmark compile stubs return explicit success or failure, matching the script's use of Cargo exit status.
- The obsolete unrun `tests/bump_deps_test.sh` and `tests/benchmark_ci_test.sh` suites are replaced by Cargo integration tests. The remaining behaviour includes avoiding a second update of an already moved partner and preserving Cargo's failure reason, covered by `tests/issue_2532_shell_script_coverage.rs::reapply_holds_young_partners_and_transitives_while_keeping_safe_updates`; grouped success is covered by `tests/issue_2532_shell_script_coverage.rs::safe_lockstep_family_is_reapplied_in_one_multi_package_update`.

## Test Plan

- Review regression checks: forcing grouped success to roll back failed `safe_lockstep_family_is_reapplied_in_one_multi_package_update`; removing the already-moved guard and discarding the Cargo error each failed `reapply_holds_young_partners_and_transitives_while_keeping_safe_updates`. The production script was restored after each mutation.
- `cargo fmt --all -- --check && cargo test --test issue_2532_shell_script_coverage --test issue_1909_quarantine_second_precision --test issue_1918_benchmark_threshold_validation -- --test-threads=1 < /dev/null` — passed; 15 new coverage tests, 6 second-precision tests and 9 threshold-validation tests.
- Lockstep mutation: changed the production grouped `cargo update` invocation to a single-package update; `safe_lockstep_family_is_reapplied_in_one_multi_package_update` failed on the grouped-update expectation. Restored the script.
- Quarantine-branch mutation: changed `if [[ -n "$plan" ]]` to `if false` in `reapply_stuck_lock_changes`; `lockstep_update_that_drags_in_a_young_package_is_rolled_back_and_rejected` failed. Failure-branch mutation: inverted the grouped `cargo update` failure test (`! true || cargo …`, so a successful call took the failure branch); both new lockstep tests failed. Restored the script.
- Benchmark mutation: forced the compile failure counter to remain zero; `benchmark_mode_selection_and_compile_results_use_cargo_stub` failed because the script returned success instead of failure. Restored the script.
- `timeout 900 ./quality.sh < /dev/null` — completed after passing formatting, ShellCheck, Cargo checks, the full Rust test suite, documentation and release build.
- Updated the existing comments in `tests/issue_1909_quarantine_second_precision.rs` and `tests/issue_1918_benchmark_threshold_validation.rs` to remove the stale statements about shell scripts. Deleted `tests/bump_deps_test.sh` and `tests/benchmark_ci_test.sh`; their assertions are replaced by the Rust test mappings above.
- The quality gate exposed an existing rustdoc warning promoted to an error in `src/parquet_format/footer.rs`; changed the intra-doc link to plain code text so the documented symbol remains clear without a private-item link.
