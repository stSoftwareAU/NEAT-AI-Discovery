## Summary

Serialised the GPU environment smoke and setup tests, and GPU initialisation tests that read the same environment variables, to prevent the `Once` setters and setup probes from racing. Added a shared guard that restores all four GPU setup variables to either their original value or unset state. Test-only change. Closes #2538.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The issue #713 smoke tests could set `XDG_RUNTIME_DIR` between the two issue #1873 setup calls, changing their verdict. Serialisation protects the shared process environment.
- GPU initialisation also reads these variables through `setup_gpu_environment`; its callers in the `gpu` integration binary use the same serial lock.

### Essential Design Decisions

- The lock assertions make each serialization requirement fail deterministically if its attribute is removed.
- `EnvGuard` preserves both present and absent values for `XDG_RUNTIME_DIR`, `EGL_LOG_LEVEL`, `MESA_GLSL_CACHE_DISABLE` and `MESA_DEBUG`.

### Undiscoverable Facts

- The original intermittent verdict mismatch was not directly reproduced; its trigger depends on test scheduling and the process-wide `Once`.

## Evidence

Backend/test-infrastructure change; no UI. `src/analysis/gpu/analyzer.rs` calls `setup_gpu_environment` from GPU availability checks and analyzer construction, and `src/analysis/gpu/device.rs` does so before adapter probing. Those indirect GPU initialisation tests now use the same serial lock as the environment setters.

- #713: Deduplicate env::set_var code between memory.rs and platform.rs.
- #1873: Safe GPU setup fns assert an unenforceable thread-free invariant around unsafe env writes.
- #2538: GPU integration test binary race between the issue #713 smoke tests and issue #1873 verdict test.

## Test Plan

- `timeout 900 cargo test --locked --all-features --test gpu env_setup -- --test-threads=2`: 6 passed.
- Same targeted command in a separate process with all four setup variables and `NEAT_AI_DISCOVERY_QUIET_GPU` unset: 6 passed.
- Same targeted command in a separate process with `NEAT_AI_DISCOVERY_QUIET_GPU=1` and the four setup variables unset: 6 passed.
- `timeout 900 cargo test --locked --all-features --test gpu -- --test-threads=2`: 66 passed.
- `tests/gpu/env_guard.rs::tests::capture_gpu_setup_vars_restores_set_and_unset_values`: passed. Removing either the present-value restore or absent-value removal made its corresponding restoration assertion fail.
- Removed each of the six issue #713/#1873 serial attributes individually and ran its test: each failed at its `is_locked_serially` assertion; all attributes were restored.
- Removed each of the 13 serial attributes on indirect GPU initialisation tests individually: each failed at the shared serial-lock assertion in its GPU availability helper; all attributes were restored.
- Existing assertions were preserved. No UI or production code changed.
- `cargo fmt --all -- --check`: passed.
- `cargo clippy --locked --all-features --test gpu -- -D warnings`: passed.
- `./quality/fuzz_lockfile.sh check`: passed; `fuzz/Cargo.lock` remains synchronised with the retained version bump.
- `timeout 900 ./quality.sh < /dev/null`: all 1,848 library tests and integration test targets passed. The gate then failed in rustdoc at `src/parquet_format/footer.rs:66`: public documentation for `uuid_heap_bytes` links to private `neuron_uuid_heap_bytes`. That line is present on `origin/Develop` and the file is unchanged by this branch; the scope restriction kept this PR test-only. The release-library build was not reached.
- The original intermittent race itself was not reproduced.
