# PR Summary — Issue #2332

## Summary

Closes #2332

The GPU capability probe called `pollster::block_on` on wgpu's
`request_adapter` and `request_device` with no deadline. If a driver never
answered, `check_gpu_availability()` and `get_adapter_info_internal()` blocked
forever. Their `OnceLock::get_or_init` callers (`gpu_is_available`,
`get_adapter_info`, `supports_unified_memory`) then wedged every later analysis
call.

The probe now uses the same bound as `GpuWorkQueue::new`:

- `run_gpu_probe_with_timeout` (`src/analysis/gpu/device.rs`) runs the probe on
  a named thread and waits with `recv_timeout(GPU_INIT_TIMEOUT_SECS)`.
- **On timeout**, it trips `global_gpu_breaker()` with
  `GpuTripReason::InitTimeout` and leaks the stuck thread. The verdict is
  `is_error: true` with the reason
  "GPU capability probe timed out after 30s". Over FFI this becomes
  `success: false`, `gpu_permanent`, `retryable: false`.
- **If the probe thread panics or cannot spawn**, the call fails with an error
  and the breaker is not tripped.
- `setup_gpu_environment()` stays on the caller thread and runs before the
  probe thread is spawned (Issue #1873).
- `check_gpu_availability_with(timeout, breaker, probe)` is the injectable
  seam for tests.

**PR review follow-up:** the first cut above only tripped the breaker on a
new timeout — it never *checked* the breaker before spawning. On a hung
driver, `check_gpu_available` is called uncached on every analysis pass
(`RustDiscoveryOperations.ts`), so after the first trip every later call still
spawned a fresh `gpu-capability-probe` thread, waited out the full
`GPU_INIT_TIMEOUT_SECS` again, and leaked that thread — one leaked thread per
call instead of the one stuck thread this PR was meant to bound. Fixed by
checking `breaker.is_tripped()` at the top of `run_gpu_probe_with_timeout`
(the shared root both `check_gpu_availability_with` and
`get_adapter_info_internal` call through), returning a new
`BoundedProbe::Tripped` variant without spawning. `check_gpu_availability_with`
maps `Tripped` to a new `tripped_probe_result(breaker)` (`is_error: true`,
`gpu_permanent`, reason naming the original trip cause);
`get_adapter_info_internal` maps it to `None`, alongside `TimedOut`/`Failed`.

Version `0.74.273`.

## Spec

### Intent and Rationale

- A driver that never answers must not stop the process from answering.
  Instead, the probe gives a bounded, typed verdict.
- The fix reuses the existing init-timeout pattern and breaker reason, so there
  is one contract for "the GPU never came up".

### Essential Design Decisions

- The stuck thread is leaked rather than joined. Joining would block on the
  same hung driver call.
- A timeout trips the breaker, so later `GpuWorkQueue::new` calls fail fast
  instead of spawning more threads on dead hardware. A panic does not trip it.
- The timeout verdict is `is_error: true` (permanent, not retryable), so the
  host does not retry it with a longer deadline.

### Undiscoverable Facts

None.

## Evidence

This is a backend-only change with no visual surface, so there is no
screenshot.

- **Regression test:**
  `src/analysis/gpu/issue_2332_probe_timeout_test.rs::hung_probe_times_out_and_trips_breaker`.
  It injects a probe that never resolves (the original trigger: an
  adapter/device request that never returns). It asserts that the call returns
  the timeout verdict within the bound, and that the breaker is tripped with
  `InitTimeout`.
  - This regression test reproduces the flaw. It fails against the unfixed
    code and passes after the fix.
  - Against base, the unbounded `block_on` has no deadline, so the call never
    returns, and the seam does not exist.
  - Removing the trip, or mapping `TimedOut` to a non-error result, turns it
    red.
- **The original trigger is closed, with no trivial bypass.** Every
  adapter/device request in the probe paths now goes through
  `run_gpu_probe_with_timeout`:
  - `check_gpu_availability` and `get_adapter_info_internal` have no unbounded
    `block_on` left.
  - The `OnceLock` callers can therefore no longer wedge.
- **Other tests in that file:**
  - `src/analysis/gpu/issue_2332_probe_timeout_test.rs::resolving_probe_returns_its_result_without_tripping`
  - `src/analysis/gpu/issue_2332_probe_timeout_test.rs::panicking_probe_fails_without_tripping`
  - `src/analysis/gpu/issue_2332_probe_timeout_test.rs::run_gpu_probe_with_timeout_times_out_on_a_hung_closure`
  - `src/analysis/gpu/issue_2332_probe_timeout_test.rs::run_gpu_probe_with_timeout_completes_on_a_resolving_closure`
- **Regression test for the review follow-up (pre-tripped breaker must not
  spawn):**
  `src/analysis/gpu/issue_2332_probe_timeout_test.rs::a_pretripped_breaker_skips_the_probe_without_spawning`.
  Pre-trips an isolated breaker, then calls `run_gpu_probe_with_timeout` with a
  probe closure that flips an `AtomicBool`. Asserts the outcome is `Tripped`,
  the flag is still `false` (the closure never ran), the call returned in
  under 1s against a 30s timeout, and the original trip reason survived.
  - Against the pre-follow-up code (breaker check reverted), this test fails:
    confirmed locally — reverting the `breaker.is_tripped()` guard in
    `run_gpu_probe_with_timeout` makes the outcome `Completed(42)` instead of
    `Tripped`, so the assertion on the matched variant fails.
  - `src/analysis/gpu/issue_2332_probe_timeout_test.rs::check_gpu_availability_with_short_circuits_on_a_pretripped_breaker`
    is the same check through the `check_gpu_availability_with` seam the FFI
    entry point actually calls: asserts `available: false`, `is_error: true`,
    a reason naming the original trip cause, the probe closure never ran, and
    the call returned in under 1s. Also confirmed red against the reverted
    guard (fails on `assert!(!result.available)`).
- **Docs updated:**
  - `docs/GPU_GUIDE.md:401`: new breaker trip-table row, and the follow-on
    sentence covers the probe's FFI verdict.
  - `docs/FFI_API.md:485`: "Probe timeout" bullet.
  - `src/analysis/gpu/breaker.rs:54`: the `InitTimeout` doc covers both trip
    sites.
- **Sweep hits read and kept as still true:**
  - `src/analysis/gpu/breaker.rs:99`: still true, because the
    "GPU initialisation timed out" label is generic to both trip sites.
  - `src/ffi_internal/gpu.rs:39`: still true, because "Hard error (`is_error`,
    e.g. macOS …)" gives an example, not a complete list.
  - `docs/FFI_API.md:1422`: still true, because `check_gpu_available` is still
    a hardware probe.
  - `src/analysis/gpu/queue/scheduling.rs:89`: still true, because the queue
    init trip site is unchanged.

**Docs sweep (review follow-up)** — grep: `BoundedProbe`, `run_gpu_probe_with_timeout`, `is_tripped`, `check_gpu_availability_with`, `get_adapter_info_internal`; section: `docs/GPU_GUIDE.md` "Process-wide circuit breaker" (re-read in full). No doc text changed: `docs/GPU_GUIDE.md:401`'s trip-table row and the following sentence ("the capability probe instead answers `check_gpu_available` with `success: false` and `errorKind: "gpu_permanent"`") already described a tripped-breaker short-circuit — that sentence was true going forward but not yet true for a call made *after* the first timeout, which is exactly the bug this follow-up fixes; it is now true for every call. `docs/FFI_API.md:485`'s "Probe timeout" bullet is unaffected — it describes the first timeout, not repeat calls after a trip. No other doc names `run_gpu_probe_with_timeout` or `BoundedProbe`.

**Docs sweep** — grep: `check_gpu_availability`, `get_adapter_info_internal`, `InitTimeout`, `GPU_INIT_TIMEOUT_SECS`, `check_gpu_available`, `gpu_permanent`, "GPU queue creation timed out", "probe timed out"; section: `docs/FFI_API.md` `check_gpu_available` section (the capability verdict), `docs/GPU_GUIDE.md` "Process-wide circuit breaker", `README.md#minimum-system-requirements`; updated: `docs/FFI_API.md`, `docs/GPU_GUIDE.md`; `docs/GPU_GUIDE.md:400` — still true because the queue-init trip row is unchanged and the probe has its own new row below it; `README.md:59` — still true because discovery is still skipped when no usable GPU answers the probe; `README.md:113` — still true because it only names the probe entry point; `README.md:147` — still true because it describes the requirements-not-met verdict, which this change leaves untouched; `README.md:149` — still true because the listed kinds are unchanged and the timeout reuses `gpu_permanent`; `docs/FFI_API.md:17` — still true because it only names the entry point; `docs/FFI_API.md:439` — still true because the symbol is unchanged; `docs/FFI_API.md:464` — still true because it is the GPU-less example payload, unchanged; `docs/FFI_API.md:479` — still true because a timed-out probe is also "no usable GPU on this host"; `docs/FFI_API.md:547` — still true because the timeout reuses `gpu_permanent` with the same meaning; `docs/FFI_API.md:557` — still true because the timeout verdict follows the same skip branch; `docs/FFI_API.md:1422` — still true because `check_gpu_available` still takes no input and is a hardware probe; `docs/audits/security-sweep-chunk-02-ffi-entry.md:75` — still true because the entry point still takes no input and is panic-contained; `src/analysis/gpu/analyzer.rs:325` — still true because `gpu_is_available` still caches `check_gpu_availability().available`, which is now bounded; `src/analysis/gpu/analyzer.rs:419` — still true because the host still calls the probe before discovery; `src/analysis/gpu/analyzer.rs:447` — still true for the same reason; `src/ffi_types/responses/gpu.rs:84` — still true because the payload shape is unchanged; `src/ffi_internal/gpu.rs:2` — still true because the FFI wrapper still calls the internal probe; `src/ffi_internal/gpu.rs:40` — still true because a timed-out probe is an `is_error` result and takes this hard-error mapping; `src/ffi_internal/gpu.rs:137` — still true because it documents the GPU-less test case, unchanged; `tests/infrastructure/issue_988_graceful_gpu_error.rs:5` — still true because `GpuUnavailable` still maps to `gpu_permanent`; `tests/infrastructure/issue_988_graceful_gpu_error.rs:51` — still true for the same reason; `tests/gpu/issue_1930_gpu_circuit_breaker.rs:31` — still true because the constant's value is unchanged; `tests/gpu/issue_1930_gpu_circuit_breaker.rs:57` — still true because `GpuWorkQueue::new` still waits up to the constant; `tests/issue_1935_wedged_gpu_harness.rs:226` — still true for the same reason; `tests/issue_2311_gpu_init_timeout_single_source.rs:1` — still true because the constant still has a single definition; "GPU queue creation timed out" — no hits left at the head (the reworded `breaker.rs` doc was its only occurrence)

## Test Plan

- [x] Bound the adapter/device requests in `check_gpu_availability` and
      `get_adapter_info_internal`
- [x] On timeout, trip the breaker with `InitTimeout`, return `is_error`, and
      leak the thread
- [x] Unit tests with an injected probe that never resolves
- [x] Review follow-up: check the breaker before spawning, so a tripped
      breaker stops new probe threads immediately instead of only after the
      next timeout
- [x] Unit tests with a pre-tripped isolated breaker asserting the probe
      closure never runs and the call returns without waiting
- [x] `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`
- [x] `timeout 900 ./quality.sh < /dev/null`

Branch outcomes:

- `src/analysis/gpu/device.rs:456` (breaker already tripped →
  `Tripped`, no thread spawned):
  `a_pretripped_breaker_skips_the_probe_without_spawning` and
  `check_gpu_availability_with_short_circuits_on_a_pretripped_breaker`.
  Removing the `breaker.is_tripped()` guard (reverted locally) turned both
  tests red — the outcome became `Completed(42)` and `result.available`
  became `true` respectively.
- `src/analysis/gpu/device.rs:481` (`Ok` → `Completed`):
  `run_gpu_probe_with_timeout_completes_on_a_resolving_closure` and
  `resolving_probe_returns_its_result_without_tripping`. Flipping
  `Completed(result)` to the timeout result went red.
- `src/analysis/gpu/device.rs:483` (`Timeout` → trip + `TimedOut`):
  `hung_probe_times_out_and_trips_breaker` and
  `run_gpu_probe_with_timeout_times_out_on_a_hung_closure`. Removing the trip
  went red.
- `src/analysis/gpu/device.rs:493` (`Disconnected` → `Failed`, no trip):
  `panicking_probe_fails_without_tripping`. Adding a trip went red.
- `src/analysis/gpu/device.rs:476` (spawn failure → `Failed`): **not
  covered**. Reaching it needs OS thread exhaustion. It shares the `Failed`
  mapping that the panic test covers.
- `src/analysis/gpu/analyzer.rs:219` (`Completed` → result):
  `resolving_probe_returns_its_result_without_tripping`. Flipping it went red.
- `src/analysis/gpu/analyzer.rs:220` (`TimedOut` → `probe_timeout_result`):
  `hung_probe_times_out_and_trips_breaker`. Mapping it to `no_gpu_result` went
  red.
- `src/analysis/gpu/analyzer.rs:221` (`Failed` → `no_gpu_result`):
  `panicking_probe_fails_without_tripping`.
- `src/analysis/gpu/analyzer.rs:222` (`Tripped` → `tripped_probe_result`):
  `check_gpu_availability_with_short_circuits_on_a_pretripped_breaker`.
  Mapping it to `no_gpu_result` instead would drop `is_error`, which the
  test's `assert!(result.is_error)` catches.
- `src/analysis/gpu/device.rs:620-621` (`get_adapter_info_internal` arms,
  `Tripped` folded into the existing `None` arm alongside `TimedOut`/`Failed`):
  **not covered**. They need real GPU hardware, and the function takes no
  injectable probe. All three variants are mechanical `Option` mappings and
  share the fate of the already-uncovered `TimedOut`/`Failed` arms noted in
  the original round of this PR.
