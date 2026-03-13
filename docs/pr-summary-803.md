## Summary

Replace `Arc<Mutex<bool>>` with `Arc<AtomicBool>` for the `analysis_timed_out` flag in
the neuron evaluation module, aligning it with the lock-free pattern already used in the
synapse module. This eliminates unnecessary mutex contention on a simple boolean flag
during parallel neuron evaluation. Closes #803.

Regarding `Arc<Mutex<HashMap>>` for `helpful_map`: during the parallel evaluation loop
all accesses are writes (via `upsert_candidate`), with the single read happening after
the loop completes. Since the access pattern is write-heavy during parallelism, switching
to `RwLock` would not provide a benefit and the `Mutex` is retained.

## Changes

- `src/analysis/neuron/evaluation.rs` — changed `analysis_timed_out` parameter from
  `Arc<Mutex<bool>>` to `Arc<AtomicBool>`, replaced `lock_or_bail` with `.store(true, Ordering::Relaxed)`
- `src/analysis/neuron/post_processing.rs` — changed `NeuronResultParams.analysis_timed_out`
  from `Arc<Mutex<bool>>` to `Arc<AtomicBool>`, replaced `lock_or_bail` with `.load(Ordering::Relaxed)`
- `src/analysis/neuron/preparation.rs` — changed `load_source_records` parameter from
  `Arc<Mutex<bool>>` to `Arc<AtomicBool>`, replaced `lock_or_bail` with atomic operations,
  removed unused `Mutex` and `lock_or_bail` imports
- `src/analysis/neuron/mod.rs` — changed `analysis_timed_out` from `Arc::new(Mutex::new(false))`
  to `Arc::new(AtomicBool::new(false))`, updated all call sites to use atomic operations,
  removed unused `lock_or_bail` import

## Evidence

- All existing tests pass (verified via `quality.sh`)
- The synapse module already uses the identical `AtomicBool` pattern (`synapse/orchestration.rs` line 76)
- `Ordering::Relaxed` is sufficient because the flag is monotonic (false -> true only) and
  does not guard any other shared data

## Test Plan

- All existing tests pass unchanged — this is a pure internal refactor with no behavioural change
- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
