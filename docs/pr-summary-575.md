## Summary

Add structured logging with the `tracing` crate, replacing all `eprintln!` calls in
production source files with levelled, structured tracing macros. Closes #575.

### What changed

- **Dependencies**: Added `tracing` and `tracing-subscriber` (with `env-filter` and `fmt` features)
  to `Cargo.toml`.
- **Subscriber initialisation**: `observability::init_tracing()` installs a human-readable
  `fmt` subscriber writing to stderr, controlled by the `RUST_LOG` environment variable
  (default level: `warn`). Called once during `log_version_once()` in the FFI entry path.
- **eprintln! replacement**: All ~140 `eprintln!` calls in 30 production source files replaced
  with appropriate tracing macros (`error!`, `warn!`, `info!`, `debug!`, `trace!`), using
  structured fields instead of inline format strings.
- **Span instrumentation**: `#[tracing::instrument]` added to `analyze_all()`,
  `run_discovery_module()`, `run_discovery_modules_parallel()`, and `RecordCache::new_adaptive()`.
- **Backward compatibility preserved**: Default log level is `warn`, so stderr output is minimal
  unless the caller sets `RUST_LOG`. Signal handlers and deadlock detection in `debug.rs` retain
  `eprintln!` because they run in contexts where the tracing subscriber may be unavailable.
- **Documentation**: `RUST_LOG` added to environment variable tables in `AGENTS.md` and `README.md`.
- **Australian English** used throughout.

### Level mapping

| Context | tracing level |
|---------|---------------|
| Failures, panics, watchdog stalls | `error!` |
| Degraded mode (memory fallback, GPU unavailable) | `warn!` |
| Key lifecycle events (library init, GPU info, parquet loading mode) | `info!` |
| Verbose diagnostics (previously gated by `NEAT_AI_DISCOVERY_VERBOSE`) | `debug!` |
| Per-neuron / per-candidate / per-batch details | `trace!` |

## Evidence

This is a backend/CLI change with no visual output. Evidence is provided by test results:

- All 5 new integration tests pass (`tests/issue_575_structured_logging.rs`)
- All 501 existing unit tests pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- Added `tests/issue_575_structured_logging.rs` with 5 integration tests:
  - `init_tracing_is_idempotent` — verifies multiple init calls don't panic
  - `phase_timer_uses_tracing_without_panic` — exercises PhaseTimer with tracing backend
  - `gpu_metrics_report_uses_tracing_without_panic` — exercises GpuMetrics.report()
  - `profile_data_report_uses_tracing_without_panic` — exercises ProfileData.report()
  - `log_version_once_initialises_tracing` — verifies FFI entry path initialises tracing
- All existing tests continue to pass unchanged
