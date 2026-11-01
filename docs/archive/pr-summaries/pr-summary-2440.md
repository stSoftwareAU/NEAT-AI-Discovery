## Summary

Classify untyped `anyhow` failures from their full context-and-cause chain and
include that chain in the host-visible error message. Typed `DiscoveryError`
classification still takes precedence. Closes #2440.

Version bump: `Cargo.toml` remains at `0.74.317`, matching the base branch.
The patch bump is pending and will be managed by CI when the PR is raised.

## Spec

### Intent and Rationale

- Wrapped GPU, timeout, memory, and Parquet I/O failures previously exposed
  only their outer context, hiding their cause from the host and preventing
  retry classification.
- Formatting the anyhow error with `{err:#}` preserves its full diagnostic
  chain while retaining the existing string classifier and priority.

### Essential Design Decisions

- A typed `DiscoveryError` found anywhere in the chain determines the kind
  before text classification, including `GpuWedged` with timeout wording and
  `Timeout` wrapped in conflicting device-lost context.
- The host-visible `error` includes outer context and root cause for any
  `anyhow::Error` returned through the shared helper.

### Undiscoverable Facts

- The regression tests reproduced the reported `GPU device lost and recovery
  failed after 3 attempts: Device is lost` cause under two analysis contexts.

## Evidence

- `tests/ffi/issue_2440_error_classification_context.rs` exercises serialised
  `AnalyzeParallelOutput::failure` responses without requiring a GPU.
- `tests/ffi/issue_2440_error_classification_context.rs::typed_timeout_takes_precedence_over_conflicting_device_lost_context` asserts
  that the full chain text classifies as `gpu_transient`, while the serialised
  response retains the typed `timeout` kind, retryability, context and cause.
- Existing helper callers in analysis, recording, GPU and utility FFI paths all
  receive the same structured tuple; they pass its message directly into their
  response fields, with no caller-specific truncation or parsing.

## Test Plan

- `cargo test --locked --test ffi --all-features issue_2440_error_classification_context -- --nocapture` — passed, 3 tests.
- Against the unfixed production code, the original two tests failed: wrapped GPU
  failure returned `unknown`, and the typed wrapped error's host message omitted
  the `GPU wedged:` cause.
- Classification mutation check: restoring `classify_error(&err.to_string())`
  made the wrapped failure test fail (`unknown` instead of `gpu_transient`).
- Message mutation check: restoring `err.to_string()` made the wrapped failure
  test fail because the inner context was absent.
- Typed-precedence mutation check: removing the downcast branch left the
  original two tests passing. With the new conflicting-context fixture, the
  same mutation failed on the serialised `errorKind` (`gpu_transient` instead
  of `timeout`). The downcast branch was restored afterwards.
- `cargo fmt --all -- --check` — passed.
- `cargo clippy --locked --test ffi --all-features -- -D warnings` — passed.
- `timeout 900 ./quality.sh < /dev/null` — timed out twice at 900 seconds during
  compilation of the full test targets. The targeted regression tests and
  Clippy check passed; the full quality gate did not complete.
