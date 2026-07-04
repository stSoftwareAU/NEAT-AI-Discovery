## Summary

Documentation-only change closing the README ↔ `docs/FFI_API.md` coverage gap
identified by the Documentation Posture Audit (finding `BP-3be62eced7f5`). Five
public `pub extern "C"` FFI entry points exported by the `neat_ai_discovery`
cdylib were absent from **both** the README's `## 🔌 FFI API Summary` table and
the authoritative interface spec (`docs/FFI_API.md`), leaving them
undiscoverable through the documented consumer path (README summary →
`docs/FFI_API.md`). Closes #1506.

The five newly-documented entry points:

| Entry point | Declared at | Purpose |
|-------------|-------------|---------|
| `get_calibration_summary` | `src/ffi/utilities.rs` | Per-module prediction-calibration summary from discovery history |
| `cleanup_discovery_lib` | `src/ffi/utilities.rs` | Shut down background threads — **must** be called before process exit |
| `cleanup_discovery_dir` | `src/ffi/utilities.rs` | Atomically remove a single discovery temp directory |
| `clean_orphaned_discovery_dirs` | `src/ffi/utilities.rs` | Sweep and remove orphaned discovery directories |
| `cancel_analysis_memory_pressure` | `src/ffi/analysis.rs` | Cancel in-flight analysis under CRITICAL memory pressure |

### Changes

- **README.md** — extended the `## 🔌 FFI API Summary` table with new
  **Cancellation**, **Calibration**, and **Lifecycle / cleanup** rows so every
  exported symbol is represented, flagging `cleanup_discovery_lib` as
  "call before process exit".
- **docs/FFI_API.md** —
  - added the missing symbols to the Exported Symbols list;
  - documented `cancel_analysis_memory_pressure` in the Cancellation Signal
    section (behaviour + `memoryPressureCancelled` output);
  - added a **Calibration Summary** section with input/output JSON contract;
  - added a **Library Lifecycle & Directory Cleanup** section documenting
    `cleanup_discovery_lib`, `cleanup_discovery_dir`, and
    `clean_orphaned_discovery_dirs` with their JSON contracts.

No source/behaviour change — documentation only.

## Evidence

Backend/documentation change; no web UI to screenshot. Verified via the new
regression test suite, which fails against the pre-change docs and passes after.

```mermaid
flowchart LR
    C[Consumer] --> R[README FFI API Summary]
    R --> F[docs/FFI_API.md]
    F --> S[All pub extern C symbols]
    style S fill:#c8e6c9
```

Before this change the five symbols terminated the discovery path at `README`
with no onward reference; now every exported symbol is reachable through the
documented path.

## Test Plan

Added `tests/issue_1506_ffi_doc_coverage.rs` (TDD — failed before the doc edits,
passes after):

- `readme_documents_all_previously_missing_ffi_entry_points` — asserts README
  mentions each of the five symbols.
- `ffi_api_doc_documents_all_previously_missing_ffi_entry_points` — asserts
  `docs/FFI_API.md` mentions each of the five symbols.
- `readme_flags_cleanup_before_exit` — asserts the README flags
  `cleanup_discovery_lib` as a before-process-exit call (the most material
  omission per the finding).
- `documented_cleanup_symbol_is_a_real_ffi_entry_point` — calls
  `cleanup_discovery_lib()` (idempotent, no args) to prove the documented symbol
  is a genuine, callable FFI entry point rather than a phantom reference.
- `documented_memory_pressure_symbol_is_a_real_ffi_entry_point` — binds
  `cancel_analysis_memory_pressure` as an `extern "C" fn()` pointer to prove it
  resolves with the correct signature, without mutating global cancellation
  state from a parallel test.
