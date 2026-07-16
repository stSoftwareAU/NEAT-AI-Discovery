## Summary

Add a distinct `NoTargetRecords` diagnostic reason code for synapse analysis when the target neuron has zero activation records in the Parquet file. Previously, this case was misleadingly reported as `NoEligibleSources` because the early exit bypassed the `set_total_eligible_sources()` call, leaving the counter at its default of 0. The new reason code clearly indicates that the recording phase may have timed out or produced insufficient data. Closes #1101.

## Changes

- **`src/analysis/shared/metadata.rs`**: Added `NoTargetRecords` variant to `SynapseNoCandidateReason` enum
- **`src/ffi_types/responses/analysis.rs`**: Added `NoTargetRecords` variant to `SynapseDiagnosticReasonJson` (serialises as `"no_target_records"`)
- **`src/ffi_types/responses/mod.rs`**: Added conversion mapping for the new reason code
- **`src/analysis/diagnostics/rejection.rs`**: Updated `no_candidate_summaries()` to check `target_record_count == 0` before the `NoEligibleSources` fallback; added warning log in `emit_logs()` for zero-records case
- **`src/analysis/synapse/target_analysis/mod.rs`**: Added `tracing::warn!` when a focus neuron has zero Parquet records

## Evidence

This is a backend-only change with no UI components. Correctness is verified through unit and integration tests (see below). The JSON output now serialises the new reason as `"no_target_records"` in snake_case, consistent with existing reason codes.

## Test Plan

- **`synapse_diagnostics_reports_no_target_records_when_zero_records`** (unit): Verifies `NoTargetRecords` reason when target_record_count is 0
- **`synapse_diagnostics_no_target_records_does_not_mask_eligible_sources`** (unit): Verifies `NoEligibleSources` is still used when records exist but no sources are eligible
- **`synapse_diagnostics_no_target_records_json_serialises_correctly`** (unit): Verifies `NoTargetRecords` serialises to `"no_target_records"` in JSON
- **`zero_target_records_reports_no_target_records_reason`** (integration): Full pipeline test writing Parquet with no records for the focus neuron, asserting `NoTargetRecords` diagnostic
- **`json_output_serialises_no_target_records_reason`** (integration): Full pipeline JSON test verifying the reason appears correctly in `synapseDiagnostics`
- All existing tests continue to pass unchanged
