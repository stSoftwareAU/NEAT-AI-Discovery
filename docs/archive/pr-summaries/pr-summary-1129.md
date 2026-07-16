# Issue #1129 — Structured rejection-reason breakdown on analysis metadata

Closes #1129.

## Problem

When discovery returned zero candidates, operators had no structured way to
know why. The `synapseMetadata` / `neuronMetadata` JSON surfaced only totals
and timing, so diagnosing "no candidates found" meant re-running with verbose
logging and scraping log lines.

## Change

Added a `rejection_breakdown` map and a one-sentence `top_level_summary` to
both `SynapseAnalysisMetadata` and `NeuronAnalysisMetadata`.

- **`rejection_breakdown: HashMap<String, u32>`** — keyed by stable documented
  reason names (`below_expected_gain_floor`, `below_multi_op_floor`,
  `non_positive_gain`, `saturation_discounted_to_zero`,
  `pessimism_discounted_to_zero`, `interference_filtered`,
  `below_improved_ratio`, `duplicate_of_failure_cache`, `add_synapse_gated`,
  `budget_truncated`, `no_samples`, `zero_improvement`, `below_threshold`,
  `no_target_records`, `no_eligible_sources`, `input_neuron_filtered`,
  `hidden_neuron_filtered`, `constant_neuron_filtered`, `no_diagnostics`).
  The full catalogue lives in `analysis::diagnostics::rejection_reasons`.

- **`top_level_summary: Option<String>`** — e.g. `"32 of 34 candidates
  rejected by expected-gain floor of 1e-5"`. Populated whenever any
  candidate was rejected, including when zero candidates survive.

### Counter wiring

- `candidate_aggregation::apply_coordinated_gain_floor` now returns the drop
  count and the caller in `orchestration::analyze_all` feeds it into
  `rejection_breakdown` under `below_expected_gain_floor`.
- `candidate_aggregation::merge_coordinated_structural_replacements` records
  `non_positive_gain` and `below_multi_op_floor` drops.
- `discovery_dispatch::run_discovery_module` + `merge_discovery_module_results`
  record `below_expected_gain_floor` and `budget_truncated` drops.
- `orchestration::analyze_all` records `add_synapse_gated` drops from
  `gate_add_synapse_candidates`.
- `orchestration::aggregate_synapse_rejection_breakdown` and
  `aggregate_neuron_rejection_breakdown` lift existing per-target
  `no_candidate_reasons` into the breakdown and compute `top_level_summary`.
- `neuron::preparation::build_empty_result` populates the breakdown with
  input/hidden/constant pre-filter counts so the early-return path still
  surfaces a meaningful summary.

### FFI

Both metadata structs in `ffi_types::responses::analysis` gained a
`rejection_breakdown` (skip-serialize when empty) and `top_level_summary`
(skip-serialize when `None`) field. `ffi_internal::analysis::analyze_parallel_internal`
copies them into the response.

## Tests

- `tests/issue_1129_rejection_breakdown.rs` — three integration tests:
  - `floor_dominated_case_records_below_expected_gain_floor` runs a discovery
    module emitting 32 candidates at 5e-7 (below the 1e-5 floor). Asserts
    zero survivors, ≥32 `below_expected_gain_floor` count, matching dominant
    reason, and "expected-gain floor" in the summary.
  - `mixed_reason_case_records_each_reason_independently` populates three
    distinct reasons and asserts exact per-reason counts and a
    dominant-reason summary.
  - `apply_coordinated_gain_floor_returns_drop_count` verifies the helper's
    new `u32` return carries the right drop count back to callers.
- `src/analysis/diagnostics/rejection_reasons.rs` — six unit tests covering
  `record`, `record_many` (zero skip), `dominant_reason`, `top_level_summary`
  (floor and empty cases), and `merge_from`.

## Validation

- `./quality.sh < /dev/null` — PASS (fmt, clippy, check, 2000+ unit/integration
  tests, doc build, release build).
- No new warnings introduced. Pre-existing cargo duplicate-crate warnings are
  unrelated.

## Australian English

`analyse`, `behaviour`, `favour`, `centre` used throughout comments.
