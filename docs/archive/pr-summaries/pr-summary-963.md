## Summary

Add cross-detection candidate synthesis for co-flagged neurons. When multiple
detection modules independently flag the same neuron (e.g., saturation +
restricted range + high error), the system now synthesises combined remediation
candidates that address multiple issues simultaneously, rather than only
generating independent single-issue candidates. Closes #963.

### What changed

- **New module** `src/analysis/detection/cross_detection_synthesis.rs`:
  - Groups single-operation detection candidates by target neuron UUID
  - When 2+ modules flag the same neuron with compatible operations, synthesises
    a combined `CoordinatedStructuralCandidateJson` containing all operations
  - Compatibility rules: modifications (ChangeSquash, SetBias, SetWeight) merge
    with each other; removals (RemoveNeuron, RemoveSynapse) merge with each
    other; removals do NOT merge with modifications
  - Individual candidates are always preserved (synthesis is additive)
  - Synthesised candidates are tagged with `cross-detection synthesis` in their
    comment for success rate tracking via `ModuleOutcomeTracker`

- **Orchestration integration** (`src/analysis/orchestration.rs`):
  - Synthesis runs after discovery module dispatch and before cross-module
    deduplication and ensemble scoring

- **Pipeline wrapper** (`src/analysis/module_dispatch_specs/mod.rs`):
  - Added `synthesise_cross_detection_candidates()` wrapper with watchdog beats,
    phase timing, and verbose logging

## Evidence

All 21 tests pass (15 integration + 6 unit). `quality.sh` passes cleanly.

## Test Plan

- **Integration tests** (`tests/analysis/issue_963_cross_detection_candidate_synthesis.rs`):
  - `test_group_candidates_by_neuron_groups_correctly` — verifies grouping by UUID
  - `test_group_candidates_empty_input` — empty input passthrough
  - `test_group_candidates_multi_op_candidates_excluded` — multi-op excluded from grouping
  - `test_synthesise_produces_combined_candidate_for_coflagged_neuron` — co-flagged synthesis
  - `test_synthesise_preserves_individual_candidates` — synthesis is additive
  - `test_single_detection_neuron_produces_no_synthesis` — no false positives
  - `test_empty_input_passthrough` — empty input
  - `test_merge_change_squash_and_set_bias` — ChangeSquash + SetBias compatibility
  - `test_merge_change_squash_and_set_weight` — ChangeSquash + SetWeight compatibility
  - `test_merge_remove_neuron_and_remove_synapse` — removal compatibility
  - `test_incompatible_remove_and_modify_not_merged` — incompatible ops rejected
  - `test_synthesised_candidate_gain_uses_best_individual` — gain scoring
  - `test_synthesised_candidate_comment_identifies_source_modules` — attribution
  - `test_synthesis_result_counts` — result structure validation
  - `test_three_way_coflagging_produces_synthesis` — 3+ module co-flagging
- **Unit tests** (`src/analysis/detection/cross_detection_synthesis.rs`):
  - `primary_neuron_uuid_for_change_squash` — UUID extraction
  - `primary_neuron_uuid_for_set_weight_uses_target` — target UUID for synapse ops
  - `primary_neuron_uuid_none_for_multi_op` — multi-op exclusion
  - `compatible_modifications` — modification compatibility check
  - `compatible_removals` — removal compatibility check
  - `incompatible_removal_with_modification` — incompatibility check
