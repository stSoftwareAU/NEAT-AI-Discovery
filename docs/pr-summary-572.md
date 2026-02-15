## Summary

Implements ensemble candidate scoring as a post-processing step in the analysis pipeline. When multiple discovery modules independently identify the same target neuron or synapse and suggest the same type of structural change, their predictions are combined with an agreement boost. When modules disagree (e.g. one suggests removing a neuron while another suggests changing its activation), both candidates are penalised. Single-module candidates pass through unchanged.

Closes #572.

## Design

### Ensemble scoring module (`src/analysis/ensemble_scoring.rs`)

- **Target grouping**: Candidates are grouped by a target key derived from their operations (neuron UUID or synapse from-to pair)
- **Agreement detection**: Groups where all candidates share the same operation type (e.g. all SetBias) are classified as agreement
- **Disagreement detection**: Groups with mixed operation types (e.g. RemoveNeuron vs ChangeSquash) are classified as disagreement
- **Agreement boost**: Best individual score is multiplied by `1.0 + 0.3 * (n-1)/n` where n is the number of agreeing modules (15% boost for 2 modules, scaling up)
- **Disagreement penalty**: Each conflicting candidate's score is multiplied by 0.7
- **Success rate weighting**: Per-module historical success rates from `ModuleOutcomeTracker` influence which candidate template is selected during agreement combination
- **Metrics**: Tracks ensemble vs single-module candidate counts for validation

### Pipeline integration

Ensemble scoring runs in `orchestration.rs` after cross-module deduplication (Issue #489) and before candidate clustering (Issue #224), ensuring it operates on deduplicated candidates and its results are properly clustered.

## Evidence

This is a backend/library change with no visual output. Evidence is provided by the test suite:

- 9 integration tests verify all core behaviours
- 5 unit tests verify internal functions (target key extraction, operation classification, module name parsing)
- All 509 existing unit tests continue to pass
- All existing integration tests continue to pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

### New integration tests (`tests/issue_572_ensemble_candidate_scoring.rs`)
- `single_module_candidate_score_unchanged` — single-module candidates pass through with original score
- `empty_input_produces_empty_output` — empty input produces empty output
- `agreement_boosts_score_for_same_target_neuron` — multi-module agreement boosts score above best individual
- `disagreement_penalises_conflicting_candidates` — conflicting operations are penalised below original score
- `mixed_candidates_handled_correctly` — mixed agreement/disagreement groups are handled independently
- `module_success_rates_influence_ensemble_weight` — per-module success rates influence ensemble scoring
- `candidates_targeting_same_synapse_are_ensembled` — synapse-level matching works for add-synapse candidates
- `candidates_for_different_targets_are_independent` — candidates for different targets are not grouped together
- `result_tracks_ensemble_vs_single_module_counts` — result metrics correctly track ensemble vs single-module counts

### New unit tests (`src/analysis/ensemble_scoring.rs`)
- `target_key_for_set_bias` — correct key for neuron operations
- `target_key_for_add_synapse` — correct key for synapse operations
- `classify_agreement` — same operation types detected as agreement
- `classify_disagreement` — mixed operation types detected as disagreement
- `extract_module_name_from_comment` / `extract_module_name_no_comment` — module name parsing
