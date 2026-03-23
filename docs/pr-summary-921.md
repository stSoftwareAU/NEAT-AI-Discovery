## Summary

Implement candidate compression for IDENTITY squash candidates. When multiple independently-discovered `CandidateSynapseJson` candidates target the same output neuron, compress them into a single `CoordinatedStructuralCandidateJson` using a hidden IDENTITY neuron that sums all inputs. Closes #921.

### What changed

- **New module `src/analysis/candidate_compression.rs`**: Detection of compressible groups (candidates sharing a target from distinct sources), compression into coordinated structural candidates with deterministic FNV-1a UUIDs, and operation-count discount validation.
- **New constants in `src/analysis/constants.rs`**: `MIN_COMPRESSED_SOURCES` (2) and `MAX_COMPRESSION_INPUTS` (5).
- **Pipeline integration in `src/analysis/orchestration.rs`**: Compression step runs after synapse analysis and neuron-to-coordinated conversion, before discovery module dispatch. Compressed candidates are merged via `merge_coordinated_structural_replacements()`.
- **Re-export in `src/analysis/mod.rs`**: `candidate_compression` module added to the analysis module tree.

### Design decisions

- Uses IDENTITY activation with bias=0 for the hidden neuron (sum passthrough).
- Combined gain = sum of individual gains (linear additivity for IDENTITY).
- Output synapse weight = 1.0 (IDENTITY passes sum directly).
- Original individual candidates are preserved alongside compressed ones for fallback.
- Deterministic UUID uses `compress-` prefix to distinguish from `fan-in-` candidates.
- Operation-count discount (`0.65^(N+1)`) filters out candidates whose discounted gain falls below `MIN_COORDINATED_MULTI_OP_GAIN` (1e-3).

## Evidence

All 22 unit + integration tests pass. `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build).

## Test Plan

### Unit tests (in `src/analysis/candidate_compression.rs`)
- `test_detect_compressible_groups_basic` — detects groups with multiple distinct sources
- `test_detect_no_groups_for_single_candidate` — single candidate is not compressible
- `test_detect_no_groups_for_same_source` — same source duplicates are not compressible
- `test_detect_multiple_groups` — multiple target groups detected independently
- `test_detect_empty_input` — empty input produces no groups
- `test_deterministic_uuid` — same inputs produce same UUID
- `test_uuid_order_independent` — input order does not affect UUID
- `test_uuid_differs_for_different_targets` — different targets produce different UUIDs
- `test_uuid_starts_with_compress_prefix` — UUID has correct prefix
- `test_compress_produces_valid_operations` — correct operation structure
- `test_compress_gain_calculation` — operation-count discount applied correctly
- `test_compress_below_min_gain_threshold` — below-threshold candidates filtered
- `test_compress_preserves_original_weights` — original candidate weights preserved
- `test_compress_caps_at_max_inputs` — capped at `MAX_COMPRESSION_INPUTS`
- `test_compress_has_descriptive_comment` — compressed candidate has comment
- `test_compress_insert_before_target` — hidden neuron inserted before target

### Integration tests (in `tests/analysis/issue_921_candidate_compression.rs`)
- `test_detect_compressible_groups` — end-to-end group detection
- `test_compressed_candidate_structure` — validates full operation structure
- `test_deterministic_compression_uuid` — UUID reproducibility
- `test_operation_count_discount` — discount calculation verification
- `test_original_candidates_preserved` — originals not consumed
- `test_empty_input_no_compression` / `test_single_candidate_no_compression` / `test_same_source_no_compression` — edge cases
- `test_below_min_gain_filtered` — gain threshold filtering
- `test_max_inputs_capped` — input cap enforcement
- `test_compressed_uses_identity_activation` — IDENTITY squash + zero bias
- `test_multiple_target_groups` — independent groups produce separate candidates
