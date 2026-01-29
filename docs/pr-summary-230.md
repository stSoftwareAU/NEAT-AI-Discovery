## Summary

Implements multi-hop candidate analysis for deeper network improvements (Issue #230).

Current discovery only considers single-hop improvements (adding one synapse or neuron).
For deep networks, multi-hop improvements (adding a path of 2-3 connections) can be more
effective. This adds a new analysis pass that finds neurons whose activations correlate
with a target's error but are not directly connected, then recommends bypass synapses or
relay neurons.

### Key design decisions

- **Correlation-based intermediate selection**: Uses Pearson correlation between neuron
  activations and target errors to identify useful intermediates (threshold |r| >= 0.3).
- **Aggressive pruning**: Max 10 intermediates per target, max 50 total candidates, max
  3 hops depth to control exponential growth.
- **Reuses existing candidate types**: Emits `CoordinatedStructuralCandidateJson` with
  `AddNeuron` and/or `AddSynapse` operations — no new candidate types needed.
- **Two-hop and three-hop paths**: Two-hop adds a bypass synapse; three-hop adds a relay
  neuron with incoming and outgoing synapses.

### Files changed

- `src/analysis/multi_hop.rs` — New module with `detect_multi_hop_candidates` and
  `multi_hop_to_coordinated_candidates` functions.
- `src/analysis/mod.rs` — Registered module and integrated into `analyze_all` pipeline.
- `tests/issue_230_multi_hop_candidate_analysis.rs` — 13 TDD tests covering detection,
  edge cases, pruning, coordinated operations, and determinism.
- `README.md` — Added multi-hop analysis documentation section.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. The
feature is a new analysis pass integrated into the existing `analyze_all` pipeline.

## Test Plan

- `test_detects_two_hop_candidates_via_error_correlation` — Verifies two-hop candidates
  are detected when an intermediate's activation correlates with a target's error.
- `test_no_candidates_when_fully_connected` — Already-connected pairs are excluded.
- `test_insufficient_samples_no_candidates` — Too few samples produce no candidates.
- `test_empty_records_no_candidates` — Empty input handled gracefully.
- `test_estimated_improvement_positive` — All detected candidates have positive improvement.
- `test_candidates_produce_coordinated_operations` — Coordinated structural operations
  contain valid `addSynapse`/`addNeuron` operations.
- `test_candidate_path_depth_bounded` — Path length is bounded to max 4 nodes (3 hops).
- `test_candidates_sorted_by_improvement` — Results are sorted by estimated improvement.
- `test_no_hidden_neurons_no_candidates` — Network without hidden neurons handled correctly.
- `test_handles_neurons_without_errors` — Neurons without error records don't cause panics.
- `test_output_neurons_only_as_targets` — Output neurons never appear as intermediate nodes.
- `test_three_hop_candidate` — Three-hop paths through two intermediates are detected.
- `test_coordinated_candidates_have_deterministic_uuids` — Same input produces identical output.
