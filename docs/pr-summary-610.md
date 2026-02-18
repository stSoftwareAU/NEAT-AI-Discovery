## Summary

Candidate diversity enforcement — penalise structurally similar candidates. Closes #610.

When the discovery pipeline returns multiple candidates that are structurally similar
(e.g., removing adjacent synapses to the same target), they waste evaluation budget by
exploring the same region of mutation space. This PR adds diversity-aware reranking that
penalises lower-ranked candidates that are structurally similar to higher-ranked ones,
promoting exploration of diverse mutation regions.

### Changes

- **New module `src/analysis/candidate_diversity.rs`**: Implements structural similarity
  metric (Jaccard similarity of affected neurons + operation types) and diversity-aware
  reranking with configurable penalty strength.
- **Integration in `orchestration.rs`**: Diversity reranking runs after ensemble scoring
  and before candidate clustering, fitting naturally into the existing pipeline.
- **Configurable penalty**: `DiversityConfig::penalty_strength` (default 0.5) controls
  the trade-off between exploitation (keeping similar high-scoring candidates) and
  exploration (promoting diverse alternatives).

### How it works

1. Candidates are sorted by `expected_creature_score_gain` (best first).
2. For each candidate from position 1 onward, the maximum structural similarity to any
   higher-ranked candidate is computed.
3. The candidate's effective score is penalised: `effective = score × (1 - penalty × similarity)`.
4. Candidates are re-sorted by effective score.
5. All candidates are preserved (none removed), only the ranking changes.

## Evidence

This is a backend-only change with no visual output. Evidence is provided by the 13
integration tests that verify correctness of the similarity metric and reranking behaviour.

## Test Plan

- Added `tests/issue_610_candidate_diversity_enforcement.rs` with 13 tests:
  - Structural similarity: same target (high), identical (1.0), different (low), same neuron different ops (moderate), multi-op, symmetric
  - Reranking: demotes similar, preserves diverse ordering, empty/single passthrough, configurable penalty, zero penalty preserves order, preserves all candidates
- All existing tests pass unchanged
- `cargo clippy` and `./quality.sh` pass cleanly
