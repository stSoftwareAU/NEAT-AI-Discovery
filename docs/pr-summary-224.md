## Summary

Implements candidate clustering to reduce redundant ablation tests (Issue #224).

When discovery returns many similar candidates (e.g., multiple synapses from the same
source region targeting the same neuron), the TypeScript controller would otherwise test
each independently, wasting CPU cycles. This change groups similar candidates into
clusters so the controller can test a representative first and skip redundant tests
if the representative fails.

### What changed

- **New module `src/analysis/candidate_clustering.rs`**: Groups candidates by target
  neuron, source type, and improvement similarity. Each cluster identifies a
  representative (highest improvement) and computes an internal correlation score.
- **New JSON field `candidateClusters`**: Optional field in `AnalyzeParallelOutput`
  that lists detected clusters with representative UUID, member count, member UUIDs,
  and internal correlation.
- **Post-processing integration**: Clustering runs as a final step in `analyze_all`
  after all candidates have been gathered, so it sees the complete candidate set.
- **Backward compatible**: The `candidateClusters` field is `skip_serializing_if`
  `Option::is_none`, so existing consumers are unaffected.

### Clustering criteria

1. **Same target neuron** — candidates must share `toNeuronUuid`
2. **Same source type** — input vs hidden neurons are not mixed
3. **Similar improvement** — candidates with >5x improvement ratio are split into
   separate sub-clusters

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

No performance benchmark needed — clustering is a lightweight O(n) grouping pass
over the existing candidate arrays with no GPU involvement.

## Test Plan

- 13 integration tests in `tests/issue_224_candidate_clustering.rs`:
  - `test_same_target_candidates_clustered` — candidates targeting same neuron cluster
  - `test_representative_is_best_candidate` — highest improvement is representative
  - `test_different_targets_separate_clusters` — different targets form different clusters
  - `test_internal_correlation_range` — correlation is between 0.0 and 1.0
  - `test_single_candidate_no_cluster` — no cluster for lone candidates
  - `test_empty_candidates_no_clusters` — empty input produces no clusters
  - `test_cluster_json_serialisation` — JSON output has expected camelCase fields
  - `test_mixed_types_sub_clustering` — input vs hidden sources form separate clusters
  - `test_dissimilar_improvements_separate_clusters` — large improvement differences split
  - `test_cluster_members_sorted_by_improvement` — members sorted best-first
  - `test_clusters_sorted_by_best_improvement` — clusters sorted by representative
  - `test_similar_improvements_yield_high_correlation` — similar values yield high correlation
  - `test_large_candidate_set` — 100 candidates across 2 targets clustered correctly
- 2 inline unit tests in `src/analysis/candidate_clustering.rs`:
  - `test_split_by_improvement_similarity_basic`
  - `test_compute_internal_correlation_identical`
- Existing contract test updated: `issue_337_candidate_type_contract.rs`
