## Summary

Replaced recursive simple-path enumeration with a single iterative depth pass in validated neuron evaluation order. The detector now computes input reachability and maximum hidden depth once, then reuses the result for each output. Closes #2383.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The former traversal revisited shared paths exponentially and overflowed the stack on a deep chain. Evaluation order is already validated, so dynamic programming computes the same longest hidden depth in O(N + S).

### Essential Design Decisions

- Track input reachability separately from depth so disconnected hidden chains do not count as input paths.
- Resolve synapse endpoints through one neuron index and omit dangling endpoints from the depth graph. Existing depth thresholds, candidate fields, and ranking remain unchanged.
- Replace the proposed timing-ratio regression test with behavioural coverage. CONTRIBUTING.md prohibits timing assertions; Criterion measures performance separately.

### Undiscoverable Facts

- The base implementation was tested by temporarily using `origin/Develop`'s detector with the new small-stack test. It aborted with `thread has overflowed its stack` and SIGABRT.
- Criterion baseline and final measurements used the same release command and 10 samples per case on this host.

## Evidence

- Regression test `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::deep_hidden_chain_does_not_overflow_a_small_stack` failed against the base detector with SIGABRT and passed with the iterative detector.
- The 50,000-hidden-neuron chain, width-2 shared-path ladder, mixed shallow/deep outputs with a diamond and disconnected chains, and a 50,000-endpoint dangling chain are covered by the four tests in that file. The mixed-output test also checks a shallow-output candidate as its positive control.
- Criterion command for both before and after: `cargo bench --bench topology_traversal -- topology_traversal --sample-size 10 --measurement-time 0.5 --warm-up-time 0.5`.

  | Case | Recursive baseline | Iterative final |
  | --- | ---: | ---: |
  | Width 2, 4 layers | 4.96 µs | 3.40 µs |
  | Width 2, 8 layers | 52.6 µs | 5.77 µs |
  | Width 2, 12 layers | 738 µs | 7.68 µs |
  | Width 2, 16 layers | 11.8 ms | 9.86 µs |
  | Width 2, 8 layers, 32 outputs | 210 µs | 25.4 µs |

- Original trigger closed: the detector no longer recursively enumerates paths. The graph is indexed once, each resolved edge is processed once in the depth pass, and each neuron receives one depth value reused by all outputs. Dangling endpoints are omitted, so they cannot create a second traversal path.
- Updated traversal documentation in `docs/DISCOVERY_TYPES.md` and the chunk 8a audit ledger. Updated the sweep contract to track `input_reachable_hidden_depths` and its three storage buffers.

## Test Plan

- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::deep_hidden_chain_does_not_overflow_a_small_stack` — passed on the final detector; observed SIGABRT with the base detector.
- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::width_two_shared_path_ladder_keeps_deep_outputs_unflagged` — passed.
- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::mixed_outputs_diamond_and_disconnected_chains_keep_candidate_results` — passed; the shallow output remains a candidate while deep and diamond outputs do not.
- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::dangling_endpoint_chains_do_not_extend_graph_traversal` — passed.
- `cargo test --all-features --test recommendation topology_diversification -- --test-threads=1` — 9 passed.
- `cargo test --all-features --test issue_2298_chunk_08a_per_neuron_b_sweep -- --test-threads=1` — 4 passed after updating the ledger counts.
- Removed from `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`: `topology_diversification_source.contains("fn dfs_max_hidden_depth")`. Required by: "Replace the recursive enumeration with an iterative, memoised longest-path computation". Replaced by `topology_diversification_source.contains("fn input_reachable_hidden_depths")`.
- Removed from `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`: expected capacity-row count `17`. Required by: "one pass in evaluation order giving depth[n] = max(depth[pred]) + is_hidden(n) is O(N+S)". Replaced by expected count `20` to include the three iterative traversal buffers.
- Removed from `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`: expected production capacity-site count `17`. Required by: "one pass in evaluation order giving depth[n] = max(depth[pred]) + is_hidden(n) is O(N+S)". Replaced by expected count `20` to include the three iterative traversal buffers.
- `cargo fmt --all -- --check` — passed.
- `./quality.sh < /dev/null` — incomplete. Shell checks, cargo-deny, debug build, formatting, Clippy, and all-target checks passed; the run budget reached 486 seconds while the all-features test targets were compiling, so the gate was stopped before tests completed.

<!-- vibe-quality-gate-skipped reason="run budget reached during ./quality.sh all-features test build" -->

Resume with `timeout 900 ./quality.sh < /dev/null` before merge; the full gate and CI were not completed in this run.
