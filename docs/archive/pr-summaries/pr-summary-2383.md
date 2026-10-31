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
- The 50,000-hidden-neuron chain, width-2 shared-path ladder, mixed shallow/deep outputs with a diamond and disconnected chains, and a 50,000-endpoint dangling chain are covered by the four tests in that file. The input-disconnected hidden chain feeds the shallow output alongside its direct input, so the test checks that this chain cannot suppress the shallow-output candidate.
- Criterion command for both before and after: `cargo bench --bench topology_traversal -- topology_traversal --sample-size 10 --measurement-time 0.5 --warm-up-time 0.5`.

  | Case | Recursive baseline | Iterative final |
  | --- | ---: | ---: |
  | Width 2, 4 layers | 4.96 µs | 3.40 µs |
  | Width 2, 8 layers | 52.6 µs | 5.77 µs |
  | Width 2, 12 layers | 738 µs | 7.68 µs |
  | Width 2, 16 layers | 11.8 ms | 9.86 µs |
  | Width 2, 8 layers, 32 outputs | 210 µs | 25.4 µs |

- Original trigger closed: the detector no longer recursively enumerates paths. The graph is indexed once, each resolved edge is processed once in the depth pass, and each neuron receives one depth value reused by all outputs. Dangling endpoints are omitted, so they cannot create a second traversal path.
- #2383: topology_diversification::dfs_max_hidden_depth enumerates every simple path recursively — exponential time and stack overflow on a deep hidden chain (title confirmed with `gh issue view 2383`).
- PR #2488 review: the audit ledger now marks all three iterative-pass buffers as added after the audit HEAD and reconciles 85 production capacity rows with 88 capacity hits. Earlier 82/85 figures describe their historical checkpoint.
- Updated traversal documentation in `docs/DISCOVERY_TYPES.md` and the chunk 8a audit ledger. Updated the sweep contract to track `input_reachable_hidden_depths` and its three storage buffers.

## Test Plan

- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::deep_hidden_chain_does_not_overflow_a_small_stack` — passed on the final detector; observed SIGABRT with the base detector.
- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::width_two_shared_path_ladder_keeps_deep_outputs_unflagged` — passed.
- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::mixed_outputs_diamond_and_disconnected_chains_keep_candidate_results` — passed; the shallow output remains the sole candidate with hidden depth zero even when fed by an input-disconnected hidden chain, while deep and diamond outputs do not qualify.
- `tests/issue_2383_topology_diversification_dfs_path_explosion.rs::dangling_endpoint_chains_do_not_extend_graph_traversal` — passed.
- `cargo test --all-features --test recommendation topology_diversification -- --test-threads=1` — 9 passed.
- `cargo test --all-features --test issue_2298_chunk_08a_per_neuron_b_sweep -- --test-threads=1` — 4 passed after updating the ledger counts.
- Removed from `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`: `topology_diversification_source.contains("fn dfs_max_hidden_depth")`. Required by: "Replace the recursive enumeration with an iterative, memoised longest-path computation". Replaced by `topology_diversification_source.contains("fn input_reachable_hidden_depths")`.
- Removed from `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`: expected capacity-row count `17`. Required by: "one pass in evaluation order giving depth[n] = max(depth[pred]) + is_hidden(n) is O(N+S)". Replaced by expected count `20` to include the three iterative traversal buffers.
- Removed from `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`: expected production capacity-site count `17`. Required by: "one pass in evaluation order giving depth[n] = max(depth[pred]) + is_hidden(n) is O(N+S)". Replaced by expected count `20` to include the three iterative traversal buffers.
- Full head regression file against `origin/Develop`'s detector: aborted with `thread has overflowed its stack` and SIGABRT while running `dangling_endpoint_chains_do_not_extend_graph_traversal`; the base run could not complete.
- Reachability regression: the strengthened mixed-output test passes with `origin/Develop`'s detector. Removing both `input_reachable` guards from the iterative helper makes it fail at the candidate-count assertion (`left: 0`, `right: 1`), then restoring the helper makes all four tests pass. No production branch was added in this review fix.
- `tests/issue_2154_chunk_08a_finalisation.rs` (8 tests at head) — passed, including production capacity-row reconciliation. The existing per-neuron-b sweep remains at 20 capacity sites and retains its traversal checks.
- `cargo fmt --all -- --check` — passed.
- Markdown lint and formatting checks — passed for this summary and `docs/audits/security-sweep-chunk-08a-detection-neuron.md` using the repository's Markdown rules.
- `timeout 900 ./quality.sh < /dev/null` — timed out (exit 124) while compiling the all-features test targets. Shell checks, cargo-deny, debug build, formatting, Clippy and all-target type checks passed. Full-suite execution, documentation and release builds remain unverified in this run.

<!-- vibe-quality-gate-skipped reason="full gate attempted but did not complete within the 900-second command limit" -->
