## Summary

`detect_topology_diversification_candidates` rescanned the whole creature for every eligible output (a reverse-adjacency rebuild in `has_unhealthy_intermediates`, an input-set rebuild plus synapse filter in `count_direct_input_paths`, a synapse filter with a linear neuron scan per match in `best_source_input`) and never checked the analysis deadline, so cost grew with outputs times synapses and could not be cancelled. The scan now builds one `SynapseIndex` (reverse adjacency over all synapses, and per-output direct-input lists) in a single pass, reuses it for every output, and checks `deadline_passed` at the top of each per-output iteration, keeping candidates found so far. The dispatch spec forwards the discovery deadline into the scan and logs a partial result. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version. Closes #2489.

## Spec

### Intent and Rationale

- With the exponential `dfs_max_hidden_depth` gone (#2383), the per-output rescans were the dominant uncancellable cost on this path. Building the index once removes the multiplier; the per-output deadline check makes the remaining work stoppable.
- The public `detect_topology_diversification_candidates` keeps its signature (wrapper over a new `_with_deadline` function that returns a `TopologyDiversificationScan { candidates, stop }`), following the `skip_connection` precedent, so the bench and existing tests are unaffected.

### Essential Design Decisions

- `reverse_adj` is keyed by uuid and includes dangling endpoints, so the unhealthy-intermediate walk sees exactly the predecessors it saw before; `direct_inputs` keeps duplicate synapses so `direct_path_count` is unchanged.
- `best_source_input` now runs before the unhealthy-intermediate walk (it returns `None` exactly when the output has no direct inputs, replacing the old `direct_paths == 0` exit). Both are pure, so results are unchanged.
- Assumption: the deadline is checked per output only. The one-off O(N+S) setup and a single output's backwards walk are not interrupted (marked `SIMPLE-ON-PURPOSE`; the ledger rows say so).

### Undiscoverable Facts

- `RecordCache::get` calls `deadline_passed` once per neuron while loading, so the dispatch-level test scripts one `false` per neuron before the scan's own checks.

## Evidence

Backend-only change; no UI. Step counts are exact and host-independent (`on_step` observer, as in the #2368 skip-connection tests): for K outputs each with one direct input synapse the scan makes `2K` steps (K at 32 gives 64, K at 128 gives 256, ratio 4). With the index build moved back inside the per-output loop the same test reads 1056 steps at K=32 instead of 64.

## Test Plan

- Added `src/analysis/topology_diversification_deadline_tests.rs` (in-crate, wired from `topology_diversification.rs`): `work_is_linear_in_outputs_not_quadratic`; `deadline_stops_at_per_output_check_and_keeps_earlier_candidates` (covers `[false, true]` keeps `out-0` only, `[true]` returns none, no override matches the wrapper); `duplicate_direct_synapse_counts_as_two_paths`; `source_is_highest_activation_variance_input`; `dangling_endpoint_still_reaches_unhealthy_intermediate`.
- Added `src/analysis/module_dispatch_specs/structural_specs_issue_2489_tests.rs`: `topology_spec_returns_partial_candidates_when_scan_stops` and `topology_spec_returns_none_at_elapsed_deadline` drive the registered `topology_diversification_detection` closure with a scripted deadline (the changed call site).
- Added `topology_diversification_unhealthy_intermediates_row_is_checked_and_the_code_checks_the_deadline` to `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs`; ledger rows and the `## Issues filed` entry in `docs/audits/security-sweep-chunk-08a-detection-neuron.md` updated, and the `candidates` capacity row now names `detect_topology_diversification_observed`.
- Red checks: index build moved into the loop → work-count test red (1056 vs 64); per-output `deadline_passed` deleted → deadline test and both dispatch tests red; `max_by` → `min_by` in `best_source_input` → variance test red (`lo` vs `hi`).
- Existing suites kept green: `tests/issue_2383_topology_diversification_dfs_path_explosion.rs`, `tests/recommendation/issue_549_topology_diversification.rs`, and the chunk 8a ledger suites. No existing assertion was removed or edited.
- No new capacity sites (`with_capacity(`, `vec![..;`, `.reserve(`) were added, so the #2480 capacity figure is unchanged.
- Docs sweep: `docs/discoveries/topology-diversification.md` and `CHANGELOG.md` checked; neither states the per-output rescan or a per-PR entry for comparable issues.

Run locally on the final head: `cargo test --lib -- topology_diversification structural_specs` (11 passed); `cargo test --test issue_2383_topology_diversification_dfs_path_explosion` (4 passed); `cargo test --test recommendation issue_549` (9 passed); eight chunk 8a ledger suites (`issue_2298`, `issue_2154`, `issue_2280`, `issue_2088`, `issue_2282`, `issue_2284`, `issue_2294`, `issue_2300`: 61 passed); `cargo clippy --all-targets -- -D warnings` clean; `cargo fmt --check` clean. The new ledger test goes red when the `has_unhealthy_intermediates` Cancellation cell is changed from `yes` to `no`.

<!-- vibe-quality-gate-skipped reason="./quality.sh hit the 595s foreground cap twice (exit 124) while compiling; targeted checks above were run instead; CI runs the full gate" -->

Not done / risk: full `./quality.sh` not completed locally. The one-off O(N+S) setup and a single output's backwards walk are still uninterrupted by the deadline. The ledger's per-file `Lines` column (472) was not refreshed.
