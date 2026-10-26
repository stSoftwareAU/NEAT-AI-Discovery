## Summary

Bounds and makes cancellable the skip-connection detector (CWE-400). Closes #2368.

- `compute_depths_from_inputs` is now a single O(V+E) Kahn-style longest-path pass. The old BFS re-pushed a node on every depth rise with no visited set, so a cyclic creature never returned and a DAG cost O(V·E). A cycle is detected and reported (`SkipConnectionScan::depth_cycle_detected`) and the detector fails closed with no candidates.
- Shallow sources are indexed once into depth-0 and depth-1 lists; each deep target stops at the first unconnected source, so the nested O(H·(I+H)) scan is bounded by the target's in-degree. Results match the old scan on acyclic creatures.
- New `detect_skip_connection_candidates_with_deadline` checks the analysis deadline and global cancellation once per deep target and returns partial candidates with `ScanTruncation::DeadlinePassed`. The old function keeps its signature and delegates with no deadline. The dispatch spec in `src/analysis/module_dispatch_specs/structural_specs.rs` passes the discovery deadline and logs truncation or a detected cycle.
- Docs: `docs/DISCOVERY_TYPES.md`, `docs/ANALYSIS_DEEP_DIVE.md`, `docs/discoveries/skip-connection.md`.
- Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The hang is a cyclic synapse graph reaching the public Rust API without `validate_creature`; the fix makes the detector safe itself rather than relying on the FFI trust boundary.
- A single topological pass removes the re-queueing at the root, instead of adding an iteration cap to the BFS.

### Essential Design Decisions

- A cyclic creature yields no candidates (fail closed) because depths are undefined; this is reported via `depth_cycle_detected` and a `tracing::warn!` in the dispatch spec.
- Source selection preserves the old tie-break: largest depth gap wins (depth 0 beats depth 1), first unconnected source in creature order within a depth. Unreached neurons still count as depth 0.
- Assumption: the issue's suggested "bound the shallow-source set" is met by indexing plus early exit rather than an arbitrary cap, which would change results.

### Undiscoverable Facts

- None.

## Evidence

```mermaid
flowchart LR
    A[creature] --> B[Kahn depth pass O(V+E)]
    B -->|cycle| C[no candidates, depth_cycle_detected]
    B -->|DAG| D[per deep target: deadline check]
    D -->|elapsed| E[partial candidates + DeadlinePassed]
    D --> F[first unconnected depth-0 / depth-1 source]
```

Commands run on the final head:

- `cargo test --test issue_2368_skip_connection_scan_bounds`: 1 passed.
- `cargo test --lib skip_connection`: 5 passed. `cargo test --test recommendation issue_570`: 13 passed (existing skip-connection behaviour unchanged).
- `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean.
- Base red run: copied the new test onto a detached worktree of `origin/Develop`; it failed with `detect_skip_connection_candidates must return on a cyclic creature: Timeout` after 30.01s.

## Security Fix Evidence

- Regression test: `tests/issue_2368_skip_connection_scan_bounds.rs::cyclic_creature_returns_without_hanging_and_proposes_nothing`. Red at base (timeout, above), green at head. It uses only the pre-existing public API so it compiles against the base.
- Original trigger closed: a cyclic synapse graph can no longer loop in the depth pass. Each node is processed once when its in-degree reaches zero, so the pass terminates in O(V+E) for any input; nodes left unprocessed mean a cycle, which returns no candidates. There is no other depth computation in the detector, and the pair scan is bounded by in-degree and checks the deadline per target. No equivalent bypass exists: every path to candidate creation goes through `compute_depths_from_inputs` first.

## Test Plan

- `tests/issue_2368_skip_connection_scan_bounds.rs::cyclic_creature_returns_without_hanging_and_proposes_nothing` (new): described above.
- `src/analysis/detection/skip_connection_tests.rs` (new): `depths_match_naive_longest_path_on_dense_layered_dag`, `cyclic_creature_sets_depth_cycle_detected_and_returns_no_candidates`, `elapsed_deadline_mid_scan_keeps_partial_candidates_and_reports_truncation`, `far_future_deadline_matches_unbounded_scan_without_truncation`, `shallow_source_selection_skips_connected_sources_and_falls_back_to_depth_one`. All pass.
- No existing test was edited or had assertions removed.
- Callers of `detect_skip_connection_candidates` checked: `tests/recommendation/issue_570_skip_connection_discovery.rs` (4 tests, pass) and the dispatch spec, which now calls the deadline variant.
- Not done: no work-count assertion or mutation red-check was run for the per-target deadline check beyond `elapsed_deadline_mid_scan_keeps_partial_candidates_and_reports_truncation`; the issue's suggested H vs 4H ratio test was replaced by the cycle-hang test, since the depth pass is now linear by construction and the cycle hang is the demonstrable base failure.
- `./quality.sh` stops at `cargo deny` on RUSTSEC-2026-0391/-0398/-0403/-0410 (zstd-sys 2.1.0 via parquet), present on base; this branch changes no dependency. fmt and clippy stages passed.

<!-- vibe-quality-gate-skipped reason="cargo deny advisories fail on base-inherited zstd-sys; later gate stages not reached locally, CI runs the full gate" -->
