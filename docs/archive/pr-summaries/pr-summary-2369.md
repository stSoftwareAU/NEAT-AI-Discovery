## Summary

`detect_topology_issues` no longer does an O(H·O) / O(H·I) scan per hidden neuron, and its per-neuron loops now honour the analysis deadline and global cancellation. Closes #2369.

- Skip-connection target and extra-input source are found by scanning output/input UUID lists that are sorted once (lazily). The scan stops after at most (already-connected candidates + 1) steps, so per-neuron cost is O(degree).
- New `detect_topology_issues_with_deadline(creature, records, topo, &Option<SystemTime>)` checks `deadline_passed` at the start of every iteration of both loops; `detect_topology_issues` is kept as an unbounded wrapper (`&None`). The dispatch spec in `structural_specs.rs` now calls the deadline variant with the module's `deadline`.
- On expiry it returns the candidates found so far, sorted best-first (each is valid alone); expiry in the long-path loop also skips imbalance detection.
- Docs: `docs/DISCOVERY_TYPES.md` § Topology-Aware Structure Analysis gains a "Cost and cancellation" paragraph.
- Version bump: not edited here; CI's `version-increment` job bumps `Cargo.toml` on the PR.

## Spec

### Intent and Rationale

- Hostile/corrupt creatures with up to 1,000,000 inputs/outputs and unbounded hidden neurons could pin a CPU in two uncancellable scans (CWE-834). Fix = make each scan O(degree) and add cancellation.
- Chosen over a per-neuron `HashSet` of inputs: a sorted list + early-exit `find` needs no per-neuron allocation and also makes the choice deterministic.

### Essential Design Decisions

- Choice is now deterministic (smallest UUID; outputs ordered by (path distance, UUID)); previously it followed `HashSet` iteration order. Any unconnected candidate was equally valid, so behaviour is compatible.
- A long-path neuron can never have a direct output synapse (it would be at distance 1), so every output is unconnected for it and the first sorted output is the answer.
- Partial results on deadline expiry rather than an error, matching module-level cancellation, where the dispatcher already treats truncation as best-effort.

### Undiscoverable Facts

- `MIN_DISCOVERY_SAMPLE_COUNT` is 20 (`src/analysis/constants/sample_thresholds.rs`).
- Before the base merge, `./quality.sh` stopped at `cargo deny` on `zstd` RUSTSEC advisories via `parquet 59.3.0`. The base branch has since moved `parquet` to 60 with the `zstd` feature omitted, and the merge took that manifest and lockfile, so those advisories no longer apply to this PR.

## Evidence

Backend-only change; no UI files touched.

- `tests/issue_2369_topology_scan_bounds.rs::long_path_skip_target_is_the_smallest_unconnected_output`, `tests/issue_2369_topology_scan_bounds.rs::starved_neuron_source_is_the_smallest_unconnected_input` — the sorted scans choose the smallest unconnected output/input. They use only the pre-existing `detect_topology_issues`, so they compile and run on base, and repeat 40 times because base order is a random `HashSet` order.
- `src/analysis/detection/topology_deadline_tests.rs` — two `#[serial]` tests using `DeadlineOverrideGuard` prove each loop's own deadline check stops the scan (loop 1: `[false, true]`; loop 2: N×`false` then `true`).

Original trigger is closed with no trivial bypass: the only per-neuron work left is a `find` over a pre-sorted list bounded by that neuron's own degree, and both loops check `deadline_passed` on every iteration before doing it, so a hostile creature can neither force an O(H·O)/O(H·I) pass nor keep the loops running past the deadline. The one-off sort is O(O log O) / O(I log I) over capped (≤ 1,000,000) lists and runs lazily after a deadline check; the BFS and fan-in passes were already linear.

No wall-clock or ratio assertion is used (CONTRIBUTING forbids timing in tests); the integration tests pin the deterministic choice and the in-crate tests pin the deadline behaviour.

## Test Plan

- `cargo test --test issue_2369_topology_scan_bounds`: 2 passed.
- `cargo test --lib topology`: 7 passed (includes both new in-crate tests).
- `cargo test --test recommendation` (249), `--test infrastructure` (223), `--test issue_2351_sibling_detectors_non_finite_gain_a` (7): all passed.
- `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`: clean.
- Red on base (base `topology.rs` and `structural_specs.rs` restored, test file unchanged): both integration tests failed in 3 of 3 runs; both pass on head. The deadline checks are proven by the two in-crate tests in `topology_deadline_tests.rs`, which need the `#[cfg(test)]` deadline seam and cannot run on base (the new symbol and module do not exist there).
- Callers checked: `structural_specs.rs` dispatch (now passes `&deadline`); tests in `tests/recommendation/issue_422_topology_aware_discovery.rs`, `tests/infrastructure/issue_754_topology_cache.rs`, `tests/issue_2351_sibling_detectors_non_finite_gain_a.rs` use the unchanged wrapper.
- Quality gate: `./quality.sh` was last run before the base merge and stopped at `cargo deny` on the `zstd` advisories (since removed by the base's parquet 60 change); it has not been re-run on the merged head. The equivalent fmt, clippy and test commands above were run directly and passed before the merge.

Not done: full test suite, version bump (CI), dependency advisory fix (out of scope).
