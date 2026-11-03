## Summary

`merge_discovery_module_results` truncated each module's candidates to `max_candidates`, applied the gain floor and the per-target cap, and recorded soft failures before any non-finite gain was dropped (that only happened later, in `merge_coordinated_structural_replacements`). A `+∞` or NaN gain therefore took a budget or cap slot from a finite candidate, and the displaced finite candidates were charged to the module as soft failures. The non-finite-gain gate now runs first, in both `merge_discovery_module_results` and the twin `run_discovery_module`. Closes #2486.

Version bump: `Cargo.toml` and `Cargo.lock` patch version incremented 0.74.335 → 0.74.336.

```mermaid
flowchart LR
    R[module candidates] --> O[reject non-finite operations]
    O --> G[reject non-finite gains - new]
    G --> B[max_candidates truncation]
    B --> F[gain floor]
    F --> C[per-target cap]
    C --> S[soft failures for the remainder]
    S --> M[merge_coordinated_structural_replacements]
```

## Spec

### Intent and Rationale

- The non-finite-operation gate (Issue #2375) already ran first; the gain gate is placed beside it so every invalid candidate is removed before it can consume a slot.
- `run_discovery_module` (single-module path) had the same ordering flaw for the per-target cap, so it is fixed too rather than leaving one copy of the rule behind.

### Essential Design Decisions

- Drops are counted under `REJECTION_NON_FINITE_GAIN` and excluded from the soft-failure count: `initial_count` is now taken after both non-finite rejections. This also stops non-finite-operation drops being charged as soft failures (previously they were, via `initial_count`); the module did not earn those either.
- NaN is already removed by the gain floor (`NaN >= floor` is false) before the cap, so the per-target-cap tests use `+∞`, which passes the floor and is the case that displaced candidates.
- `merge_coordinated_structural_replacements` keeps its own `reject_non_finite_gains` call as the final safety net.

### Undiscoverable Facts

- The audit ledger (`docs/audits/security-sweep-chunk-08a-detection-neuron.md`) describes audited code, so the #2351 text was kept and a note added that #2486 moved the gate.

## Evidence

Backend-only change; no UI. Tests added in `src/analysis/discovery_dispatch_parallel_tests.rs`; red checks below.

Twinned rule: the two dispatch paths (`merge_discovery_module_results`, `run_discovery_module`) both changed; the ledger rows that quote the old order were updated. No other copy was found (`grep -rn "after per-module" docs src` outside the archive).

## Test Plan

- `src/analysis/discovery_dispatch_parallel_tests.rs::non_finite_gain_does_not_displace_finite_candidate_under_module_budget` (new): `max_candidates` 1, `[+∞, 1.0]` returns the finite candidate, one `non_finite_gain` rejection, no `budget_truncated`, zero soft failures. Red with the merge-path gate removed (`left: 0, right: 1`).
- `…::non_finite_gain_does_not_take_a_per_target_cap_slot` (new): `+∞` plus three finite candidates on one target keep all three finite ones. Red with the merge-path gate removed (got `[3.0, 2.0]`).
- `…::run_discovery_module_non_finite_gain_does_not_take_a_per_target_cap_slot` (new): same scenario through `run_discovery_module`. Red with only that function's gate removed (got `[3.0, 2.0]`); the other two stay green, so each path is reached separately.
- Existing callers of the changed functions re-run and passing: `cargo test --lib -- discovery_dispatch` (51 passed), `tests/issue_2375_non_finite_operations.rs` (10), `tests/issue_1925_module_skip_attribution.rs` (8), `tests/issue_1367_non_finite_gain_rejection.rs` (2), `tests/analysis` filter `issue_1060` (22). No assertion was removed from any existing test.
- `cargo fmt --check` and `cargo clippy --lib --tests -- -D warnings`: clean.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 590 s foreground cap while compiling (exit 124); fmt, clippy and the targeted tests above were run; CI runs the full gate" -->

Not done / risk: the full `./quality.sh` did not complete; the soft-failure count for non-finite-operation drops changed as a side effect (no existing test pinned it).
