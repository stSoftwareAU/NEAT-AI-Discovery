## Summary

Adds a bypass-weight floor to the 1-in/1-out hidden-neuron collapse detector
so it stops emitting 4-op coordinated-structural candidates whose
`addSynapse(a→b, weight=w)` would carry `|w|` below a meaningful threshold.
When the bypass weight is near-zero, the chain `a→h→b` was contributing
essentially nothing through `h`, so the proposal is functionally equivalent
to a 1-op `remove-neuron` but still carries the much higher implementation-
risk profile of a 4-op coordinated change. production creature `bcbca347`
captured 41 consecutive such failures (bypass weights as low as `2.1e-3`
producing actual error changes ~6,500× worse than predicted).

The new compile-time constant `MIN_BYPASS_WEIGHT_FOR_COLLAPSE` defaults to
`0.01` and is overridable via
`NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE` (clamped to `[0.0, 0.1]`).
Rejected chains are recorded under the new stable rejection reason
`coordinated_collapse_bypass_weight_below_floor` on the synapse metadata's
`rejection_breakdown`, so the drop surfaces in the drought diagnostic
without re-running analysis.

Closes #1270.

## Evidence

Backend-only Rust change. No web UI to screenshot; verified via the new
unit tests plus the full `quality.sh` gate (Clippy, full test suite, doc
build, release build).

```mermaid
flowchart LR
    A[detect_collapsible_hidden_neurons] --> B{|weight| >= MIN_BYPASS_WEIGHT_FOR_COLLAPSE?}
    B -- No --> C[Reject + bypass_weight_below_floor_drops++]
    C --> M[Metadata: rejection_breakdown[coordinated_collapse_bypass_weight_below_floor]]
    B -- Yes --> D[Emit 4-op coordinated candidate]
```

### Filter placement

The new floor is checked **after** the optimal bypass weight is computed
and **before** the `improvement > 0` gate, so the rejection is recorded
exactly when the dispatch-side counter says — not silently absorbed by a
later filter.

| Stage | Filter | Issue |
|-------|--------|-------|
| Pre-merge | `expected_creature_score_gain > 0.0` | existing |
| Per-merge | `COORDINATED_MIN_EXPECTED_GAIN` (1e-5) | #1110 |
| Post-discount | `COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS` (5e-6) | #1128, #1272 |
| **Per-candidate (new)** | **`MIN_BYPASS_WEIGHT_FOR_COLLAPSE` (0.01)** | **#1270** |

The new filter is orthogonal: `COORDINATED_MIN_EXPECTED_GAIN` and the
post-discount tiers filter by *predicted gain*; this filter targets the
failure mode where the bypass weight itself signals the collapse is risky
regardless of predicted gain.

## Test Plan

Added inline unit tests in `src/analysis/synapse/structural_patterns.rs`
(four tests, all passing under `cargo test --lib`):

- `rejects_collapse_when_bypass_weight_below_default_floor` — synthetic
  chain whose computed bypass weight is `0.005`; expects empty candidate
  list and `bypass_weight_below_floor_drops == 1` (acceptance criterion).
- `reproduces_bcbca347_failure_cache_pattern` — regression test for the
  `bcbca347` failure-cache pattern at bypass weight `0.0021`; asserts no
  candidate is emitted and the counter is incremented (acceptance
  criterion).
- `emits_candidate_when_bypass_weight_above_floor` — guards against the
  floor accidentally rejecting useful candidates: bypass weight `0.05`
  still produces exactly one collapse candidate, zero rejections.
- `env_var_override_disables_floor` — sets
  `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE=0` and verifies the
  rejection counter stays at zero so legacy tests can opt out of the new
  floor.

The new tests use `RecordCache::with_loader` for a fully in-memory
fixture and serialise env-var mutation via a static `Mutex`-backed RAII
guard so they remain stable under `--test-threads=2`.

All existing tests continue to pass — including the
`collapse_hidden_neuron_with_identity_squash` integration test in
`tests/synapse/issue_522_synapse_structural_patterns.rs`, which uses a
bypass weight (~1.0) well above the new floor.

## Files Touched

- `src/analysis/constants/candidate_scoring.rs` — new constant, clamp
  bounds, and `min_bypass_weight_for_collapse()` helper.
- `src/analysis/diagnostics/rejection_reasons.rs` — new rejection reason
  string, `ALL_REJECTION_REASONS` entry, and `friendly_reason` arm.
- `src/analysis/synapse/structural_patterns.rs` — `CollapseDetectionOutcome`
  return type, bypass-weight skip, in-line unit tests.
- `src/analysis/synapse/results.rs` — destructure new outcome and pipe
  drop count into `MetadataParams`.
- `src/analysis/synapse/post_processing.rs` — new `MetadataParams` field
  and seeding into `rejection_breakdown` in `build_metadata`.
- `AGENTS.md` — env-var documentation row.
- `Cargo.toml` — patch version bump `0.74.71 → 0.74.72`.
