# PR Summary — Issue #1142

Closes #1142.

## Summary

Gate `remove-low-impact` candidates on a noise-floor net-improvement threshold so
candidates whose post-boost `net_improvement = boosted_savings − activation_weighted_impact`
is indistinguishable from floating-point noise no longer reach the FFI response.

## Change Set

- **`src/analysis/constants/candidate_scoring.rs`**
  - Add `REMOVE_LOW_IMPACT_NOISE_FLOOR: f32 = 1e-5` (matches
    `COORDINATED_MIN_EXPECTED_GAIN`) with the GRQ-sampler failure-cache
    evidence inlined in the doc comment.
  - Add `remove_low_impact_noise_floor()` helper that reads
    `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR` at call time. `0.0` is
    accepted as "disable the floor" (for tests exercising the pre-#1142
    impact/savings contract).
  - Update the doc comment on `REMOVAL_CANDIDATE_BOOST` to cross-reference
    Issue #1142.

- **`src/analysis/diagnostics/rejection_reasons.rs`**
  - Add `REJECTION_REMOVAL_BELOW_NOISE_FLOOR = "removal_below_noise_floor"`
    and list it in `ALL_REJECTION_REASONS`.
  - Extend `friendly_reason` with a human-readable description that reports
    the effective floor.

- **`src/focus/ranking/removal_candidates.rs`**
  - Apply `REMOVAL_CANDIDATE_BOOST` **before** the savings-vs-impact check.
  - After the check, gate on `net_improvement < noise_floor` and drop the
    candidate (counted under `noise_floor_rejections`).
  - Return `RemovalCandidateOutcome { candidates, noise_floor_rejections }`
    instead of a bare `Vec<RemovalCandidate>`.
  - Inline unit tests:
    - `issue_1142_evidence_candidate_is_dropped` — reproduces the GRQ-sampler
      failure-cache scenario and asserts the candidate is dropped and
      counted.
    - `well_above_noise_floor_candidate_is_kept` — net ≈ 1.5e-5 candidate
      survives.
    - `env_var_override_changes_effective_floor` — overriding the env var
      changes both drop and keep behaviour.

- **`src/focus/ranking/mod.rs`**
  - Add `rejection_breakdown: HashMap<String, u32>` to `RankFocusStats`
    (derives `Default`).
  - Build the breakdown from the `RemovalCandidateOutcome` before moving
    candidates into the stats struct.

- **FFI surface (`src/ffi_types/responses/mod.rs`, `src/ffi_internal/analysis.rs`,
  `src/ffi/analysis.rs`)**
  - Add `rejection_breakdown: Option<HashMap<String, u32>>` to
    `RankFocusNeuronsOutput`, serialised only when non-empty.

- **Tests updated for the new contract**
  - `tests/focus/focus.rs`, `tests/neuron/issue_132_cost_of_growth.rs`,
    `tests/neuron/issue_414_remove_neuron_high_error.rs`,
    `tests/regression/regression_v0_1_127.rs`: pre-#1142 tests that
    deliberately exercise tiny magnitudes (below 1e-5) now wrap themselves
    in a `NoiseFloorOffGuard` (env-var override) and are marked `#[serial]`.
    The guards document that the pre-#1142 impact/savings contract still
    holds in the absence of the noise floor.
  - `tests/analysis/issue_337_candidate_type_contract.rs`: struct literal
    updated with `rejection_breakdown: None`.

## Evidence

From the GRQ-sampler failure-cache entry cited in the issue
(`v2_remove-low-impact_0ce92a87-...json`):

```
boosted_savings = 1.20e-7 × 1.5 = 1.80e-7
activation_weighted_impact = 1.14e-7
net_improvement = 6.64e-8          ← well below the 1e-5 noise floor
actualErrorReduction = −2.39e-7    ← candidate harmed the creature
```

With this PR, the scenario is rejected at the noise-floor gate:

```rust
let growth = 1e-7_f32;
let creature = creature_with_synapse_counts("h1", 1, 1);   // 2 synapses
let neuron   = ranked_neuron("h1", 0.0, 1.14e-7);           // impact
let outcome  = identify_removal_candidates(&[neuron], ..., growth);

assert!(outcome.candidates.is_empty());
assert_eq!(outcome.noise_floor_rejections, 1);
```

A symmetric keep-case test (`well_above_noise_floor_candidate_is_kept`) proves
the gate does not over-reject: a candidate with `net ≈ 1.5e-5` survives.

## Is `REMOVAL_CANDIDATE_BOOST = 1.5` Still Justified?

The issue asks us to revisit the 1.5× boost. The boost was introduced in Issue
#892 to reflect a measured 21.5% success rate for `remove-low-impact`
candidates. Several subsequent PRs changed the calibration landscape (#1112,
#1118, #1131).

**Decision: keep the boost at 1.5× for this PR.**

Rationale:

1. The failure evidence in #1142 is caused by the boost letting candidates
   **cross the savings-vs-impact line** at noise-level magnitudes. The
   noise-floor gate introduced here is a **targeted, additive** fix:
   candidates can no longer cross the line by accident of noise, regardless
   of the boost multiplier.
2. Lowering or removing the boost without fresh end-to-end success-rate
   telemetry would be a speculative change. The GRQ-sampler failure cache
   entry alone is not a statistically meaningful sample — it tells us about
   **this particular** candidate being bad, not about the marginal success
   rate of the population above the floor.
3. The boost still plays its original role of giving `remove-low-impact`
   candidates a fair chance when their raw savings are close to their
   impact and they sit above the noise floor. With the noise floor in
   place, the boost's worst-case damage is bounded.

**Follow-up recommended:** once a fresh window of GRQ-sampler data is
available post-merge, re-evaluate `REMOVAL_CANDIDATE_BOOST` against
candidates that clear the new 1e-5 floor. If the observed success rate for
above-floor candidates is materially different from the 21.5% that motivated
#892, open a separate issue to adjust the boost with that evidence.

## Quality

- `./quality.sh` passes (cargo-deny, fmt, clippy `-D warnings`, full test
  suite including doctests, docs, release build).
- New inline unit tests (3/3) cover the issue's drop and keep cases plus the
  env-var override.
- All prior focus/ranking and removal-boost integration tests still pass
  (pre-#1142 tests updated with `NoiseFloorOffGuard` + `#[serial]` to
  preserve their original contract).

## Pre-PR Security Self-Check

- **Input validation**: env var parsed via `f32::parse` with finite and
  non-negative filter; falls back to compile-time default on any failure.
- **Secrets**: none staged.
- **Injection surface**: no new SQL/shell/HTTP surface.
- **Output encoding**: `rejection_breakdown` is serialised via `serde_json`
  with the existing FFI machinery.
- **Error handling**: no user-facing messages changed.
- **Dependencies**: no new third-party dependencies.
