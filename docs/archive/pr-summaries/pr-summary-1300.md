## Summary

Fixes three correctness gaps in Discovery's impact attribution that mirror
`NEAT-AI-Explore#266`. The previous attribution used pre-squash magnitudes and
overstated influence whenever a downstream neuron was saturated, gave every
inbound synapse to a STEP/BIPOLAR neuron the full `child_impact` (sum
`= N × child_impact`), and had no way to model external `min(output, constant)`
gates. Fixes #1300.

- New `squash_emit_magnitude` helper in `src/activations.rs` exposes the
  emit ceiling (`Some(M)`) for bounded squashes (TANH/LOGISTIC/HARD_TANH/STEP/
  BIPOLAR/RELU6/...) and `None` for unbounded ones (IDENTITY/RELU/ELU/...).
- `src/focus/impact.rs` now applies a squash-bounded cap in
  `compute_impact_with_shared_cache` for both `Linear` and `Threshold`
  branches. Threshold squashes additionally switch from
  `contribution = child_impact` to the normalised
  `contribution = (|w|/T) × min(M, child_impact)`.
- New `ConsumerContract` / `OutputGate` types and
  `compute_impacts_with_contract` entry point let callers declare downstream
  `MinAgainstConstant` / `MaxAgainstConstant` / `Identity` gates. Output
  impact is scaled by the gate's pass-through probability computed from
  recorded activations.
- New `derive_regime_threshold_from_records` helper picks a percentile from
  the recorded activation distribution (replaces hard-coded "low volume"
  constants).
- `docs/IMPACT_CALCULATION.md` documents the new semantics with a Mermaid
  flowchart of the consumer-gate pipeline and a squash emit-magnitude table.
- One existing test (`test_step_neuron_uses_full_impact_not_normalised`) was
  renamed and updated to assert the new, correct squash-bounded behaviour;
  the change is documented in-place explaining why the old assertion was the
  bug.

## Evidence

This is a backend / library change — no UI to screenshot. Verification is
via the unit + integration test suite.

- 9 new tests in `tests/regression/regression_issue_1300.rs` cover:
  - the squash emit-magnitude lookup (bounded ±1, RELU6=6.0, unbounded → None,
    aggregate → None, case-insensitive);
  - TANH inbound contributions staying inside `[-1, 1]` when the TANH feeds
    multiple outputs (`child_impact > 1`);
  - STEP inbound contributions staying bounded at the STEP emit magnitude
    (this was the `N × child_impact` overstatement);
  - sole-inbound synapse retaining its full impact (no behaviour change for
    the common case);
  - `MinAgainstConstant` gate masking impact in the regime where the network
    output does not drive the consumer (verified against records crafted so
    20/100 observations are below the gate threshold);
  - `Identity` gate leaving impact unchanged versus no contract;
  - regime threshold derivation at the min/max/p25 percentiles, with
    empty-records and non-finite-input edge cases.
- The full library + integration suite (`cargo test --tests --all-features`)
  passes: 30 test binaries report `0 failed` across ~1500 tests.
- `cargo clippy --all-targets --all-features -- -D warnings` is clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` is clean.

```mermaid
flowchart LR
    A[Recorded output<br/>activations] --> B{Consumer gate?}
    B -- "Identity" --> C[Factor = 1.0]
    B -- "min(out, t)" --> D[Factor = P(out < t)]
    B -- "max(out, t)" --> E[Factor = P(out > t)]
    C --> F[Scale output<br/>initial impact]
    D --> F
    E --> F
    F --> G[Backward propagate<br/>through network<br/>with squash-bounded cap]
```

## Test Plan

- [x] `cargo test --test regression regression_issue_1300` — 9 new tests pass.
- [x] `cargo test --test activation` — 105 tests pass (existing STEP/BIPOLAR
      tests with looser assertions still pass under new normalisation).
- [x] `cargo test --test analysis impact_calculation` — 9 tests pass
      (`test_step_neuron_contribution_is_squash_bounded` replaces the old
      `test_step_neuron_uses_full_impact_not_normalised` and locks in the
      Issue #1300 fix).
- [x] `cargo test --test focus` — 126 tests pass.
- [x] `cargo test --tests --all-features` — full integration suite passes.
- [x] `cargo clippy --all-targets --all-features -- -D warnings` — clean.
- [x] `cargo fmt --all -- --check` — clean.
- [x] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` — clean.
