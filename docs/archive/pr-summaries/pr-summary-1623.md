## Summary

Implements the **removal mechanism** for functionally-constant hidden neurons
(part of the #1620 milestone): fold each neuron's constant downstream
contribution into its target neurons' biases, then remove the neuron — behind an
**evaluate-before-accept gate** as a safety net. Closes #1623.

A neuron with a constant activation `c` contributes a fixed `w_{n→t}·c` to each
target `t` on every observation. Adding `w_{n→t}·c` to `t`'s bias reproduces that
contribution exactly, so removing the neuron and applying the fold is
behaviour-preserving on the recorded window — the constant-activation special
case where mean-preserving bias compensation is *complete*, not merely
mean-cancelling (unlike the general remove-neuron weight redistribution in
#1559).

### What was added

- `src/analysis/remove_neuron_bias_fold.rs` — the new module:
  - `evaluate_constant_neuron_bias_fold` — derives the folded constant `c` as the
    mean activation over the window, computes each target's `w × c` bias delta,
    and runs the gate (max per-sample residual `|w·(a_i − c)|` vs tolerance)
    **without mutating** the creature.
  - `fold_and_remove_constant_neuron` — evaluate-before-accept: applies the fold
    (bias deltas + delete neuron and every edge touching it) **only if** the gate
    passes; a rejected or unevaluable fold leaves the creature untouched and
    reports why (fail-loud, no blind delete).
  - `BiasFoldOutcome`, `FoldedTarget`, `BIAS_FOLD_GATE_TOLERANCE`.
- Reuses the bias-compensation machinery in
  `src/analysis/remove_neuron_compensation.rs` rather than duplicating it:
  `VARIANCE_EPSILON` is now `pub(crate)`, and a new
  `ActivationCovariance::from_values` wraps the existing numerically-stable
  mean/variance accumulation for the single-variable case.

### Gate flow

```mermaid
flowchart TD
    A[flagged constant neuron + recorded window] --> B[mean c, variance over window]
    B --> C[per target t: bias_delta = w_n→t · c]
    C --> D{max residual\n=w·|a_i − c| ≤ tolerance?}
    D -- yes --> E[apply: bias_t += w·c, delete neuron + all edges]
    D -- no --> F[reject: creature unchanged, reason recorded]
```

## Evidence

Backend/library change — no web interface to screenshot. Verified via the tests
below (`./quality.sh`: fmt, clippy `-D warnings`, check, doc build, release
build all pass; the full test suite passes — one pre-existing flaky wall-clock
timing test, `focus_ranking_aborts_when_budget_exceeded`, is unrelated and passes
in isolation).

Key exactness property under test — the target pre-activation
`bias_t + Σ_s w_{s→t}·a_{s,i}` is identical before and after the fold across every
observation, because the removed term `w_{n→t}·a_{n,i}` is replaced by the folded
constant `w_{n→t}·c` and `a_{n,i} = c`.

## Test Plan

Integration tests — `tests/analysis/issue_1623_constant_neuron_bias_fold.rs`:

- `constant_neuron_fold_preserves_outputs_across_window_and_removes_neuron` —
  post-fold output pre-activations are bit-identical (within tolerance) across
  the recorded window; the neuron and all its edges (incoming and outgoing) are
  gone.
- `multi_target_fan_out_folds_each_target_independently` — each target's bias
  receives its own `w × c` delta.
- `gate_rejects_non_constant_neuron_and_leaves_network_unchanged` — a fold whose
  evaluation fails is not accepted; the network is unchanged.
- `evaluate_does_not_mutate_creature` — pure evaluation has no side effects.

Unit tests — `src/analysis/remove_neuron_bias_fold.rs` (`mod tests`): accept +
target-bias update, gate rejection leaves creature unchanged, missing-records
fail-loud without deleting.

Shared-machinery unit test — `src/analysis/remove_neuron_compensation.rs`:
`from_values_reports_mean_and_variance` covers the reused mean/variance
accumulation and `VARIANCE_EPSILON` handling for constant and varying streams.
