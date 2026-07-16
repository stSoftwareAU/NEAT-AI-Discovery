## Summary

Add activation compatibility scoring that penalises neuron candidate squash functions likely to compound clipping when feeding into bounded target activations (`HARD_TANH`, `TANH`, `LOGISTIC`, `CLIPPED`). Closes #1113.

### What changed

- **`src/analysis/activation/compatibility.rs`** (new): `activation_compatibility_score(candidate_squash, target_squash) -> f32` classifies candidates and targets into output-range categories and returns a multiplier in (0, 1]. Scores: 1.0 (fully compatible, e.g., `IDENTITY` → any), 0.8 (moderate, e.g., `TANH` → `HARD_TANH`), 0.3–0.5 (poor, e.g., `ABSOLUTE`/`Softplus` → `HARD_TANH`).
- **`src/analysis/activation/mod.rs`**: Added `compatibility` sub-module and re-export.
- **`src/analysis/neuron/evaluation.rs`**: Applied the compatibility score as a multiplier to `expected_creature_error_reduction` and `expected_creature_score_gain` in `evaluate_activation_specs()`, after the activation boost and before cross-validation penalty.

### Design decisions

- Candidates are **never filtered outright** — only deprioritised via a score multiplier, preserving exploration.
- Classification uses two small enums (`CandidateClass`, `TargetClass`) with a match table rather than a full N×N matrix, keeping the code maintainable as new activations are added.
- Unknown candidates default to `Flexible` (score 1.0) so new activations are not accidentally penalised.

## Evidence

Backend-only change with no UI. Verified by 14 unit tests covering all issue-specified combinations plus exhaustive validation across all 15 activation specs × 7 target types.

## Test Plan

- `identity_to_hard_tanh_is_fully_compatible` — IDENTITY → HARD_TANH = 1.0
- `absolute_to_hard_tanh_is_penalised` — ABSOLUTE → HARD_TANH ≤ 0.5
- `softplus_to_hard_tanh_is_penalised` — Softplus → HARD_TANH ≤ 0.5
- `tanh_to_identity_is_fully_compatible` — TANH → IDENTITY = 1.0
- `gelu_to_hard_tanh_is_penalised` — GELU → HARD_TANH ≤ 0.6
- `elu_to_hard_tanh_is_penalised` — ELU → HARD_TANH ≤ 0.6
- `tanh_to_hard_tanh_is_moderately_compatible` — TANH → HARD_TANH = 0.8
- `relu6_to_hard_tanh_is_penalised` — ReLU6 → HARD_TANH ≤ 0.5
- `logistic_to_hard_tanh_is_penalised` — LOGISTIC → HARD_TANH ≤ 0.5
- `identity_to_identity_is_fully_compatible` — IDENTITY → IDENTITY = 1.0
- `mish_to_tanh_is_penalised` — Mish → TANH ≤ 0.6
- `any_candidate_to_unbounded_target_is_fully_compatible` — 8 candidates → IDENTITY = 1.0
- `all_spec_combinations_produce_valid_scores` — all 15 specs × 7 targets in (0, 1]
- `unknown_candidate_defaults_to_flexible` — unknown → HARD_TANH = 1.0
