## Summary

Increases the change-squash suggestion rate (Issue #417). The change-squash discovery type
had an 18.2% success rate (highest among active types) but very low volume — only 11 total
samples across all experiments. The low volume was caused by overly conservative detection
thresholds and the proactive activation recommendation engine not being integrated.

### Changes

**1. Lowered saturation detection thresholds** (`src/analysis/saturation.rs`):
- TANH threshold: 0.95 → 0.85 (catches neurons approaching saturation earlier)
- LOGISTIC upper/lower: 0.95/0.05 → 0.90/0.10
- HARD_TANH threshold: 0.99 → 0.95
- Max activation std dev: 0.05 → 0.08 (accommodates near-saturated neurons)

**2. Lowered oscillation detection thresholds** (`src/analysis/oscillating_neuron.rs`):
- Sign change fraction: 0.30 → 0.15 (catches mildly oscillating neurons)
- Minority sign fraction: 0.20 → 0.10

**3. Integrated proactive activation recommendation** (`src/analysis/mod.rs`):
- The activation recommendation engine (Issue #431) was already implemented but not
  wired into the `analyze_all()` pipeline. Now integrated using the standard
  `discovery_dispatch::run_discovery_module()` pattern, generating `changeSquash`
  candidates based on input distribution analysis before saturation/oscillation occurs.

### Rationale for threshold values

- **TANH at 0.85**: At tanh(1.26) ≈ 0.85, the derivative is ~0.28 (down from 1.0 at
  x=0). This means gradient flow is already reduced by 72%. Catching these neurons
  early allows activation changes before gradient information is fully lost.
- **Oscillation sign change at 0.15**: Even 15% sign changes in consecutive samples
  indicates the neuron is meaningfully fighting between two functions. The previous
  30% threshold missed neurons with periodic (not random) oscillation patterns.
- **Minority sign at 0.10**: A neuron with 10% negative activations still has significant
  sign conflict that an activation change (e.g., ABSOLUTE) can resolve.

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface.

The expected increase in candidate volume is structural:
- Near-saturation: TANH neurons at 0.85–0.95 are now detected (previously only >0.95)
- Near-oscillation: Neurons with 15–30% sign change fraction are now detected
- Proactive recommendations: All hidden neurons are now analysed for suboptimal activation
  function choices based on input distribution

## Test Plan

- Added `tests/issue_417_increase_change_squash_rate.rs` with 15 tests:
  - `test_detects_tanh_near_saturation_at_090` — TANH at 0.90 detected
  - `test_detects_tanh_near_saturation_at_negative_088` — TANH at -0.88 detected
  - `test_detects_logistic_near_saturation_at_092` — LOGISTIC at 0.92 detected
  - `test_detects_logistic_near_saturation_at_lower_bound` — LOGISTIC at 0.08 detected
  - `test_does_not_detect_tanh_at_070` — TANH at 0.70 not flagged
  - `test_near_saturation_has_lower_improvement_than_full` — Severity grading correct
  - `test_detects_mild_oscillation_at_020_sign_change_fraction` — Mild oscillation detected
  - `test_detects_oscillation_with_lower_minority_fraction` — Lower minority sign detected
  - `test_does_not_detect_stable_neuron_with_lowered_thresholds` — Stable neurons excluded
  - `test_saturation_recommends_expanded_activations` — changeSquash operations produced
  - `test_oscillation_recommends_appropriate_activation_for_logistic` — Correct activation recommended
  - `test_proactive_recommendation_for_suboptimal_activation` — Proactive recommendation works
  - `test_proactive_recommendations_to_coordinated_candidates` — Conversion to candidates works
  - `test_lowered_thresholds_increase_candidate_volume` — Multiple neurons detected
  - `test_detects_softsign_near_saturation` — SOFTSIGN at 0.88 detected
- All 13 existing saturation tests pass (Issue #342)
- All 19 existing oscillation tests pass (Issue #358)
- All 17 existing activation recommendation tests pass (Issue #431)
- `quality.sh` passes cleanly
