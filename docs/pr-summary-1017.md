## Summary

Audit the candidate selection pipeline and evaluate MCMC applicability. After thorough analysis of 10 key source files, the finding is that **there is no explicit MCMC implementation** and **MCMC is not the correct framework** for this system. The pipeline is a one-shot optimisation search (propose, evaluate, filter, rank) with deterministic accept/reject, not a Markov chain sampling procedure. The 0% synapse success rate is best addressed through calibration improvements to the existing deterministic pipeline. Closes #1017.

### Key Findings

- **No Markov chain state**: Each analysis invocation is independent and stateless
- **Deterministic acceptance**: `MIN_IMPROVED_RATIO = 0.6` hard cutoff, no probabilistic Metropolis-Hastings acceptance
- **No detailed balance or ergodicity**: Pipeline searches for best candidates, does not sample from a posterior
- **Existing calibration is sound**: Pessimism discounts, prediction calibration, hold-out validation, and adaptive boosts from cache evidence already address the prediction-to-reality gap
- **Recommendation**: Do not introduce MCMC machinery. Focus on expanding evaluation data and refining calibration parameters

### Deliverables

- `docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md` — Full analysis document mapping the pipeline to MCMC concepts, identifying gaps, evaluating MCMC techniques (MH acceptance, simulated annealing, adaptive proposals, temperature scheduling), and providing a recommendation
- `tests/synapse/issue_1017_candidate_pipeline_mcmc_audit.rs` — 13 integration tests validating the audit findings through code

## Evidence

- All 13 new tests pass, confirming:
  - Deterministic accept/reject (no probabilistic acceptance)
  - Pessimism discount hierarchy correctly ordered by success rate (synapse < neuron < generic)
  - Prediction calibration hierarchy correctly ordered by overestimation severity
  - Hold-out validation parameters within sensible ranges
  - Full calibration stack reduces predictions by >90%
  - Calibration stack is stateless (no MCMC chain state)
- All existing tests continue to pass (quality.sh passes cleanly)

## Test Plan

- Added `tests/synapse/issue_1017_candidate_pipeline_mcmc_audit.rs` with 13 tests:
  - `accept_reject_is_deterministic_threshold` — confirms hard cutoff, not probabilistic
  - `zero_improvement_always_produces_zero_or_negative_gain` — confirms greedy rejection
  - `pessimism_floor_hierarchy_matches_success_rates` — validates floor ordering
  - `pessimism_exponent_hierarchy_matches_success_rates` — validates exponent ordering
  - `pessimism_discount_ordering_at_moderate_ratio` — validates runtime discount ordering
  - `prediction_calibration_hierarchy` — validates calibration factor ordering
  - `prediction_calibration_factors_are_scale_down` — validates all factors in (0, 1)
  - `prediction_calibration_always_reduces_gain` — confirms calibration reduces predictions
  - `holdout_validation_parameters_sensible` — validates hold-out parameters
  - `diversify_top_k_in_valid_range` — validates diversification constant
  - `coordinated_pessimism_discount_is_scale_down` — validates coordinated discount
  - `full_synapse_calibration_stack_reduces_prediction` — end-to-end calibration validation
  - `calibration_stack_is_stateless` — confirms no hidden MCMC chain state
