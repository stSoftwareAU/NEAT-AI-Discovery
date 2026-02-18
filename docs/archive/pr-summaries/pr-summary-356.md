## Summary

Adds four new structural discovery methods to the analysis pipeline, addressing issue #356's
request to itemise and implement additional discoveries beyond the existing set. Each new
method detects a specific pattern in recorded neuron/synapse data and recommends corrective
structural changes via the existing `coordinatedStructuralCandidates` mechanism.

### New Discovery Methods

1. **Oscillating Neuron Detection** — Identifies hidden neurons whose activations frequently
   change sign across samples, indicating the neuron is fighting between contradictory
   functions. Recommends changing the activation function (e.g., TANH → ABSOLUTE) to
   stabilise output.

2. **Dormant Synapse Detection** — Identifies synapses with near-zero weights (< 1e-4) that
   contribute negligible signal. Recommends removal to reduce network complexity, provided
   the target neuron has other incoming connections.

3. **Opposing Synapse Detection** — Identifies synapses whose contribution correlates
   positively with target error (Pearson r ≥ 0.3), meaning they actively worsen predictions.
   Recommends removal (strong opposition) or weight sign flip (moderate opposition).

4. **Output Bias Drift Detection** — Identifies output neurons with systematic error sign
   bias (> 70% same sign), indicating a prediction offset. Recommends bias adjustment by
   the negative of the mean error to centre predictions.

All four methods follow the established patterns:
- Detection functions with configurable thresholds
- Candidate structures with statistics and confidence
- Conversion to `CoordinatedStructuralCandidateJson` for the NEAT-AI controller
- Integration into the `analyze_all` pipeline with watchdog heartbeats, phase timers,
  and verbose logging

No new candidate types are required — all methods reuse existing operation types
(`changeSquash`, `setBias`, `removeSynapse`, `setWeight`).

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. All
changes are verified through unit tests.

## Test Plan

30 new tests across four test files:

- `tests/issue_356_oscillating_neuron_detection.rs` (9 tests)
  - Detects alternating activation patterns
  - Excludes consistently positive, dead, low-frequency, unbalanced, and insufficient-sample neurons
  - Produces correct `changeSquash` coordinated candidates
  - Recommends ABSOLUTE for symmetric activation functions
  - Only detects oscillating neurons in mixed populations

- `tests/issue_356_dormant_synapse_detection.rs` (7 tests)
  - Detects near-zero weight synapses
  - Excludes active synapses, sole connections, and insufficient samples
  - Produces correct `removeSynapse` coordinated candidates
  - Detects multiple dormant synapses and records correct fan-in counts

- `tests/issue_356_opposing_synapse_detection.rs` (5 tests)
  - Detects positive contribution–error correlation
  - Excludes helpful (negative correlation) and hidden-target synapses
  - Handles insufficient samples correctly
  - Produces removal or weight-flip candidates based on correlation strength

- `tests/issue_356_output_bias_drift_detection.rs` (9 tests)
  - Detects positive and negative bias drift
  - Excludes balanced errors, hidden neurons, insufficient samples, and noise-level errors
  - Produces correct `setBias` coordinated candidates
  - Only detects biased outputs in mixed populations
  - Records current bias correctly
