## Summary

Addressed the "Brilliant but Brittle" issue (#432) by researching the three main repositories (NEAT-AI-Discovery, NEAT-AI, and the production observation layer) and creating targeted sub-issues to reduce brittleness in production creature predictions.

Bad or missing observations can wildly affect predictions. This PR creates a comprehensive plan across all three repositories to:

1. **Improve discovery modules** (NEAT-AI-Discovery) to detect and remove high noise-to-signal neurons/synapses
2. **Enhance evolutionary algorithms** (NEAT-AI) with stability-aware mutation and validation
3. **Strengthen observation handling** (production observation layer) with better error handling and sentinel value management

## Evidence

This is a planning/issue-creation task with no code changes or UI components. Evidence is provided via the created GitHub issues:

### NEAT-AI-Discovery Issues Created
- [#434](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/434) - High noise-to-signal ratio detection for neurons and synapses
- [#435](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/435) - Input sensitivity analysis module
- [#436](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/436) - Cross-validation consistency scoring
- [#437](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/437) - Weight coherence validation

### NEAT-AI Issues Created
- [#1307](https://github.com/stSoftwareAU/NEAT-AI/issues/1307) - Adaptive mutation rate based on validation stability
- [#1308](https://github.com/stSoftwareAU/NEAT-AI/issues/1308) - Enhanced discovery candidate validation with holdout testing
- [#1309](https://github.com/stSoftwareAU/NEAT-AI/issues/1309) - Weight regularisation during mutation
- [#1310](https://github.com/stSoftwareAU/NEAT-AI/issues/1310) - Ensemble diversity scoring for species

### Production Observation-Layer Issues Created

Four follow-up issues were raised in the private production observation-layer
repository (issue links omitted — that repository is not public):

- Standardised sentinel value handling across all observation extensions
- Observation quality scoring and filtering
- Robust error handling in observation extensions
- Input normalisation consistency checks

## Research Findings

### Existing Brittleness Detection in NEAT-AI-Discovery

The codebase already has strong foundations for brittleness detection:

1. **Weight Guard Rails** (`src/analysis/weights.rs:40-54`)
   - MAX_OUTGOING_WEIGHT: 0.1 (successful discoveries have |outgoing_weight| < 0.05)
   - MIN_WEIGHT_RATIO: 50.0 (successful: 71x to 104,000x ratio)

2. **Sensible Range Filtering** (`src/analysis/utils/mod.rs:70-92`)
   - SENSIBLE_INCOMING_ABS_MAX: 20.0
   - SENSIBLE_BIAS_ABS_MAX: 10.0
   - Conservative pairing strategy for production stability

3. **Sentinel Value Handling** (`src/analysis/sentinel_gating.rs`)
   - Error-correlation analysis for sentinel detection
   - Proposes gated connections to suppress null/sentinel observations

4. **Operating Point Analysis** (`src/analysis/operating_point.rs`)
   - Detects neurons working outside their effective zone
   - Proposes bias adjustments to centre operating points

### Gaps Identified

1. Explicit noise-to-signal ratio scoring per candidate
2. Cross-neuron brittleness propagation detection
3. Temporal stability analysis across validation sets
4. Gradient-based sensitivity analysis
5. Standardised observation quality metrics across production observation extensions

## Test Plan

This PR creates issues for future implementation. No code changes were made, so no new tests are required. The parent issue (#432) has been updated with links to all sub-issues.

All created issues include:
- TDD requirements (write failing tests first)
- DRY principle reminders
- Australian English spelling requirements
- Clear acceptance criteria

## Quality Verification

- `./quality.sh` passes with all 440 tests succeeding
- No code changes made to this repository
- All issues created successfully via GitHub CLI
