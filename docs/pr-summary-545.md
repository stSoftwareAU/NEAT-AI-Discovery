## Summary

Implements local minimum escape strategies for NEAT-AI Discovery. Closes #545.

The issue identified that networks using HARD_TANH as their output activation function cannot converge when targets are generated with TANH, because weight/bias adjustments alone cannot cross the structural barrier between activation functions. This PR addresses the problem with:

1. **Six GitHub sub-issues** (#546–#551) covering different local minimum escape strategies
2. **Two new detection modules** implementing the most impactful strategies:
   - `output_squash_mismatch` (#546) — detects when output neuron activation functions don't match the target data range
   - `error_plateau` (#547) — detects when error distributions show stagnation patterns indicating the network is stuck

### Sub-issues created

| Issue | Strategy | Description |
|-------|----------|-------------|
| #546 | Output squash mismatch detection | Detects HARD_TANH→TANH and similar mismatches |
| #547 | Error stagnation plateau detection | Detects flat error landscapes |
| #548 | Coordinated multi-neuron squash exploration | Bundle squash changes with weight rescaling |
| #549 | Topology diversification | Add neurons at strategic positions |
| #550 | Weight magnitude reset | Exploratory weight jumps for stuck synapses |
| #551 | Bias perturbation regime shifts | Large bias shifts to change operating regimes |

### New detection modules

**Output squash mismatch** (`src/analysis/detection/output_squash_mismatch.rs`):
- Strategy 1: Detects bounded activations with clipping at saturation bounds (e.g., HARD_TANH clipping at ±1)
- Strategy 2: Detects bounded activations whose range doesn't cover the target data (e.g., LOGISTIC [0,1] with [-1,1] targets)
- Strategy 3: Detects unbounded activations where targets need bounding
- Generates `changeSquash` coordinated candidates with confidence scores

**Error plateau** (`src/analysis/detection/error_plateau.rs`):
- Detects high mean error with low coefficient of variation (tight clustering around non-zero error)
- Recommends structurally different activation functions to escape the flat error landscape
- Generates `changeSquash` coordinated candidates

Both modules are wired into `analyze_all()` and run in parallel with all other detection modules.

## Evidence

This is a backend change with no UI components. Evidence is provided by the test suite:

- 8 integration tests for output squash mismatch detection
- 7 integration tests for error plateau detection
- All 15 new tests pass
- All 496 existing unit tests pass
- All existing integration tests pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

### `tests/issue_545_output_squash_mismatch.rs` (8 tests)
- `test_hard_tanh_output_with_smooth_tanh_targets_detected` — positive detection of the exact scenario from the issue
- `test_tanh_output_with_tanh_targets_no_detection` — no false positive when squash matches
- `test_insufficient_samples_returns_empty` — minimum sample threshold
- `test_hidden_neurons_are_ignored` — only output neurons analysed
- `test_coordinated_candidate_conversion` — valid changeSquash candidates with issue reference
- `test_multiple_output_neurons_detected` — multiple mismatches detected
- `test_no_preactivation_data_still_analyses` — works without pre-activation values
- `test_logistic_output_with_symmetric_targets` — LOGISTIC range mismatch detection

### `tests/issue_545_error_plateau.rs` (7 tests)
- `test_stagnant_error_plateau_detected` — positive detection of plateau pattern
- `test_no_detection_for_low_error` — no false positive when error is low
- `test_insufficient_samples_returns_empty` — minimum sample threshold
- `test_coordinated_candidate_conversion` — valid candidates with issue reference
- `test_multiple_neurons_mixed_results` — only plateau neurons flagged
- `test_high_variance_errors_not_detected` — high-variance errors not confused with plateau
- `test_empty_outputs_returns_empty` — empty input handling
