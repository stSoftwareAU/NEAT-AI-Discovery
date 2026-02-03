# PR Summary: Fix Remove-Neuron (High Error) Discovery — Issue #414

## Summary

Disabled the "remove-neuron (high error)" discovery type which had a 0% success rate (0 successes from 2 attempts). The root cause was a **fundamentally flawed assumption**: high error magnitude on a neuron does not mean the neuron is harmful to the network.

### Root Cause Analysis

The discovery type selected neurons where `raw_error >= 10 × max_output_error` and proposed removing them as "exploratory ablation candidates". This was based on the assumption that high error means the neuron is destabilising the network.

**Why this assumption is wrong:**

A neuron with high recorded error is often:
1. **Handling difficult samples** — It's the only computation path for hard cases
2. **Receiving bad inputs** — The error is a symptom of upstream problems, not a cause
3. **Fighting incorrect biases** — It's compensating for problems elsewhere in the network

Removing such neurons typically makes performance **worse** because:
- Difficult samples lose their only computation path
- The network loses the only neuron attempting to handle a specific pattern

**Key insight**: Error magnitude measures how **wrong** the neuron's output is, not how **harmful** the neuron is to the network's overall score.

### Changes Made

1. **Disabled high-error exploratory ablation** in `src/focus.rs` (both occurrences)
2. **Updated existing test** in `tests/exploratory_ablation_candidates.rs` to verify the behaviour is disabled
3. **Added new tests** in `tests/issue_414_remove_neuron_high_error.rs` to verify:
   - High-error neurons are NOT returned as removal candidates
   - Low-impact neurons are STILL returned as removal candidates (legitimate removal)
   - Removal candidate reasons reflect impact, not error magnitude
4. **Updated documentation** in `docs/DISCOVERY_TYPES.md` with:
   - Root cause analysis
   - Status changed from "Not working" to "Disabled"
   - Updated recommendations table

### What Remains Active

The legitimate "remove-low-impact" discovery type remains active with a 17.6% success rate. This discovery type identifies neurons where `activation_weighted_impact < costOfGrowth` — neurons that genuinely don't contribute to the network.

## Evidence

This is a backend logic fix with no UI changes. Evidence is provided through the test suite.

## Test Plan

New and updated tests verify the fix:

| Test File | Test Name | Purpose |
|-----------|-----------|---------|
| `tests/issue_414_remove_neuron_high_error.rs` | `high_error_neuron_is_not_returned_as_exploratory_ablation_candidate` | Verifies high-error neurons are no longer suggested for removal |
| `tests/issue_414_remove_neuron_high_error.rs` | `low_impact_neurons_still_returned_as_removal_candidates` | Verifies legitimate low-impact removal still works |
| `tests/issue_414_remove_neuron_high_error.rs` | `removal_candidate_reason_reflects_impact_not_error` | Verifies removal reasons mention impact, not error |
| `tests/exploratory_ablation_candidates.rs` | `high_error_hidden_neuron_is_not_returned_as_exploratory_ablation_candidate` | Updated existing test to verify new behaviour |

All tests pass with `./quality.sh`.

## Acceptance Criteria

- [x] Either achieve >5% success rate OR disable the discovery type — **Disabled**
- [x] Document the root cause in `docs/DISCOVERY_TYPES.md` — **Done**
