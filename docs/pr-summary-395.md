## Summary

Implements bounded range discovery (Issue #395) to detect observations and hidden neurons
where activation values split into a meaningful bounded range and a sentinel/null cluster.

For example, an observation like "Debt-to-Equity" might use -1 as a sentinel for "no data",
while values in 0..1 carry actual meaning. Without gating, a linear weight treats the sentinel
as a real value, which degrades the creature's score. This module detects such bimodal
distributions and recommends inserting an IF-gated hidden neuron that passes the meaningful
signal when in the valid range and outputs zero at the sentinel value.

### Key design decisions

- **Histogram-based bimodal detection**: Uses a 20-bin histogram to find the dominant cluster
  (sentinel), then validates separation from the meaningful range.
- **IF squash function**: Recommends coordinated structural candidates using the existing `IF`
  squash function (`if condition > 0 then positive_input else negative_input`), which is
  already supported by NEAT-AI. No new candidate types required.
- **DRY**: Uses the generic `discovery_dispatch::run_discovery_module()` pattern (Issue #375)
  for dispatch, consistent with all other discovery modules.
- **Input and hidden neurons**: Both observation inputs and hidden neurons are eligible for
  bounded range detection.

### Files changed

| File | Change |
|------|--------|
| `src/analysis/bounded_range.rs` | New module: detection and candidate conversion |
| `src/analysis/mod.rs` | Register module and add dispatch call in `analyze_all()` |
| `tests/issue_395_bounded_range_detection.rs` | 13 integration tests covering all detection scenarios |

## Evidence

Unable to generate screenshot: This is a CLI-only Rust library with no visual interface.

## Test Plan

- `test_detects_sentinel_at_lower_bound` — Sentinel at -1, meaningful range 0..1
- `test_uniform_spread_not_flagged` — Uniform distribution is not flagged
- `test_insufficient_samples_not_flagged` — Too few samples rejected
- `test_output_neurons_not_flagged` — Output neurons excluded
- `test_detects_sentinel_at_upper_bound` — Sentinel at +1, meaningful range -1..0
- `test_detects_sentinel_at_zero` — Sentinel at 0, meaningful range 0.3..1.0
- `test_candidates_produce_if_gated_operations` — Coordinated candidate has IF squash
- `test_mixed_neurons_only_bounded_range_detected` — Only bimodal neurons flagged
- `test_hidden_neuron_with_bounded_range_detected` — Hidden neurons also eligible
- `test_estimated_improvement_positive` — All improvements are positive
- `test_sample_count_recorded` — Sample count matches records
- `test_confidence_increases_with_more_samples` — Confidence scales with sample size
- `test_neuron_with_downstream_found` — Downstream neurons correctly identified
