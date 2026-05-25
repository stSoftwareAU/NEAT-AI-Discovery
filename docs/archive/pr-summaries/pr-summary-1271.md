## Summary

Cap coordinated-structural candidates per final-operation target neuron
within a single batch (Issue #1271). Mirrors the per-target add-neuron cap
(Issue #1140) for the coordinated-structural pipeline so that a single
problematic target neuron — e.g. output `533d8616…` in creature
`bcbca347` (GRQ-sampler commit `e85c5d2`), which consumed 41 of 41
coordinated-structural failure slots — cannot monopolise the batch budget
while other targets go unexplored.

The cap runs inside `run_discovery_module` (and
`merge_discovery_module_results`) immediately after the existing
`COORDINATED_MIN_EXPECTED_GAIN` post-discount floor, **before** the cross-
target diversity spread (Issue #1193) operates downstream. Dropped
candidates are recorded under the new stable rejection reason
`coordinated_target_cap_exceeded`.

Closes #1271.

## Evidence

This is a backend / CLI change with no UI to screenshot. The evidence is
the test results:

- `coordinated_per_target_cap_reduces_ten_same_target_to_three` — synthetic
  10-candidate batch against one target is reduced to 3.
- `coordinated_per_target_cap_regression_bcbca347_41_same_target` —
  reproduces the 41-failure pattern from creature `bcbca347` and asserts
  it is capped at 3.
- `coordinated_per_target_cap_admits_full_quota_per_distinct_target` —
  two targets with four candidates each retain 3+3 = 6, confirming the
  cap is grouped by target rather than global.
- `coordinated_per_target_cap_env_override` — the
  `NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET` env var override is
  honoured and clamped to the documented range.
- `max_coordinated_per_target_output_default_is_three` — sanity check
  on the documented constant default.

`./quality.sh` passes cleanly with the new tests in place.

### Pipeline placement

```mermaid
flowchart LR
    A[Discovery module<br>detection] --> B[Per-module budget<br>truncation<br>Issue #967]
    B --> C[Expected-gain floor<br>COORDINATED_MIN_EXPECTED_GAIN<br>Issue #1110]
    C --> D[Per-final-target cap<br>MAX_COORDINATED_PER_TARGET_OUTPUT<br>Issue #1271 — this PR]
    D --> E[merge_coordinated_structural_replacements]
    E --> F[Cross-target diversity spread<br>MIN_DISTINCT_TARGETS_PER_BATCH<br>Issue #1193]
```

## Test Plan

- Added `tests` in `src/analysis/discovery_dispatch_tests.rs`:
  - `coordinated_per_target_cap_reduces_ten_same_target_to_three`
  - `coordinated_per_target_cap_regression_bcbca347_41_same_target`
  - `coordinated_per_target_cap_admits_full_quota_per_distinct_target`
  - `max_coordinated_per_target_output_default_is_three`
  - `coordinated_per_target_cap_env_override`
- `cargo test`, `cargo clippy`, `./quality.sh` all pass.
