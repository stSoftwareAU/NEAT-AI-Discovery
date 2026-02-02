## Summary

Implement a dynamic threshold for folding constant-source synapses into bias operations, based on the overall source variance profile of the creature.

**Issue #199**: The threshold for folding constant-source synapses into bias operations now scales based on the creature's source variance profile using the formula:

```
dynamic_threshold = 1e-7 × max(1.0, source_std_dev_avg / 0.05)
```

This means:
- For creatures with mostly low-variance sources: threshold stays at 1e-7
- For creatures with high-variance sources: threshold scales up proportionally

This captures more coordinated candidates in creatures where "constant" is relative to the overall variance distribution.

### Implementation Details

1. **Source Variance Profile Calculation**: Sample up to 50 input neurons to compute the average standard deviation of source activations during synapse analysis initialisation.

2. **Dynamic Threshold Function**: Added `compute_dynamic_constant_source_threshold()` that scales the default threshold based on the source variance profile.

3. **Env Var Override**: The existing `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD` environment variable still works:
   - unset/empty: uses dynamic threshold based on source variance profile
   - `0`: disables folding entirely
   - `> 0`: uses explicit fixed threshold (overrides dynamic calculation)

### Expected Impact

- Better adaptation to different creature architectures
- More coordinated structural candidates discovered in high-variance creatures
- No impact on creatures with typical variance distributions

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface. The feature is verified through unit tests.

## Test Plan

Added comprehensive tests in `tests/issue_199_dynamic_constant_source_threshold.rs`:

1. **test_low_variance_sources_use_default_threshold**: Verifies that creatures with uniformly low variance sources use the default threshold
2. **test_high_variance_sources_scale_threshold**: Verifies that creatures with high variance sources get a scaled-up threshold
3. **test_env_var_override_takes_precedence**: Verifies that explicit env var override takes precedence over dynamic calculation
4. **test_env_var_zero_disables_folding**: Verifies that setting threshold to 0 disables constant-source folding

Added unit tests in `src/analysis/samples.rs`:
- `test_compute_dynamic_constant_source_threshold_low_variance`
- `test_compute_dynamic_constant_source_threshold_at_reference`
- `test_compute_dynamic_constant_source_threshold_high_variance`
- `test_compute_dynamic_constant_source_threshold_edge_cases`
- `test_compute_source_std_dev_from_records`
- `test_get_constant_source_threshold_no_env_var`

All tests pass with `./quality.sh`.
