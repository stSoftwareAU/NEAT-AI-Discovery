# PR Summary: Tighten weight constraints to match successful candidate patterns

Closes #888

## Problem

GRQ-sampler discovery cache analysis shows that successful add-neuron candidates
cluster in a narrow weight range that is far tighter than the previous constraints
allowed. Specifically:

| Parameter       | Successes       | Failures       | Old Limit | New Limit |
|-----------------|-----------------|----------------|-----------|-----------|
| Outgoing weight | 0.001–0.005     | 0.01–10+       | 0.1       | 0.01      |
| Incoming weight | 1.5–3.0         | 5–200+         | 20.0      | 5.0       |
| Bias magnitude  | 0.0–1.5         | 5–50+          | 10.0      | 2.0       |

The previous constraints allowed "Extreme" pattern candidates (large incoming,
large outgoing, large bias) that almost always fail in production.

## Changes

### New constants (`constants.rs`)
- `MAX_INCOMING_WEIGHT = 5.0` — caps add-neuron incoming weights
- `MAX_BIAS_MAGNITUDE = 2.0` — caps add-neuron bias
- `MICRO_NUDGE_VARIANT_BOOST = 1.5` — scoring boost for the dominant success pattern

### Tightened `MAX_OUTGOING_WEIGHT` (`scoring/weights/mod.rs`)
- Reduced from `0.1` to `0.01` — outgoing weights now clamped to ±0.01

### Variant generation (`variant_generation.rs`)
- **Conservative**: outgoing_abs_max 0.05→0.005, min_outgoing_fallback 0.01→0.001
- **Gentle Nudge**: incoming 20→5, bias 10→2, outgoing 0.02→0.01, min_fallback 0.005→0.002
- **Micro-Nudge**: expected_multiplier 0.25→0.5 (boosted to reflect cache dominance)
- **Sensible ranges**: incoming 20→5, bias 10→2, outgoing 0.1→0.01

### Gradient discovery (`gradient_discovery.rs`)
- Separated synapse weight adjustment clamping from add-neuron outgoing weight
  clamping. Synapse adjustments now use `MAX_GRADIENT_ADJUSTED_WEIGHT = 10.0`
  instead of `MAX_OUTGOING_WEIGHT`, since existing synapses can have larger weights.

### Test updates
- New `tests/scoring/issue_888_weight_constraints.rs` with 17 tests covering
  compile-time constant validation, sensible-range filtering, variant config
  constraints, variant clamping, and weight calculation with tightened ceiling
- Updated 8 existing test files to reflect tightened constraint values
- Restructured IDENTITY affine fit test to directly test the calculation function
- Updated parallel determinism test tolerance for tightened candidate scoring

## Test plan

- [x] All 17 new tests pass
- [x] All existing tests updated and passing
- [x] `./quality.sh` passes cleanly (build, fmt, clippy, check, test, doc, release)
