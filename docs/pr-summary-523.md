## Summary

Add targeted tests for 4 focus sub-modules: `layers`, `allocation`, `gradient`, and `ranking`. These sub-modules were split out in Issue #491 but lacked dedicated test files. Closes #523.

## Evidence

This is a test-only change with no UI or performance implications. All 48 new tests pass across 4 test files:

- `tests/focus_layers.rs` — 10 tests
- `tests/focus_allocation.rs` — 14 tests
- `tests/focus_gradient.rs` — 11 tests
- `tests/focus_ranking.rs` — 13 tests

`./quality.sh` passes cleanly (fmt, clippy, check, tests, release build).

## Test Plan

### `tests/focus_layers.rs` (10 tests)
- Linear chain assigns increasing BFS depths
- Fan-out groups sibling neurons at same depth
- Skip connections use maximum depth (not shortest path)
- Skip connections don't affect intermediate depths
- Multiple inputs contribute to depth calculation
- Disconnected neurons assigned to unreachable layer
- Input and constant neurons excluded from layers
- Empty creature returns no layers
- Output-only creature produces a layer
- Diamond topology assigns correct depths

### `tests/focus_allocation.rs` (14 tests)
- Equal distributes budget evenly across layers
- Equal distributes remainder to deeper layers
- Proportional gives more to larger layers
- Proportional handles uneven division
- OutputFirst prioritises deeper layers
- OutputFirst fills deep layers before shallow
- Zero budget returns empty
- Empty layers returns empty
- Budget exceeding total neurons selects all
- Single layer gets entire budget
- Higher-scored neurons selected first within layer
- Neurons with no score treated as zero
- All strategies respect max_focus
- All strategies return unique neurons

### `tests/focus_gradient.rs` (11 tests)
- RELU all-negative values fully dead (dead_ratio ≈ 1.0)
- RELU all-positive values not dead (gradient = 1.0)
- RELU mixed values partial dead (dead_ratio ≈ 0.5)
- TANH extreme values saturated
- TANH moderate values not saturated
- LOGISTIC extreme values saturated
- IDENTITY always has gradient 1.0
- Mixed activations produce independent stats
- Default GradientFlowStats assumes full flow
- Neuron with no finite values gets default stats
- ELU negative values have non-zero gradient (never dead)

### `tests/focus_ranking.rs` (13 tests)
- SynapseCounts correct for simple chain
- SynapseCounts for hub neuron (3 incoming, 2 outgoing)
- SynapseCounts returns zero for nonexistent neuron
- Removal savings matches NEAT-AI formula
- Removal savings scales with growth_cost
- Output neuron has impact 1.0
- Hidden neuron impact bounded (0, 1]
- Disconnected neuron has zero impact
- High error + high impact neuron ranked first
- Low impact neuron identified as removal candidate
- High impact neuron not a removal candidate
- max_results limits output size
- Ranked neurons have correct error and impact
