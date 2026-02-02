## Summary

Implements hierarchical focus neuron selection for large creatures (500+ neurons) as described in Issue #222.

### Problem

For large creatures, the flat focus neuron selection approach had several limitations:
1. Computed impact for ALL neurons (expensive for large creatures)
2. Didn't consider network structure
3. Often over-represented neurons from certain layers while under-representing others

### Solution

Added hierarchical focus selection that organises neurons into layers based on their network depth (distance from inputs) and allocates focus budget proportionally across layers.

#### New Features

1. **`compute_network_layers()`** - Organises neurons by depth using BFS from inputs
   - Handles cycles gracefully with max-depth protection
   - Handles disconnected components (assigns to special layer)
   - Returns layers sorted by depth (shallowest first)

2. **`AllocationStrategy` enum** - Three allocation strategies:
   - `Equal`: Divides focus budget evenly across layers
   - `Proportional`: Allocates based on layer size (larger layers get more)
   - `OutputFirst`: Prioritises layers closest to output (for output-biased discovery)

3. **`hierarchical_focus_selection()`** - Main selection function
   - Allocates budget across layers using specified strategy
   - Within each layer, selects top-scoring neurons
   - Redistributes unused slots if a layer has fewer neurons than allocated
   - Guarantees coverage across all network depths

4. **`HIERARCHICAL_SELECTION_THRESHOLD`** - Constant (100 neurons) for auto-switching between flat and hierarchical selection

### Benefits

- **Better coverage**: Guaranteed analysis of neurons at all network depths
- **Output neuron guarantee**: Output-adjacent neurons are always represented
- **Proportional representation**: Hidden layers are represented based on strategy
- **Improved discovery quality**: Better distribution leads to more diverse candidates

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. Performance improvements would require dedicated benchmarks with production-scale creatures (500+ neurons).

The implementation provides the building blocks for hierarchical selection. Integration with `rank_focus_neurons` can be done as a follow-up change to enable automatic hierarchical selection for large creatures.

## Test Plan

Added 14 comprehensive tests in `tests/issue_222_hierarchical_focus.rs`:

### Layer Computation Tests
- `test_compute_network_layers_simple_chain` - Verifies basic chain topology
- `test_compute_network_layers_parallel_paths` - Verifies parallel paths with different depths
- `test_compute_network_layers_handles_cycles` - Verifies cycle detection doesn't cause infinite loops
- `test_compute_network_layers_disconnected_component` - Verifies orphan neurons are included
- `test_compute_network_layers_deep_network` - Verifies deep networks (10 layers)

### Allocation Strategy Tests
- `test_allocation_strategy_equal` - Equal distribution across layers
- `test_allocation_strategy_proportional` - Size-based allocation
- `test_allocation_strategy_output_first` - Output-prioritised allocation

### Selection Tests
- `test_hierarchical_focus_selection_guarantees_layer_coverage` - All layers have representation
- `test_hierarchical_selection_respects_max_focus` - Doesn't exceed budget
- `test_hierarchical_selection_uses_scores_within_layer` - Top scores win within each layer
- `test_hierarchical_selection_handles_empty_layers` - Edge case handling
- `test_hierarchical_selection_improves_layer_distribution` - Prevents layer domination
- `test_large_creature_hierarchical_selection` - Full integration with 500+ neurons

All 14 tests pass, plus all 349 existing tests continue to pass.
