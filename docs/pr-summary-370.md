## Summary

Consolidates discovery type documentation (DRY) by making `docs/DISCOVERY_TYPES.md`
the single source of truth for all discovery types (Issue #370).

**Changes**:

- **`docs/DISCOVERY_TYPES.md`**: Expanded from a tracking document into a comprehensive
  reference covering all 16 discovery types with detection criteria, recommended actions,
  candidate output format, and production success rates. Added 10 previously undocumented
  types: Saturated Neuron, Bottleneck Neuron, Dead Neuron, Dormant Synapse, Opposing
  Synapse, Output Bias Drift, Oscillating Neuron, Correlated Error Pattern, Multi-Hop
  Candidate Analysis, and Redundant Path Pruning.

- **`README.md`**: Replaced the category-level Discovery Types table with a per-type
  summary table that links directly to the corresponding section in
  `docs/DISCOVERY_TYPES.md`. No detection criteria or detailed descriptions remain in
  README.md.

- **Rust source modules** (10 files): Added `docs/DISCOVERY_TYPES.md` cross-references
  to each discovery analysis module's doc comment (`saturation.rs`, `bottleneck.rs`,
  `dead_neuron.rs`, `dormant_synapse.rs`, `opposing_synapse.rs`, `output_bias_drift.rs`,
  `oscillating_neuron.rs`, `correlated_error.rs`, `multi_hop.rs`, `redundant_path.rs`).

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- Added `tests/discovery_types_doc_consistency.rs` with 4 tests:
  - `each_discovery_module_references_discovery_types_md` — verifies each Rust source
    module references `docs/DISCOVERY_TYPES.md` and its section name
  - `discovery_types_md_documents_each_module` — verifies `DISCOVERY_TYPES.md` contains
    a heading and source file reference for each module
  - `readme_links_to_discovery_types_md` — verifies README links to
    `docs/DISCOVERY_TYPES.md` and does not contain detailed detection criteria
  - `discovery_types_md_is_single_source_of_truth` — verifies `DISCOVERY_TYPES.md`
    contains detection criteria, recommended actions, and output format information
- `./quality.sh` passes cleanly
