## Summary

Updated all documentation to cover the full set of 38+ detection and
recommendation modules. Previously only 13 discovery types were listed in
the README; all modules are now documented and grouped by category.
Closes #757.

## Changes

- **README.md**: Replaced flat 13-row discovery types table with categorised
  tables (Activation & Neuron State, Weight & Synapse, Structural & Topology,
  Range & Input Analysis, Scoring & Recommendation) covering all modules.

- **docs/DISCOVERY_TYPES.md**: Added 22 new detailed module descriptions for
  previously undocumented modules (bimodal neuron, restricted range, operating
  point, activation mismatch, monotonicity, error plateau, output range
  compression, output squash mismatch, bias perturbation, squash + weight
  rescale, weight coherence, weight magnitude reset, weight polarity flip,
  fan-in polarity conflict, topology diversification, skip connection,
  symmetry breaking, co-adaptation, output conflict, hard sample cluster,
  bounded range, sentinel gating, observation utilisation, input sensitivity,
  sample-weighted discovery). Updated the summary table and table of contents
  to reflect all modules grouped by category.

- **docs/ANALYSIS_DEEP_DIVE.md**: Added comprehensive "Detection Module
  Reference" section covering algorithm descriptions for all 38+ detection
  modules across 5 categories.

- All synapse/ sub-modules already had module-level doc comments (no changes
  needed).

## Evidence

Documentation-only changes (no code changes). `quality.sh` passes cleanly.

## Test Plan

- No code changes, so no new tests required
- Verified `quality.sh` passes (including `cargo doc --no-deps` which validates
  all doc references)
