## Summary

Comprehensive documentation audit and update across all project docs. Closes #884.

### What changed

- **Archived 12 PR summaries** (PRs #841-#846, #872-#877) from `docs/` into `docs/archive/pr-summaries/`
- **AGENTS.md**: Updated detection module listing from 18 to 46 files (added 28 missing modules including activation_mismatch, bias_perturbation, bimodal_neuron, co_adaptation, error_plateau, fanin_polarity_conflict, hard_sample_cluster, high_error_squash_exploration, low_impact_neuron, monotonicity, observation_utilisation, output_conflict, output_range_compression, output_squash_mismatch, skip_connection, squash_weight_rescale, symmetry_breaking, topology_cache, topology_diversification, weight_magnitude_reset, weight_polarity_flip, and infrastructure modules). Added 5 missing analysis-level modules (candidate_diversity, candidate_cache, ensemble_scoring, module_weights, neuron_fingerprint, system). Updated test count (~97 to ~242 files) and benchmark count (7 to 22 suites).
- **BENCHMARKS.md**: Updated benchmark suite list from 12 to 22, adding 10 missing suites (async_pipeline, bfs_allocation, error_collection, gpu_shader_workgroup, squash_normalisation, synapse_lookup, synapse_preparation, topology_cache, uuid_hashing, weight_coherence_cache) with accurate descriptions.
- **README.md**: Added 2 missing discovery types to the Activation & Neuron State table: High Error Squash Exploration and Low-Impact Neuron.
- **DISCOVERY_TYPES.md**: Added High Error Squash Exploration (Issue #788) and Low-Impact Neuron Detection (Issue #793) — table of contents entries, summary table rows, and full detailed description sections with detection criteria. Updated last-updated date.

## Evidence

All documentation changes verified against actual source code modules and Cargo.toml benchmark definitions. `quality.sh` passes cleanly — no code changes were made.

## Test Plan

- No test changes required (documentation-only PR)
- Verified `quality.sh` passes cleanly including `cargo doc --no-deps` (all doc links resolve)
