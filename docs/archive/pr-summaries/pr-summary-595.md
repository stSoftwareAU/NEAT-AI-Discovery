## Summary

Split `src/analysis/module_dispatch_specs.rs` (~49KB, 1181 lines) into focused sub-modules grouped by concern. Closes #595.

The monolithic file is now a directory module with four sub-modules:

- `module_dispatch_specs/mod.rs` — public API, `build_discovery_module_specs()`, and orchestration functions (deduplication, clustering, ensemble scoring)
- `module_dispatch_specs/neuron_specs.rs` — neuron-focused dispatch specs (14 modules: saturation, bottleneck, dead, oscillating, restricted range, operating point, unbounded capping, noisy neurons, activation recommendation/mismatch, bias perturbation, symmetry breaking, co-adaptation, squash weight rescale)
- `module_dispatch_specs/synapse_specs.rs` — synapse-focused dispatch specs (7 modules: dormant, opposing, weight coherence ×3, noisy synapses, weight magnitude reset)
- `module_dispatch_specs/structural_specs.rs` — structural discovery specs (5 modules: correlated error, multi-hop, topology, topology diversification, skip-connection)
- `module_dispatch_specs/scoring_specs.rs` — scoring and recommendation specs (10 modules: output bias drift, bounded range, sentinel gating, observation utilisation, input sensitivity ×2, sample-weighted, gradient, output squash mismatch, error plateau)

The public API is completely unchanged — all existing callers (`orchestration.rs`) continue to work without modification.

## Evidence

This is a pure refactoring with no behaviour changes. All existing tests pass without modification:
- `cargo test --lib --tests --all-features -- --test-threads=1` — all tests pass
- `cargo clippy` — clean
- `cargo fmt` — clean
- `./quality.sh` — all checks passed

## Test Plan

- No new tests required — this is a structural refactoring only
- All existing tests continue to pass unmodified, confirming backward compatibility
