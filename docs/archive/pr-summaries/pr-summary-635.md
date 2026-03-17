## Summary

Add discovery scenario documentation for 21 previously undocumented detection modules, bringing the `docs/discoveries/` directory to full coverage of all 35 active detection modules wired into the dispatch pipeline. Closes #635.

### New Scenario Docs (21 files)

**Neuron-focused (8):**
- `activation-mismatch.md` — RELU negative bias and bounded underutilisation
- `bias-perturbation.md` — Large bias shifts to escape saturated regimes
- `co-adaptation.md` — Correlated neuron pairs wasting capacity
- `operating-point.md` — Pre-activation misalignment with active zone
- `restricted-range.md` — Neurons confined to narrow output band
- `squash-weight-rescale.md` — Coordinated activation change + weight rescaling
- `symmetry-breaking.md` — Neurons with near-identical weight configurations
- `unbounded-capping.md` — Unbounded activations producing extreme values

**Synapse-focused (2):**
- `weight-coherence.md` — Incoherent ratios, constant paths, symmetric cancellation
- `weight-magnitude-reset.md` — Exploratory weight changes to escape plateaus

**Scoring/recommendation (6):**
- `bounded-range.md` — Sentinel boundary detection with gating neurons
- `input-sensitivity.md` — Dominant inputs and threshold cliff effects
- `error-plateau.md` — Output neurons stuck at uniformly high error
- `observation-utilisation.md` — Sentinel-dominated input bias compensation
- `output-squash-mismatch.md` — Output activation/target-data incompatibility
- `sample-weighted.md` — Importance-weighted analysis for hard samples

**Structural (2):**
- `skip-connection.md` — Residual connections for gradient-attenuated deep neurons
- `topology-diversification.md` — Adding hidden neurons to flat linear paths

**Already in DISCOVERY_TYPES.md (3):**
- `activation-recommendation.md` — Proactive squash matching to input distribution
- `noise-signal.md` — Noise-to-signal ratio detection for neurons and synapses
- `gradient-discovery.md` — Gradient-based synapse weight adjustment

### Updated Files
- `docs/discoveries/README.md` — Expanded index with 6 categorised sections covering all 35 scenarios, updated pipeline diagram

## Evidence

This is a documentation-only change. No code was modified. `./quality.sh` passes cleanly.

## Test Plan

- `./quality.sh` passes (no code changes, only documentation added)
- All 21 new files follow the existing format (plain-English explanation, ASCII diagrams, detection criteria, candidate operations, example, references)
- All files use Australian English spelling conventions
- All links to source modules and cross-references between related docs are correct
- `docs/discoveries/README.md` index lists all 35 scenario docs in categorised sections
