## Summary

Triaged all open issues in the repository per issue #340. Closed 9 issues that were
obsolete or risky with minimal performance gains, updated labels, and created 4 new
discovery category issues.

### Issues Closed (9)

| Issue | Title | Reason |
|-------|-------|--------|
| #225 | Speculative execution for GPU pipeline | Risky GPU architecture change, uncertain gains |
| #223 | SIMD-accelerated CPU fallback for sample building | Library is GPU-only by design; marginal gains |
| #220 | Async GPU command submission with CPU overlap | Risky GPU architecture change, performance reasonable |
| #209 | Dynamic GPU memory allocation based on VRAM | Risky change, current 256MB limit works well |
| #207 | Optimize LazyRecordProvider LRU cache | Superseded by tiered/streaming loader (#215/#335) |
| #205 | Pre-warm Rayon thread pool | Self-acknowledged <1% improvement for typical workloads |
| #203 | Reduce minimum GPU timeout for tight deadlines | Edge case only, risky to reduce safety timeouts |
| #197 | Chunked Parquet reading for large files | Obsolete - already implemented via #215/#335 |
| #191 | Pre-compile specialised GPU pipelines | Risky shader changes, high maintenance burden |

### Labels Updated

- #213 (benchmark suite): Added `enhancement` label
- #212 (deadline checking): Added `enhancement` label
- #168 (Plateau Escape Challenge): Kept `question` label (requirements unclear)
- #166 (Interference Cancellation Task): Kept `question` label (requirements unclear)

### New Discovery Category Issues Created (4)

| Issue | Title | Description |
|-------|-------|-------------|
| #341 | Dead neuron detection for removal candidates | Identify neurons with near-zero activation across all samples |
| #342 | Saturated neuron detection for activation function change | Detect neurons stuck at activation ceiling/floor |
| #343 | Bottleneck neuron detection limiting information flow | Find single-neuron convergence points limiting capacity |
| #344 | Correlated error pattern detection for shared-cause identification | Group outputs with correlated errors to find missing features |

### Issues Kept Open (6)

| Issue | Title | Reason |
|-------|-------|--------|
| #230 | Multi-hop candidate analysis | Valuable new discovery category |
| #226 | Adaptive improvement threshold | Clear requirements, useful feature |
| #224 | Candidate clustering | Reduces redundant ablation tests |
| #213 | End-to-end benchmark suite | Infrastructure improvement |
| #212 | Fine-grained deadline checking | Reliability improvement |
| #200 | Cache hit rate monitoring | Observability improvement |
| #198 | Pre-compute source variance | Clear, focused optimisation |

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface.
Issue triage actions were performed via GitHub CLI (`gh`) commands.

## Test Plan

- Added `tests/issue_340_discovery_categories.rs` with 5 tests validating the discovery
  category contract:
  - `candidate_synapse_json_serialisation_contract` - verifies helpful/harmful synapse candidate format
  - `candidate_neuron_json_serialisation_contract` - verifies add-neuron candidate format
  - `synapse_weight_update_serialisation_contract` - verifies weight update candidate format
  - `coordinated_structural_operations_contract` - verifies all 7 coordinated operation types
  - `coordinated_structural_candidate_optional_fields` - verifies optional field omission
