## Summary

Archive PR summaries and audit all documentation for accuracy. Closes #998.

### What changed

- **Archived 48 PR summaries** (PRs #884–#994) from `docs/` into `docs/archive/pr-summaries/`
- **AGENTS.md**: Updated source layout to reflect recent refactors:
  - Added `config/` module (5 files, split from config.rs per Issue #981)
  - Added `detection/compound_degradation.rs` (Issue #929) and `detection/cross_detection_synthesis.rs` (Issue #963)
  - Added `recommendation/batch_successful/` (3 files, Issue #965) and `recommendation/fan_in.rs` (Issue #908)
  - Added `analysis/scale_outcomes.rs` (Issue #964)
  - Added `candidate_compression/` directory (5 files, split per Issue #939)
  - Expanded `synapse/` module with 9 new files (orchestration, preparation, evaluation, holdout validation, metadata, results, tests)
  - Added `synapse/scoring/test_helpers.rs` and `synapse/scoring/tests.rs`
  - Updated shader listing to include `activation_reduce.wgsl` and `relu_reduce.wgsl`
  - Updated test count (~242 → ~277 files) and benchmark count (22 → 28 suites)
- **BENCHMARKS.md**: Updated from 22 to 28 suites, adding 6 missing benchmarks (analysis_pipeline_clones, candidate_pipeline_clones, impact_cache_contention, impact_uuid_cloning, queue_submission_copies, topology_traversal)
- **DISCOVERY_TYPES.md**: Added 4 new discovery types with table of contents entries, summary table rows, and detailed description sections:
  - Compound Degradation Detection (Issue #929)
  - Fan-in Candidates (Issue #908)
  - Cross-Detection Synthesis (Issue #963)
  - Batch-Successful Grouping (Issue #965)
  - Updated last-updated date to 5 Apr 2026
- **README.md**: Added 4 new discovery types to the discovery tables (Compound Degradation, Fan-in Candidates, Cross-Detection Synthesis, Batch-Successful Grouping)

## Evidence

All documentation changes verified against actual source code modules and Cargo.toml benchmark definitions. `quality.sh` passes cleanly — no code changes were made.

## Test Plan

- No test changes required (documentation-only PR)
- Verified `quality.sh` passes cleanly including `cargo doc --no-deps` (all doc links resolve)
