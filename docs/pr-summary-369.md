## Summary

Slimmed down README.md from 2,102 lines to 426 lines (80% reduction) to make it
human-readable, as requested in issue #369. All essential user-facing information
is preserved — deep-dive content has been extracted into focused docs/ files with
links from the README.

### Changes

- **README.md**: Restructured to focus on what a human reader needs — project
  overview, quick start, FFI API summary, GPU requirements, discovery types,
  configuration (environment variables table), troubleshooting (quick-reference
  table), development instructions, and links to detailed documentation.

- **docs/ANALYSIS_DEEP_DIVE.md** (new): Extracted detailed analysis workflow,
  coordinated structural discovery algorithms (epistatic pairs, redundant path
  pruning, saturated/bottleneck/dead neuron detection, correlated error patterns,
  multi-hop candidates, oscillating neurons, dormant/opposing synapses, output
  bias drift, candidate clustering), discrete activation function handling, error
  distribution analysis, and tiered loading strategy.

- **docs/GPU_GUIDE.md** (new): Extracted GPU performance tuning (automatic
  adaptation, manual tuning, M4 Mac tuning, kernel profiling), detailed
  troubleshooting (EGL/DRI permissions, OOM errors, analysis timeouts, GPU
  timeouts), debugging deadlocks (automatic detection, SIGUSR1 thread dumps,
  hang watchdog, manual inspection), and parquet file memory checks.

- **docs/FFI_API.md** (new): Extracted full FFI API reference (exported symbols,
  GPU availability checking, JSON input/output formats, streaming recording API,
  critical requirements for atomic record writes and forward-only activation
  order, and file format/debugging information).

- **tests/issue_367_changelog_extraction.rs**: Updated two tests that previously
  checked for deep-dive sections in README.md to instead verify the content exists
  in docs/ANALYSIS_DEEP_DIVE.md and that README links to it. This is a documented
  business logic change required by issue #369's acceptance criteria to move deep
  dives to docs/ files.

### Acceptance Criteria

- [x] README.md is under 500 lines (426 lines)
- [x] All essential user-facing information is preserved
- [x] Deep dives moved to appropriate docs/ files
- [x] Links to CHANGELOG.md, AGENTS.md, and docs/ files are present
- [x] `./quality.sh` passes
- [x] Australian English spelling used throughout

## Evidence

Unable to generate screenshot: This is a documentation restructuring with no
visual interface.

## Test Plan

- Updated `tests/issue_367_changelog_extraction.rs` to verify deep-dive content
  now lives in `docs/ANALYSIS_DEEP_DIVE.md` instead of README.md
- All existing tests in `tests/issue_346_readme_project_goal.rs` continue to pass
  (README retains project mission, speed goal, candidate quality, GPU/SIMD mentions)
- `./quality.sh` passes cleanly (434 unit tests + all integration tests)
