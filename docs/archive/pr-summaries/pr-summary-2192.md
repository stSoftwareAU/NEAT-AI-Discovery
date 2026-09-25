# PR Summary — Issue #2192: delete the dead epistatic interference-detection lever

## Summary

Closes #2192

`detect_interfering_pairs`, `check_saturation_risk`, `InterferenceType` and
`InterferencePairResult` (Issue #415) had no production caller. Following the
AGENTS.md **Dead Levers** rule, this deletes them, their `pub use` re-export and
`SATURATION_RISK_THRESHOLD` (used only by `check_saturation_risk`).

The live half of the Issue #415 work is unchanged:
`filter_interfering_epistatic_pairs` and
`filter_interfering_synergistic_candidates`, both called from
`candidate_selection::detect_epistatic_and_synergistic`.

- [x] Remove the dead functions, types, constant and re-export (`epistatic/scoring.rs`, `epistatic/mod.rs`)
- [x] Remove the unit tests of the deleted function
- [x] Bump the version to `0.74.257`
- [x] `./quality.sh` green

### Test file: trimmed rather than deleted

The issue suggested deleting `tests/neuron/issue_415_combo_successful_interference.rs`.
Four of its six tests called only `detect_interfering_pairs`, and those are
deleted. The other two (`combo_successful_filters_interfering_pairs` and
`combo_successful_allows_compatible_pairs`) are the **only end-to-end tests of
the live filter path**: redundant inputs must not be paired, and complementary
inputs must be. Deleting them would remove coverage of production code, so they
are kept. The file is renamed to `issue_415_combo_interference_filter.rs` to
match what it now tests.

The two `scoring.rs` unit tests of `detect_interfering_pairs` are deleted too.
The correlation, filter and pre-screen unit tests are kept.

### Security ledger

On `Develop`, the epistatic rows in
`docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md` are
still `pending`. The rows that cite #2192 exist only on
`milestone/2083-security-scan-overflow-8-chunks-not-reached`, and they already
describe this deletion, so this PR leaves the ledger untouched.

## Evidence

No references remain outside archived PR summaries (the only hit is the test
file's note explaining the deletion):

```text
$ grep -rn "detect_interfering_pairs\|check_saturation_risk\|InterferenceType\|InterferencePairResult\|SATURATION_RISK_THRESHOLD" src tests benches fuzz docs | grep -v pr-summaries
tests/neuron/issue_415_combo_interference_filter.rs:10://! The unit tests of the never-called `detect_interfering_pairs` that used to
```

```text
$ cargo clippy --all-targets -- -D warnings
No issues found
$ cargo test --lib epistatic
13 passed, 1578 filtered out
$ cargo test --test neuron issue_415
2 passed, 39 filtered out
$ ./quality.sh
✅ All quality checks passed!
```

## Test Plan

- `cargo test --lib epistatic`: the correlation, `filter_interfering_*` and pre-screen unit tests still pass.
- `cargo test --test neuron issue_415`: the e2e tests of the live interference filter still pass.
- `./quality.sh`: the full gate passes (fmt, clippy, build, tests).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
