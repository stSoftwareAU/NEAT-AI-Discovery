## Summary

Deduplicate 58 test function names that appeared across multiple integration test files, making CI output ambiguous. Added module-specific prefixes to 182 test functions across 45 files so every test name is unique and self-describing. Closes #815.

For example:
- `test_empty_records_no_candidates` in `issue_358_oscillating_neuron_detection.rs` became `test_oscillating_neuron_empty_records_no_candidates`
- `test_insufficient_samples_not_flagged` in `issue_341_dead_neuron_detection.rs` became `test_dead_neuron_insufficient_samples_not_flagged`

No test logic was changed — only function names were updated with module-specific prefixes.

## Evidence

- Zero duplicate test names remain across all integration test files
- All 1017 integration tests pass
- `quality.sh` passes cleanly (fmt, clippy, check, tests, docs, release build)

## Test Plan

- No new tests added — this is a pure rename of existing tests
- Verified no duplicate test function names remain: `grep -rn '#\[test\]' tests/ -A1 | grep 'fn test_' | sed 's/.*fn \(test_[a-zA-Z0-9_]*\).*/\1/' | sort | uniq -c | awk '$1 > 1'` returns empty
- All existing tests pass with their new names
