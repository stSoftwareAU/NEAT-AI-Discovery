## Summary

Extract a shared `dispatch_analyses` function in `orchestration.rs` to eliminate
three near-identical synapse/neuron dispatch blocks. The new function takes a
`synapse_first` flag and runs the two analyses in the requested order, reducing
~50 lines of duplicated code to a single parameterised call site. Closes #774.

## Evidence

This is a pure refactoring with no behavioural change — backend/CLI only, no UI.
All existing tests pass, including the integration tests that exercise
`analyze_all` with and without deadlines:

- `tests/analyze_all_deadline_prioritises_synapses.rs`
- `tests/issue_419_parallel_discovery_execution.rs`
- Unit tests for `choose_deadline_order_synapse_first` in `src/analysis/mod_tests.rs`

`quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build).

## Test Plan

- Existing unit tests for `choose_deadline_order_synapse_first` verify ordering logic
- Existing integration tests for `analyze_all` verify end-to-end behaviour is preserved
- Full `quality.sh` run confirms no regressions
