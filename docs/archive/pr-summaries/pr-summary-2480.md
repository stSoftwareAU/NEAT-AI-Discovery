## Summary

The chunk 8a audit record's capacity grep used `[^\]]*`, which GNU grep reads as `[^\]` followed by `]*`, so it missed every `vec![value; n]` with a multi-character value and its "Verify this record" figure could not be reproduced. The record now quotes the portable `[^]]*`, states the figure the corrected command prints (102 = 89 production capacity rows + 13 named test-only hits), and a new test pins both. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version. Closes #2480.

## Spec

### Intent and Rationale

- An auditor re-running the record's command must get the recorded number, or an uncited capacity site can hide in the gap. Fixing the bracket (rather than replacing the recipe with the Rust scanner) keeps the shell recipe usable and makes it agree with `is_capacity_site`.

### Essential Design Decisions

- The recorded figure is derived from the live tree, not from earlier checkpoints: earlier reconciliation paragraphs are reworded as checkpoints so they no longer claim to be live (84 rows / 87 hits were stale; the ledger holds 89 `capacity` rows).
- The new drift test computes the figure with `is_capacity_site` over every `.rs` file in scope (test modules included) rather than typing it in, so it follows the code.

### Undiscoverable Facts

- The interactive `grep` in the worker image is ugrep; the figures were taken with GNU grep 3.11 (`/usr/bin/grep`): `[^\]]*` prints 87, `[^]]*` prints 102.

## Evidence

Command run: `/usr/bin/grep -rnE 'with_capacity\(|vec!\[[^]]*;|\.reserve\(' src/analysis/detection/ src/analysis/neuron/ | wc -l` → `102`; the same with `[^\]]*` → `87`.

The 13 test-only hits (no ledger row): one each in `symmetry_breaking_deadline_tests.rs`, `redundant_path_deadline_tests.rs`, `compound_degradation_tests.rs`; five `vec![false; N]` in the inline test module of `correlated_error.rs`; two in `fanin_polarity_conflict.rs`; three in `input_sensitivity.rs`. Production sites: 89, equal to the 89 `capacity` rows.

## Test Plan

- Added `verify_this_record_capacity_command_is_portable_and_its_figure_reproducible` in `tests/issue_2154_chunk_08a_finalisation.rs`. Ran `cargo test --test issue_2154_chunk_08a_finalisation`: 9 passed. The six other ledger suites (`issue_2282`, `issue_2294`, `issue_2284`, `issue_2298`, `issue_2300`, `issue_2088`) passed: 45 tests.
- Red checks: with the doc reverted to base the new test fails at the `[^]]*` assertion; with only the figure changed to `87` it fails on `The recorded capacity figure is`; with only `13 test-only hits` changed to `12` it fails on that phrase.
- Only comments were edited in existing tests (the quoted regex); no assertion was removed.
- `cargo clippy --test issue_2154_chunk_08a_finalisation -- -D warnings`: no issues. `cargo fmt --check`: clean.
- Docs sweep: grepped `docs/` and `tests/` for `[^\]]`; remaining hits are the historical explanation of the bug in the record.

<!-- vibe-quality-gate-skipped reason="./quality.sh hit the 500s foreground cap (exit 124) while still compiling dev-dependencies; targeted checks above were run instead; CI runs the full gate" -->

Not done / risk: full `./quality.sh` not completed locally.
