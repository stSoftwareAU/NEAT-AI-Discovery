## Summary

The skip-connection detector (`src/analysis/detection/skip_connection.rs`) seeded depth 0 only from `input`-typed entries in `creature.neurons`. Production creatures list only non-input neurons and give the input count in `creature.input`, so no node ever got a depth, every hidden neuron was dropped from `deep_attenuated`, and the detector never emitted a candidate. The depth pass and the depth-0/1 source lists now include the inputs synthesised from `creature.input` (`input-0 .. input-{N-1}`, the naming used by `src/analysis/synapse/preparation.rs`), and still accept any listed `input` neuron. Closes #2523.

Version bump: `Cargo.toml` and the root `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Input nodes only ever appear as synapse endpoints in production, so the fix has to create them from the count rather than rely on `neurons`.
- The change is confined to the two places that decide "which nodes are inputs": the Kahn depth pass and the shallow-source split.

### Essential Design Decisions

- Synthesised inputs are interned alongside listed neurons and de-duplicated (a `HashSet` of seeded indices, and a listed-uuid check in the source split), so a creature that does list `input-0` is seeded and offered once and the O(V + E) bound and step counts are unchanged.
- Synthesised inputs come first in the depth-0 source list, ahead of `creature.neurons`, so `input-0` is the first source offered to each deep target.
- The fixtures in `src/analysis/skip_connection_deadline_tests.rs` named their inputs `in0`/`in1` while declaring `input: 1`/`2`, which is inconsistent with the FFI naming; they were renamed to `input-0`/`input-1`. Fixture names and expected pair only; no assertion was weakened, and the `legacy_*` reference functions are unchanged.

### Undiscoverable Facts

- None.

## Evidence

Backend change, no UI. The new test `tests/recommendation/issue_570_skip_connection_discovery.rs::test_deep_network_without_listed_input_neurons_emits_skip_from_input_0` builds the issue's creature (`input: 1`, no input entry in `neurons`, `input-0` only as a synapse endpoint). Against the unfixed code it failed: `Issue #2523: unlisted input neurons must still seed depth 0, so candidates must be emitted` (13 passed, 1 failed). After the fix it passes and asserts a candidate with source `input-0` at depth 0 to a deep target, plus an `AddSynapse` from `input-0` after `skip_connections_to_coordinated_candidates`.

## Test Plan

- Added `test_deep_network_without_listed_input_neurons_emits_skip_from_input_0` (red before the fix, green after, as above).
- Edited `src/analysis/skip_connection_deadline_tests.rs`: fixture input uuids `in0`/`in1` → `input-0`/`input-1` (and the expected pair `("hc","in1",3)` → `("hc","input-1",3)`). Without the rename the two tests `acyclic_result_matches_legacy_algorithm` and `deadline_at_per_target_check_keeps_completed_depth_pass_and_candidates_so_far` failed, because the fixture's declared `input` count now synthesised phantom `input-N` nodes. Removed assertions: none.
- Results on the final tree: `cargo test --test recommendation issue_570` → 14 passed; `cargo test --lib skip_connection` → 7 passed; `cargo fmt --all -- --check` clean; `cargo clippy --all-targets --all-features -- -D warnings` clean (executor run).
- Full gate: `timeout 595 ./quality.sh` passed `bash -n`, shellcheck, `cargo deny check` and the build stages, then reached `cargo test --lib` (376 tests ok, no failure) before hitting the 595 s cap (exit 124); the remaining test, doc and release stages did not run here.
- Docs: `CHANGELOG.md`, `docs/DISCOVERY_TYPES.md` (Skip Connection Detection), `docs/discoveries/skip-connection.md` note that inputs come from the `input` count and need not be listed.

Not done / risk: the rest of the full gate (remaining tests, `cargo doc`, release build) was not run; `fuzz/Cargo.lock` is untouched (out of scope, separate issue).

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 595 s foreground cap (exit 124) during cargo test --lib after a cold build in a fresh worktree; stages before it passed and no failure was seen; targeted tests, fmt and clippy were run separately; CI runs the full gate" -->
