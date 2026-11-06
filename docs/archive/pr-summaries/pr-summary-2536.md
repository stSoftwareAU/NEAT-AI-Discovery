## Summary

The Observation Utilisation reference and module docs said the module emits `addNeuron` / `addSynapse` gating-neuron candidates. It emits one `setBias` candidate per downstream target (`observation_utilisation_to_coordinated_candidates_observed`, `src/analysis/detection/observation_utilisation.rs`). The docs now match the code, and a drift test pins them to the module's real output. Documentation and comment change only; no behaviour changes, so no `Cargo.toml` version bump (CI's `version-increment` job bumps on the PR). Closes #2536.

## Spec

### Intent and Rationale

- `docs/DISCOVERY_TYPES.md` is the reference `AGENTS.md` points at, so a wrong operation list there misleads anyone auditing NEAT-AI's candidate handlers. The fix is to the prose, not the code: `docs/discoveries/observation-utilisation.md` already describes `setBias` correctly.

### Essential Design Decisions

- The drift test builds the expected operation list from the serialised output of the real conversion function (`type` tag of each operation) rather than typing `setBias` in, so a change to the emitted operations fails the test until the reference follows.
- `docs/ANALYSIS_DEEP_DIVE.md` step 3 ("propose gating neurons") had the same drift for the same module, so it is fixed and pinned too.

### Undiscoverable Facts

- None.

## Evidence

Files changed:

- `docs/DISCOVERY_TYPES.md`: summary row now lists `setBias` only; the section's purpose, recommended action and output line describe one `setBias` per downstream target (target's absolute bias = current bias minus effective centre x weight). The Detection criteria list is unchanged. The Sentinel Gating row keeps `addNeuron` / `addSynapse`, which that module does emit.
- `docs/ANALYSIS_DEEP_DIVE.md`: Observation Utilisation algorithm step 3.
- `src/analysis/detection/observation_utilisation.rs`: `//!` header, two field/function doc comments and one inline comment only; no code changed.
- `tests/issue_1938_discovery_doc_contract.rs`: new test `observation_utilisation_documents_the_operations_it_emits`.

Docs sweep: grepped `README.md`, `docs/*.md` and `docs/discoveries/` for the module name and for `gating` / `addNeuron` near it. `README.md:276` and `docs/discoveries/README.md:175` already say "observation utilisation" / "Bias-compensate downstream neurons", which is still true. Remaining `gating` mentions in `docs/DISCOVERY_TYPES.md` belong to Sentinel Gating and Bounded Range.

## Test Plan

- `cargo test --test issue_1938_discovery_doc_contract`: 8 passed.
- `cargo test --lib observation_utilisation`: 7 passed.
- `cargo fmt --check`: clean. `cargo clippy --test issue_1938_discovery_doc_contract -- -D warnings`: clean. `cargo check --lib`: clean.
- New test goes red without its change (docs restored to the base branch, test kept): the summary-row assertion fails (`must list the emitted operation \`setBias\``). Per pin: with only `docs/ANALYSIS_DEEP_DIVE.md` reverted the deep-dive assertion fails; with the row fixed but the `DISCOVERY_TYPES.md` section reverted the detail-section assertion fails.
- The test also asserts the emitted absolute biases (hidden-0: 0.25 - 0.5 x 0.8 = -0.15; output-0: 0.0 + 0.2) with different current biases per target, so a module that emitted the compensation alone would fail it. I did not break the module on purpose to confirm that assertion goes red.
- No existing test was edited; only two `use` lines in `tests/issue_1938_discovery_doc_contract.rs` were merged and extended.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeds the 590 s foreground cap on this host (as in #2496/#2497/#2508/#2509/#2515/#2533); targeted tests, fmt and clippy were run; CI runs the full gate" -->

Not checked: the full `./quality.sh` locally; the `emitted bias` assertions were not red-checked against a broken module.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
