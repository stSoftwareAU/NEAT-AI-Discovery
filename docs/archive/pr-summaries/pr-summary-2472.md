## Summary

`get_constant_source_threshold` now delegates to `config::constant_source_threshold_with_dynamic`, so an explicit `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD=1e-7` (equal to the default) is honoured instead of falling through to the variance-scaled dynamic value. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version. Closes #2472.

## Spec

### Intent and Rationale

- The old arm `Some(v) if v != DEFAULT` could not tell "explicit 1e-7" from "unset/invalid", because the accessor returns `Some(DEFAULT)` for all three. The twin already parses the raw value, so one rule is shared rather than a second parser being added.

### Essential Design Decisions

- Production delegates to the twin (the issue's first suggested fix). Behaviour is unchanged for unset, empty, invalid, non-default and `0` values (pinned by the existing positive controls in the 2463 test file).
- The verbose debug log is kept but renamed "Resolved constant-source threshold", since the value may now be an explicit override.
- `docs/audits/security-sweep-chunk-13-config-env.md` is a historical sweep record that the ledger tests read, so it is left as written.

### Undiscoverable Facts

- None.

## Evidence

Backend-only change, no UI. Red check: with the old arm restored, `explicit_default_override_is_honoured_like_its_twin` fails with `get_constant_source_threshold returned Some(1e-6), expected Some(1e-7)`; with the fix it passes.

## Test Plan

- `tests/issue_2463_constant_source_threshold_explicit_default.rs`: removed `#[ignore]` from `explicit_default_override_is_honoured_like_its_twin`; added `explicit_default_override_with_whitespace_is_honoured`. Both go red with the old logic restored (verified).
- `tests/issue_2124_chunk_13d_ledger.rs::the_filed_finding_is_linked_and_has_its_probe`:
  - Removed from `tests/issue_2124_chunk_13d_ledger.rs`: the assertion that the probe contains `#[ignore = "finding: #2463`
    - Required by: "update the `#[ignore = \"finding: #2463` pin in `tests/issue_2124_chunk_13d_ledger.rs`"
    - Replaced by: `probe.contains("fn explicit_default_override_is_honoured_like_its_twin")` and `!probe.contains("#[ignore")`
- Passed on the final head: `cargo test --test issue_2463_constant_source_threshold_explicit_default` (6), `--test issue_2124_chunk_13d_ledger` (10), `--test issue_2096_config_doc_parity` (17), `--lib samples::` (19); `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.
- Quality gate: `./quality.sh` exceeded the 570s foreground cap (exit 124) and did not report; CI runs it.
  <!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the foreground time cap; targeted checks passed, CI runs the gate" -->
- Callers of `get_constant_source_threshold`: `src/analysis/synapse/preparation.rs:193` (inherits the fix) and `src/analysis/samples/mod.rs` unit tests (`--lib samples::` passed).
- Docs: `docs/CONFIGURATION.md` row updated with the override rules.
