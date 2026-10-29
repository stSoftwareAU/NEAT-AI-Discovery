## Summary

Chunk 13a-2a: adds `tests/issue_2272_config_small_modules.rs`, 31 `#[serial]` tests that pin what the un-cached accessors in `src/config/detection.rs` and `src/config/observability.rs` return today for hostile environment values. Probe (a) was traced to its three consumer gates and filed as finding #2438. Closes #2272.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

### Probe (a) outcome — finding #2438

[#2438](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2438) — "Detection-threshold env overrides accept NaN, inf, zero and negative values and flip three detectors to fire-on-everything or never (`src/config/detection.rs`)". Labels: `security`, `lang:rust`, `severity:low`, `confidence:high`. Failing-first test named in the finding: `tests/issue_<n>_detection_threshold_hostile_values.rs`. The finding states that its fix must update the threshold pins in this PR's test file.

No consumer guards the value. The three gates, as at this branch's head:

- `src/analysis/detection/noise_signal.rs:164` — `noise_to_signal_ratio <= threshold` skips; the factor at `:171` divides by the threshold.
- `src/analysis/detection/input_sensitivity.rs:277` — `sensitivity_score <= config.dominance_threshold` skips; the factor at `:291-293` divides by the threshold.
- `src/analysis/detection/input_sensitivity.rs:449` — `effective_gradient < config.gradient_threshold` skips; the factor at `:484-486` divides by the threshold.

A NaN fails every comparison, so no gate skips and the detector fires on every candidate with a NaN factor. A negative threshold never skips and yields a negative factor. `inf` skips everything and silently disables the detector.

## Spec

### Intent and Rationale

- Pin today's hostile-value behaviour so that the fix for #2438 must change the pins deliberately, rather than drifting silently.
- Tests only; the accessors are not changed (fixing them belongs to #2438).

### Essential Design Decisions

- `calibration_miss_threshold` is pinned against the literal `10.0` because `DEFAULT_CALIBRATION_MISS_THRESHOLD` is `pub(crate)` and unreachable from an integration test.
- Positive-control tests (a legal value is read) sit beside each `None`/default group so a mis-named variable cannot make a pin pass vacuously.
- The module doc cites the gates by symbol (CONTRIBUTING.md forbids line numbers in code); the `file:line` citations the issue requires live in this summary only.

### Undiscoverable Facts

- `timing`, `profile_mode` and `gpu_metrics` are cached in a `OnceLock`, so they cannot be exercised in-process and are not tested here.

## Evidence

Backend-only change; no UI. Evidence is test output below.

- `cargo test --test issue_2272_config_small_modules`: 31 passed.
- `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- #2438: "Detection-threshold env overrides accept NaN, inf, zero and negative values…" (looked up with `gh issue view 2438`).

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The new test file has one test for each accessor/value pair listed above, and every test passes against today's code. — evidence: `tests/issue_2272_config_small_modules.rs` (31 tests, all passed) — reviewer: met
- **met** — Probe (a) has a written outcome in the PR body: either exactly one #2078-shape finding, with all four labels and a failing-first test path; or the guarding `file:line`. — evidence: the Probe (a) outcome section above and #2438 — reviewer: partial — reason: the reviewer saw only the diff, where the outcome is not yet present; it is in this summary, and #2438 carries the four labels and the failing-first test path
- **met** — That outcome cites all three consumer gates by `file:line`. — evidence: the Probe (a) outcome section above — reviewer: partial — reason: the reviewer saw only the diff; the three gates are cited by `file:line` in this summary, checked against the head source
- **partial** — `./quality.sh` passes. — evidence: targeted checks only — reviewer: missing — reason: CI not yet verified — remaining: merge-verification-only; the full gate exceeded the 600s foreground cap on a cold build in three attempts here, so it was not completed locally

<!-- vibe-quality-gate-skipped reason="full ./quality.sh did not complete within the 600s foreground cap (cold build plus lib tests); targeted test, clippy and fmt checks pass; CI runs the full gate" -->

- **unrequested** — Positive-control tests and extra module-doc paragraphs — reviewer: unrequested — reason: harmless guards so the `None`/default pins cannot pass vacuously
- **unrequested** — Version bump 0.74.291 → 0.74.292 in `Cargo.toml` and `Cargo.lock` — reviewer: unrequested — reason: AGENTS.md requires a bump on any code change

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — PR summary file missing from the diff — evidence: `docs/archive/pr-summaries/pr-summary-2272.md` — reason: fixed in this diff
- **clean** — Cite code by symbol, Australian English, version bump, `#[serial]` on env-mutating tests, positive controls, no named-but-absent tests, no changed existing expectations, no dead levers; this repository has no `CODING-STANDARDS.md`, so `CONTRIBUTING.md` and `AGENTS.md` were used

## Test Plan

- Added `tests/issue_2272_config_small_modules.rs` (31 tests): three float thresholds × NaN/inf/-1; module tiering 0 and garbage; three cooldown/batch readers × 0/-1/garbage plus positive controls; `calibration_miss_threshold` NaN/1.0/0.5 plus positive control; `strict_candidate_reconciliation` 0/false/garbage plus positive control.
- Result on this head: `cargo test --test issue_2272_config_small_modules` — 31 passed.
- Red check for probe (a) was not run: no fix for #2438 exists yet, so the pins have not been shown to go red against a fix.
- No existing test is edited; no assertions removed.
- Twinned rules: none found.

Not done: the full `./quality.sh` gate was not completed locally (see the skip note above); the optional rewording of a module-doc sentence flagged by the Standards reviewer was not applied.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
