## Summary

`.github/workflows/semgrep.yml` no longer sets `SEMGREP_APP_TOKEN` on the scan step. `semgrep ci` exits fatally when an app token is combined with an explicit `--config`, and the step runs `semgrep ci --config p/default`, so the check would fail on every PR once the secret was set. The step is now a plain scan. Docs that said the workflows hold that token are corrected. Version bump: `Cargo.toml` updated for this change; the merged commit records the final version. Closes #2558.

## Spec

### Intent and Rationale

- Removing the token (the issue's suggested fix) keeps the existing `--config p/default` and rule excludes untouched, so scan coverage is unchanged.

### Essential Design Decisions

- The workflow had no job-level, top-level or `workflow_call`/`environment:` use of the token, so the only removal is the step `env:` block.
- The regression test reads the YAML as text (no YAML parser in the tree) and ignores comments, so the explanatory comment in the workflow does not trip it.

### Undiscoverable Facts

- `fuzz/Cargo.lock` was already behind (0.74.359); `./quality/fuzz_lockfile.sh sync` brought it to the new version.
- The Develop merge retains `0.74.362` in `fuzz/Cargo.lock`, matching the merged `Cargo.toml`.

## Evidence

- Backend/CI-only change; no UI files touched.
- `tests/issue_2558_semgrep_no_app_token_with_config.rs` runs against the real workflow file.
- Related workflow tests still pass: `issue_1648`, `issue_1654`, `issue_1288`, `issue_1287`, `issue_2035`, `issue_1234`, `issue_1727`, `issue_1486`.

## Test Plan

- Added `tests/issue_2558_semgrep_no_app_token_with_config.rs` (7 tests): real file has no token outside comments and still has the `semgrep ci --config` invocation; fixtures for step-level, job-level, top-level and re-wrapped `run: |` token placement; full-line and trailing comments are not flagged, while a live token with a trailing comment is.
- Red-check: re-adding the `env:` block made `real_semgrep_yml_puts_no_app_token_in_scope` fail (offending line 77); restored afterwards.
- Edited `tests/issue_1486_codeowners.rs` doc comment and one assertion message text only (no assertion logic changed).
- Ran the new suite plus `issue_1648`, `issue_1486`, `issue_2035`, `issue_1234`, `issue_1288`, `issue_1287`, `issue_1654`, `issue_1727`: 51 passed.
- Twinned-copy search: no other workflow references the token (`grep -rn SEMGREP_APP_TOKEN`); remaining hits are archived PR summaries (historical) and the new test.
- Docs updated: `CONTRIBUTING.md` (Code Owners section), `.github/CODEOWNERS` comment, `CHANGELOG.md`.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh takes about 15 minutes; targeted tests only run here, CI runs the full gate" -->
