# PR Summary — Issue #2364: warn on invalid or zero `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT`

Closes #2364

- [x] Extract pure `resolve_gpu_retry_limit(raw: Option<&str>) -> u32` in
  `src/config/user_facing.rs`
- [x] Warn (naming the variable) on unparsable or above-10 values, stating the
  default used
- [x] Warn on `0`, stating that device-lost recovery is disabled
- [x] `gpu_retry_limit()` accessor delegates to the new resolver with no
  behaviour change
- [x] Regression test `tests/issue_2364_gpu_retry_limit_warn_test.rs`
- [x] Structured `value` field on the unparsable warn is Debug-formatted
  (`?s`), so control characters cannot forge log lines
- [x] Docs: `docs/CONFIGURATION.md` row updated

## Spec

### Intent and Rationale

`NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` was parsed silently: `""`, `"abc"` and
`"999999"` fell back to the default of 3 with no log, and `"0"` silently
disabled the #647 device-lost recovery guard (CWE-778,
SEC-c01db5943e3c). An operator tuning this lever during an incident had no
signal that their value was ignored, or that `0` turned recovery off.

### Essential Design Decisions

- **A pure resolver taking `Option<&str>`.** `resolve_gpu_retry_limit` does no
  environment access, so tests exercise every branch without fighting the
  process-wide `OnceLock` behind the accessor.
- **`MAX_GPU_RETRY_LIMIT: u32 = 10` constant** replaces the inline `10` that
  used to live in the accessor's `.filter(|&n| n <= 10)`.
- **No behaviour change beyond the logging.** The returned values are
  identical to the base
  (`parse_env::<u32>(..).filter(|&n| n <= 10).unwrap_or(DEFAULT)`).
- **`0` is still honoured.** It is a legitimate operator choice, so the
  resolver still returns `0` — it just now announces the consequence.

### Undiscoverable Facts

- `parse_env` trims whitespace and treats an empty string as unparsable.
- The accessor is memoised, so only one in-process test may touch it:
  `accessor_routes_env_through_resolver`, marked `#[serial]`, uses the value
  `"abc"`.
- `tests/issue_2006_numeric_env_override_trim.rs:15-18` documents the same
  `OnceLock` limitation for this accessor.

## Evidence

- Security regression test paragraph, used here verbatim as required by the
  security-fix gate: "The regression test
  `tests/issue_2364_gpu_retry_limit_warn_test.rs::zero_warns_that_recovery_is_disabled`
  (with its siblings `empty_value_warns_and_uses_default`,
  `unparsable_value_warns_and_uses_default`, `above_maximum_warns_and_uses_default`
  and `accessor_routes_env_through_resolver`) reproduces the original trigger:
  it fails against the unfixed code and passes after the fix. The original
  trigger is closed with no trivial bypass — every value of the variable
  reaches the resolver through the single accessor, and each non-honoured or
  recovery-disabling value now warns."
- Baseline red: with every warn removed from the resolver, 5 tests FAILED
  (`empty_value_warns_and_uses_default`, `above_maximum_warns_and_uses_default`,
  `unparsable_value_warns_and_uses_default`, `accessor_routes_env_through_resolver`,
  `zero_warns_that_recovery_is_disabled`); `unset_does_not_warn` and
  `valid_values_do_not_warn` passed. With the fix: 8 passed (the eighth,
  `unparsable_value_field_escapes_control_characters`, was added afterwards
  for log-injection escaping; flipping `?s` back to `%s` turns it red).
- Quality gate: `./quality.sh < /dev/null` failed (exit 101) in this run on the branch
  head, on its first build step and not on this change. The first error was
  `error: rustc 1.98.0 is not supported by the following package:
  neat_ai_discovery@0.74.278 requires rustc 1.99`. This host has rustc 1.98.0
  and no `rustup`, so the toolchain pinned in `rust-toolchain.toml` (1.99.0) is
  unavailable. The steps before the build passed: bash syntax, shellcheck,
  install pinning, PR-summary layout and `cargo deny check`
  (`advisories ok, bans ok, licenses ok, sources ok`). With
  `cargo test --ignore-rust-version --test issue_2364_gpu_retry_limit_warn_test`
  on 1.98.0, all 8 tests passed. The remaining gate steps were run by hand on
  1.98.0: `cargo fmt --all -- --check` (clean),
  `cargo clippy --ignore-rust-version --all-targets --all-features -- -D warnings`
  (clean), `cargo test --ignore-rust-version --lib` (1634 passed, 0 failed),
  `--test issue_1684_doc_dedup` and `--test issue_2006_numeric_env_override_trim`
  (14 passed each). CI builds on the pinned 1.99.0.

**Docs sweep** — grep: `GPU_RETRY_LIMIT`, `gpu_retry_limit`, `get_gpu_retry_limit`, "Accepted range `0–10`"; section: `docs/CONFIGURATION.md#gpu`; updated: `docs/CONFIGURATION.md`, `src/analysis/gpu/queue/recovery.rs`, `src/config/user_facing.rs`

- Docs sweep detail: I read the `## GPU` table in `docs/CONFIGURATION.md`
  through. Its `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` row (line 42) now states the
  range, the warn on invalid or above-10 values and the warn on `0`. No other
  row in that section mentions the retry limit. Also updated:
  `src/analysis/gpu/queue/recovery.rs:88-91` and the doc comment on
  `gpu_retry_limit()` in `src/config/user_facing.rs`. Each remaining hit is
  listed as "file:line — still true because …":
  - `CHANGELOG.md:151` — historical #2006 entry listing `gpu_retry_limit` among
    the accessors that lost the trim before that fix.
  - `src/analysis/gpu/mod.rs:84-85` — re-exports only.
  - `src/analysis/gpu/queue/execution.rs:10`, `:283` — env var configures the
    limit, default 3; the resolver change does not alter the wording.
  - `src/analysis/gpu/queue/execution.rs:30`, `:311` — call `get_gpu_retry_limit`.
  - `src/analysis/gpu/queue/recovery.rs:10`, `:17` — `DEFAULT_GPU_RETRY_LIMIT`
    and `GPU_RETRY_LIMIT_ENV` constants are unchanged.
  - `src/analysis/gpu/queue/recovery.rs:46-48` — "Delegates to
    `crate::config::gpu_retry_limit()`" still true.
  - `src/analysis/gpu/queue/recovery.rs:95` — asserts the env var name.
  - `src/analysis/gpu/queue/stale_skip_tests.rs:236`, `:283` — default and
    accessor call.
  - `src/analysis/gpu/queue/staleness.rs:7` — retried that many times.
  - `src/config/mod.rs:15` — table row "(0–10)" still accurate;
    `docs/CONFIGURATION.md` is the single source for the warn behaviour detail.
  - `tests/gpu/issue_647_gpu_device_lost_recovery.rs:9-10`, `:91`, `:96` —
    constant asserts.
  - `tests/issue_1684_doc_dedup.rs:89` — env var name list.
  - `tests/issue_2006_numeric_env_override_trim.rs:15-18` — the `OnceLock`
    accessor cannot be exercised for the trim test; still true — the new test
    reaches it once under `#[serial]` and covers the logic through the pure
    resolver instead.
  - `docs/archive/pr-summaries/pr-summary-647.md:7`,
    `docs/archive/pr-summaries/pr-summary-1684.md:28`,
    `docs/archive/pr-summaries/pr-summary-1469.md:29`,
    `docs/archive/pr-summaries/pr-summary-2006.md:41`,
    `docs/archive/pr-summaries/pr-summary-2006.md:136`,
    `docs/archive/pr-summaries/pr-summary-717.md:15` — archive records of
    earlier PRs; still true historically and not rewritten.

## Test Plan

- `cargo test --ignore-rust-version --test issue_2364_gpu_retry_limit_warn_test`
  on rustc 1.98.0: 8 passed. `--ignore-rust-version` was needed because this
  host lacks the pinned 1.99.0 toolchain (see Quality gate under Evidence).

**Branch outcomes:**
- `src/config/user_facing.rs:65` — absent (variable unset) → default 3, no log — `tests/issue_2364_gpu_retry_limit_warn_test.rs::unset_does_not_warn` — flipped to `return 0`, test went red
- `src/config/user_facing.rs:70` — `Ok(0)` → WARN "device-lost recovery is disabled", returns 0 — `tests/issue_2364_gpu_retry_limit_warn_test.rs::zero_warns_that_recovery_is_disabled` — arm disabled (`Ok(0) if false`, so 0 is returned silently), test went red
- `src/config/user_facing.rs:78` — error (above the maximum of 10) → WARN, returns default 3 — `tests/issue_2364_gpu_retry_limit_warn_test.rs::above_maximum_warns_and_uses_default` — arm disabled (`if false && …`), test went red
- `src/config/user_facing.rs:89` — success (`1..=10`, trimmed) → value returned, no log — `tests/issue_2364_gpu_retry_limit_warn_test.rs::valid_values_do_not_warn` — flipped to return the default, test went red
- `src/config/user_facing.rs:90` — error (unparsable, including empty) → WARN, returns default 3 — `tests/issue_2364_gpu_retry_limit_warn_test.rs::empty_value_warns_and_uses_default`, `tests/issue_2364_gpu_retry_limit_warn_test.rs::unparsable_value_warns_and_uses_default`, `tests/issue_2364_gpu_retry_limit_warn_test.rs::accessor_routes_env_through_resolver` — warn removed, all three tests went red; `tests/issue_2364_gpu_retry_limit_warn_test.rs::unparsable_value_field_escapes_control_characters` — `value = ?s` flipped to `value = %s`, test went red (7 passed, 1 failed: "value field must not contain a raw newline")

I ran every flip in this run against the branch head with
`--ignore-rust-version` on rustc 1.98.0, restoring the file after each one. `src/analysis/gpu/queue/recovery.rs`
changes a comment only, so it adds no branch.

- Entry points checked: `gpu_retry_limit()` accessor →
  `accessor_routes_env_through_resolver`; reverting to the old inline parse
  makes it go red, as the baseline run showed.
- Security self-check: no new external input surface beyond the existing env
  read. The operator's raw value is Debug-formatted in both the message
  (`{s:?}`) and the structured `value` field (`?s`), so control characters
  are escaped and cannot forge log lines. The above-maximum arm logs the
  parsed `u32`, not the raw string. No secrets are logged.
- Deno regression avoided: N/A (Rust repository).
