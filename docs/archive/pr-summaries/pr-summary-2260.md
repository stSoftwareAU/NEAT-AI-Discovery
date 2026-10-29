## Summary

Chunk 13b-3: swept `src/config/user_facing.rs` from the top to the end of `focus_ranking_memory_margin_mb` and recorded the result in `docs/audits/security-sweep-chunk-13-config-env.md`. The 13b Files-swept row is no longer `pending`, the 13b settings table has a ten-column row for each of the 31 variables read in the range plus a row for the removed `OUTLIER_PERCENTILE`, and a new "Probe dispositions (13b)" section records probes (b)–(j). No new issue was filed: probes (c)–(j) found nothing new to file (negative-result). Closes #2260.

Version bump: not applicable (documentation only); the merged commit records the final version.

## Spec

### Intent and Rationale

- The sweep record had a scaffolded, empty 13b region. This fills it from the code at HEAD (`aa6b117`), with every consumer traced by symbol rather than assumed from the issue text.
- Edits outside the 13b markers are limited to the "Swept by" line, the "Sweep status" paragraph and the `## Outcome` paragraph, so the record stays truthful about what is swept (the 13c sub-issue did the same).

### Essential Design Decisions

- **Decode ceiling accepted.** `MAX_PARQUET_DECODE_MB=u64::MAX` is recorded as an accepted operator choice: same local principal as the process, a caller budget above 0 still wins in `DecodeBudget::resolve`, the ceiling is only compared with decoded bytes, and the design already decodes unbounded when total RAM is unknown (`default_decode_limit_bytes(0)`).
- **Tiered cache not reachable.** `RecordCache::new_tiered` has no caller in `src`, and `TieredRecordCache::new` is called only from two benches, so the issue's conditional doc-drift entry is not triggered; `docs/CONFIGURATION.md` is correct and the record says so.
- **Class sites go on #2276.** The class issue #2213 is a closed planning issue; the open issue that files the class findings is #2276, so the sites were added there as a comment ([comment](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2276#issuecomment-6469353019)). The comment also lists four newly found zero-disables sites (`MAX_SOURCES_PER_TARGET`, `MAX_ACTIVATION_CONFIGS_PER_TARGET`, `CONSTANT_SOURCE_EFFECT_THRESHOLD`, `HIDDEN_SQUASH_PRUNE`), described as low-weight caps and tuning knobs.
- **Helper-test wording.** For cached or un-parsed accessors that bypass `src/config/helpers.rs` (`verbose`, `gpu_timing`, `mh_temperature`, `gpu_retry_limit`), the "on invalid input" cell says so instead of repeating "covered by the helper unit tests", which would be false. `gpu_retry_limit` cites its own tests (`tests/issue_2364_gpu_retry_limit_warn_test.rs`, `tests/issue_2006_numeric_env_override_trim.rs`) and #2245.

### Undiscoverable Facts

- #2096 is closed, and no new finding exists to link, so no comment was posted on it.
- Doc drift found (not security findings) was reported on #2125 ([comment](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2125#issuecomment-6469372712)): the `FOCUS_RANKING_MEMORY_BUDGET_MB` docs say `info` where `plan_loading` emits `warn!`, the `user_facing.rs` module doc still mentions "outlier analysis", and the watchdog and `MH_TEMPERATURE` rows in `docs/CONFIGURATION.md` omit ranges.

## Evidence

Documentation-only change. Consumer traces were done by reading the code at HEAD (`DecodeBudget::resolve`, `plan_loading`, `plan_cache_preload`, `parquet_preload_fits_available`, `apply_source_budget`, `build_scan_configs`, `get_adjusted_batch_size`, `GpuAnalyzerFactory::create`, `StreamingRecordCache`). The ledger contract tests were run (see Test Plan).

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The `user_facing L1-664` Files-swept row is not `pending`. — evidence: `docs/audits/security-sweep-chunk-13-config-env.md` (13b Files-swept row reads `swept (Issue #2260)`) — reviewer: met
- **met** — Every variable read in the range has a `13b` settings row with all ten columns filled, including the accepted range and the on-invalid behaviour. — evidence: the 13b settings table in `docs/audits/security-sweep-chunk-13-config-env.md` (31 variables plus `OUTLIER_PERCENTILE`, each with 10 cells) — reviewer: met — reason: the reviewer's one low-severity note on the `MH_TEMPERATURE` cell was fixed in this diff
- **met** — Probe (g) has a written decode-ceiling disposition. — evidence: "Probe dispositions (13b)" (g) in `docs/audits/security-sweep-chunk-13-config-env.md` — reviewer: met
- **met** — The class findings are linked from these rows: the four presence-only booleans; `GPU_STALL_WINDOW_SECS=0`; `CPU_PRE_REJECT=0`; invalid `WATCHDOG_STALL_SECS`. — evidence: the `VERBOSE`, `GPU_TIMING`, `NEURON_TARGETS_OUTPUT_ONLY`, `QUIET_GPU`, `GPU_STALL_WINDOW_SECS`, `CPU_PRE_REJECT` and `WATCHDOG_STALL_SECS` rows cite "class #2213 / #2276" — reviewer: met
- **met** — The two watchdog rows link both #2259 findings. — evidence: the `WATCHDOG_STALL_SECS` row links #2405 and cross-links #2404; the `WATCHDOG_ABORT_DELAY_SECS` row links #2404 and cross-links #2405 — reviewer: partial — reason: the reviewer saw the `WATCHDOG_STALL_SECS` row without #2404; the cross-link was added in this diff
- **met** — Every surviving finding is filed and linked in the ledger and on #2096. — evidence: negative-result bullet in the 13b Issues-filed region; no surviving finding exists, and #2096 is closed — reviewer: met
- **met** — `./quality.sh` passes, including the #2224 scaffold test and `tests/issue_2088_sweep_ledger_contract.rs`. — evidence: `tests/issue_2211_chunk_13_ledger_scaffold.rs` and `tests/issue_2088_sweep_ledger_contract.rs` pass (15 tests) — reviewer: missing — reason: the reviewer was read-only and could not run the gate. Only the full test stage of `./quality.sh` did not complete within its 900 s cap (see Test Plan); CI runs it. Remaining: merge-verification-only
- **unrequested** — the doc-drift entries (and the #2125 comment) beyond the conditional `LruRecordCache` entry — reviewer: unrequested — reason: found while tracing consumers; they sit in the 13b Doc drift region the issue assigns and are reported to the owner of doc reconciliation (#2125)
- **unrequested** — "Swept by", "Sweep status" and `## Outcome` text updates — reviewer: unrequested — reason: keeps the record truthful once 13b is swept, as the 13c sweep did
- **unrequested** — the "Probe dispositions (13b)" heading — reviewer: unrequested — reason: the issue asks for dispositions but names no location; it mirrors the 13c layout

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — `CODING-STANDARDS.md` is absent in this repository, so the reviewer used `CONTRIBUTING.md` and `AGENTS.md`. Australian English, cite code by symbol (not line number), no Mermaid blocks added, every named test file exists, cited symbols exist in `src/`. No violations.

## Test Plan

- No test files were added or edited.
- Ran `cargo test --test issue_2211_chunk_13_ledger_scaffold --test issue_2088_sweep_ledger_contract` on the final ledger: 15 passed.
- Full `./quality.sh` gate: started on the final ledger content and passed the script-syntax, shellcheck, cargo-install-pinning, PR-summary-layout, `cargo deny`, build, format, clippy and type-check stages. It was killed by its 900 s `timeout` while `cargo test --lib --tests --all-features` was still running, so the full test stage did not complete here. <!-- vibe-quality-gate-skipped reason="full test stage exceeded the 900 s foreground cap; docs-only change, CI runs the same checks" -->
- Scaffold and ledger tests are the failure detection for lost markers or a broken settings header; they are unchanged.
- Twinned-rule search: none found (documentation record only).
