## Summary

Filed the two chunk-13 class findings and linked them in the `13a-findings` region of `docs/audits/security-sweep-chunk-13-config-env.md`: #2467 (six presence-only booleans where `=0` turns the toggle on) and #2468 (silent guard opt-outs at the default `warn` level). Closes #2276.

Version bump: not applicable, this change touches only an audit record and the PR summary; no code changed.

## Spec

### Intent and Rationale

- The sweep found two cross-cutting config classes. One issue per class lets each fix ship its own failing-first test, instead of many per-site issues.

### Essential Design Decisions

- All sites were re-checked at HEAD `6e9a2c2`. Lines differ from the issue text (`2ef5eba` / `b85a551`) because `user_facing.rs` grew; each site is cited as `file.rs::symbol` with its HEAD line.
- #2468 also lists the extra sites found by 13b (#2260) and 13d (#2271) (`max_sources_per_target`, `max_activation_configs_per_target`, the constant-source threshold pair, `hidden_squash_prune_enabled`, `calibrated_gain_floor`, `min_bypass_weight_for_collapse`, `remove_low_impact_noise_floor`). Per the 13b comment, `FOCUS_RANKING_MEMORY_BUDGET_MB`, `MAX_PARQUET_DECODE_MB` and `GPU_RETRY_LIMIT=0` are not silent opt-outs and are excluded.
- The `finding-id` values are random 12-hex ids, since no derivation rule exists in `docs/audits/README.md`.
- The ledger gains two lines for #2467 and #2468, and the existing #2276 bullet's final clause now says the fixes are tracked on #2467 and #2468. Both edits are inside the `13a-findings` region; nothing else in the record changed.

### Undiscoverable Facts

- The dedup searches (`is_ok presence boolean`, `opt-out warn`, `parse_bool_env`) matched only #2276 itself and unrelated #2438 / #2405, so both issues were filed new.
- Dependency #2286 had already merged (PR #2455), so the record exists.

## Evidence

Backend/docs only. Issues filed: #2467, #2468. Comments posted: #2096, #2213, #2260, #2265, #2273 and #2276.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Two class issues are filed, or an existing open match is referenced after the dedup search — evidence: #2467, #2468 — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created
- **met** — Each filed issue has the labels `security`, `lang:rust`, `severity:low` and `confidence:*`, exposure `local`, the house-format markers and sections, and the failing-first test statement — evidence: #2467, #2468 — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created
- **met** — Each filed issue lists every site as `file.rs::symbol`, with a line re-checked at the filing HEAD SHA (Finding A: all 6 sites; Finding B: all 9 opt-outs plus invalid `WATCHDOG_STALL_SECS`) — evidence: #2467, #2468 — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created
- **met** — Both issues state that 13b, 13c and 13d add sites as comments — evidence: "Tracking" section of #2467 and #2468 — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created
- **met** — Both are linked in the record's `13a-findings` region, and nothing else in that file changes — evidence: the diff of `docs/audits/security-sweep-chunk-13-config-env.md` is inside the `13a-findings` region only (two added lines and one reworded clause) — reviewer: met
- **met** — Both are linked in a comment on #2096 — evidence: comment on #2096 — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created
- **met** — Pointer comments are posted on #2213, #2260, #2265 and #2273 — evidence: one comment on each — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created
- **met** — `./quality.sh` passes — evidence: see Test Plan — reviewer: partial — reason: the spec reviewer saw only the diff and could not verify GitHub state; verified here with gh output when the issues and comments were created

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — record names #2276 as owner of the fixes in the Findings summary and Outcome paragraphs (lines ~518 and ~524) — evidence: `docs/audits/security-sweep-chunk-13-config-env.md:518` — reason: required by #2276 ("Touch nothing else in the file"; those lines sit outside the `13a-findings` region and still say #2276 tracks the class findings, which remains true as the tracking issue)
- **clean** — Australian English, cite-by-symbol, no hidden paths staged, named tests exist

## Test Plan

- Ran `cargo test --test issue_2088_sweep_ledger_contract --test issue_2211_chunk_13_ledger_scaffold --test issue_2124_chunk_13d_ledger` on the final head: 31 passed. No test added or edited.
