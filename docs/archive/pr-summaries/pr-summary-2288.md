## Summary

Adds the chunk-9 (GPU dispatch + WGSL shaders) sweep record
`docs/audits/security-sweep-chunk-9-gpu-wgsl.md` so the other chunk-9 slices
have somewhere to write, and registers it in `docs/audits/lib-sweep-coverage.json`.
Closes #2288.

- **Record header:** chunk `9`, sweep date `2026-09-28`, baseline
  `a7c3f65108023b93c50e9ac23e6561e0c803e22e`, exposure `internal`, tracker
  #2094, plus a short methodology paragraph and the defect classes the issue
  lists.
- **Inventory:** 34 rows of `pending — <owner>`, grouped under
  `### shaders` / `evaluation` / `device` / `queue-core` / `queue-lifecycle`.
  Line counts come from `git show <baseline>:<path> | wc -l`: 24 `.rs` files
  (9,658 lines, up from 9,639 at `aef0077`) and 10 `.wgsl` files (1,338 lines).
- **Section regions:** the five markers appear under `## Audit sections`, and
  again under both `## Ledger` (`finding-id | file:line | CWE | severity | status`)
  and `## Refuted / not findings`. Both tables are empty.
- **Files the issue's owner list does not name:**
  - `queue/fake_evaluator.rs` → queue-core (#2114). It is the
    `RequestEvaluator` test double for `executor.rs`, driven through
    `execution.rs::run_work_loop`.
  - `queue/wedge_tests.rs` → queue-core (#2114). It drives that loop and the
    bounded wait in `submission.rs`.
  - `queue/stale_skip_tests.rs` → queue-lifecycle (#2115). It tests
    `staleness.rs`.
- **Index entry:** the chunk `"9"` line now sets `last_swept`,
  `baseline_commit` and `record`. It stays on one line and keeps its key order.

No audit verdicts or struct-parity table: those belong to the dependent slices.

## Evidence

This change touches only docs and tests, so there is no UI to capture.

- `cargo test --test issue_2088_sweep_ledger_contract`: 9/9 pass, including
  `prose_records_and_index_entries_match_both_ways` and
  `a_claimed_sweep_pins_a_baseline_commit_and_a_record`.
- `cargo test --test issue_2288_chunk_09_ledger_scaffold`: 4/4 pass. It failed
  before the record existed.
- The Spec and Standards reviewers each re-checked every inventory line count
  against `git show a7c3f65…:<path> | wc -l`, and all 34 match.
- `./quality.sh` passed: `✅ All quality checks passed!`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `security-sweep-chunk-9-gpu-wgsl.md` exists with every `## Record` field, a 34-row `## Files swept` table whose counts equal `wc -l` at the baseline, all five markers in the body, and both tables carrying all five markers — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`, `tests/issue_2288_chunk_09_ledger_scaffold.rs` — reviewer: met
- **met** — the chunk `"9"` index entry names the record, the full baseline SHA and the sweep date, on one line with the key order unchanged — evidence: `docs/audits/lib-sweep-coverage.json`, `tests/issue_2088_sweep_ledger_contract.rs::every_entry_is_one_line_with_a_stable_key_order` — reviewer: met
- **met** — `cargo test --test issue_2088_sweep_ledger_contract` passes, including both named tests — evidence: 9/9 pass locally — reviewer: met
- **met** — `./quality.sh` passes — evidence: full gate run after the final code edit, `✅ All quality checks passed!` — reviewer: missing — reason: the reviewer only saw the diff and could not run the gate. I ran it here and it passed.
- **unrequested** — new scaffold contract test `tests/issue_2288_chunk_09_ledger_scaffold.rs` — reviewer: unrequested — reason: TDD for the scaffold's shape (header/index agreement, section grouping, marker order), following the chunk-11 scaffold precedent `tests/issue_2233_chunk_11_ledger_scaffold.rs`; the issue's file area lists `tests/`.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — `docs/audits/README.md` says only a real sweep justifies a non-null `last_swept`, but this sets it while every row is still `pending` — evidence: `docs/audits/lib-sweep-coverage.json:10` — reason: it stands. Issue #2288 step 2 explicitly requires setting `last_swept`, and the chunk-11 scaffold (#2233) did the same. The record's `### Sweep status — IN PROGRESS` says the date marks when the scaffold was cut, not a finished sweep.
- **clean** — required record fields; all 34 line counts and paths; owner mapping, including the three unnamed files and why they were assigned; marker order across all three regions; the index line's format and key order; the falsifying `git diff` command; Australian English; CI and dependencies left alone. The repo has no `CODING-STANDARDS.md`, so the reviewer checked against `CONTRIBUTING.md`, `AGENTS.md` and `docs/audits/README.md`. Optional notes: the `chunk-9` filename is not zero-padded, as the issue requires and the record explains. The template's optional `## Related remediations` section is left out.

## Test Plan

- Added `tests/issue_2288_chunk_09_ledger_scaffold.rs`. It checks:
  - every record field is present, and the date, baseline, exposure and record path agree with the index;
  - every inventory row sits under one of the five `###` groups, names a real in-scope file once, and has a positive line count;
  - the five markers appear in order in `## Audit sections`, `## Ledger` and `## Refuted / not findings`.
- Existing `tests/issue_2088_sweep_ledger_contract.rs` passes unchanged.
