## Summary

Stages the chunk-7 Parquet sweep record at
`docs/audits/in-progress/security-sweep-chunk-07-parquet.md`: record header, the
nine files with their line counts at baseline `b85a551` (total 2,642), the six
related remediations with guard and comment citations, and a 22-row Decode entry
points table covering every decode path in `src/parquet_format/`. Defect
classes probed, Outcome and Issues filed are explicit `pending #2143 / #2144`
placeholders. Docs only: no `.rs`, `Cargo.toml`, workflow or
`lib-sweep-coverage.json` change, so chunk `7` keeps `last_swept`,
`baseline_commit` and `record` as `null`. Version bump: not applicable, no code
change. Closes #2279.

## Spec

### Intent and Rationale

- The ledger lives under `in-progress/` because `record_files` in
  `tests/issue_2088_sweep_ledger_contract.rs` scans only the top level of
  `docs/audits/`; #2145 later moves it and flips chunk `7`.

### Essential Design Decisions

- Citations are `file:line` pinned to `b85a551`, as the issue requires, though
  `CONTRIBUTING.md` prefers symbols; the record states the departure and the
  command that falsifies it.
- A `### Sweep status — IN PROGRESS` note sits under `## Record` so the file
  cannot be read as a finished sweep.

### Undiscoverable Facts

- `src/analysis/cache/mod.rs` has drifted since `b85a551` (its `#1048` comments
  are now at L119, 134, 235, 311, 637). The ledger cites the `b85a551` lines the
  issue gives, confirmed with `git grep -n '#1048' b85a551 -- src/analysis/cache`.

## Evidence

Docs-only change; no UI or performance surface.

- `git diff --stat b85a551 HEAD -- src/parquet_format` is empty.
- `git show b85a551:src/parquet_format/<file>.rs | wc -l` gives 393, 106, 95, 135,
  558, 324, 596, 276, 159 for the nine files.
- Every cited line was checked against the source; the reviewer also confirmed
  them.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `docs/audits/in-progress/security-sweep-chunk-07-parquet.md` exists, with every section above filled or explicitly marked pending — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md` — reviewer: met
- **met** — No top-level `docs/audits/security-sweep-chunk-07*.md` is created — evidence: diff touches only the `in-progress/` path — reviewer: met
- **met** — `docs/audits/lib-sweep-coverage.json` is unchanged — evidence: not in the diff — reviewer: met
- **met** — Every file:line citation matches `b85a551`. There is no `src/parquet_format` diff between `b85a551` and HEAD — evidence: empty `git diff b85a551 HEAD -- src/parquet_format`; the `src/analysis/cache` citations are the `b85a551` lines — reviewer: met
- **met** — No `.rs`, `Cargo.toml` or `.github/workflows/ci.yml` change — evidence: single `.md` file plus this summary — reviewer: met
- **met** — `cargo test --test issue_2088_sweep_ledger_contract` passes — evidence: 9 passed, run on the final head content — reviewer: met — reason: the reviewer could not run it; it was run here
- **partial** — `./quality.sh < /dev/null` passes — evidence: `./scripts/doc-check.sh` (9 passed, "Documentation build passed") and markdownlint-cli2 (0 issues) — reviewer: partial — reason: quality gate not yet run; it takes about 15 minutes, longer than the 10 minute foreground command cap, so CI runs it on the PR — remaining: merge-verification-only
- **met** — Verify this record: commands that reproduce the line counts and citations — evidence: the section's `git show … | wc -l`, `git grep` for each issue number, and a definitions grep, whose nine hits were checked — reviewer: partial — reason: the reviewer found the first draft's definitions grep did not reproduce several cited lines; a symbol grep for them was added and run (hits at `writer.rs:325`, `reader.rs:117`, `shared_records.rs:62` and `:99`, `decode_budget.rs:75` and `:91`, `footer.rs:53`, `batch_columns.rs:113` and `:145`)
- **unrequested** — a "Citation convention" note, a "Sweep status — IN PROGRESS" note and an Exposure field in the record — reviewer: unrequested — reason: the template carries Exposure; the notes keep the staged file from reading as a finished sweep and explain the line-number citations

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — `CONTRIBUTING.md` "Cite Code by Symbol, Never by Line Number" — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md:70` — reason: required by #2279 (its Decode entry points column is `File:line` and its remediations name line numbers); the record names the symbol beside each line and pins it to `b85a551`
- **clean** — Australian English, relative links resolve, ledger README rules (one file per chunk, required fields, remediations kept apart from coverage), index entry untouched, no hidden files staged. `CODING-STANDARDS.md` is absent from this checkout, so `CONTRIBUTING.md` and `docs/audits/README.md` were used.

## Test Plan

- `cargo test --test issue_2088_sweep_ledger_contract`: passed, 9 tests.
- `./scripts/doc-check.sh`: passed.
- `markdownlint-cli2` on the new record: 0 issues.
- No test added or edited: the staged file is outside the top-level scan the contract test reads, and that test is the guard.
- Not run: `./quality.sh`, because it takes about 15 minutes and a foreground command is capped at 10.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh takes about 15 minutes, beyond the 10 minute foreground command cap; docs-only change, doc-check and the ledger contract test were run; CI runs the gate on the PR" -->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
