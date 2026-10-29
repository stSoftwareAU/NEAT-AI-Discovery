## Summary

Audits all 10 `src/shaders/*.wgsl` kernels (1,338 lines) for out-of-range
guards, `workgroupBarrier()` uniformity and NaN/Inf/divide-by-zero handling,
and writes the result into the `shaders` region of
`docs/audits/security-sweep-chunk-9-gpu-wgsl.md`. Closes #2290.

- **Kernel table:** one row per kernel, giving its `@workgroup_size` line,
  bounds guard (`file:line`), barrier-safety verdict, arithmetic handling and
  verdict. Every kernel uses one of three guard shapes: an element-wise early
  `return`, zero-padding in the `*_reduce.wgsl` loads, or `bias.wgsl`'s
  `in_range`.
- **Barriers:** all 10 `workgroupBarrier()` sites are uniform. No kernel that
  has a barrier returns early, and each barrier sits at function scope or in a
  loop bounded by a constant or a uniform. The record notes that naga 30 does
  not check barrier uniformity, so this table is the only record of it.
- **Arithmetic:** the `uniforms.epsilon` guards are recorded. `bias.wgsl`'s
  unused `epsilon` (L22) and its L228 ceil-div are both refuted with reasons.
  The ceil-div cannot wrap because `wgpu::Limits::default()` caps the
  `samples` binding at 2^24 elements.
- **One finding:** #2308 (CWE-754, low; filed, not fixed here). `is_finite_value`
  is a float self-comparison that fast-math may fold away. It is linked from a
  `## Ledger` row.
- **Dead shaders:** `matching.wgsl` is recorded as dead and `relu_reduce.wgsl`
  as unused. One removal follow-up covering both is filed as #2309. Neither
  file is deleted.
- **Inventory:** none of the 10 `.wgsl` rows in `## Files swept` reads
  `pending` any more.

This change is docs only; no code changes.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The kernel table has exactly 10 rows, each with a guard `file:line`, a barrier-safety verdict, an arithmetic entry and a verdict — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (`#### Kernel bounds and arithmetic (Issue #2290)`) — reviewer: met
- **met** — Each of the 10 `workgroupBarrier()` sites has a verdict — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (the **Barrier sites** table, 10 rows, all uniform) — reviewer: met
- **met** — The unused `bias.wgsl` `epsilon` field and the L228 ceil-div each appear as a ledger finding or a refutation with a reason — evidence: two rows in the `shaders` region of `## Refuted / not findings`, citing `bias_evaluation.rs:125`/`:130`, `analyzer.rs:291`/`:394` and wgpu-types `limits.rs:441` — reviewer: met
- **met** — `matching.wgsl` is recorded as dead and `relu_reduce.wgsl` as unused, and exactly one removal follow-up is linked — evidence: the **Dead shaders** paragraph and the `## Files swept` rows link #2309 — reviewer: met
- **met** — Every surviving finding has its own issue, linked from a `## Ledger` row — evidence: the #2308 ledger row reads `open — #2308`. #2308 is in house format, with its markers, all five sections and the `security`, `lang:rust`, `severity:low` and `confidence:low` labels — reviewer: met
- **met** — None of the 10 `.wgsl` rows in `## Files swept` reads `pending` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (`### shaders` inventory) — reviewer: met
- **met** — `./quality.sh` passes, including `tests/issue_2088_sweep_ledger_contract.rs` — evidence: the worker's `./quality.sh` run on this branch; `cargo test --test issue_2088_sweep_ledger_contract --test issue_2288_chunk_09_ledger_scaffold --test issue_2289_gpu_struct_layout` passes 9/9, 4/4 and 5/5 — reviewer: partial — reason: the reviewer ran only those three test targets (all passing), not the full `./quality.sh`. The worker's quality gate did run and passed on this branch.
- **unrequested** — Five extra refutation rows: NaN tile padding, barrier uniformity, out-of-range index, `activation.wgsl`'s unused `epsilon`, and `relu.wgsl`'s unused `threshold` — reviewer: unrequested — reason: these suspicions came up while sweeping the same kernels and are recorded so no later slice re-opens them. They are within the audit's scope.
- **unrequested** — A "Defence in depth" paragraph on wgpu runtime bounds checks, plus `## Outcome` and `## Issues filed` updated to list #2308 and #2309 — reviewer: unrequested — reason: record housekeeping and context. The reviewer checked every citation and found them accurate.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

This repository has no `CODING-STANDARDS.md`, so the reviewer used
`CONTRIBUTING.md` and `AGENTS.md`.

- **violation** — PR summary file missing (CONTRIBUTING.md "PR Summary File") — evidence: `docs/archive/pr-summaries/pr-summary-2290.md` — reason: fixed here; this file was added.
- **violation** — The change cites code by line number, against "Cite Code by Symbol, Never by Line Number" (CONTRIBUTING.md, Issue #1942) — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (about 44 new `file:N` citations) — reason: stands. The chunk-9 record's own scaffold (#2288) requires a `file:line` ledger column, and the issue asks for `file:line` guards. The record is pinned to a baseline SHA and has a `## Verify this record` drift check. The reviewer judged this justified, but no written exemption for SHA-pinned audit records exists yet.
- **violation** — Edits outside the slice's own regions — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (the shared `## Outcome` paragraph and the intro line of `## Issues filed`) — reason: stands. This is minor: a merge-conflict risk for sibling chunk-9 slices, not a correctness problem.
- **violation** — The ledger row's `file:line` cell holds more than one location — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (the #2308 row) — reason: stands as a cosmetic point. The row keeps the required five columns, and the contract test passes.
- **clean** — Areas the reviewer checked and found compliant:
  - Australian English.
  - No Mermaid was added.
  - The new ledger, refuted and inventory rows stay in the `shaders` regions, and the row formats are correct.
  - About 65 WGSL citations, plus the Rust and third-party crate citations (naga, wgpu-hal, wgpu-types), were verified at HEAD.
  - `.github/workflows/ci.yml` is untouched.
  - No version bump is needed for a docs-only change, because CI's `version-increment` job bumps on the PR.
  - The Dead Levers rule is applied correctly: the unused fields are not operator levers, and the dead shaders have a removal follow-up.

## Test Plan

- Ran `cargo test --test issue_2088_sweep_ledger_contract --test issue_2288_chunk_09_ledger_scaffold --test issue_2289_gpu_struct_layout`.
- Ran `./quality.sh`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
