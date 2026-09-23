# Chunk 8b: audit `src/analysis/recommendation/` core detectors (Issue #2108)

## Summary

Swept all 8 files of `src/analysis/recommendation/` core (3,463 lines) for the
five chunk 8b defect classes, with ranking integrity as the primary lens, and
filled the `recommendation core` section of
`docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md` — the
eight per-file rows, the capacity table, the float-comparison table and a new
outcome section. Closes #2108.

**Three security findings filed**, each demonstrated against this tree with a
crafted input built from values the FFI boundary accepts:

- **#2181** (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `fan_in.rs::detect_fan_in_candidates` ranks its per-target inputs with
  `partial_cmp(…).unwrap_or(Equal)`, which is **not a total order**, while the
  `corr.abs() < THRESHOLD` filter above it **fails open** on NaN. A NaN
  correlation is reachable from finite records: an activation swing near
  `±2e30` overflows the `f32` covariance accumulator in `pearson_correlation`,
  whose `denom < f32::EPSILON` guard a NaN loses and whose `clamp` propagates
  it. Measured: 25 genuine fan-in candidates with the poisoned neurons listed
  *after* the honest ones, **0** with them listed *first* — the caller picks
  the order the unspecified sort produces.
- **#2182** (same labels) — the `recommendation core` counterpart of #2167.
  The descending `total_cmp` sorts in `output_bias_drift.rs`, `multi_hop.rs`
  and `gradient_discovery.rs` rank a `+inf` `estimated_improvement` first, and
  each detector manufactures that `+inf` from **finite** records by overflowing
  an `f32` accumulator. Measured at rank 0 for all three;
  `output_bias_drift.rs` additionally emits a `-inf` bias in its `SetBias`
  payload.
- **#2183** (same labels) — the same shape as #2161 and #2169: the directory
  contains **zero** `deadline_passed` / `is_cancelled` calls, and
  `discovery_dispatch.rs::detect_discovery_modules_parallel` checks the
  deadline only *before* a module's closure runs, so the three live detectors'
  scans cannot be interrupted once started.

**Two out-of-class observations**, filed as ordinary (non-security) issues:
**#2184** (a `HashMap`-iteration-order tie-break makes the activation
recommendation non-deterministic, and the `"ReLU6"` penalty key is dead because
every insertion spells `"RELU6"`) and **#2185** (`output_competition.rs`, 325
lines from Issue #1321, has no production caller at all — AGENTS.md § *Dead
Levers*).

The findings are linked in the ledger and in a comment on the parent issue
#2093.

## Evidence

Backend/library change with no web interface, so no screenshot applies. The
evidence is the contract test suite plus the measured triggers recorded in the
ledger and in each filed issue.

`tests/issue_2108_chunk_08b_recommendation_core_sweep.rs` (13 tests, 0.03 s)
splits into two halves:

- **Record contract** — the eight rows are present in record order and none
  reads `pending`; every capacity site and every float-comparison site in the
  production half of those files is cited in the matching table; the 22 symbols
  the outcome traces still exist; the three findings are linked from the
  outcome and from `## Issues filed`; and a ranking verdict is recorded for
  every file that carries a ranking comparator.
- **Behaviour contract** — three tests pin the reachability the findings rest
  on (including one that drives the FFI boundary itself, so "reachable from
  finite records" is asserted where the gate actually lives) and three pin the
  `clean` verdicts.

```mermaid
flowchart TD
    A["Parquet record stream<br/>finite values only<br/>FFI gates #2134 / #2135"] --> B["pearson_correlation<br/>f32 covariance accumulator"]
    A --> C["sum_error / sum_min / mean_abs_error<br/>f32 accumulators"]
    B -->|"overflow to inf, inf/inf"| D["NaN correlation"]
    C -->|"overflow"| E["+inf estimated_improvement"]
    D --> F["corr.abs() &lt; THRESHOLD<br/>fails open, input kept"]
    F --> G["sort_by partial_cmp unwrap_or Equal<br/>not a total order"]
    G --> H["#2181 — window emptied,<br/>0 genuine candidates"]
    E --> I["&lt;= 0.0 and &lt; MIN gates<br/>fail open on +inf"]
    I --> J["descending total_cmp<br/>+inf above every finite gain"]
    J --> K["#2182 — crafted candidate at rank 0,<br/>non-finite SetBias emitted"]
```

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — All 8 rows non-`pending` with a one-line reason — evidence: `docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md` § `recommendation core`, gated by `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::every_recommendation_core_row_is_swept_with_a_reason` — reviewer: met
- **met** — A ranking-integrity conclusion recorded per detector: can a crafted input reach rank 1 unchecked — yes/no, with the path — evidence: the **Ranking integrity** verdict table in the outcome section, gated by `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::the_recommendation_core_outcome_records_a_ranking_verdict_per_detector` — reviewer: met
- **met** — Every capacity and float-comparison site in these files has a table row with its NaN handling — evidence: the two `<!-- section: recommendation core -->` regions, gated by `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::every_capacity_site_in_the_swept_files_has_a_table_row` and `::every_float_comparator_in_the_swept_files_has_a_table_row` — reviewer: met
- **met** — Findings filed with required labels, linked in the ledger and in a comment on #2093 — evidence: #2181, #2182, #2183 each carry `security`, `lang:rust`, `severity:medium`, `confidence:high`; linked in `## Issues filed` (gated by `::the_recommendation_core_outcome_links_its_filed_findings`) and in stSoftwareAU/NEAT-AI-Discovery#2093 (comment 5795152721) — reviewer: met
- **partial** — `./quality.sh` passes — evidence: every stage ran green except the final `cargo test` sweep, which aborted twice on a `Permission denied (os error 13)` executing a freshly linked test binary under `/var/tmp/vibe-cargo-target` — a different, unrelated binary each time (`issue_1931_gpu_breaker_partial_result`, then `focus`), each of which passes when re-run on its own — reviewer: met — reason: the reviewer read the diff and could not run the gate; it was run here and is recorded honestly as `partial`, see the **Quality gate** note below

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — CONTRIBUTING.md § PR Summary File: no `docs/archive/pr-summaries/pr-summary-2108.md` — evidence: `docs/archive/pr-summaries/pr-summary-2108.md` — reason: fixed here; this file is that summary
- **violation** — CONTRIBUTING.md § Guard Wiring at the Shipped Entry Point: the behaviour tests call the detectors directly rather than crossing the FFI boundary, so "reachable from finite records" was asserted in prose only — evidence: `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs:445` — reason: fixed here by adding `::the_ffi_boundary_still_accepts_the_magnitudes_both_findings_are_triggered_with`, which drives `serde_json::from_str::<NeuronData>` with the exact trigger magnitudes and asserts they deserialise, and that a saturating `1e39` is refused. The remaining direct detector calls match the sibling shape in `tests/issue_2105_*.rs` and `tests/issue_2107_*.rs`
- **violation** — CONTRIBUTING.md § Boy Scout Rule: the record's *Sweep status — IN PROGRESS* paragraph still names only two swept sections — evidence: `docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md:36` — reason: **stands, deliberately**. Issue #2108 instructs "Edit only the `recommendation core` section of the ledger" precisely because the chunk's sub-issues run concurrently; every sibling (#2105–#2107) left the same line for the same reason, and rewriting a shared paragraph guarantees a conflict with whichever sibling PR merges next. It is the finalisation sub-issue's line to fix
- **violation** — `docs/audits/README.md` § *When a sweep must write to the ledger*: `lib-sweep-coverage.json` is untouched — evidence: `docs/audits/lib-sweep-coverage.json` — reason: **stands**. The chunk 8b entry already exists from the #2103 scaffold, and the record itself (lines 39–45) documents that the index cannot express per-section progress — the chunk is complete only when no row reads `pending`, which is two sections away. `tests/issue_2088_sweep_ledger_contract.rs` passes
- **violation** — CONTRIBUTING.md § Commit Messages: the two commits carrying this work read `WIP checkpoint: periodic agent progress snapshot (Issue #4170)` — evidence: `git log` at `HEAD` and `HEAD~1` — reason: those are the worker's own periodic auto-commits, not authored here; the final commit on this branch references #2108
- **clean** — Australian English throughout the added prose and test messages; symbol-not-line citation (`file.rs::symbol`) everywhere, with 22 `(file, declaration)` pairs pinned in the test; the ledger-parsing helpers are byte-identical to the sibling sweeps' and are backed by tests that call real library functions; Issue #1799 preconditions pinned before every citation loop and before both negative assertions; no `Instant` / `Duration` / wall-clock threshold anywhere; version bumped `0.74.250` → `0.74.251` in `Cargo.toml` and `Cargo.lock`; no hidden paths staged; no CI workflow touched; no dependency resolution moved

## Quality gate

`./quality.sh` was run twice end to end. Both runs passed the shell-syntax,
ShellCheck, cargo-install-pinning, PR-summary-location, `cargo deny check`,
`cargo build`, `cargo fmt`, `cargo clippy --all-targets --all-features -D
warnings` and `cargo check --all-targets --all-features` stages, and both
aborted partway through `cargo test` with:

```text
Caused by:
  could not execute process `…/debug/deps/<test>-<hash> --test-threads=2` (never executed)
Caused by:
  Permission denied (os error 13)
```

A **different** unrelated test binary each time (`issue_1931_gpu_breaker_partial_result`,
then `focus`), both of which pass when re-run individually, and both with mode
`755` on disk. It is an execution-permission flake in this sandbox on freshly
linked binaries, not a test failure. The suites this change touches were run
directly and pass: `issue_2108_chunk_08b_recommendation_core_sweep` (13),
`issue_2103_chunk_08b_ledger_scaffold` (11), `issue_2107_chunk_08b_scoring_sweep`
(17), `issue_1931_gpu_breaker_partial_result` (8). CI runs the same gate on this
PR.

## Test Plan

Added `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs` — 13 tests:

**Record contract** (fails against the unswept ledger, passes after this
change — `every_recommendation_core_row_is_swept_with_a_reason` is red while
the eight rows read `pending`):

- `the_recommendation_core_section_owns_exactly_the_files_issue_2108_swept`
- `every_recommendation_core_row_is_swept_with_a_reason`
- `every_capacity_site_in_the_swept_files_has_a_table_row`
- `every_float_comparator_in_the_swept_files_has_a_table_row`
- `the_recommendation_core_outcome_cites_symbols_that_still_exist`
- `the_recommendation_core_outcome_links_its_filed_findings`
- `the_recommendation_core_outcome_records_a_ranking_verdict_per_detector`

**Behaviour contract** — the reachability the findings rest on, and the guards
the `clean` verdicts rest on:

- `a_finite_record_set_still_drives_the_fan_in_correlation_to_nan`
- `the_ffi_boundary_still_accepts_the_magnitudes_both_findings_are_triggered_with`
- `fan_in_candidates_still_depend_on_the_order_the_caller_lists_neurons_in`
- `sample_weighted_launders_every_unusable_error_before_it_ranks`
- `the_synapse_gradient_rejects_every_unusable_mean_before_returning_it`
- `the_activation_recommender_ranks_over_a_constant_score_space`

No existing test was modified or removed.

## Security note — the original trigger, and why it is still open

This is an **audit** PR: it records defects and files them; it does not change
any production code, so no trigger is closed here. That is deliberate and
matches the process the issue sets out — "file one house-format issue (#2078
shape) per surviving finding … stating that the fix ships
`tests/issue_<n>_*.rs` failing before the fix". Each of #2181, #2182 and #2183
carries that statement and the regression test it must ship.

What this PR does close is the **audit** gap, and the regression test for that
is `every_recommendation_core_row_is_swept_with_a_reason`: it fails against the
unfixed ledger (eight rows reading `pending`) and passes after this change.
There is no trivial bypass of it — `the_recommendation_core_outcome_records_a_ranking_verdict_per_detector`
and `::cites_symbols_that_still_exist` reject a row filled in with prose that
names no symbol and no ranking verdict, and the two table tests reject a file
whose capacity or comparator sites are not cited, so a later edit that adds a
sort or an allocation without recording it turns the suite red.
