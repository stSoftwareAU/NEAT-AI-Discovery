# PR Summary — Issue #2350: O(C·N) candidate/neuron squash lookups in fan-in polarity (and seven siblings)

## Summary

Closes #2350.

Eight detection converters looked up each candidate's neuron (squash,
incoming synapses, outgoing synapses, downstream targets) with
`creature.neurons.iter().find(..)` or `creature.synapses.iter().filter(..)`
inside the per-candidate loop, giving `O(C·N)` or `O(C·S)` work for `C`
candidates against `N` neurons / `S` synapses (CWE-407). Each converter now
builds the lookup once per call as a `HashMap`/`HashSet`, so the per-candidate
cost drops to `O(1)`:

- `src/analysis/detection/fanin_polarity_conflict.rs` — squash lookup;
  gained a deadline-aware `_with_deadline` entry point.
- `src/analysis/detection/bottleneck.rs` — squash lookup.
- `src/analysis/detection/monotonicity.rs` — squash, outgoing-synapse and
  incoming-synapse lookups.
- `src/analysis/detection/operating_point.rs` — incoming-synapse lookup.
- `src/analysis/detection/restricted_range.rs` — incoming-synapse lookup.
- `src/analysis/detection/output_range_compression.rs` — incoming-synapse
  lookup.
- `src/analysis/detection/sentinel_gating.rs` — downstream-target lookup.
- `src/analysis/detection/squash_weight_rescale.rs` — the per-neuron
  `feeds_downstream_aggregate` predicate (an `O(N·S)` scan repeated per
  hidden neuron) became `feeds_downstream_aggregate_set`, a single
  `HashSet<&str>` built once; plus a records-by-uuid and
  incoming-synapses-by-uuid map.

## Spec

### Intent and Rationale

- Each converter runs once per candidate list but was re-walking the whole
  neuron/synapse vector per candidate, so a creature with many flagged
  candidates against a large topology paid quadratic cost on every discovery
  pass.
- `squash_weight_rescale.rs`'s `feeds_downstream_aggregate` was worse: it
  re-scanned `creature.synapses` for every hidden neuron being evaluated,
  not just every candidate, so it was `O(N·S)` before any candidate existed.
- The fix follows the repo's existing `_with_deadline` pattern (co-adaptation,
  Issue #2348): a new deadline-aware fan-in function, with the old public
  signature delegating to it with `&None`, so no caller's signature changes.

### Essential Design Decisions

- `entry(..).or_insert(..)` for the squash map keeps the **first** neuron
  with a duplicate uuid, matching the semantics of the prior `.find()` scan.
- `entry(..).or_default().push(..)` for synapse maps preserves the original
  creature iteration order within each per-key `Vec`, because
  `operating_point.rs`/`restricted_range.rs`/`output_range_compression.rs`/
  `monotonicity.rs` all index into `incoming[0]`/`outgoing[0]` — changing the
  order would silently pick a different synapse.
- The fan-in deadline is threaded through a new
  `fanin_polarity_conflicts_to_coordinated_candidates_with_deadline` function
  rather than added as a parameter to the existing public one, and wired into
  `module_dispatch_specs/synapse_specs.rs` via a `custom:` dispatch-spec
  closure (the `guard:`/`records:`/`detect:`/`convert:` macro shorthand has
  no hook for passing a deadline through a second detection stage).
- `feeds_downstream_aggregate_set` returns `HashSet<&str>` (the set of
  neuron uuids that feed an aggregate-squash neuron), not a per-neuron
  bool predicate, so the aggregate-squash scan over `creature.neurons` and
  the synapse scan each run exactly once per call regardless of how many
  hidden neurons are evaluated afterwards.

### Undiscoverable Facts

- `fanin_polarity_conflicts_to_coordinated_candidates` (no deadline) is kept
  as the public entry point purely for ABI/call-site stability; the live
  dispatch path (`synapse_specs.rs:107`) calls
  `..._with_deadline` directly, so the no-deadline function is now exercised
  only by `tests/detection/issue_641_fanin_polarity_conflict.rs` and the new
  growth test, not by production dispatch.
- The growth regression test
  (`tests/issue_2350_fanin_polarity_squash_lookup_growth.rs`) deliberately
  calls only the pre-existing `fanin_polarity_conflicts_to_coordinated_candidates`
  signature (no `_with_deadline`, no new imports), so it compiles and runs
  unmodified against both the unfixed base and the fixed tree.
- On the unfixed base, the growth test measured a ratio of roughly 14.98
  (11.25 ms at N=2000 vs 168.44 ms at 4N=8000) — close to the expected ~16x
  quadratic blow-up; on the fixed tree it measured roughly 4.03, consistent
  with linear growth. Both numbers are timing-dependent and will vary run to
  run; the test's `ratio < 8.0` threshold sits between the two.
- `squash_weight_rescale.rs`'s `records_by_uuid` map uses
  `entry(..).or_insert_with(..)`, matching the first-match semantics of the
  prior `.iter().find()` over the records list exactly as the neuron-squash
  maps do for neurons.

## Security Fix Evidence

**Security regression test:**
`tests/issue_2350_fanin_polarity_squash_lookup_growth.rs::squash_lookup_work_grows_linearly_with_candidates`
(new in this diff).

- **Fails on the unfixed code.** Copying only this test onto `origin/Develop`
  measured roughly 11.25 ms at N=2000 vs 168.44 ms at 4N=8000 — a ratio of
  about 14.98, well past the `ratio < 8.0` assertion, showing the quadratic
  cost of the per-candidate `.find()` scan.
- **Passes after the fix.** It measured roughly 4.03 after the squash lookup
  became a once-per-call `HashMap`.
- **No trivial bypass:** the test builds candidates in increasing count
  alongside a proportionally larger neuron/synapse list, placing hidden
  neurons early in evaluation order (the worst case for a linear `find`), so
  there is no way to keep `N` small while growing `C`, nor to dodge the
  lookup — every candidate has a `neuron_uuid` that must resolve to a squash
  value for the comment/recommendation text it emits.

## Evidence

- `tests/detection/issue_641_fanin_polarity_conflict.rs` — existing
  fan-in-polarity converter tests, unchanged signatures, still pass.
- `tests/detection/issue_643_activation_error_monotonicity.rs` — gained
  `test_lookup_miss_drives_change_squash_branch` (below).
- `tests/detection/issue_399_restricted_range_detection.rs`,
  `tests/detection/issue_400_sentinel_value_gating.rs`,
  `tests/detection/issue_401_operating_point_analysis.rs`,
  `tests/detection/issue_645_output_range_compression.rs`,
  `tests/recommendation/issue_548_squash_weight_rescale.rs` — existing
  converter tests for the remaining six files, unchanged signatures, still
  pass.
- `tests/issue_2350_fanin_polarity_squash_lookup_growth.rs` — new growth
  regression test (above).
- `src/analysis/module_dispatch_specs/mod.rs::fanin_polarity_spec_detects_conflicts_before_the_deadline`
  and
  `src/analysis/module_dispatch_specs/mod.rs::fanin_polarity_spec_honours_an_elapsed_deadline`
  — new dispatch-spec tests covering the deadline threading.

## Callers checked

- `append_synapse_specs` (`src/analysis/module_dispatch_specs/synapse_specs.rs:17`)
  has exactly one caller:
  `build_discovery_module_specs` (`src/analysis/module_dispatch_specs/mod.rs:71`),
  which now passes the dispatch `deadline` through.
- `fanin_polarity_conflicts_to_coordinated_candidates`
  (`src/analysis/detection/fanin_polarity_conflict.rs:195`) is called by
  `tests/detection/issue_641_fanin_polarity_conflict.rs` and
  `tests/issue_2350_fanin_polarity_squash_lookup_growth.rs`; the live
  dispatch path calls the `_with_deadline` variant directly from
  `src/analysis/module_dispatch_specs/synapse_specs.rs:107`, so both the
  deadline-aware and deadline-free call paths are exercised by tests.

## Docs sweep

`grep -rn "feeds_downstream_aggregate\b" --include='*.md' --include='*.rs' .`
found three hits besides this file: `docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md:222`
(updated in this diff to name `feeds_downstream_aggregate_set`, built once
per detection call per Issue #2350) and
`docs/archive/pr-summaries/pr-summary-1713.md:28,32`, which are historical
archive entries for a prior PR and are left untouched.

## Test Plan

- `cargo test --test detection issue_643 < /dev/null`: 11 passed.
- `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings < /dev/null`.

Branch outcomes:

- `src/analysis/detection/fanin_polarity_conflict.rs:237` (deadline check in
  `_with_deadline`) — deadline elapsed → loop breaks early. Reached by
  `src/analysis/module_dispatch_specs/mod.rs::fanin_polarity_spec_honours_an_elapsed_deadline`
  (asserts `None`). Flipping the dispatch call in
  `src/analysis/module_dispatch_specs/synapse_specs.rs:107` from `&deadline`
  to `&None` turned this test red; restoring it turned it green.
- Same site — deadline in the future → conversion proceeds normally. Reached
  by
  `src/analysis/module_dispatch_specs/mod.rs::fanin_polarity_spec_detects_conflicts_before_the_deadline`.
- `src/analysis/detection/fanin_polarity_conflict.rs:272` — squash map hit →
  the neuron's own squash is used. Reached by every existing
  `tests/detection/issue_641_fanin_polarity_conflict.rs` test (every fixture's
  hidden uuid is present in `creature.neurons`); no existing test reaches the
  map-miss `"TANH"` fallback, which preserves the base code's behaviour of
  defaulting to `"TANH"` when the neuron cannot be found.
- `src/analysis/detection/monotonicity.rs:211-218` — both the
  `outgoing_by_uuid` and `incoming_by_uuid` maps miss (a neuron with no
  synapses at all) → `ChangeSquash` branch with an empty-slice fallback.
  Reached by the new
  `tests/detection/issue_643_activation_error_monotonicity.rs::test_lookup_miss_drives_change_squash_branch`
  (flipping the fallback to panic on miss turned it red — see below).
- Same site — both maps hit (a neuron with at least one outgoing and one
  incoming synapse) → `AddNeuron` branch. Reached by the pre-existing tests
  in the same file that build a hidden neuron with incoming and outgoing
  synapses.
- `src/analysis/detection/operating_point.rs:301`,
  `src/analysis/detection/restricted_range.rs:286`,
  `src/analysis/detection/output_range_compression.rs:250` —
  `synapses_by_to_uuid` hit (an incoming synapse exists) → the `SetWeight`
  candidate is produced. Reached by the existing fixtures in
  `tests/detection/issue_401_operating_point_analysis.rs`,
  `tests/detection/issue_399_restricted_range_detection.rs` and
  `tests/detection/issue_645_output_range_compression.rs` respectively —
  every fixture's hidden neuron under test has at least one incoming
  synapse. No existing test reaches the miss (empty) path for these three;
  the miss preserves the base behaviour of an empty `.iter().filter(..)`
  result, so the `SetWeight` candidate is simply skipped, unchanged from the
  unfixed code.
- `src/analysis/detection/sentinel_gating.rs:244-245` —
  `downstream_by_from` hit → downstream targets found, candidate proceeds.
  Reached by the existing fixtures in
  `tests/detection/issue_400_sentinel_value_gating.rs` (every candidate
  neuron under test has an outgoing synapse). The miss path (`continue` on
  an empty result) preserves the base behaviour of an empty
  `.iter().filter(..)` result; no existing test reaches it.
- `src/analysis/detection/squash_weight_rescale.rs:277-278` —
  `incoming_by_to` hit → `SetWeight`/rescale candidates evaluated. Reached by
  the existing fixtures in
  `tests/recommendation/issue_548_squash_weight_rescale.rs`. The miss path
  (`continue` on an empty result) preserves the base behaviour; no existing
  test reaches it.
- `src/analysis/detection/squash_weight_rescale.rs` (`feeds_downstream_aggregate_set`
  membership) — a hidden neuron whose uuid is in the returned set is skipped
  before any records lookup. Reached by the existing
  `tests/recommendation/issue_548_squash_weight_rescale.rs` fixtures that
  feed an aggregate-squash downstream neuron.

## Pre-PR Security Self-Check

- [x] **Input validation:** no change to validated FFI entry points; these
  converters only run after `validate_creature` has already accepted the
  creature.
- [x] **Secrets:** none staged.
- [x] **Injection surface:** no new shell, SQL or HTTP calls.
- [x] **Logging:** no new log lines.
- [x] **Dependencies:** no new dependencies.
