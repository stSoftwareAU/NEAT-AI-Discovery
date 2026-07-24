## Summary

Closes #1139.

Applied the coordinated-structural gain floor unconditionally after variant
generation so sub-floor `Gentle Nudge` / `Micro Nudge` variants can no longer
leak to the FFI response when the memory budget is exceeded or the
post-processing deadline passes.

Root cause: `pair_coordinated_structural_with_weight_variants` scales the base
candidate's `expected_creature_score_gain` by `0.75×` / `0.5×` / `0.25×` /
`0.1×`, producing variants below `COORDINATED_POST_DISCOUNT_NOISE_FLOOR`
(`5e-7`). The final sweep that removed those variants
(`apply_coordinated_gain_floor_with_multiplier`) lived **inside** the
`!memory_budget_exceeded && !post_processing_deadline_passed` guard in
`analyze_all`, so whenever either condition tripped the filter was skipped.
Production discovery-cache commit `744ac60d` (discoveryVersion `0.74.16`) captured two such
variants with gains of `1.4e-7` / `1.3e-7` damaging the creature when tested.

Changes:

- Added `apply_final_coordinated_gain_floor(syn, mode, multiplier)` in
  `src/analysis/candidate_aggregation.rs`. It computes the mode-aware
  multiplier, applies the floor, records the drop in
  `metadata.rejection_breakdown[REJECTION_BELOW_EXPECTED_GAIN_FLOOR]`, and
  refreshes `metadata.candidates_returned`. All three updates happen together
  so the FFI caller sees consistent metadata in either code path.
- `analyze_all` now calls the helper **outside** the memory/deadline guard,
  unconditionally, right after the guarded post-processing block closes. The
  duplicated inline block inside the guard was removed.
- Bumped `Cargo.toml` to `0.74.17` so remote workers pick up the fix.

## Evidence

This is a backend/FFI change with no web interface. Verification was via the
new TDD regression tests plus the existing floor suite, all green:

```
cargo test --test issue_1139_coordinated_floor_always_applied
  running 4 tests ... 4 passed
cargo test --test coordinated_min_gain_floor
  running 5 tests ... 5 passed
./quality.sh
  ✅ All quality checks passed!
```

The new precondition test (`variant_generation_can_produce_subfloor_gains`)
directly reproduces the leak by feeding a `1.9e-6` base candidate through
`pair_coordinated_structural_with_weight_variants` and asserting that at
least one of the six paired variants lands below `5e-7` — demonstrating the
bug mechanism captured by the production failure cache.

## Test Plan

New file `tests/issue_1139_coordinated_floor_always_applied.rs`:

- `variant_generation_can_produce_subfloor_gains` — reproduces the
  precondition for the Issue #1139 leak.
- `final_floor_removes_subfloor_variants_and_updates_metadata` — exercises
  the helper against a populated `AnalyzeSynapsesResult`, asserting no
  sub-floor candidate survives, that the rejection-breakdown count matches
  the removed count, and that `candidates_returned` is refreshed.
- `final_floor_honours_conservative_multiplier` — the conservative-mode `10×`
  multiplier drops a candidate exactly at the normal-mode floor.
- `final_floor_noop_when_all_above_floor` — no false-positive removals and
  `candidates_returned` still refreshed when nothing is filtered.

Existing suites continue to pass:

- `tests/coordinated_min_gain_floor.rs` (5 tests)
- Full `./quality.sh` (fmt, clippy, check, test, docs, release build)

## Security Self-Check

- [x] Input validation — new helper accepts only internal types; `multiplier`
      is clamped to `>= 1.0` by the downstream floor helper so a misconfigured
      env var cannot relax the floor.
- [x] No secrets staged.
- [x] No new SQL / shell / filesystem / HTTP surface.
- [x] No output encoding changes; metadata fields use existing typed structs.
- [x] No new authentication / authorisation surface.
- [x] No change to user-facing error messages; existing tracing remains.
- [x] No new third-party dependency.
