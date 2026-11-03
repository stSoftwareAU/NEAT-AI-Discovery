## Summary

Rewrote the `Skip Connection Detection` detection criteria and recommended action in `docs/DISCOVERY_TYPES.md` so they describe what `src/analysis/detection/skip_connection.rs` does: depth measured from the inputs, attenuation ratio against the shallow mean error, and an `addSynapse` from a shallow source to the deep neuron (never to an output). Documentation only; no code or version change. Closes #2475.

## Spec

### Intent and Rationale

- The section said a deep neuron is "several hops from the nearest output" and is connected "directly to an output"; the detector does the opposite (source shallow, target deep). Matches the earlier fix to `docs/ANALYSIS_DEEP_DIVE.md` (#2368).

### Essential Design Decisions

- Constants are named as in the code (`MIN_DEEP_DEPTH` = 3, `ATTENUATION_THRESHOLD` = 0.5, `MIN_DISCOVERY_SAMPLE_COUNT`); the Purpose, Cost and cancellation and Output paragraphs are untouched.
- The Topology section (`docs/DISCOVERY_TYPES.md` ~1282-1295) legitimately adds a skip to the output (`topology_issues_to_coordinated_candidates`, `skip_target_uuid`), so it was left alone.

### Undiscoverable Facts

- None.

## Evidence

Docs-only change; each claim checked against `src/analysis/detection/skip_connection.rs`: depth ≥ `MIN_DEEP_DEPTH` and ratio `< ATTENUATION_THRESHOLD` (deep_attenuated filter), shallow mean over depth ≤ 2 hidden neurons with error > 0, source choice depth-0 then depth-1 first unconnected, improvement = gap × (1 − ratio) × 0.005, weight `0.01 * target_mean_error.min(1.0)` in `skip_connections_to_coordinated_candidates`.

## Acceptance Criteria

The issue states no acceptance-criteria heading; nothing it asks is left undone.

## Test Plan

- No test added or edited: documentation prose only.
- Docs sweep: grepped `docs/`, `README.md` for "skip connection"; `docs/ANALYSIS_DEEP_DIVE.md:673-687` already correct; `docs/DISCOVERY_TYPES.md` Topology lines are a different detector and still true.
- `./quality.sh` not run (docs-only change): <!-- vibe-quality-gate-skipped reason="documentation-only change to docs/DISCOVERY_TYPES.md; no code, tests or dependencies touched; CI runs the gate" -->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
