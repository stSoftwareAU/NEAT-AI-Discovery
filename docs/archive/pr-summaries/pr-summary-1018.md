## Summary

Implement Metropolis-Hastings probabilistic acceptance for marginal synapse
candidates during candidate selection. Closes #1018.

When enabled via the `NEAT_AI_DISCOVERY_MH_TEMPERATURE` environment variable,
candidates with improvement between 0 and the threshold are accepted with
probability `min(1, exp(improvement / temperature))` instead of always
proceeding. This allows the search to explore marginal candidates
probabilistically while preserving existing deterministic behaviour as the
default.

### Key changes

- **`candidate_scoring.rs`**: Added `DEFAULT_MH_TEMPERATURE` constant (0.01)
- **`config/user_facing.rs`**: Added `mh_temperature()` accessor for the
  `NEAT_AI_DISCOVERY_MH_TEMPERATURE` environment variable (cached via `OnceLock`)
- **`evaluation.rs`**: Implemented Metropolis-Hastings acceptance logic in the
  below-threshold code path, using a deterministic FNV-1a hash of source+target
  UUIDs for reproducible pseudo-random decisions
- **`rejection.rs`**: Added `accepted_below_threshold_count` counter to
  `TargetDiagnosticEntry` and `record_accepted_below_threshold()` method to
  `TargetDiagnostics` for monitoring acceptance rates

### Feature gating

- When `NEAT_AI_DISCOVERY_MH_TEMPERATURE` is **unset** (default): existing
  deterministic threshold-based acceptance is preserved — no behaviour change
- When set to a positive value (e.g. `0.01`): marginal candidates are
  probabilistically accepted or rejected based on the Metropolis-Hastings formula

## Evidence

All 171 unit tests and 5 integration tests pass. `./quality.sh` passes cleanly
including clippy, fmt, doc build, and release build.

## Test Plan

- `mh_acceptance_hash_is_deterministic` — same inputs produce same hash
- `mh_acceptance_hash_differs_for_different_inputs` — different inputs diverge
- `mh_acceptance_hash_avoids_prefix_collision` — separator byte prevents collisions
- `acceptance_probability_above_threshold_is_always_one` — above-threshold always accepted
- `acceptance_probability_marginal_candidates` — probability proportional to improvement
- `zero_or_negative_improvement_rejected` — zero/negative improvements never accepted
- `hash_produces_varied_random_values` — hash covers reasonable [0, 1) range
- `diagnostics_tracks_accepted_below_threshold_count` — counter increments correctly
- `diagnostics_accepted_below_threshold_defaults_to_zero` — counter starts at zero
