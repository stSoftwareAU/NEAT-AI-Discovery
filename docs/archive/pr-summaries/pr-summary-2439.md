## Summary

Built the topology cache's input and output UUID sets from listed neurons rather
than declared widths, preventing sparse creatures from paying for oversized
HashSet capacity during scans. Preserved width validation and implicit-input
semantics. Closes #2439.

Version bump: `Cargo.toml` updated for this change; the merged commit records the
final version.

## Spec

### Intent and Rationale

- Declared widths can be much larger than the listed topology, and HashSet
  iteration cost follows capacity. Empty construction lets inserted UUIDs set
  capacity while preserving the existing single-pass classification.

### Essential Design Decisions

- Input and output membership still comes only from neurons listed in
  `creature.neurons`; declared widths remain authoritative for observation
  width validation and implicit inputs remain unlisted.
- The existing `1..=1,000,000` input and output bounds remain in force.

### Undiscoverable Facts

- None.

## Evidence

Backend-only performance change; no visual surface changed. Criterion benchmarks
used sparse fixtures with one listed input and one listed output. Each repeated
scan iteration traversed both UUID sets 30 times.

| Case | Before | After |
| --- | ---: | ---: |
| Ordinary-width construction | 171.79 ns | 215.88 ns |
| Maximum input-width construction | 271.89 µs | 223.88 ns |
| Maximum output-width construction | 266.06 µs | 212.10 ns |
| Ordinary-width repeated scans | 48.257 ns | 60.369 ns |
| Maximum input-width repeated scans | 1.6524 ms | 69.117 ns |
| Maximum output-width repeated scans | 2.0150 ms | 58.497 ns |

The ordinary-width changes are benchmark noise on this run; maximum-width
construction and scans improved substantially. The width validator still
rejects unsupported counts, and the cache no longer reserves from those counts.

## Test Plan

- `tests/infrastructure/issue_754_topology_cache.rs` — both new regressions
  failed against the original constructor at the bounded-capacity assertions,
  then passed after the fix. The tests also verify unchanged UUID membership and
  zero capacity for empty listed sets.
- `timeout 900 cargo test --test infrastructure issue_754_topology_cache -- --nocapture < /dev/null` — passed, 8 tests.
- `timeout 900 cargo test --test issue_2369_topology_scan < /dev/null` — passed, 4 tests; cached and uncached candidate equivalence and cancellation coverage remain intact.
- `timeout 900 cargo test --lib topology_deadline_tests < /dev/null` — passed, 8 deadline and cancellation tests.
- `cargo fmt --all -- --check` and `git diff --check` — passed.
- `cargo bench --bench topology_cache -- declared_width < /dev/null` — before and after measurements recorded above.
- `timeout 900 ./quality.sh < /dev/null` — timed out three times with exit 124 while compiling the full integration-test matrix. Shell checks, dependency audit, debug build, formatter, Clippy and type checks passed before the test compilation stage; the full test stage and remaining gate steps did not complete.
- No existing test assertions were removed or changed.

## Security

No new input surface or dependency was added. Existing input/output width
validation and the implied-input representation remain unchanged.
