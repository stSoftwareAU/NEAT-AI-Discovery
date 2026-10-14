# PR Summary — Issue #2355

## Summary

`assess_sentinel_cluster` (`src/analysis/detection/sentinel_cluster.rs`) built
the non-sentinel indices with `(0..n).filter(|i| !sentinel_indices.contains(i))`.
That is a linear `Vec::contains` for every sample, and the density rule
guarantees k ≥ 0.15·n sentinels, so the partition was O(n²) with no
cancellation point (CWE-407).

Both halves of the partition are now built in one `enumerate` pass, so the
cost is O(n). The accept/reject rule (density, separation, error
decorrelation) is unchanged, and both index lists stay in ascending order, so
the output is identical.

Closes #2355

## Spec

### Intent and Rationale

- The sentinel partition must cost O(n), not O(n·k), so a large observation
  cannot pin a worker inside an uncancellable loop.
- Behaviour is unchanged: the same samples are classified the same way, in the
  same order.

### Essential Design Decisions

- One loop pushes each index into `sentinel_indices` or
  `non_sentinel_indices`. This replaces the `HashSet` alternative, which would
  have cost an extra allocation and hashing for no benefit.
- No wall-clock assertion lives in `tests/`. CONTRIBUTING.md's Unit Tests vs
  Benchmarks table says tests must not use `Instant`/`elapsed` for pass/fail, and
  a ratio of two timings fails under host load with no code defect (#2296,
  #2320). The N vs 4N growth comparison is a Criterion benchmark,
  `benches/sentinel_cluster_partition.rs` (N = 4000, 16000, 64000), and the
  deterministic partition pin stays in `tests/`.

### Undiscoverable Facts

- The sibling partition in `observation_range.rs:164` looks the same but is
  already O(1) per lookup, because its `sentinel_indices` is a
  `HashSet<usize>`.

## Evidence

- **Behaviour pin (regression test):**
  `tests/issue_2355_sentinel_cluster_quadratic.rs::assess_sentinel_cluster_partitions_every_sample_once`
  (new file in this diff). For n in {101, 10, 5}, it asserts that sentinels are
  exactly the even indices, non-sentinels are exactly the odd indices, together
  they cover `0..n` once in ascending order, and the reported fraction matches.
  The old filter produced the same output, so this test pins the unchanged
  behaviour; it does not reproduce the quadratic cost.
- **Cost evidence:** `benches/sentinel_cluster_partition.rs` (new file in this
  diff, registered in `Cargo.toml`). Earlier timings from the removed ratio
  test: unfixed N=4000 took 30,968,500 ns and N=16000 took 491,187,584 ns
  (ratio 15.86); fixed N=4000 took 156,916 ns and N=16000 took 627,208 ns
  (ratio 4.00).
- **No trivial bypass:** the trigger was the only `Vec::contains` partition in
  `sentinel_cluster.rs`, and the file now has no `contains` call.
  `sentinel_gating.rs:98` uses `contains` on an unrelated `input_uuids` check.
- **Existing behaviour:** `cargo test --test detection` gave 503 passed and
  0 failed. This includes the #2042 sentinel tests.

**Docs sweep** — grep: `assess_sentinel_cluster`, `sentinel_indices`, `non_sentinel_indices`, `sentinel_cluster`, `MIN_SENTINEL_FRACTION`, `SENTINEL_TOLERANCE`, "sentinel cluster"; section: `docs/DISCOVERY_TYPES.md#sentinel-gating-detection` (criterion 2 names `sentinel_cluster.rs` as the single accept/reject rule — still true, the rule is unchanged and only the partition's cost changed); siblings: `input_uuid_precompute`, `record_arc_sharing` (the `[[bench]]` set in `Cargo.toml` and the bench table in `docs/BENCHMARKS.md` gained `sentinel_cluster_partition`; the `docs/BENCHMARKS.md` row is added in this change); no hits needing change — "sentinel cluster" hits in `docs/ANALYSIS_DEEP_DIVE.md:768`, `docs/DISCOVERY_TYPES.md:1559,1588`, `docs/discoveries/observation-utilisation.md:48` and `docs/discoveries/sentinel-gating.md:11,18` describe detection behaviour, which is unchanged; no hits for the other names in README.md, `docs/` (excluding `docs/archive/`) or `*/README.md`; updated: `docs/BENCHMARKS.md` (new `sentinel_cluster_partition` row)

## Test Plan

- [x] `cargo test --test issue_2355_sentinel_cluster_quadratic`: 1 passed. It is
      green on the base code too, because the output is unchanged. Inverting the
      sentinel `if` condition in `sentinel_cluster.rs` turns it red.
- [x] `cargo bench --bench sentinel_cluster_partition` on the head: N=4000
      about 33.7 µs, N=16000 about 108.6 µs, N=64000 about 459 µs, so about
      3.2x then 4.2x per 4x step in N (linear, not 16x).
- [x] `cargo test --test detection`: 503 passed.
- [x] `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- [x] `./quality.sh`: all checks passed, `cargo doc -D warnings` included.
- Version bumped to 0.74.288 (`Cargo.toml`, `Cargo.lock`), one patch above the base branch's 0.74.287.

Branch outcomes:

- `src/analysis/detection/sentinel_cluster.rs` loop `if` (sentinel arm) is
  reached by `assess_sentinel_cluster_partitions_every_sample_once` (the even
  indices). Swapping the arms turns it red.
- The same loop's `else` (non-sentinel arm) is reached by the same test (the
  odd indices).

The `if`/`else` replaces the old filter predicate one for one. It adds no new
accept or reject outcome.

Related: #2042: 🟠 Diverged sentinel-detection logic makes the error-variance
check a no-op in one copy. #873: Enable additional pedantic clippy lints for
numeric safety and code clarity.
