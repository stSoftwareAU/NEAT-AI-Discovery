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
- The regression test uses `std::time::Instant` for a **ratio** check (N=4000
  vs N=16000, best of 5, fail at ratio ≥ 10). CONTRIBUTING.md's testing section
  discourages timing assertions in unit tests. This test departs from that
  because the issue names this ratio test explicitly. It compares two readings
  of the same work, never a reading against a constant, so a slow host inflates
  both and stays green.

### Undiscoverable Facts

- The sibling partition in `observation_range.rs:164` looks the same but is
  already O(1) per lookup, because its `sentinel_indices` is a
  `HashSet<usize>`.

## Evidence

- **Regression test:**
  `tests/issue_2355_sentinel_cluster_quadratic.rs::assess_sentinel_cluster_grows_linearly`
  (new file in this diff).
  - **Red against the unfixed code:**
    `assess_sentinel_cluster grew super-linearly: N=4000 took 30968500ns,
    N=16000 took 491187584ns, ratio 15.86 (linear ~4, quadratic ~16)
    (Issue #2355)`
  - **Green after the fix:** N=4000 took 156,916 ns and N=16000 took
    627,208 ns, a ratio of 4.00.
- **Behaviour pin:**
  `tests/issue_2355_sentinel_cluster_quadratic.rs::assess_sentinel_cluster_partitions_every_sample_once`.
  For n in {101, 10, 5}, it asserts that sentinels are exactly the even
  indices, non-sentinels are exactly the odd indices, together they cover
  `0..n` once in ascending order, and the reported fraction matches.
- **No trivial bypass:** the trigger was the only `Vec::contains` partition in
  `sentinel_cluster.rs`, and the file now has no `contains` call.
  `sentinel_gating.rs:98` uses `contains` on an unrelated `input_uuids` check.
- **Existing behaviour:** `cargo test --test detection` gave 503 passed and
  0 failed. This includes the #2042 sentinel tests.
- **Base-branch gate fix:** `./quality.sh` failed at `cargo doc` on Develop
  itself (`public documentation for resolve_gpu_retry_limit links to private
  item MAX_GPU_RETRY_LIMIT --> src/config/user_facing.rs:56:15`).
  `src/config/user_facing.rs:56` now names the private const in plain backticks
  instead of an intra-doc link. This is a doc-only change, needed for the gate to
  pass.

**Docs sweep** — grep: `assess_sentinel_cluster`, `sentinel_indices`, `non_sentinel_indices`, `sentinel_cluster`, `MIN_SENTINEL_FRACTION`, `SENTINEL_TOLERANCE`, "sentinel cluster", `MAX_GPU_RETRY_LIMIT`, `resolve_gpu_retry_limit`, `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT`; section: `docs/DISCOVERY_TYPES.md#sentinel-gating-detection` (criterion 2 names `sentinel_cluster.rs` as the single accept/reject rule — still true, the rule is unchanged and only the partition's cost changed) and `docs/CONFIGURATION.md` row `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` (still true — the `user_facing.rs` change only turns an intra-doc link into plain backticks); siblings: none — no existing set gained a member, the diff only rewrites the body of one function and one rustdoc link; no hits needing change — "sentinel cluster" hits in `docs/ANALYSIS_DEEP_DIVE.md:768`, `docs/DISCOVERY_TYPES.md:1559,1588`, `docs/discoveries/observation-utilisation.md:48` and `docs/discoveries/sentinel-gating.md:11,18` describe detection behaviour, which is unchanged; no hits for the other names in README.md, `docs/` (excluding `docs/archive/`) or `*/README.md`; updated: none

## Test Plan

- [x] `cargo test --test issue_2355_sentinel_cluster_quadratic`. The ratio test
      fails on the unfixed code (ratio 15.86) and passes after the fix (4.00).
- [x] `cargo test --test detection`: 503 passed.
- [x] `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- [x] `./quality.sh`: all checks passed, `cargo doc -D warnings` included.
- Version bumped to 0.74.286 (`Cargo.toml`, `Cargo.lock`), one patch above the base branch's 0.74.285.

Branch outcomes:

- `src/analysis/detection/sentinel_cluster.rs` loop `if` (sentinel arm) is
  reached by `assess_sentinel_cluster_partitions_every_sample_once` (the even
  indices). Swapping the arms turns it red.
- The same loop's `else` (non-sentinel arm) is reached by the same test (the
  odd indices) and by `assess_sentinel_cluster_grows_linearly`.

The `if`/`else` replaces the old filter predicate one for one. It adds no new
accept or reject outcome.

Related: #2042: 🟠 Diverged sentinel-detection logic makes the error-variance
check a no-op in one copy. #873: Enable additional pedantic clippy lints for
numeric safety and code clarity.
