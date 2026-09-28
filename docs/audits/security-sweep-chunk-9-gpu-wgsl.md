# Security sweep — chunk `9`: GPU dispatch + WGSL shaders

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `9` — matches the `id` in the index.
- **Human name:** GPU dispatch + WGSL shaders — `src/analysis/gpu/**/*.rs`
  (including `queue/`) and `src/shaders/*.wgsl`.
- **Sweep date:** `2026-09-28`
- **Baseline commit:** `a7c3f65108023b93c50e9ac23e6561e0c803e22e`
- **Exposure:** `internal`
- **Swept by:** Issue #2094 (chunk 9 of the #2083 overflow tracker), planned
  under #2231 and split across its slices: shader layer #2232, evaluation
  #2237/#2238, device #2240–#2242, queue core #2243/#2244, queue lifecycle
  #2245/#2246, completeness #2249; scaffolded by Issue #2288.
- **Tracker issue:** `#2094`

### Sweep status — IN PROGRESS

This record is a scaffold. Every file row below reads `pending — <owner>` until
its owning slice sweeps it; the index's `last_swept` date marks when the
scaffold was cut, not a finished sweep. Each slice edits only its own `###`
group in `## Files swept`, its own region under `## Audit sections`, and its own
marked region of the `## Ledger` and `## Refuted / not findings` tables, so
concurrent PRs do not conflict.

### Methodology

Each slice reads its files in full at the baseline commit above and checks them
against the defect classes below, pairing every host-side buffer layout,
binding list and dispatch size with the WGSL declaration it feeds. A finding is
filed in house format (`<!-- finding-id: SEC-… -->`, `<!-- cwe: … -->`) and gets
a `## Ledger` row in the slice's region; a refuted candidate gets a
`## Refuted / not findings` row naming the code that refutes it. Line numbers
are re-verified at the baseline before they are cited.

The filename is `chunk-9`, not the zero-padded `chunk-09` the ledger README
prescribes: the chunk-9 slices already cite this path, and the ledger
contract's `normalise_id` accepts both spellings.

## Defect classes probed

- Host/shader layout mismatch — a Rust struct or buffer whose size, alignment
  or field order disagrees with the WGSL declaration it is bound to, letting a
  shader read or write past the data it was given (CWE-125 / CWE-787).
- Out-of-range invocations — a `global_invocation_id` that is not bounds-checked
  against the real element count before it indexes a storage buffer.
- Barrier divergence — `workgroupBarrier` / `storageBarrier` reached under
  non-uniform control flow.
- NaN / Inf / divide-by-zero — non-finite values produced in a kernel or its
  host-side reduction, and divisors that can be zero.
- Binding-order mismatch — a `build_compute_pipeline` caller whose buffer order
  disagrees with its shader's `@binding(n)`.
- Dispatch-size overflow — workgroup-count arithmetic that can overflow or
  exceed the device's dispatch limit.
- Timeout / availability — a GPU wait, queue or device-loss path that can hang,
  silently fall back, or report an empty result as success.

## Files swept

Line counts as at the baseline commit (`git show <baseline>:<path> | wc -l`):
24 Rust files (9,658 lines) and 10 WGSL shaders (1,338 lines), 34 in total.

Three files are not named in the #2288 owner list and are assigned to the
group of the module they test or support: `queue/fake_evaluator.rs` (the
`RequestEvaluator` test double for `executor.rs`, driven through
`execution.rs::run_work_loop`) and `queue/wedge_tests.rs` (drives that loop and
the `submission.rs` bounded wait) go to **queue-core**;
`queue/stale_skip_tests.rs` (tests `staleness.rs`) goes to **queue-lifecycle**.

### shaders

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/gpu/mod.rs` | 168 | pending — #2111 |
| `src/analysis/gpu/pipeline_builder.rs` | 157 | pending — #2111 |
| `src/analysis/gpu/shaders.rs` | 392 | pending — #2111 |
| `src/shaders/activation.wgsl` | 232 | pending — #2111 |
| `src/shaders/activation_reduce.wgsl` | 101 | pending — #2111 |
| `src/shaders/bias.wgsl` | 303 | pending — #2111 |
| `src/shaders/harmful.wgsl` | 64 | pending — #2111 |
| `src/shaders/harmful_reduce.wgsl` | 92 | pending — #2111 |
| `src/shaders/helpful.wgsl` | 96 | pending — #2111 |
| `src/shaders/helpful_reduce.wgsl` | 116 | pending — #2111 |
| `src/shaders/matching.wgsl` | 135 | pending — #2111 |
| `src/shaders/relu.wgsl` | 89 | pending — #2111 |
| `src/shaders/relu_reduce.wgsl` | 110 | pending — #2111 |

### evaluation

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/gpu/activation_evaluation.rs` | 718 | pending — #2112 |
| `src/analysis/gpu/bias_evaluation.rs` | 225 | pending — #2112 |
| `src/analysis/gpu/harmful_evaluation.rs` | 453 | pending — #2112 |
| `src/analysis/gpu/helpful_evaluation.rs` | 591 | pending — #2112 |
| `src/analysis/gpu/relu_evaluation.rs` | 242 | pending — #2112 |

### device

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/gpu/analyzer.rs` | 498 | pending — #2113 |
| `src/analysis/gpu/budget.rs` | 245 | pending — #2113 |
| `src/analysis/gpu/breaker.rs` | 511 | pending — #2113 |
| `src/analysis/gpu/device.rs` | 638 | pending — #2113 |

### queue-core

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/gpu/queue/mod.rs` | 423 | pending — #2114 |
| `src/analysis/gpu/queue/submission.rs` | 1048 | pending — #2114 |
| `src/analysis/gpu/queue/execution.rs` | 928 | pending — #2114 |
| `src/analysis/gpu/queue/executor.rs` | 137 | pending — #2114 |
| `src/analysis/gpu/queue/scheduling.rs` | 171 | pending — #2114 |
| `src/analysis/gpu/queue/fake_evaluator.rs` | 308 | pending — #2114 |
| `src/analysis/gpu/queue/wedge_tests.rs` | 489 | pending — #2114 |

### queue-lifecycle

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/gpu/queue/recovery.rs` | 248 | pending — #2115 |
| `src/analysis/gpu/queue/staleness.rs` | 266 | pending — #2115 |
| `src/analysis/gpu/heartbeat.rs` | 274 | pending — #2115 |
| `src/analysis/gpu/inflight.rs` | 166 | pending — #2115 |
| `src/analysis/gpu/queue/stale_skip_tests.rs` | 362 | pending — #2115 |

## Audit sections

Each slice writes its audit prose only between its own marker and the next.

### shaders

<!-- section: shaders -->

Pending — #2111 (shader layer, #2232).

### evaluation

<!-- section: evaluation -->

Pending — #2112 (#2237, #2238).

### device

<!-- section: device -->

Pending — #2113 (#2240–#2242).

### queue-core

<!-- section: queue-core -->

Pending — #2114 (#2243, #2244).

### queue-lifecycle

<!-- section: queue-lifecycle -->

Pending — #2115 (#2245, #2246).

## Ledger

Each slice appends rows only inside its own marked region.

| finding-id | file:line | CWE | severity | status |
| --- | --- | --- | --- | --- |
<!-- section: shaders -->
<!-- section: evaluation -->
<!-- section: device -->
<!-- section: queue-core -->
<!-- section: queue-lifecycle -->

## Refuted / not findings

Each slice appends rows only inside its own marked region.

| Candidate | Refuting file:line | Why it is not a finding |
| --- | --- | --- |
<!-- section: shaders -->
<!-- section: evaluation -->
<!-- section: device -->
<!-- section: queue-core -->
<!-- section: queue-lifecycle -->

## Outcome

In progress — no file has been swept yet. Each slice records its outcome in its
region under `## Audit sections`.

## Issues filed

None yet — the sweep is in progress; each slice lists the issues it files here.
This line is replaced by the filed issues, or by `negative-result`, when the
sweep completes.

## Verify this record

```bash
git diff a7c3f65108023b93c50e9ac23e6561e0c803e22e..HEAD -- src/analysis/gpu src/shaders
```

An empty diff means this record still describes the current code.
