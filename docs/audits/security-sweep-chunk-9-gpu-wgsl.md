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
| `src/shaders/activation.wgsl` | 232 | finding filed — #2308 (`is_finite_value` at L35 is a float self-comparison fast-math may fold, so the L223 output guard can pass an overflowed Inf/NaN as `valid`); `sample_count` guard L195, no barrier, unused `epsilon` refuted |
| `src/shaders/activation_reduce.wgsl` | 101 | audited, no finding — zero-padded load L76–L80 keeps every read in bounds and adds a neutral element; barriers L83/L94 sit under uniform control flow |
| `src/shaders/bias.wgsl` | 303 | finding filed — #2308 (the `is_finite_value` skips at L253/L264/L273 are its only non-finite handling); `in_range` L216 guards every `bias_idx` access, barriers L244/L283 are uniform, unused `epsilon` L22 and the L228 ceil-div refuted |
| `src/shaders/harmful.wgsl` | 64 | audited, no finding — `length` guard L40 before any access, no barrier, `epsilon` comparisons at L50 reject NaN, no division |
| `src/shaders/harmful_reduce.wgsl` | 92 | audited, no finding — zero-padded load L67–L71; barriers L74/L85 sit under uniform control flow |
| `src/shaders/helpful.wgsl` | 96 | audited, no finding — `length` guard L48 before any access, no barrier, `epsilon` comparisons at L67/L74/L79 reject NaN, no division |
| `src/shaders/helpful_reduce.wgsl` | 116 | audited, no finding — zero-padded load L91–L95; barriers L98/L109 sit under uniform control flow |
| `src/shaders/matching.wgsl` | 135 | dead — no `include_str!` in `shaders.rs`, absent from `ALL_SHADERS`, no pipeline builds it; removal tracked by #2309, file kept |
| `src/shaders/relu.wgsl` | 89 | finding filed — #2308 (the L63 `is_finite_value` input skip); `length` guard L45, no barrier, `epsilon` guards L73/L81, unused `threshold` refuted |
| `src/shaders/relu_reduce.wgsl` | 110 | unused — registered as `RELU_REDUCE_SHADER` (`shaders.rs:91`) and in `ALL_SHADERS` (`shaders.rs:214`) but no `build_compute_pipeline` call builds it; kernel itself is sound (zero-padded load, barriers L92/L103 uniform); removal tracked by #2309, file kept |

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

#### Host ↔ WGSL struct parity (Issue #2289)

The 12 `#[repr(C)] bytemuck::Pod` structs in
`src/analysis/samples/gpu_types.rs` against every WGSL struct that mirrors them
(23 declarations across 9 shaders; `matching.wgsl` is out of scope here and
is covered by #2232). Field order, scalar types and per-field offsets were
compared member by member; the WGSL size/alignment column is naga 30's `Layouter` output, the
same layout wgpu validates against. No struct uses a `vec3` (or any vector or
nested struct) — every member is a 4-byte `f32`/`u32`, so every WGSL struct
aligns to 4 and its size is `4 × members`, exactly as `repr(C)` lays it out.
Pinned by `tests/issue_2289_gpu_struct_layout.rs`, which fails CI if either
side drifts.

| Struct | Rust file:line | WGSL file:line | Rust size/align | WGSL size/align | Verdict |
| --- | --- | --- | --- | --- | --- |
| `GpuHelpfulSample` | `src/analysis/samples/gpu_types.rs:13` | `helpful.wgsl:1`, `relu.wgsl:1`, `activation.wgsl:1`, `bias.wgsl:4` (`HelpfulSample`); `harmful.wgsl:1` (`HarmfulSample`) | 8 / 4 | 8 / 4 | parity |
| `HelpfulContribution` | `src/analysis/samples/gpu_types.rs:30` | `helpful.wgsl:6`, `helpful_reduce.wgsl:11` | 48 / 4 | 48 / 4 | parity |
| `HelpfulUniforms` | `src/analysis/samples/gpu_types.rs:48` | `helpful.wgsl:21` | 16 / 4 | 16 / 4 | parity |
| `HarmfulContribution` | `src/analysis/samples/gpu_types.rs:58` | `harmful.wgsl:6`, `harmful_reduce.wgsl:11` | 16 / 4 | 16 / 4 | parity |
| `HarmfulUniforms` | `src/analysis/samples/gpu_types.rs:68` | `harmful.wgsl:13` | 16 / 4 | 16 / 4 | parity |
| `ReluContribution` | `src/analysis/samples/gpu_types.rs:78` | `relu.wgsl:6`, `relu_reduce.wgsl:11` | 40 / 4 | 40 / 4 | parity |
| `ReluUniforms` | `src/analysis/samples/gpu_types.rs:94` | `relu.wgsl:19` | 16 / 4 | 16 / 4 | parity |
| `BiasResult` | `src/analysis/samples/gpu_types.rs:104` | `bias.wgsl:9` | 16 / 4 | 16 / 4 | parity |
| `BiasUniforms` | `src/analysis/samples/gpu_types.rs:126` | `bias.wgsl:16` | 32 / 4 | 32 / 4 | parity |
| `ActivationOutput` | `src/analysis/samples/gpu_types.rs:140` | `activation.wgsl:6`, `activation_reduce.wgsl:11` | 28 / 4 | 28 / 4 | parity (see decision) |
| `ActivationUniforms` | `src/analysis/samples/gpu_types.rs:153` | `activation.wgsl:16` | 28 / 4 | 28 / 4 | parity (see decision) |
| `ReductionUniforms` | `src/analysis/samples/gpu_types.rs:166` | `helpful_reduce.wgsl:26`, `harmful_reduce.wgsl:18`, `relu_reduce.wgsl:24`, `activation_reduce.wgsl:21` | 16 / 4 | 16 / 4 | parity |

WGSL paths are under `src/shaders/`. Every struct-typed `array<T>` global
(`samples`, `contributions`, `outputs`, `partial_sums`, `results`, the
`var<workgroup> shared_data` arrays and `bias.wgsl`'s `shared_samples`) has a
naga stride equal to the Rust `size_of::<T>()` the host multiplies by the
element count.

**Decision — the 28-byte `ActivationUniforms` and `ActivationOutput` are
valid.** WGSL's 16-byte rule for the `uniform` address space
(`RequiredAlignOf(S, uniform) = roundUp(16, AlignOf(S))`, and the uniform
array-stride rule) constrains a struct *nested* inside a uniform buffer and an
array *element* stored there; it does not round up the top-level struct a
`var<uniform>` binds. `activation.wgsl:31` binds `ActivationUniforms` directly
(`var<uniform> uniforms: ActivationUniforms`), and the host uploads exactly
`bytemuck::bytes_of(&uniforms)` — 28 bytes — at
`src/analysis/gpu/activation_evaluation.rs:156` and `:490`, with
`min_binding_size: None` (`src/analysis/gpu/pipeline_builder.rs:41`), so wgpu
checks the 28-byte buffer against naga's 28-byte span. `ActivationOutput` is
only ever an element of `storage`/`workgroup` arrays
(`activation.wgsl:29`, `activation_reduce.wgsl:30`, `:33`, `:39`), where the
stride is `roundUp(AlignOf(T), SizeOf(T)) = roundUp(4, 28) = 28`; the host
sizes those buffers as `size_of::<ActivationOutput>() * n`
(`activation_evaluation.rs:211`, `:277`, `:517`, `:539`, `:596`, `:646`).
naga's validator (`ValidationFlags::all()`) accepts both shaders — it would
report `Disalignment::ArrayStride` or `MemberOffsetAfterStruct` if either rule
were broken — and `uniform_bindings_accept_28_byte_activation_uniforms` pins
that. No padding change is needed.

**Outcome: negative result** — parity holds on all 12 structs; no ledger row
and no issue filed for struct layout.

#### Kernel bounds and arithmetic (Issue #2290)

All 10 `src/shaders/*.wgsl` kernels, read in full at the baseline. Line
numbers are unchanged at the #2289 head (`abd698d`). Every kernel declares
`@compute @workgroup_size(256)`, matching `WORKGROUP_SIZE`
(`src/analysis/gpu/shaders.rs`). Three guard shapes are used:

- **element-wise** (`activation`, `harmful`, `helpful`, `relu`, `matching`):
  `if idx >= uniforms.<count> { return; }` before the first buffer access;
- **reduction** (the four `*_reduce.wgsl`): a lane past `contribution_count`
  loads a zero struct into `shared_data` instead of reading the input, so the
  tree reduction reads only `shared_data[local_idx + stride]` with
  `local_idx < stride <= 128` (max index 255) and never returns early;
- **tiled** (`bias.wgsl`): `in_range = bias_idx < uniforms.bias_count` (L216)
  gates the per-candidate reads and writes, while every lane stays in the tile
  loop to reach both barriers.

Defence in depth: `pipeline_builder.rs:26` uses `create_shader_module`, not the
`_trusted` variant, so wgpu keeps runtime bounds checks on: naga's `Restrict`
policy for storage-buffer access on Metal (`wgpu-hal-30.0.1/src/metal/device.rs:186`)
and on Vulkan without `robustBufferAccess2`, which otherwise supplies the same
guarantee in hardware (`src/vulkan/adapter.rs:2839`–`:2844`). A missed guard
would clamp or read zero rather than read out of bounds.

| Kernel (lines) | `@workgroup_size` line | Bounds guard (`file:line`) | Barrier-safe? | NaN/Inf/div-by-zero handling | Verdict |
| --- | --- | --- | --- | --- | --- |
| `activation.wgsl` (232) | L192 | `activation.wgsl:195` (`idx >= uniforms.sample_count`) | n/a — no barrier, so the early `return` is safe | `is_finite_value` (L35) skips non-finite input (L210) and gates `valid = 1u` on the output (L223); the non-constant divisions are in `logistic`/`swish`/`softsign` with denominators `1 + exp(..)` / `1 + abs(x)` ≥ 1 (`bent_identity` divides by the constant `2.0`); `epsilon` (L21) unread (refuted) | finding — #2308 (the self-comparison guard is foldable under fast-math) |
| `activation_reduce.wgsl` (101) | L67 | zero-padding `activation_reduce.wgsl:76`–`:80` | yes — L83 at function scope, L94 in a constant-bound loop after the non-uniform `if` closes | sums only; the zero pad is the additive identity; `valid` sums ≤ 256 per workgroup, no u32 overflow; no division | clean |
| `bias.wgsl` (303) | L207 | `in_range` `bias.wgsl:216` (gates L224, L247, L287); sample load `bias.wgsl:233`; `tile_end` `bias.wgsl:248` | yes — L244/L283 sit in the tile-loop body outside `if (in_range)`; the loop bound `tile_count` (L228) derives only from a uniform | `is_finite_value` (L38) skips at L253, L264, L273; the NaN pad (L239) is never read because `tile_end` stops the inner loop at the real sample count; `epsilon` (L22) unread (refuted); L228 ceil-div cannot wrap (refuted) | finding — #2308 |
| `harmful.wgsl` (64) | L37 | `harmful.wgsl:40` (`idx >= uniforms.length`) | n/a — no barrier | `abs(..) > uniforms.epsilon` on activation and error (L50) is false for NaN, so a NaN sample contributes zeros; inputs pre-filtered finite on the host; no division | clean |
| `harmful_reduce.wgsl` (92) | L58 | zero-padding `harmful_reduce.wgsl:67`–`:71` | yes — L74 at function scope, L85 in a constant-bound loop | sums only; flag sums ≤ 256 per workgroup; no division | clean |
| `helpful.wgsl` (96) | L45 | `helpful.wgsl:48` (`idx >= uniforms.length`) | n/a — no barrier | `epsilon` comparisons at L67, L74 and L79 are false for NaN; inputs pre-filtered finite on the host; no division | clean |
| `helpful_reduce.wgsl` (116) | L82 | zero-padding `helpful_reduce.wgsl:91`–`:95` | yes — L98 at function scope, L109 in a constant-bound loop | sums only; flag sums ≤ 256 per workgroup; no division | clean |
| `matching.wgsl` (135) | L79 | `matching.wgsl:82` (`idx >= uniforms.from_count`); error reads gated by `error_idx < total_errors` (L106) | n/a — no barrier | `avg_error` division L118 guarded by `error_count > 0u` (L117); non-finite skips via `is_finite_value` (L47) | dead — never compiled into a pipeline; removal #2309 |
| `relu.wgsl` (89) | L42 | `relu.wgsl:45` (`idx >= uniforms.length`) | n/a — no barrier | `is_finite_value` (L35) input skip at L63; `relu_positive`/`relu_negative > uniforms.epsilon` at L73/L81; no division (the host's `error_activation / activation` in `relu_evaluation.rs` is guarded by `activation > EPSILON`); `threshold` (L21) unread (refuted) | finding — #2308 |
| `relu_reduce.wgsl` (110) | L76 | zero-padding `relu_reduce.wgsl:85`–`:89` | yes — L92 at function scope, L103 in a constant-bound loop | sums only; count sums ≤ 256 per workgroup; no division | unused — no pipeline builds it; removal #2309 |

**Barrier sites.** A barrier reachable only after a thread-dependent early
`return`, or inside a non-uniform branch, would be a finding. None is:

| Barrier | Control flow reaching it | Verdict |
| --- | --- | --- |
| `activation_reduce.wgsl:83` | function scope; no preceding `return`; the L76 `if/else` closes before it | uniform |
| `activation_reduce.wgsl:94` | `for` loop with constant bounds (`stride` 128 → 1); the `local_idx < stride` branch closes at L93 | uniform |
| `harmful_reduce.wgsl:74` | function scope; no preceding `return` | uniform |
| `harmful_reduce.wgsl:85` | constant-bound loop; the branch closes at L84 | uniform |
| `helpful_reduce.wgsl:98` | function scope; no preceding `return` | uniform |
| `helpful_reduce.wgsl:109` | constant-bound loop; the branch closes at L108 | uniform |
| `relu_reduce.wgsl:92` | function scope; no preceding `return` | uniform |
| `relu_reduce.wgsl:103` | constant-bound loop; the branch closes at L102 | uniform |
| `bias.wgsl:244` | tile-loop body, outside `if (in_range)`; loop bound from `uniforms.sample_count` only; `main` has no `return` | uniform |
| `bias.wgsl:283` | same loop body, after `if (in_range)` closes at L280 | uniform |

These verdicts rest on reading the code, not on a tool. naga 30 does not
enforce barrier uniformity: `test_shaders_are_valid_wgsl`
(`src/analysis/gpu/shaders.rs`) validates with `ValidationFlags::all()`, but
naga's `CONTROL_FLOW_UNIFORMITY` check raises `NonUniformControlFlow` only for
expressions that carry a requirement, such as derivatives
(`naga-30.0.1/src/valid/analyzer.rs:915`–`:927`). A barrier statement records
`WORK_GROUP_BARRIER` (`:960`) without being checked against the enclosing
control flow. A future barrier moved under a non-uniform branch would
therefore pass CI, so the table above is the record.

**Arithmetic decisions.**

- *`uniforms.epsilon` guards* — `harmful.wgsl:50` and `relu.wgsl:73`/`:81`
  (also `helpful.wgsl:67`/`:74`/`:79`) compare `>` against a host constant
  `EPSILON = 1e-8` (`src/analysis/samples/mod.rs:29`). A `>` comparison is
  false for NaN, so NaN samples fall through to the zeroed contribution.
- *`bias.wgsl` non-finite handling* — the real mechanism is the
  `is_finite_value` skip at L253 (input), L264 (activated output, charged the
  baseline error) and L273 (corrected error, charged the baseline error). The
  `epsilon` field it declares at L22 is never read (refuted below). The skips
  share the fast-math weakness filed as #2308.
- *`bias.wgsl:228` ceil-div* — `(sample_count + 255u) / 256u` would wrap for
  `sample_count > u32::MAX - 255`. It cannot: the device is requested with
  `wgpu::Limits::default()` (`src/analysis/gpu/analyzer.rs:291`, `:394`),
  whose `max_storage_buffer_binding_size` is 128 MiB
  (`wgpu-types-30.0.1/src/limits.rs:441`). At 8 bytes per
  `GpuHelpfulSample` the `samples` binding holds at most 2^24 samples, and
  `sample_count` is that buffer's own length (`bias_evaluation.rs:125`), so a
  wrapping value never reaches a dispatch. Were it reached, `tile_count`
  would be 0, no sample is read, and every candidate fails
  `min_sample_count` (L292) — fail-safe, not out of bounds.

**Dead shaders.** `matching.wgsl` is dead: `shaders.rs` has no
`include_str!` for it and it is absent from `ALL_SHADERS`, so it is not even
naga-validated. `relu_reduce.wgsl` is unused: it is registered as
`RELU_REDUCE_SHADER` (`shaders.rs:91`) and validated through `ALL_SHADERS`
(`shaders.rs:214`), but no `build_compute_pipeline` call builds it —
`relu_evaluation.rs:33` builds only `RELU_SHADER` — and only `shaders.rs` and
the #2289 layout test reference it. AGENTS.md: "Delete a never-constructed
component unless a concrete writer can be named". No writer is named, so one
removal follow-up covering both is filed as #2309; neither file is deleted
here.

**Outcome: one finding** — #2308 (`SEC-e8e1dd84a447`, CWE-754, low). Bounds
guards and barrier placement are sound in all 10 kernels.

Pending — 9a-2b (#2291): `mod.rs`, `pipeline_builder.rs`, `shaders.rs`.

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
| SEC-e8e1dd84a447 | `src/shaders/activation.wgsl:35` (also `relu.wgsl:35`, `bias.wgsl:38`) | CWE-754 | low | open — #2308 |
<!-- section: evaluation -->
<!-- section: device -->
<!-- section: queue-core -->
<!-- section: queue-lifecycle -->

## Refuted / not findings

Each slice appends rows only inside its own marked region.

| Candidate | Refuting file:line | Why it is not a finding |
| --- | --- | --- |
<!-- section: shaders -->
| Host/shader layout mismatch in any of the 12 `Pod` structs (CWE-125 / CWE-787) | `src/analysis/samples/gpu_types.rs:13`–`:172`; `tests/issue_2289_gpu_struct_layout.rs` | Every WGSL mirror has the same member names, order, scalar types, offsets, size and alignment as its Rust struct, per naga's `Layouter` (Issue #2289) |
| 28-byte `ActivationUniforms` invalid as a `var<uniform>` (not a multiple of 16) | `src/shaders/activation.wgsl:31`; `src/analysis/gpu/activation_evaluation.rs:156` | The 16-byte rounding applies to structs nested in, or arrays stored in, uniform space — not the top-level bound struct; naga validation passes and the host uploads exactly 28 bytes |
| 28-byte `ActivationOutput` gives a misaligned `array<ActivationOutput>` stride | `src/shaders/activation.wgsl:29`; `src/shaders/activation_reduce.wgsl:30` | Storage/workgroup arrays have stride `roundUp(4, 28) = 28`, matching the host `size_of::<ActivationOutput>() * n` sizing |
| A `vec3` member forcing 16-byte alignment on the WGSL side | `src/shaders/*.wgsl` struct declarations | No struct member in the 23 mirrors is a vector; `vec3<u32>` appears only as `@builtin` entry-point parameters, which carry no host layout |
| `bias.wgsl` declares `epsilon` in `BiasUniforms` but never reads it (dead config surface) | `src/shaders/bias.wgsl:22`; `src/analysis/gpu/bias_evaluation.rs:130` | The host always writes the compile-time constant `EPSILON` (`src/analysis/samples/mod.rs:29`); no env var, config field or caller can set it, so it is not an operator lever that silently does nothing. Non-finite handling is the `is_finite_value` skips at `bias.wgsl:253`/`:264`/`:273` (their fast-math weakness is #2308). The field is layout padding pinned by `tests/issue_2289_gpu_struct_layout.rs`; dropping it would change the 32-byte `BiasUniforms` for no safety gain |
| u32 wrap in the `bias.wgsl` tile-count ceil-div `(sample_count + TILE_SIZE - 1u) / TILE_SIZE` | `src/shaders/bias.wgsl:228`; `src/analysis/gpu/analyzer.rs:291`, `:394` | The device uses `wgpu::Limits::default()`, whose 128 MiB `max_storage_buffer_binding_size` caps the 8-byte-per-sample `samples` binding at 2^24 elements; `sample_count` is that buffer's own length (`bias_evaluation.rs:125`), far below `u32::MAX - 255`. Even if it wrapped, `tile_count = 0` reads nothing and `min_sample_count` (`bias.wgsl:292`) rejects every candidate |
| `bias.wgsl` NaN tile padding read as a sample | `src/shaders/bias.wgsl:239`, `:248` | Padded slots sit at `shared_samples[i]` with `i >= tile_end`, and the inner loop stops at `tile_end = min(TILE_SIZE, sample_count - tile * TILE_SIZE)`, so the pad is never read; its NaN is a second guard, not the only one |
| `workgroupBarrier()` reached under non-uniform control flow (10 sites) | `activation_reduce.wgsl:83`/`:94`, `harmful_reduce.wgsl:74`/`:85`, `helpful_reduce.wgsl:98`/`:109`, `relu_reduce.wgsl:92`/`:103`, `bias.wgsl:244`/`:283` | No kernel with a barrier returns early; each barrier is at function scope or in a loop whose bound is a constant or a uniform, after the lane-dependent `if` has closed (per-site table in the shaders audit section) |
| Out-of-range `global_invocation_id` indexes a storage buffer | `activation.wgsl:195`, `harmful.wgsl:40`, `helpful.wgsl:48`, `relu.wgsl:45`, `bias.wgsl:216`/`:233`, `*_reduce.wgsl` zero-pad loads | Every element-wise kernel returns before its first access; reductions read the input only when `global_idx < contribution_count` and index `shared_data` at most 255; `bias.wgsl` gates `bias_candidates`/`results` on `in_range` and `samples` on `sample_load_idx < sample_count` |
| `activation.wgsl` declares `epsilon` but never reads it | `src/shaders/activation.wgsl:21`; `src/analysis/gpu/activation_evaluation.rs:150` | Host-written constant, not an operator lever; the kernel's non-finite handling is `is_finite_value` (L210/L223, #2308) |
| `relu.wgsl` declares `threshold` but never reads it, so the caller's threshold is dropped | `src/shaders/relu.wgsl:21`; `src/analysis/synapse/relu_evaluation.rs:88`, `:135` | `threshold` is an improvement floor applied on the host after the GPU returns (`best_improvement = threshold`), not a per-sample filter, so the kernel has no use for it |
| Division by zero in a kernel | `activation.wgsl:77`/`:80`/`:150`, `bias.wgsl:80`/`:83`/`:157`, `matching.wgsl:118` | Activation-function denominators are `1 + exp(..)` or `1 + abs(x)`, never below 1 for finite input; `matching.wgsl`'s average is guarded by `error_count > 0u` (L117) and the kernel is never compiled |
<!-- section: evaluation -->
<!-- section: device -->
<!-- section: queue-core -->
<!-- section: queue-lifecycle -->

## Outcome

In progress — the 10 `src/shaders/*.wgsl` kernels are swept (#2290: one
finding, #2308); every other file is pending its slice. Each slice records its
outcome in its region under `## Audit sections`.

## Issues filed

The sweep is in progress; each slice lists the issues it files here.

- #2308 — `SEC-e8e1dd84a447` (CWE-754, low): WGSL `is_finite_value` guards are
  float self-comparisons fast-math may fold away (shaders slice, #2290).
- #2309 — removal follow-up for the dead `matching.wgsl` and unused
  `relu_reduce.wgsl` (not a security finding; shaders slice, #2290).

## Verify this record

```bash
git diff a7c3f65108023b93c50e9ac23e6561e0c803e22e..HEAD -- src/analysis/gpu src/shaders
```

An empty diff means this record still describes the current code.
