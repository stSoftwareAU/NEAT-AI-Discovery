# PR Summary — Issue #2136 (SEC-2090-05: Temperature Field Infinity)

## Summary

Closes #2136.

`temperature` was the last float on the analysis FFI boundary with no finitude
check. It was protected only by the downstream range clamps in
`analysis::constants::temperature`, which is inconsistent with every other float
field on that boundary (Issues #2132, #2133, #2134) and leaves NaN unhandled —
`f32::clamp` propagates NaN.

This adds `deserialise_temperature` in `src/ffi_types/mod.rs` and wires it via
`#[serde(deserialize_with = …)]` onto the `temperature` field of all four
analysis request structs in `src/ffi_types/requests.rs`
(`AnalyzeParallelInput`, `AnalyzeSynapsesInput`, `AnalyzeNeuronsInput`,
`AnalyzeAllInput`). A non-finite temperature is now rejected at the boundary
with an error naming the field and Issue #2136, instead of being silently
narrowed to `Infinity`.

```mermaid
flowchart LR
    J["JSON payload<br/>temperature"] --> P{"serde_json<br/>number parser"}
    P -- "1e400, Infinity, NaN tokens" --> R1["rejected — no such JSON literal"]
    P -- "finite f64 (1e39, 1e300)" --> D{"deserialise_temperature<br/>(Issue #2136)"}
    D -- "narrows to ±inf" --> R2["rejected — 'temperature must be finite'"]
    D -- "finite f32" --> S["scale_*_by_temperature<br/>clamp [0.01, 5.0]"]
    S --> A["analysis"]
```

The downstream clamps are **kept** deliberately — see acceptance criterion 3
below.

### Change surface

```text
 Cargo.lock                                   |   2 +-
 Cargo.toml                                   |   2 +-
 docs/FFI_API.md                              |  13 ++
 src/analysis/constants/temperature.rs        |  11 ++
 src/ffi_types/mod.rs                         |  26 +++
 src/ffi_types/requests.rs                    |  35 +++-
 tests/ffi/issue_2136_temperature_finitude.rs | 273 +++++++++++++++++++++++++++
 tests/ffi/main.rs                            |   1 +
```

## Evidence

**Rejection mechanisms — measured, not assumed.** JSON has no `Infinity`/`NaN`
literal, so the issue's stated reachability premise (`1e400` →
`f32::INFINITY`) does not hold: `serde_json` refuses `1e400`, `-1e400` and the
bare `Infinity` / `-Infinity` / `NaN` tokens before any custom validator runs.
The genuine hole is a magnitude that is finite as an `f64` but saturates when
narrowed to `f32` — `1e39`, `-3.5e38`, `1e300`. Both classes are covered by the
new suite.

**Pre-fix (RED).** With the test file in place and the `deserialize_with`
attributes absent, `deserialise_rejects_f32_saturating_temperature` and
`analyze_parallel_rejects_non_finite_temperature` failed: `1e39` deserialised
to `inf` and `analyze_parallel_internal` returned `success: true`.

**Post-fix (GREEN).**

| Check | Command | Result |
| --- | --- | --- |
| New regression suite | `cargo test --test ffi issue_2136` | 7 passed, 217 filtered out |
| Existing clamp behaviour | `cargo test --test analysis issue_1020` | 22 passed, 634 filtered out |
| Formatting | `cargo fmt --all -- --check` | clean |
| Lints | `cargo clippy --all-features --all-targets` | 0 warnings, 0 errors |
| Full gate | `timeout 1800 ./quality.sh < /dev/null` | ✅ All quality checks passed! (first attempt) |

The gate built `neat_ai_discovery v0.74.248` — "Finished `release` profile
[optimized] target(s) in 2m 15s".

**Version bump.** `Cargo.toml` `0.74.247 → 0.74.248`, per the `AGENTS.md`
mandate that any code change bumps the crate version. `Cargo.lock` picked up the
matching one-line change during the build.

**No screenshots.** This is a backend Rust/FFI change with no web interface or
rendered surface; the command output above is the evidence.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

1. **Add custom `Deserialize` impl that validates temperature field for
   finitude** — reviewer: `met` — evidence: `deserialise_temperature()` at
   `src/ffi_types/mod.rs:384-397`, wired at `src/ffi_types/requests.rs:148-152`
   (Parallel), `:237-241` (Synapses), `:280-284` (Neurons), `:347-351` (All). It
   is a `deserialize_with` helper rather than a hand-written `Deserialize` impl,
   which satisfies the intent and matches the SEC-2090-01..04 pattern.
2. **Reject JSON with Infinity or NaN in temperature at FFI boundary** —
   reviewer: `met` — evidence: tests
   `deserialise_rejects_f32_saturating_temperature`
   (`tests/ffi/issue_2136_temperature_finitude.rs:157`),
   `deserialise_rejects_bare_infinity_and_nan_tokens` (`:144`), and end-to-end
   `analyze_parallel_rejects_non_finite_temperature` (`:252`).
3. **Remove downstream clamps once FFI-boundary validation is in place** —
   reviewer: `missing` — reason: the diff deliberately keeps all three clamps and
   only adds justifying doc comments (`src/analysis/constants/temperature.rs:130-134`,
   `:148-150`, `:166-168`); the rationale is defensible and documented at
   `docs/FFI_API.md:189-195`, but the criterion as written is not satisfied and
   the deviation should be signed off explicitly.

   **Deviation, signed off explicitly.** Removing the clamps was rejected for
   four independent reasons: (a) they are *range* enforcement, not infinity
   mitigation — removing them makes `scale_threshold_by_temperature(0.05, 0.0)`
   divide by zero and return `Infinity`, reintroducing the exact defect class
   this issue exists to eliminate; (b) three existing tests in
   `tests/analysis/issue_1020_temperature_scheduling.rs`
   (`extreme_temperatures_are_clamped`,
   `zero_temperature_does_not_cause_division_by_zero`,
   `negative_temperature_is_clamped_to_minimum`) pin the clamping, and existing
   tests must not be deleted; (c) `f32::clamp` propagates NaN, so the clamp was
   never a NaN mitigation — the boundary check is strictly additive, not a
   replacement; (d) the sole production consumer,
   `collect_and_process_helpful_results`, passes `ctx.temperature` from a
   context the boundary validator does not exclusively own, so defence in depth
   still applies.
4. **Add regression test: JSON `"temperature": 1e400` → deserialization fails** —
   reviewer: `met` — evidence: `deserialise_rejects_f64_overflowing_temperature`
   at `tests/ffi/issue_2136_temperature_finitude.rs:126`. The reviewer notes the
   issue's stated reachability premise ("1e400 → f32::INFINITY") is contradicted
   by the evidence — `serde_json` rejects `1e400` itself — and the fix covers the
   real (f32-narrowing) variant instead.
5. **Verify analysis still produces correct results with valid finite
   temperature values** — reviewer: `partial` — evidence:
   `tests/ffi/issue_2136_temperature_finitude.rs:168` (finite values accepted),
   `:182` (missing field defaults), `:200` (scaling identity, ordering and range
   across `[MIN_TEMPERATURE, 0.5, DEFAULT_TEMPERATURE, 2.0, MAX_TEMPERATURE]`) —
   reason: verification stops at the scaling helpers and the parse round-trip;
   no test in the diff runs an actual analysis to completion with a finite
   temperature and checks its output.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

`CODING-STANDARDS.md` does not exist at this repository's root (re-verified this
run), so the diff was reviewed against the repository's actual standards
documents: `AGENTS.md` and `CONTRIBUTING.md`.

**Findings addressed in this PR**

- **violation — missing PR summary file** (`CONTRIBUTING.md:483`, enforced by
  quality-gate step 4 `./scripts/check-pr-summary-location.sh`,
  `CONTRIBUTING.md:158`). **Resolved by this file.**
- **concern — `docs/FFI_API.md` overstated the FFI surface** by naming four
  camelCase entry points (`analyzeParallel`, `analyzeSynapses`,
  `analyzeNeurons`, `analyzeAll`), clashing with the Validated FFI Surface table
  at `docs/FFI_API.md:1326-1347`. **Confirmed independently**: enumerating
  `extern "C" fn` across `src/ffi/` yields 22 entry points and no
  `analyze_synapses` / `analyze_neurons` / `analyze_all`; only `analyze_parallel`
  (`src/ffi/analysis.rs:162`) deserialises an analysis-request payload
  (`src/ffi_internal/analysis.rs:27`), the other three structs being built in
  Rust at `src/ffi_internal/analysis.rs:543` and
  `src/analysis/orchestration.rs:832,848`. **Resolved by rewording that bullet**
  to name the four request *structs* and state which single one is a live FFI
  boundary.

**Findings accepted as-is, with reasoning**

- **concern — the three non-`AnalyzeParallelInput` guards are unreachable in
  production** (near the `AGENTS.md` "Dead Levers" guidance and
  `CONTRIBUTING.md:329-333` on over-engineering). Accepted: the issue names
  "`AnalyzeParallelInput::temperature` and variants" explicitly, and a guard on
  every struct carrying the field is cheaper than a future struct being promoted
  to an entry point without one. The doc reword makes the asymmetry visible
  rather than implied.
- **concern — no test pins `scale_threshold_by_temperature(x, 0.0)` or a
  negative temperature staying finite**, which is the load-bearing claim of the
  new comments. Accepted: `tests/analysis/issue_1020_temperature_scheduling.rs`
  already pins exactly these cases (`zero_temperature_does_not_cause_division_by_zero`,
  `negative_temperature_is_clamped_to_minimum`), verified green (22 passed);
  adding a third copy in this suite would duplicate them.
- **concern — NaN is guarded only on the JSON door**, unlike #2133's
  `validate_neuron_biases`, so a Rust caller using the re-exported types
  (`src/lib.rs:50,52`) can still push NaN. Accepted as out of scope: this issue
  is scoped to the FFI boundary. Noted here rather than silently widened.
- **concern — DRY**: `deserialise_temperature` (`src/ffi_types/mod.rs:384-395`)
  duplicates `deserialise_finite_f32` (`:352-363`) apart from its message.
  Accepted: the shared detail helper hardcodes "neuron data" and "Issue #2134",
  which reads wrongly for temperature; this mirrors `deserialise_synapse_weight`,
  the precedent set by #2132.
- **concern — duplicated assertions** at
  `tests/ffi/issue_2136_temperature_finitude.rs:205-220` vs
  `src/analysis/constants/temperature.rs:243-262`. Accepted: the new assertions
  exercise the post-deserialisation path required by criterion 5; the in-module
  tests cover the helpers in isolation.

**Passes**

Australian English throughout (`CONTRIBUTING.md:293`); crate version bumped
(`Cargo.toml:3`); tests call real functions and assert on real behaviour,
including guard wiring at the shipped entry point (`CONTRIBUTING.md:418-429`),
7 passed; fail-loud error handling; documentation obligation met; citations by
symbol not line number (`CONTRIBUTING.md:318-325`); `cargo fmt --check` clean and
`cargo clippy --all-features --all-targets` reporting zero warnings.

## Test Plan

New suite `tests/ffi/issue_2136_temperature_finitude.rs` (7 tests, registered at
`tests/ffi/main.rs:30`):

1. `deserialise_rejects_f64_overflowing_temperature` — `1e400`, `-1e400`
   rejected by all four request structs (criterion 4).
2. `deserialise_rejects_bare_infinity_and_nan_tokens` — `Infinity`,
   `-Infinity`, `NaN` rejected (criterion 2).
3. `deserialise_rejects_f32_saturating_temperature` — `1e39`, `-3.5e38`,
   `1e300` rejected with a message containing `finite`, `temperature` and
   `Issue #2136`. **RED before the fix.**
4. `deserialise_accepts_finite_temperatures` — `1.0`, `0.5`, `2.0`, `1.25`,
   `0.01`, `5.0`, `3.4e38` all parse (no over-rejection).
5. `deserialise_defaults_missing_temperature` — an omitted field still yields
   `DEFAULT_TEMPERATURE`.
6. `finite_temperatures_still_scale_analysis_thresholds` — identity at the
   default, `hot < base < cold` ordering, finiteness and range across the valid
   band (criterion 5).
7. `analyze_parallel_rejects_non_finite_temperature` — end-to-end through
   `analyze_parallel_internal`, asserting `success == false` and an error naming
   `temperature` and `Issue #2136`. **RED before the fix.**

Regression coverage for the retained clamps: `cargo test --test analysis
issue_1020` (22 passed), which includes the three tests pinning clamping of
finite out-of-range temperatures.

Run:

```bash
cargo test --test ffi issue_2136 < /dev/null
cargo test --test analysis issue_1020 < /dev/null
./quality.sh < /dev/null
```
