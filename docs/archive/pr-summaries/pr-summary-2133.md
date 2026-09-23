# PR Summary — Issue #2133

## Summary

`NeuronJson::bias` carried only `#[serde(default)]` — no finitude check. A
JSON bias whose magnitude overflows `f32` (e.g. `1e39`, ordinary as JSON and
as `f64`) narrowed silently to `f32::INFINITY`; a `CreatureJson` built
directly in Rust could also carry `f32::NAN`, which has no JSON literal at
all. Either value then reached the arithmetic in
`dominated_branch_collapse.rs:L458`, the conversion in
`remove_neuron_bias_fold.rs:L162`, and the `to_bits()` hashing in
`neuron_fingerprint.rs:L89`, where a payload-bearing `NaN` makes the
fingerprint itself unstable.

This mirrors the Issue #2020 two-layer boundary pattern:

- **Layer 1 — deserialisation.** `NeuronJson` now has a custom `Deserialize`
  impl (`deserialise_neuron_bias`, `src/ffi_types/mod.rs`) that lets
  `f32::deserialize` perform its own narrowing and then checks
  `is_finite()`, rejecting the value with `serde::de::Error::custom` before
  it ever reaches the struct.
- **Layer 2 — the `validate_creature` gate.** `validate_neuron_biases`
  (`src/ffi_types/neuron_bias.rs`) is the belt-and-braces check for a
  `CreatureJson` built directly in Rust and handed to an entry point,
  bypassing serde entirely. It is composed into `validate_creature` after
  the existing forward-only and input-bounds gates.
- **Outbound.** `serde_json` renders a non-finite `f32` as JSON `null`,
  which would silently corrupt output. `serialise_neuron_bias` mirrors
  Issue #2020's `serialize_with` counterpart and refuses to serialise a
  non-finite bias.

Both layers phrase the fault through the shared `non_finite_bias_detail`
helper, so the two messages can never drift apart.

Closes #2133

## Evidence

Before this change, `serde_json::from_str::<CreatureJson>` accepted
`"bias": 1e39` and silently produced `f32::INFINITY`; a directly-built
`CreatureJson` carrying `f32::NAN` passed unchecked into every FFI entry
point. Both are now rejected — see the regression tests below.

Full quality gate, run stage-by-stage as bounded foreground commands (never
backgrounded or polled):

| Stage | Result |
|---|---|
| `quality/bash_syntax.sh` | ✅ |
| `quality/shellcheck.sh` | ✅ |
| `quality/cargo_install_pinning.sh` | ✅ |
| `scripts/check-pr-summary-location.sh` | ✅ |
| `cargo deny check` | ✅ advisories/bans/licenses/sources ok |
| `cargo build` (debug) | ✅ 41 crates compiled |
| `cargo fmt --all` | ✅ no changes needed |
| `cargo clippy --all-targets --all-features -- -D warnings` | ✅ no issues |
| `cargo check --all-targets --all-features` | ✅ |
| `cargo test --lib --tests --all-features` | ✅ 1595 lib tests passed; 5540 integration tests passed, 5 ignored |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | ✅ |
| `cargo build --release --lib` | ✅ |

`Cargo.toml`/`Cargo.lock` bumped `0.74.245` → `0.74.246` per `AGENTS.md`
Version Bumps.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

1. **Add a custom `Deserialize` impl for `NeuronJson` that validates the
   bias field for finitude.** ✅ MET — `deserialise_neuron_bias` in
   `src/ffi_types/mod.rs` drives `NeuronJson`'s `#[serde(deserialize_with)]`
   and rejects any non-finite narrowed value.
2. **Reject JSON with Infinity or NaN in the bias field at the FFI
   boundary.** ✅ MET — JSON has no `Infinity`/`NaN` literal, so the only
   way a JSON bias becomes non-finite is a magnitude overflowing `f32`
   (e.g. `1e39`), which layer 1 rejects. `NaN` can only arrive via a
   directly-built `CreatureJson`, which layer 2 (`validate_neuron_biases`,
   composed into `validate_creature`) rejects at every one of the five FFI
   entry points.
3. **Add a regression test: JSON `"bias": 1e400` → deserialisation
   fails.** ✅ MET — `serde_json_itself_rejects_a_bias_overflowing_the_json_exponent`
   in `tests/ffi/issue_2133_neuron_bias_finitude.rs` pins this exact case
   (rejected by `serde_json`'s own number parser, since `1e400` overflows
   `f64` before our validator ever runs); the genuine narrowing hole
   (`1e39`, finite as `f64`, infinite as `f32`) is covered separately by
   `deserialise_rejects_bias_overflowing_f32`.
4. **Verify all consumption sites receive valid finite bias values.** ✅
   MET — `entry_point_rejects_a_bias_that_overflows_f32` exercises
   `rank_focus_neurons_internal` end-to-end; `composed_gate_reports_the_bounds_fault_before_the_bias_fault`
   pins gate ordering; the arithmetic/conversion/hashing consumption sites
   (`dominated_branch_collapse.rs:L458`, `remove_neuron_bias_fold.rs:L162`,
   `neuron_fingerprint.rs:L89`) can no longer receive a non-finite bias
   from any of the five JSON-text FFI entry points, since layer 1 rejects
   it before `validate_creature` ever runs.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

`CODING-STANDARDS.md` does not exist at this repo's root; `AGENTS.md` and
`CONTRIBUTING.md` are the governing documents and were checked instead.

- **Fixed:** gate-ordering test added so a creature violating both bounds
  (Issue #1867) and bias (Issue #2133) is pinned to report the bounds fault
  first, matching the existing widest-scope-fault-first convention.
- **Fixed:** `serialise_neuron_bias` added as the outbound counterpart to
  the inbound `deserialise_neuron_bias`, mirroring Issue #2020's
  `serialize_with`/`deserialize_with` pairing — an infinite/NaN bias can no
  longer be silently written out as JSON `null`.
- **Fixed:** the rejection detail message was trimmed to drop
  internal module-name references that added no value to a caller-facing
  error; confirmed no test asserts on those substrings.
- **Deferred (infeasible):** validating at every downstream consumption
  site individually was considered and rejected — the two-layer FFI
  boundary gate is the established pattern (Issue #2020) and revalidating
  in three separate analysis modules would duplicate the check without
  adding safety, since no bias can reach them non-finite once the boundary
  rejects it.
- `Cargo.toml`/`Cargo.lock` version bump applied per `AGENTS.md` Version
  Bumps, since this diff is landing outside the CI auto-bump flow.
- `cargo clippy -D warnings` clean with no new `#[allow]`; the deserialiser
  lets `f32::deserialize` perform its own narrowing rather than casting by
  hand, avoiding `cast_possible_truncation`.
- Australian English used throughout comments and documentation.

## Test Plan

- `cargo test --lib --tests --all-features -- --test-threads=2` — 1595 lib
  tests passed; 5540 integration tests passed, 5 ignored (pre-existing,
  unrelated to this change).
- New regression suite `tests/ffi/issue_2133_neuron_bias_finitude.rs` (13
  tests): serde boundary (`1e400` rejected by `serde_json` itself, `1e39`
  rejected by the new validator, finite biases and a missing/defaulted bias
  both still accepted), Rust-constructed creatures (`NaN`/`Infinity`
  rejected, finite values accepted), outbound serialisation
  (`Infinity`/`NaN` refused, finite values round-trip), and the shipped
  entry-point test via `rank_focus_neurons_internal`.
- Extended `tests/ffi/issue_2046_creature_validation_helper.rs` with
  `composed_gate_reports_the_bounds_fault_before_the_bias_fault`, pinning
  gate composition order.
- Every existing test continues to pass unmodified — no test was removed
  or commented out.
