# Security sweep chunk 4 — remaining FFI request/response types (`src/ffi_types`)

## Summary

Two defects found in the `src/ffi_types` sweep scope are fixed here, both in
the deserialise-without-invariant / duplicate-identifier classes the chunk
issue names. Part of #2090.

**SEC-2090-01 — `ReadDiscoveryInput.neuron_uuid` documented a contract it did
not enforce.** The field carries a doc comment stating that the identifier
"must be a stable UUID or descriptive identifier — numeric integer IDs are not
permitted (Issue #952)", but it was a plain `#[derive(Deserialize)]`
`String` with no `deserialize_with` guard. Every other neuron-identity field
reaching the FFI (`NeuronJson::uuid`, `NeuronData::neuron_uuid`) routes through
`ffi_types::deserialise_neuron_uuid`; this one did not, so a stringified
integer arrived intact and was used for exact-match filtering against records
that can never contain one. Fixed by attaching the same
`#[serde(deserialize_with = "super::deserialise_neuron_uuid")]` guard, so the
rejection happens during deserialisation — before any path resolution,
allocation or Parquet read.

**SEC-2090-02 — duplicate neuron UUIDs silently shadowed one another in
`validate_forward_only_synapses`.** The forward-only gate built its
`uuid -> index` map with `HashMap::insert`, which is last-wins: a creature
repeating a UUID discarded the earlier neuron's index without a word. Because
the recurrence check compares the resolved source and destination indices, a
genuinely recurrent synapse could resolve against the surviving (later)
occurrence's index and validate as forward-only — defeating the Issue #1184
gate with input the caller fully controls. Fixed by rejecting a repeated UUID
outright while the index is being built, as `DiscoveryError::InvalidInput`
naming both colliding indices.

## Evidence

Backend/FFI change with no web interface, so there is no screenshot to
capture. The evidence is the red/green regression run below.

Both guards are asserted at the shipped entry point *and* at the guard itself,
per the issue's Failure Detection note that "a test that merely asserts `Err`
on the happy wrapper is not evidence".

```mermaid
sequenceDiagram
    participant Host as Deno host
    participant FFI as ffi::read_discovery_records
    participant De as serde — ReadDiscoveryInput
    participant Guard as ffi_types::deserialise_neuron_uuid
    Host->>FFI: {"neuron_uuid":"12345", …}
    FFI->>De: from_str
    De->>Guard: neuron_uuid field
    Guard--xDe: Err — numeric integer neuron ID
    De--xFFI: deserialisation error
    FFI-->>Host: success false, error names the UUID-only contract
```

Observed **red** with only the two source guards reverted
(`src/ffi_types/requests.rs` attribute removed, the duplicate-UUID check
removed from `src/ffi_types/forward_only_validation.rs`) — the tests unchanged:

```text
$ cargo test --test ffi -- ffi_read_discovery_rejects_numeric_neuron_uuid \
      read_discovery_input_deserialise ffi_record_discovery_rejects_duplicate_neuron_uuid
test issue_950_numeric_neuron_ids::read_discovery_input_deserialise_rejects_numeric_neuron_uuid ... FAILED
test issue_950_numeric_neuron_ids::ffi_read_discovery_rejects_numeric_neuron_uuid ... FAILED
test issue_1184_recurrent_synapse_rejection::ffi_record_discovery_rejects_duplicate_neuron_uuid ... FAILED

numeric neuron_uuid must fail ReadDiscoveryInput deserialisation
error must explain the UUID-only contract (Issue #952): Parquet file removed: …
recording with a duplicate neuron uuid must fail:
  {"success":true,"schemaVersion":"2","tempDir":"/tmp/.tmpxhjNUD","file":"discovery_data.parquet"}

test result: FAILED. 0 passed; 3 failed

$ cargo test --lib -- forward_only_validation::tests::duplicate
test ffi_types::forward_only_validation::tests::duplicate_neuron_uuid_is_rejected ... FAILED
test ffi_types::forward_only_validation::tests::duplicate_neuron_uuid_masking_a_back_edge_is_rejected ... FAILED

test result: FAILED. 0 passed; 2 failed
```

Note the unfixed `ffi_read_discovery_rejects_numeric_neuron_uuid` failure: the
numeric ID was accepted by deserialisation and the call proceeded all the way
to the Parquet layer, which is the defect in one line.

Observed **green** after the fix, with the same commands:

```text
$ cargo test --test ffi -- ffi_read_discovery_rejects_numeric_neuron_uuid \
      read_discovery_input_deserialise ffi_record_discovery_rejects_duplicate_neuron_uuid
test issue_950_numeric_neuron_ids::ffi_read_discovery_rejects_numeric_neuron_uuid ... ok
test issue_950_numeric_neuron_ids::read_discovery_input_deserialise_rejects_numeric_neuron_uuid ... ok
test issue_1184_recurrent_synapse_rejection::ffi_record_discovery_rejects_duplicate_neuron_uuid ... ok
test result: ok. 3 passed; 0 failed

$ cargo test --lib -- forward_only_validation::tests
test result: ok. 9 passed; 0 failed
```

**The original trigger is closed with no trivial bypass — SEC-2090-01.** The
trigger was the JSON body `{"parquet_file": …, "neuron_uuid": "12345"}`
reaching `read_discovery_records`. It is now rejected inside
`serde::Deserialize for ReadDiscoveryInput`, by the same
`ffi_types::deserialise_neuron_uuid` that already guards `NeuronJson::uuid`
and `NeuronData::neuron_uuid`, so the three neuron-identity fields share one
predicate and cannot drift apart. There is no equivalent bypass: the guard runs
on the *deserialised* `String`, so it sees the value after JSON unescaping —
`"12345"` and `"12345"` are the same `&str` by the time `is_numeric_id`
sees them — and `is_numeric_id` tests `bytes().all(is_ascii_digit)`, a byte
predicate with no locale, Unicode-digit or whitespace-trim behaviour to exploit.
A non-string JSON type (`"neuron_uuid": 12345`) fails earlier still, in
`String::deserialize`. The field is not `#[serde(default)]`, so omitting it is
a deserialisation error rather than a silent empty-string pass, and there is no
second constructor for `ReadDiscoveryInput` — the type is only ever built by
`serde_json::from_str` on host input, so the guard is unavoidable.

**The original trigger is closed with no trivial bypass — SEC-2090-02.** The
trigger was a creature whose `neurons` array repeats a `uuid` while a synapse
targets it. The check now runs while the index map is built, on the first
repeat, *before* any synapse is inspected — so the masking can never occur
regardless of neuron ordering, of which occurrence the duplicate is, or of how
many duplicates there are. Rejecting rather than merging is what removes the
bypass class: there is no "which occurrence wins" decision left to get wrong.
`validate_forward_only_synapses` is the single chokepoint all three FFI
creature entry points route through (`record_discovery`, `analyze_parallel`,
`rank_focus_neurons` — each pinned by an existing test in
`tests/ffi/issue_1184_recurrent_synapse_rejection.rs`), so there is no
unguarded sibling path. UUID comparison is exact `&str` equality, the same
equality the `HashMap` used, so no normalisation gap exists between the guard
and the behaviour it protects.

## Test Plan

- **Added** `tests/ffi/issue_950_numeric_neuron_ids.rs::ffi_read_discovery_rejects_numeric_neuron_uuid`
  — drives the shipped `read_discovery_records` entry point with
  `"neuron_uuid": "12345"` and asserts the FFI response is
  `success: false` with an error naming the UUID-only contract. Reproduces
  SEC-2090-01: **fails against the unfixed code** (the numeric ID is accepted
  and the call reaches the Parquet layer) and **passes after the fix**.
- **Added** `tests/ffi/issue_950_numeric_neuron_ids.rs::read_discovery_input_deserialise_rejects_numeric_neuron_uuid`
  — asserts on the guard itself rather than the wrapper, per the issue's
  Failure Detection note: `serde_json::from_str::<ReadDiscoveryInput>` on the
  hostile body must return `Err`. **Fails against the unfixed code**
  (deserialisation succeeds) and **passes after the fix**.
- **Added** `tests/ffi/issue_1184_recurrent_synapse_rejection.rs::ffi_record_discovery_rejects_duplicate_neuron_uuid`
  — records a creature repeating the uuid `"shared"` at indices 0 and 2 with a
  synapse into it, and asserts `success: false` with
  `errorKind: "data_validation"`. Reproduces SEC-2090-02: **fails against the
  unfixed code** — the recording succeeds and returns
  `{"success":true,…}` — and **passes after the fix**.
- **Added** `src/ffi_types/forward_only_validation.rs::duplicate_neuron_uuid_is_rejected`
  — unit-level pin that the guard fires while the index is built, with no
  synapses present at all. Red before the fix, green after.
- **Added** `src/ffi_types/forward_only_validation.rs::duplicate_neuron_uuid_masking_a_back_edge_is_rejected`
  — the masking scenario itself: uuid `"a"` at indices 0 and 2 with a synapse
  `b -> a` that would resolve to the forward pair `1 -> 2` under last-wins.
  Red before the fix, green after.
- **Unchanged** — no existing test was modified, commented out or removed; the
  full `forward_only_validation` unit suite (9 tests) and the full `ffi`
  integration suite still pass.
