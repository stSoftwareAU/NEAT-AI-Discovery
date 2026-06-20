## Summary

Replaced `.to_string()` with `.into_owned()` on the `Cow<str>` returned by
`String::from_utf8_lossy` when decoding a cached record's `neuron_uuid` in
`src/analysis/cache/serialisation.rs:105`. `into_owned()` is the idiomatic way
to take ownership of a `Cow`: it returns the existing `String` unchanged when
the `Cow` is already `Owned` (the invalid-UTF-8 path) and only allocates when
`Borrowed`. This saves one allocation-and-copy on the invalid-UTF-8 branch of
the per-record cache-deserialisation loop and is never worse than `.to_string()`.

Closes #1456.

## Evidence

Backend/library change only — no web interface to screenshot. Verified via
`./quality.sh < /dev/null` (fmt, clippy with `-D warnings`, check, test, release
build) which passed cleanly: `✅ All quality checks passed!`

Behaviour is unchanged on the normal (valid UTF-8) path and on the lossy
(invalid UTF-8) path — both produce the same decoded `String`. The new
regression test asserts the lossy path still decodes correctly.

## Test Plan

- Added `src/analysis/cache/serialisation.rs::deserialise_records_invalid_utf8_uuid`
  — builds a record with invalid UTF-8 bytes (`0xFF`, `0xFE`) in the UUID field
  and asserts the bytes are decoded lossily to the U+FFFD replacement character,
  exercising the `Cow::Owned` branch that `into_owned()` optimises.
- Existing `deserialise_records_round_trip` and the truncation tests continue to
  pass, confirming the valid-UTF-8 path and error paths are unaffected.
