## Summary

Replaced `.to_string()` with `.into_owned()` on the `Cow<str>` returned by
`String::from_utf8_lossy` in `src/analysis/cache/serialisation.rs`
(`deserialise_records`). `into_owned()` is the idiomatic way to take ownership
of a `Cow`: it returns the existing `String` unchanged when the value is already
`Cow::Owned` (the invalid-UTF-8 path) and only allocates when it is
`Cow::Borrowed`. `.to_string()` always allocates a fresh buffer and copies,
throwing away the buffer the `Cow` just produced on the owned path. This runs
once per cached record on every cache load, so it avoids an allocation-and-copy
on the invalid-UTF-8 path while being never worse on the valid path. Behaviour is
unchanged.

Closes #1456.

## Evidence

Backend/CLI change with no web interface — no screenshot applicable.

The change is behaviour-preserving; the new regression test exercises the
invalid-UTF-8 (`Cow::Owned`) branch and asserts the deserialised `neuron_uuid`
matches the lossy-decoded value (with the Unicode replacement character),
confirming `into_owned()` produces the same result as the previous
`.to_string()`.

```
test analysis::cache::serialisation::tests::deserialise_records_invalid_utf8_uuid_uses_lossy_replacement ... ok
```

All 14 `serialisation` module tests pass, and `./quality.sh` passes cleanly
(fmt, clippy `-D warnings`, check, tests, doc build, release build).

## Test Plan

- Added `src/analysis/cache/serialisation.rs::tests::deserialise_records_invalid_utf8_uuid_uses_lossy_replacement`
  — builds a serialised record whose UUID field carries invalid UTF-8 bytes
  (driving `from_utf8_lossy` into the `Cow::Owned` branch) and asserts the
  decoded `neuron_uuid` equals `String::from_utf8_lossy(..)` and contains the
  replacement character. This pins the ownership-idiom change to observable
  behaviour rather than implementation detail.
- Existing `deserialise_records_round_trip` and the truncation tests continue to
  pass, covering the normal valid-UTF-8 (`Cow::Borrowed`) path.
