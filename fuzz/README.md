# Parquet decode budget fuzzing

Run the bounded reader target with cargo-fuzz:

```bash
cargo +nightly fuzz run --locked fuzz_parquet_predecode_budget
```

The seed corpus in `fuzz/corpus/fuzz_parquet_predecode_budget/` includes a
two-row, two-row-group Parquet file with a 20,000,000-value error list. The
target writes each fuzz input to a temporary file and sends it through the real
1 MB bounded Parquet reader.
