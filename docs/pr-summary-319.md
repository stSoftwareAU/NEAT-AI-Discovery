## Summary

Fixed the "Gnuplot not found, using plotters backend" warning that appeared during benchmark builds.

The criterion crate's default features include `plotters` which tries to use gnuplot for generating HTML reports. When gnuplot is not installed, it prints a warning message that can be confusing during CI builds. This change disables the plotters feature by using `default-features = false` and explicitly enabling only the features needed:
- `rayon` - for parallel benchmark execution
- `cargo_bench_support` - for cargo bench integration

## Evidence

Unable to generate screenshot: This is a CLI tool. The fix was verified by running benchmarks before and after the change:

**Before (with plotters feature):**
```
Running benches/synapse_counts.rs
Gnuplot not found, using plotters backend
Benchmarking synapse_counts/old_O(n×m)/100n_500s
```

**After (without plotters feature):**
```
Running benches/synapse_counts.rs
Benchmarking synapse_counts/old_O(n×m)/100n_500s
```

The warning is no longer present and all benchmarks run successfully.

## Test Plan

- Ran `./quality.sh` which executes all 379 unit tests - all passed
- Ran `cargo bench --bench synapse_counts -- --quick` to verify benchmarks work without the gnuplot warning
- All existing tests continue to pass (no functionality was changed, only a dev-dependency configuration)
