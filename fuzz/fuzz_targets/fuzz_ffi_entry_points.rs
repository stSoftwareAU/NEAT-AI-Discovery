//! Fuzz target for FFI business-logic entry points.
//!
//! Feeds arbitrary byte strings directly to the `*_internal` functions that
//! sit behind the FFI boundary. These functions accept `&str` JSON and must
//! return `Ok(json_string)` or `Err` — never panic. This catches issues in
//! both the deserialisation layer *and* any downstream processing that runs
//! before deeper validation (e.g., file I/O) rejects the input.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Only valid UTF-8 can be JSON — skip non-UTF-8 quickly.
    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };

    // Exercise each internal entry point. All must return Ok or Err, never panic.
    let _ = neat_ai_discovery::record_discovery_internal(input);
    let _ = neat_ai_discovery::merge_discovery_parquet_internal(input);
    let _ = neat_ai_discovery::rank_focus_neurons_internal(input);
    let _ = neat_ai_discovery::export_visualisation_snapshot_internal(input);
    let _ = neat_ai_discovery::read_discovery_records(input);

    // analyze_parallel_internal is expensive (GPU) — skip in fuzz to keep throughput high.
    // Its JSON parsing is already covered by fuzz_ffi_deserialisation.
});
