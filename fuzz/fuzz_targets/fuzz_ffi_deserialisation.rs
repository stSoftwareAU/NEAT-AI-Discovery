//! Fuzz target for FFI JSON deserialisation.
//!
//! Feeds arbitrary byte strings to `serde_json::from_str` for every FFI input
//! type. The goal is to verify that no combination of bytes causes a panic
//! during deserialisation — all malformed inputs must produce `Err`, never a
//! crash.

#![no_main]

use libfuzzer_sys::fuzz_target;
use neat_ai_discovery::{
    AnalyzeParallelInput, AppendRecordsInput, CancelSessionInput, ExportVisualisationSnapshotInput,
    FinishSessionInput, MergeParquetInput, RankFocusNeuronsInput, ReadDiscoveryInput,
    RecordDiscoveryInput, StartSessionInput,
};

fuzz_target!(|data: &[u8]| {
    // Only valid UTF-8 can be JSON — skip non-UTF-8 quickly.
    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };

    // Attempt deserialisation of every FFI input type.
    // None of these should panic — they must return Ok or Err.
    let _ = serde_json::from_str::<RecordDiscoveryInput>(input);
    let _ = serde_json::from_str::<StartSessionInput>(input);
    let _ = serde_json::from_str::<AppendRecordsInput>(input);
    let _ = serde_json::from_str::<FinishSessionInput>(input);
    let _ = serde_json::from_str::<CancelSessionInput>(input);
    let _ = serde_json::from_str::<MergeParquetInput>(input);
    let _ = serde_json::from_str::<RankFocusNeuronsInput>(input);
    let _ = serde_json::from_str::<AnalyzeParallelInput>(input);
    let _ = serde_json::from_str::<ReadDiscoveryInput>(input);
    let _ = serde_json::from_str::<ExportVisualisationSnapshotInput>(input);
});
