#![no_main]

use libfuzzer_sys::fuzz_target;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_grouped_by_neuron_bounded,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct InputFile(std::path::PathBuf);

impl Drop for InputFile {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            panic!("failed to remove fuzz input {}: {error}", self.0.display());
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("neat-parquet-fuzz-{}-{id}.parquet", std::process::id()));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let _input = InputFile(path.clone());
    // SAFETY: libFuzzer invokes this target synchronously in one process.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "1") };
    let _ = read_all_records_grouped_by_neuron_bounded(
        path.to_str().expect("temporary path is UTF-8"),
        None,
        ColumnProfile::Full,
        None,
    );
});
