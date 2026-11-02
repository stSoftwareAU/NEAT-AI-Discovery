//! Fuzzes Parquet footer parsing with bounded input files.

#![no_main]

use libfuzzer_sys::fuzz_target;
use neat_ai_discovery::parquet_format::footer::read_footer_stats;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_INPUT_BYTES: usize = 65_536;
static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let file_id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "neat-ai-footer-fuzz-{}-{file_id}.parquet",
        std::process::id()
    ));
    std::fs::write(&path, data).expect("write fuzz input");
    let path_string = path.to_str().expect("temporary path is UTF-8");
    let _ = read_footer_stats(path_string);
    std::fs::remove_file(path).expect("remove fuzz input");
});
