//! Fuzzes Parquet footer parsing with bounded input files.
#![no_main]
#[path = "../footer_input.rs"]
mod footer_input;
use libfuzzer_sys::fuzz_target;
use neat_ai_discovery::parquet_format::footer::read_footer_stats;

fuzz_target!(|data: &[u8]| {
    footer_input::with_input(&std::env::temp_dir(), data, |path| {
        let _ = read_footer_stats(path.to_str().expect("temporary path is UTF-8"));
    })
    .expect("process bounded fuzz input");
});
