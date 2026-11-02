#![no_main]

use libfuzzer_sys::fuzz_target;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_grouped_by_neuron_bounded,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct InputFile {
    directory: std::path::PathBuf,
    path: std::path::PathBuf,
}

impl Drop for InputFile {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            panic!(
                "failed to remove fuzz input {}: {error}",
                self.path.display()
            );
        }
        std::fs::remove_dir(&self.directory).unwrap_or_else(|error| {
            panic!(
                "failed to remove fuzz directory {}: {error}",
                self.directory.display()
            )
        });
    }
}

fuzz_target!(|data: &[u8]| {
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("neat-parquet-fuzz-{}-{id}", std::process::id()));
    std::fs::create_dir(&directory).unwrap_or_else(|error| {
        panic!(
            "failed to exclusively create fuzz directory {}: {error}",
            directory.display()
        )
    });
    let path = directory.join("input.parquet");
    let _input = InputFile {
        directory,
        path: path.clone(),
    };
    std::fs::write(&path, data)
        .unwrap_or_else(|error| panic!("failed to write fuzz input {}: {error}", path.display()));
    let _ = read_all_records_grouped_by_neuron_bounded(
        path.to_str().expect("temporary path is UTF-8"),
        None,
        ColumnProfile::Full,
        Some(1),
    );
});
