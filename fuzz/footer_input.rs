//! Exclusive temporary input creation shared by the footer fuzzer and tests.
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAX_INPUT_BYTES: usize = 65_536;
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct PrivateDirectory(PathBuf);
impl Drop for PrivateDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn with_input(root: &Path, data: &[u8], parse: impl FnOnce(&Path)) -> io::Result<()> {
    if data.len() > MAX_INPUT_BYTES {
        return Ok(());
    }
    let directory = loop {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!("neat-ai-footer-fuzz-{}-{id}", std::process::id()));
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&path) {
            Ok(()) => break PrivateDirectory(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let path = directory.0.join("input.parquet");
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path)?;
    file.write_all(data)?;
    drop(file);
    parse(&path);
    Ok(())
}
