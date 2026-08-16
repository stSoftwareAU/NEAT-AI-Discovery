//! Issue #1987 — `docs/CACHE_TUNING.md`, `docs/GPU_GUIDE.md` and the
//! `docs/CONFIGURATION.md` § Streaming & Parquet rows documented a
//! cache/streaming subsystem that the production analysis path never reaches.
//!
//! An operator mid-OOM-incident who sets `NEAT_AI_DISCOVERY_PRELOAD_ALL=1` or
//! `_MAX_CACHED_BLOCKS` observes nothing change — the dead-lever failure mode
//! AGENTS.md § "Dead Levers" records from Issues #1792/#1793/#1818.
//!
//! Each test below first proves the *current* behaviour by calling the real
//! code, then asserts the prose agrees with what that behaviour demonstrated.

use neat_ai_discovery::analysis::cache::{
    CacheLazyReason, CachePreloadMode, LoadingStrategy, RecordCache,
    decide_cache_preload_for_available_memory, decide_cache_preload_for_budget,
    is_streaming_enabled, select_loading_strategy,
};
use neat_ai_discovery::analysis::streaming::adaptive_block_size;
use neat_ai_discovery::analysis::utils::{
    MemoryPressure, categorise_memory_pressure, would_cancel_for_memory_pressure,
};
use neat_ai_discovery::config::{DEFAULT_BLOCK_SIZE, block_size};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use serial_test::serial;
use std::sync::{Arc, Mutex};

const CACHE_TUNING: &str = include_str!("../docs/CACHE_TUNING.md");
const GPU_GUIDE: &str = include_str!("../docs/GPU_GUIDE.md");
const CONFIGURATION: &str = include_str!("../docs/CONFIGURATION.md");

const MB: u64 = 1024 * 1024;
const GB: u64 = 1024 * MB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Text of the markdown section introduced by `heading`, up to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("doc must contain the heading {heading:?}"));
    let level = heading.chars().filter(|c| *c == '#').count();
    let body = &doc[start + heading.len()..];
    body.match_indices("\n#")
        .find(|(idx, _)| body[idx + 1..].chars().take_while(|c| *c == '#').count() <= level)
        .map_or(body, |(idx, _)| &body[..idx])
}

/// Everything above the test/bench-only appendix — the part an operator is
/// expected to act on.
fn operator_facing(doc: &str) -> &str {
    let appendix = doc
        .find("\n## Appendix")
        .expect("CACHE_TUNING.md must carry the tiered-cache appendix");
    &doc[..appendix]
}

/// RAII guard that restores an environment variable on drop.
struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: these tests are `#[serial]`, so no other thread is reading
        // the environment while it is mutated.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: as above.
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: as above.
        unsafe {
            match self.previous.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[derive(Clone, Default)]
struct LogCapture(Arc<Mutex<Vec<u8>>>);

impl LogCapture {
    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("log buffer must not be poisoned")).into()
    }
}

impl std::io::Write for LogCapture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer must not be poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Run `body` with tracing captured at INFO or above, returning the log text.
fn capture_logs<T>(body: impl FnOnce() -> T) -> (T, String) {
    let capture = LogCapture::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(capture.clone())
        .with_max_level(tracing::Level::INFO)
        .finish();
    let result = tracing::subscriber::with_default(subscriber, body);
    (result, capture.contents())
}

/// A small on-disk parquet file the production cache constructor can read.
fn test_parquet() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("records.parquet");
    let file = path.to_str().expect("utf-8 path").to_string();
    let records: Vec<DiscoverRecord> = (0..4)
        .flat_map(|n| {
            (0..25).map(move |obs| {
                DiscoverRecord::new(obs, format!("neuron-{n}"), Some(0.5), 0.5, vec![0.25])
            })
        })
        .collect();
    write_records_to_parquet(&file, &records).expect("write parquet fixture");
    (dir, file)
}

// ---------------------------------------------------------------------------
// 1. Production has eager / lazy / skip — not three LRU tiers.
// ---------------------------------------------------------------------------

/// The production analysis cache chooses eager, lazy, or unworkable-skip, so
/// the operator-facing guide must describe those outcomes — not three LRU
/// tiers.
#[test]
fn production_cache_selection_is_binary_and_the_guide_says_so() {
    // Budget path: fits → eager, modest overbook → lazy, >10× → skip (Issue #2013).
    assert_eq!(
        decide_cache_preload_for_budget(2 * GB, 4096),
        (CachePreloadMode::Preload, CacheLazyReason::None),
        "a 2 GB projection under a 4 GB budget must pre-load eagerly"
    );
    assert_eq!(
        decide_cache_preload_for_budget(8 * GB, 4096),
        (CachePreloadMode::Lazy, CacheLazyReason::Budget),
        "an 8 GB projection over a 4 GB budget must fall back to lazy"
    );
    assert_eq!(
        decide_cache_preload_for_budget(50 * GB, 4096),
        (
            CachePreloadMode::SkipUnworkable,
            CacheLazyReason::Unworkable
        ),
        "a 50 GB projection over a 4 GB budget must skip as unworkable"
    );

    // Auto-detect path: eager vs lazy only (no budget → no 10× skip).
    assert_eq!(
        decide_cache_preload_for_available_memory(GB, 8 * GB, GB),
        (CachePreloadMode::Preload, CacheLazyReason::None),
        "a 1 GB projection with 8 GB available (1 GB margin) must pre-load"
    );
    assert_eq!(
        decide_cache_preload_for_available_memory(8 * GB, 8 * GB, GB),
        (CachePreloadMode::Lazy, CacheLazyReason::MemoryPressure),
        "an 8 GB projection with 8 GB available (1 GB margin) must go lazy"
    );

    let live = operator_facing(CACHE_TUNING);
    assert!(
        live.contains("eager pre-load") && live.contains("lazy") && live.contains("unworkable"),
        "the operator-facing guide must name eager, lazy, and unworkable skip"
    );
    assert!(
        !live.contains("Tier 2 — LRU Cache"),
        "the operator-facing guide must not present an LRU tier as tunable — \
         `RecordCache::new_tiered` has no production caller"
    );
    assert!(
        !live.contains("automatically selects one of\nthree tiers")
            && !live.contains("automatically selects one of three tiers"),
        "the overview must not claim production selects one of three tiers"
    );
}

/// The tiered cache still compiles and still selects three strategies — it is
/// simply never constructed by `analyze_parallel`, so the guide keeps it in an
/// appendix that says exactly that.
#[test]
fn the_tiered_model_survives_only_as_a_test_and_bench_appendix() {
    // The strategy selector is real code with real behaviour.
    assert!(
        matches!(
            select_loading_strategy(GB, 8 * GB),
            LoadingStrategy::LruCache { .. }
        ),
        "select_loading_strategy still yields three strategies"
    );

    let appendix = section(CACHE_TUNING, "## Appendix");
    let lower = appendix.to_lowercase();
    assert!(
        lower.contains("test") && lower.contains("bench"),
        "the appendix must be marked test/bench-only: {appendix}"
    );
    assert!(
        appendix.contains("new_tiered"),
        "the appendix must name the uncalled constructor so the claim is checkable"
    );
    // Issue #1939's accurate statement survives the move into the appendix.
    assert!(
        CACHE_TUNING.contains("There is no environment variable to force LRU or Streaming"),
        "the accurate statement about forcing tiers must be kept"
    );
}

// ---------------------------------------------------------------------------
// 2. The four streaming knobs are parsed but never consumed.
// ---------------------------------------------------------------------------

/// `NEAT_AI_DISCOVERY_PRELOAD_ALL` parses, but the production pre-load decision
/// is computed from the projection and the budget alone — the knob changes
/// nothing in either direction.
#[test]
#[serial]
fn preload_all_parses_but_does_not_move_the_production_decision() {
    let _guard = EnvGuard::set("NEAT_AI_DISCOVERY_PRELOAD_ALL", "1");

    assert!(
        !is_streaming_enabled(),
        "the knob parses — `is_streaming_enabled` flips with it"
    );

    // ...yet the decision the production cache actually makes is unchanged.
    assert_eq!(
        decide_cache_preload_for_budget(8 * GB, 1024).0,
        CachePreloadMode::Lazy,
        "PRELOAD_ALL=1 must not force eager pre-load past the budget"
    );
    assert_eq!(
        decide_cache_preload_for_available_memory(8 * GB, 8 * GB, GB).0,
        CachePreloadMode::Lazy,
        "PRELOAD_ALL=1 must not force eager pre-load past available memory"
    );

    let live = operator_facing(CACHE_TUNING);
    assert!(
        !live.contains("bypasses"),
        "the guide must not offer PRELOAD_ALL as a way to bypass the selection"
    );
}

/// The canonical reference must state the reach of each unconsumed knob, since
/// both guides delegate to it.
#[test]
fn the_canonical_reference_annotates_the_unreached_streaming_rows() {
    let streaming = section(CONFIGURATION, "## Streaming & Parquet");
    for var in [
        "NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS",
        "NEAT_AI_DISCOVERY_PREFETCH_DEPTH",
        "NEAT_AI_DISCOVERY_PRELOAD_ALL",
        "NEAT_AI_DISCOVERY_BLOCK_SIZE",
    ] {
        let row = streaming
            .lines()
            .find(|l| l.contains(var))
            .unwrap_or_else(|| panic!("§ Streaming & Parquet must carry a row for {var}"));
        assert!(
            row.contains("not consumed by `analyze_parallel`"),
            "the {var} row must state that it never reaches analyze_parallel: {row}"
        );
    }

    // The live recording knob in the same table keeps its plain description.
    let ttl = streaming
        .lines()
        .find(|l| l.contains("NEAT_AI_DISCOVERY_SESSION_TTL_SECS"))
        .expect("§ Streaming & Parquet must keep the session TTL row");
    assert!(
        !ttl.contains("not consumed by `analyze_parallel`"),
        "the session TTL knob is live and must not be annotated as unreached"
    );
}

// ---------------------------------------------------------------------------
// 3. Block size is fixed in production.
// ---------------------------------------------------------------------------

/// `adaptive_block_size` is never consulted in production: the block size is the
/// fixed `DEFAULT_BLOCK_SIZE` whatever the host's memory.
#[test]
#[serial]
fn block_size_is_fixed_in_production_and_the_gpu_guide_says_so() {
    let _guard = EnvGuard::unset("NEAT_AI_DISCOVERY_BLOCK_SIZE");

    assert_eq!(
        block_size(),
        DEFAULT_BLOCK_SIZE,
        "with the knob unset the block size is the fixed default"
    );
    // The adaptive helper returns something else entirely for the same host.
    assert_ne!(
        adaptive_block_size(64 * GB),
        block_size(),
        "a 64 GB host would get a different adaptive block size — nothing applies it"
    );
    assert_ne!(
        adaptive_block_size(GB),
        block_size(),
        "a 1 GB host would get a different adaptive block size — nothing applies it"
    );

    let sizing = section(GPU_GUIDE, "#### 📏 Adaptive Block Sizing");
    assert!(
        !sizing.contains("automatically tuned based on available memory"),
        "the GPU guide must not claim block size is automatically tuned: {sizing}"
    );
    assert!(
        sizing.to_lowercase().contains("not wired"),
        "the adaptive block-sizing table must be marked as not wired: {sizing}"
    );
}

// ---------------------------------------------------------------------------
// 4. Memory pressure only acts at CRITICAL.
// ---------------------------------------------------------------------------

/// Memory pressure is detected, but the only band that changes behaviour is
/// `Critical`, which cancels in-flight analysis (Issue #1099). No band resizes a
/// cache, evicts harder, or shrinks a block.
#[test]
fn memory_pressure_only_acts_at_critical_and_the_table_matches() {
    let total = 32 * GB;
    for (available, expected, cancels) in [
        (16 * GB, MemoryPressure::None, false),
        (7 * GB, MemoryPressure::Moderate, false),
        (3 * GB, MemoryPressure::High, false),
        (GB, MemoryPressure::Critical, true),
    ] {
        assert_eq!(
            categorise_memory_pressure(available, total),
            expected,
            "{available} of {total} must categorise as {expected:?}"
        );
        assert_eq!(
            would_cancel_for_memory_pressure(available, total),
            cancels,
            "only Critical pressure changes behaviour ({expected:?})"
        );
    }

    let pressure = section(GPU_GUIDE, "#### 🌡️ Memory Pressure Detection");
    let lower = pressure.to_lowercase();
    for claim in [
        "reduced cache sizes",
        "aggressive eviction",
        "smaller blocks",
        "minimal caching",
    ] {
        assert!(
            !lower.contains(claim),
            "the pressure table must not promise \"{claim}\" — nothing consumes the level: {pressure}"
        );
    }
    assert!(
        lower.contains("cancel"),
        "the pressure table must state the one real consequence — cancellation at Critical"
    );
}

/// The compressed LRU cache is never selected by production, so the guide may
/// not claim automatic selection or an unbenchmarked "2x larger creatures".
// `#[serial]` with the log-capture test below: both drive the same tracing
// callsites in the cache constructor, and this one runs with no subscriber
// installed. Concurrently they race tracing's global callsite-interest cache,
// so the capture test can observe an empty log buffer.
#[test]
#[serial]
fn the_compressed_cache_is_not_auto_selected_and_the_claim_is_gone() {
    let (_dir, file) = test_parquet();
    // The production constructor returns a plain `RecordCache` regardless of
    // how tight the budget is — never a compressed or tiered cache.
    let cache = RecordCache::new_adaptive_with_deadline_and_budget(&file, None, Some(0))
        .expect("a zero budget must still yield a working lazy cache")
        .expect("zero budget forces lazy, not skip");
    assert!(
        !cache
            .get("neuron-0")
            .expect("lazy load must succeed")
            .is_empty(),
        "the lazy cache still serves records"
    );

    let compressed = section(GPU_GUIDE, "#### 🗜️ Compressed In-Memory Cache (LZ4)");
    assert!(
        !compressed.contains("selected automatically under memory pressure"),
        "the compressed cache is constructed only by tests and benches: {compressed}"
    );
    assert!(
        !compressed.contains("2x larger creatures"),
        "the unbenchmarked 2x figure must be dropped: {compressed}"
    );
    assert!(
        compressed.to_lowercase().contains("not wired"),
        "the compressed cache section must be marked as not wired: {compressed}"
    );
}

// ---------------------------------------------------------------------------
// 5. Troubleshooting points at log lines production actually emits.
// ---------------------------------------------------------------------------

/// The lazy fallback emits both a structured WARN and the lazy-mode INFO line.
/// The guide's diagnosis steps must quote those, not the tiered line that only
/// the uncalled `new_tiered` can emit.
#[test]
#[serial]
fn troubleshooting_quotes_the_log_lines_production_emits() {
    let (_dir, file) = test_parquet();
    let (cache, logs) =
        capture_logs(|| RecordCache::new_adaptive_with_deadline_and_budget(&file, None, Some(0)));
    cache
        .expect("a zero budget must still yield a working lazy cache")
        .expect("zero budget forces lazy, not skip");

    assert!(
        logs.contains("insufficient memory for pre-loading"),
        "the lazy fallback must emit the structured WARN: {logs}"
    );
    assert!(
        logs.contains("using lazy-loading mode for parquet file"),
        "the lazy path must emit its INFO line: {logs}"
    );
    assert!(
        !logs.contains("tiered loading strategy selected"),
        "production never emits the tiered line: {logs}"
    );

    let live = operator_facing(CACHE_TUNING);
    for emitted in [
        "insufficient memory for pre-loading",
        "using lazy-loading mode for parquet file",
    ] {
        assert!(
            live.contains(emitted),
            "the troubleshooting flow must quote the emitted log line {emitted:?}"
        );
    }
    assert!(
        !live.contains("tiered loading strategy selected"),
        "the operator-facing troubleshooting must not send operators after an \
         unreachable log line"
    );
    assert!(
        !live.contains("High cache miss rate in LRU mode"),
        "the LRU cache-miss symptom cannot occur in production"
    );
}
