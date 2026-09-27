//! Issue #2168 / #2219: verbose log lines truncated caller-supplied UUIDs with a
//! byte-index slice (`&uuid[..12.min(uuid.len())]`), which panics when byte 12
//! falls inside a multi-byte character. A panic in a rayon worker that unwinds
//! past `extern "C"` aborts the host (CWE-248).
//!
//! This is its own test binary because `NEAT_AI_DISCOVERY_VERBOSE` is cached
//! in a `OnceLock` on first read, and the tracing subscriber is process-global.

use std::collections::HashMap;
use std::path::Path;

use neat_ai_discovery::CandidateSynapseJson;
use neat_ai_discovery::analysis::scoring::calibration_correction::CalibrationCorrection;
use neat_ai_discovery::analysis::synapse::post_processing::apply_impact_to_helpful;
use neat_ai_discovery::analysis::utils::char_prefix;

/// Eleven ASCII bytes then `é` (2 bytes), so byte 12 is mid-character.
const MULTIBYTE_UUID: &str = "aaaaaaaaaaaé-rest";

#[test]
fn char_prefix_keeps_a_multibyte_char_straddling_byte_twelve() {
    let prefix = char_prefix(MULTIBYTE_UUID, 12);
    assert_eq!(prefix, "aaaaaaaaaaaé");
    assert_eq!(prefix.len(), 13);
}

#[test]
fn char_prefix_takes_first_twelve_ascii_chars() {
    assert_eq!(char_prefix("0123456789abcdef-uuid", 12), "0123456789ab");
    assert_eq!(char_prefix("0123456789ab", 12), "0123456789ab");
}

#[test]
fn char_prefix_returns_short_string_whole() {
    assert_eq!(char_prefix("short", 12), "short");
}

#[test]
fn char_prefix_of_empty_is_empty() {
    assert_eq!(char_prefix("", 12), "");
}

#[test]
fn char_prefix_counts_emoji_as_single_chars() {
    let emoji = "😀".repeat(20);
    assert_eq!(char_prefix(&emoji, 12), "😀".repeat(12));
}

/// Reproduces #2168: with verbose logging on and a TRACE subscriber enabling
/// the callsite, the log fields are evaluated and the old byte slice panics.
#[test]
fn verbose_impact_log_does_not_panic_on_multibyte_uuid() {
    // SAFETY: set before any library call reads the cached flag; no other
    // thread in this binary touches the environment.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_VERBOSE", "1") };
    assert!(neat_ai_discovery::analysis::utils::verbose_enabled());

    // Without an enabled subscriber `tracing` never evaluates the fields,
    // so the buggy slice would never run and this test would pass vacuously.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(std::io::sink)
        .init();

    let mut candidate = CandidateSynapseJson {
        from_neuron_uuid: "input-0".to_string(),
        to_neuron_uuid: MULTIBYTE_UUID.to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight: 0.5,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.01,
        improved_count: 5,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.005, 0.015],
        comment: None,
        variant_key: None,
    };
    let neuron_type_map: HashMap<&str, &str> = HashMap::from([(MULTIBYTE_UUID, "hidden")]);

    apply_impact_to_helpful(
        &mut candidate,
        &HashMap::new(),
        &neuron_type_map,
        &HashMap::new(),
        &HashMap::new(),
        1.0,
        1.0,
        &CalibrationCorrection::neutral(),
    );

    assert!(candidate.expected_creature_score_gain.is_finite());
}

/// True when `line` contains `[..` + optional whitespace + digits + optional
/// whitespace + `.min(` — the regex `\[\.\.\s*\d+\s*\.min\(` without a regex dependency.
fn has_byte_min_slice(line: &str) -> bool {
    line.match_indices("[..").any(|(start, _)| {
        let rest = line[start + 3..].trim_start();
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        digits > 0 && rest[digits..].trim_start().starts_with(".min(")
    })
}

fn scan_dir(dir: &Path, hits: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            scan_dir(&path, hits);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let text =
                std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            for (n, line) in text.lines().enumerate() {
                if has_byte_min_slice(line) {
                    hits.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
    }
}

#[test]
fn byte_min_slice_matcher_matches_the_regex_shape() {
    assert!(has_byte_min_slice("&uuid[..12.min(uuid.len())]"));
    assert!(has_byte_min_slice("&s[.. 8 .min(s.len())]"));
    assert!(!has_byte_min_slice("&s[..n.min(s.len())]"));
    assert!(!has_byte_min_slice("&s[..12]"));
    assert!(!has_byte_min_slice("char_prefix(&s, 12)"));
}

#[test]
fn no_byte_index_min_slices_remain_in_src() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut hits = Vec::new();
    scan_dir(&src, &mut hits);
    assert!(
        hits.is_empty(),
        "byte-index `[..N.min(..)]` slices can split a UTF-8 char; use \
         analysis::utils::char_prefix instead:\n{}",
        hits.join("\n")
    );
}
